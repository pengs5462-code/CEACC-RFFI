"""NetTime backbone used for the matched PLSAR experiments.

The public implementation keeps the module hierarchy of the experiment
checkpoints. PLSAR replaces ``fexp`` with ``nn.Identity`` before training.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor, nn
import torch.nn.functional as F

from model.layers import (
    Add,
    Concat,
    Conv1d_new_padding,
    ConvBlock,
    GAP1d,
    SeparableConv1d,
    ifnone,
)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, value: Tensor, mask: Tensor | None = None) -> Tensor:
        queries = rearrange(self.queries(value), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(value), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(value), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd,bhkd->bhqk", queries, keys)
        if mask is not None:
            energy.masked_fill_(~mask, torch.finfo(torch.float32).min)
        attention = F.softmax(energy / self.emb_size**0.5, dim=-1)
        attention = self.att_drop(attention)
        output = torch.einsum("bhal,bhlv->bhav", attention, values)
        return self.projection(rearrange(output, "b h n d -> b n (h d)"))


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, value, **kwargs):
        return self.fn(value, **kwargs) + value


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=4, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, forward_expansion, forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )


class TransformerEncoder(nn.Module):
    def __init__(self, dim=512, depth=3):
        super().__init__()
        self.transformer_blocks = nn.Sequential(
            *[TransformerEncoderBlock(dim) for _ in range(depth)]
        )

    def forward(self, value):
        return self.transformer_blocks(value.permute(0, 2, 1)).permute(0, 2, 1)


class ShallowNet(nn.Module):
    def __init__(self, ni, nf, residual=True, ks=40, bottleneck=True):
        super().__init__()
        self.residual = residual
        self.stages = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        previous = ni
        residual_channels = ni
        for index in range(4):
            output_channels = nf * 2**index
            kernels = [max(ks // 2**j | 1, 1) for j in range(3)]
            stage = nn.ModuleDict({
                "bottleneck": Conv1d_new_padding(previous, output_channels, 1, bias=False)
                if bottleneck else nn.Identity(),
                "convs": nn.ModuleList([
                    SeparableConv1d(
                        output_channels if bottleneck else previous,
                        output_channels,
                        kernel,
                        bias=False,
                    ) for kernel in kernels
                ]),
                "maxconvpool": nn.Sequential(
                    nn.MaxPool1d(3, stride=1, padding=1),
                    Conv1d_new_padding(previous, output_channels, 1, bias=False),
                ),
                "concat": Concat(),
            })
            self.stages.append(stage)
            stage_channels = output_channels * (len(kernels) + 1)
            if self.residual and (index - 1) % 2 == 0:
                shortcut = nn.Sequential(
                    nn.BatchNorm1d(residual_channels)
                    if residual_channels == stage_channels
                    else ConvBlock(residual_channels, stage_channels, 1, act=None)
                )
                self.shortcuts.append(shortcut)
                residual_channels = stage_channels
            else:
                self.shortcuts.append(None)
            previous = stage_channels
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, value):
        residual = value
        shortcut_index = 0
        for index, stage in enumerate(self.stages):
            stage_input = value
            transformed = stage["bottleneck"](stage_input)
            branches = [conv(transformed) for conv in stage["convs"]]
            value = stage["concat"](branches + [stage["maxconvpool"](stage_input)])
            if self.residual and (index - 1) % 2 == 0 and self.shortcuts[shortcut_index]:
                value = self.act(self.add(value, self.shortcuts[shortcut_index](residual)))
                residual = value
                shortcut_index += 1
        return value


class NetTime(nn.Module):
    def __init__(self, c_in=2, c_out=10, nf=16, nb_filters=None, adaptive_size=50, **kwargs):
        super().__init__()
        nf = ifnone(nf, nb_filters)
        self.block = ShallowNet(c_in, nf, **kwargs)
        self.head_nf = nf * 32
        self.axformer = TransformerEncoder(dim=self.head_nf)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(adaptive_size),
            ConvBlock(self.head_nf, self.head_nf // 2, 1),
            ConvBlock(self.head_nf // 2, self.head_nf // 4, 1),
            ConvBlock(self.head_nf // 4, c_out, 1),
            GAP1d(1),
        )
        self.contrast_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.head_nf, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        self.pool = nn.AdaptiveAvgPool1d(512)
        self.fexp = nn.Identity()

    def forward(self, value):
        embedding = self.axformer(self.pool(self.block(value)))
        embedding = self.fexp(embedding)
        return self.contrast_head(embedding), self.head(embedding)
