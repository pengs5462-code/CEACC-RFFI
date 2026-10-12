"""Small 1-D building blocks required by the NetTime backbone."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.init import normal_


def ifnone(value, default):
    return default if value is None else value


class Add(nn.Module):
    def forward(self, x, y):
        return x.add(y)


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *values):
        return torch.cat(*values, dim=self.dim)


class Flatten(nn.Module):
    def forward(self, value):
        return value.view(value.size(0), -1)


class GAP1d(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Flatten()

    def forward(self, value):
        return self.flatten(self.gap(value))


class Conv1dNewPadding(nn.Module):
    def __init__(self, ni, nf, ks=None, stride=1, bias=False, pad_zero=True):
        super().__init__()
        self.ks = 3 if ks is None else ks
        if pad_zero:
            padding = (int((self.ks - 1) / 2), int(self.ks / 2))
            self.padding = nn.ConstantPad1d(padding, 0)
        else:
            self.padding = nn.ReplicationPad1d((0, int(self.ks - 1)))
        self.conv1d = nn.Conv1d(ni, nf, self.ks, stride=stride, bias=bias)

    def forward(self, value):
        return self.conv1d(self.padding(value))


# Preserve the historical class name because checkpoint keys and old runners use it.
Conv1d_new_padding = Conv1dNewPadding


class ConvBlock(nn.Module):
    def __init__(self, ni, nf, kernel_size=None, stride=1, act=None, pad_zero=True):
        super().__init__()
        kernel_size = 3 if kernel_size is None else kernel_size
        self.layer_list = []
        self.conv = Conv1d_new_padding(
            ni, nf, ks=kernel_size, stride=stride, pad_zero=pad_zero
        )
        self.bn = nn.BatchNorm1d(nf)
        self.layer_list += [self.conv, self.bn]
        if act is not None:
            self.layer_list.append(act)
        self.net = nn.Sequential(*self.layer_list)

    def forward(self, value):
        return self.net(value)


class SeparableConv1d(nn.Module):
    def __init__(
        self, ni, nf, ks, stride=1, padding="same", dilation=1, bias=True, bias_std=0.01
    ):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            ni, ni, ks, stride=stride, padding=padding, dilation=dilation,
            groups=ni, bias=bias
        )
        self.pointwise_conv = nn.Conv1d(ni, nf, 1, bias=bias)
        if bias:
            if bias_std:
                normal_(self.depthwise_conv.bias, 0, bias_std)
                normal_(self.pointwise_conv.bias, 0, bias_std)
            else:
                self.depthwise_conv.bias.data.zero_()
                self.pointwise_conv.bias.data.zero_()

    def forward(self, value):
        return self.pointwise_conv(self.depthwise_conv(value))
