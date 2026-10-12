"""Correct, auditable FRAME building blocks for the isolated ``~/gp/pro`` study.

The classes here intentionally do not depend on the legacy implementation.  Every
tensor operation used by the frequency split, sparse router, and fusion gate is
differentiable and exposes its diagnostics to the experiment runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class FRAEDiagnostics:
    rho: Tensor
    low_mask: Tensor
    low_router: Tensor | None = None
    high_router: Tensor | None = None
    fusion_weights: Tensor | None = None
    aux_loss: Tensor | None = None


class FeatureFrequencyDecomposer(nn.Module):
    """Differentiable feature-frequency split along the final tensor dimension.

    ``rho`` is the fraction of the shifted one-sided half-spectrum included in the
    low-frequency band.  Thus ``rho=0`` keeps only DC in the hard-mask limit and
    ``rho=1`` covers the full shifted spectrum.  Restricting it to
    ``[rho_min, rho_max]`` avoids degenerate empty/full branches.
    """

    def __init__(
        self,
        channels: int,
        rho_min: float = 0.05,
        rho_max: float = 0.50,
        mask_temperature: float = 0.5,
        fixed_rho: float | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= rho_min < rho_max <= 1.0:
            raise ValueError("expected 0 <= rho_min < rho_max <= 1")
        if mask_temperature <= 0:
            raise ValueError("mask_temperature must be positive")
        if fixed_rho is not None and not rho_min <= fixed_rho <= rho_max:
            raise ValueError("fixed_rho must lie in [rho_min, rho_max]")
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.mask_temperature = float(mask_temperature)
        self.fixed_rho = fixed_rho
        self.project = nn.Conv1d(channels, channels, 1)
        self.bandwidth_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, max(channels // 8, 8), 1),
            nn.GELU(),
            nn.Conv1d(max(channels // 8, 8), 1, 1),
        )

    def predict_rho(self, x: Tensor) -> Tensor:
        if self.fixed_rho is not None:
            return x.new_full((x.shape[0], 1, 1), self.fixed_rho)
        raw = torch.sigmoid(self.bandwidth_predictor(x))
        return self.rho_min + (self.rho_max - self.rho_min) * raw

    def build_mask(self, rho: Tensor, length: int, dtype: torch.dtype) -> Tensor:
        center = length // 2
        bins = torch.arange(length, device=rho.device, dtype=dtype)
        distance = torch.abs(bins - float(center)).view(1, 1, length)
        half_width = rho * max(length / 2.0, 1.0)
        return torch.sigmoid(
            (half_width - (distance + 0.5)) / self.mask_temperature
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        projected = self.project(x)
        rho = self.predict_rho(projected)
        spectrum = torch.fft.fftshift(
            torch.fft.fft(projected, dim=-1, norm="ortho"), dim=-1
        )
        low_mask = self.build_mask(rho, projected.shape[-1], projected.dtype)
        high_mask = 1.0 - low_mask
        low = torch.fft.ifft(
            torch.fft.ifftshift(spectrum * low_mask, dim=-1),
            dim=-1,
            norm="ortho",
        ).real
        high = torch.fft.ifft(
            torch.fft.ifftshift(spectrum * high_mask, dim=-1),
            dim=-1,
            norm="ortho",
        ).real
        return low, high, rho, low_mask, projected


class ECA(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.pool(x).transpose(1, 2)
        weights = torch.sigmoid(self.conv(weights)).transpose(1, 2)
        return x * weights


class MultiScaleExpert(nn.Module):
    def __init__(self, channels: int, eca_kernel: int, dilation: int) -> None:
        super().__init__()
        self.channel_attention = ECA(eca_kernel)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        delta = self.channel_attention(x)
        delta = F.gelu(self.norm(self.pointwise(self.depthwise(delta))))
        return self.gamma * delta


class SparseRouter(nn.Module):
    """Top-k router with selected-weight renormalization and load balancing.

    Let p_be be the dense softmax probability and q_be the selected indicator
    divided by k.  The auxiliary loss is

        L_balance = E * sum_e mean_b(p_be) * mean_b(q_be)
        L_entropy = sum_e pbar_e * log(E * pbar_e)
        L_aux = L_balance + entropy_weight * L_entropy.

    Both terms are minimized by balanced batch-level expert usage.
    """

    def __init__(
        self,
        channels: int,
        num_experts: int = 4,
        topk: int = 2,
        temperature: float = 1.0,
        entropy_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if not 1 <= topk <= num_experts:
            raise ValueError("expected 1 <= topk <= num_experts")
        self.num_experts = num_experts
        self.topk = topk
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_experts, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        logits = self.fc(self.pool(x).squeeze(-1))
        dense = F.softmax(logits / self.temperature, dim=-1)
        _, indices = torch.topk(dense, self.topk, dim=-1)
        selected = torch.zeros_like(dense).scatter_(1, indices, 1.0)
        sparse = dense * selected
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        importance = dense.mean(dim=0)
        load = (selected / self.topk).mean(dim=0)
        balance = self.num_experts * torch.sum(importance * load)
        entropy = torch.sum(
            importance * torch.log((self.num_experts * importance).clamp_min(1e-12))
        )
        aux = balance + self.entropy_weight * entropy
        return sparse, logits, aux


class ExpertBank(nn.Module):
    def __init__(self, channels: int, num_experts: int = 4, topk: int = 2) -> None:
        super().__init__()
        configs = [(3, 1), (5, 2), (7, 3), (9, 4)]
        if num_experts != len(configs):
            raise ValueError("the auditable expert bank currently requires four experts")
        self.experts = nn.ModuleList(
            MultiScaleExpert(channels, kernel, dilation)
            for kernel, dilation in configs
        )
        self.router = SparseRouter(channels, num_experts, topk)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        routed_input = self.norm(x.transpose(1, 2)).transpose(1, 2)
        weights, _, aux = self.router(routed_input)
        # Dispatch only selected samples. Computing every expert followed by a
        # multiplication by zero is sparse weighting, not sparse computation, and
        # cannot support a Top-k FLOPs/latency claim.
        update = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            selected = weights[:, expert_id] > 0
            if not selected.any():
                continue
            delta = expert(routed_input[selected])
            update[selected] = update[selected] + (
                weights[selected, expert_id, None, None] * delta
            )
        return x + update, weights, aux


class FusionGate(nn.Module):
    """Sample-wise three-source convex fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(3 * channels, 3, 1)
        nn.init.zeros_(self.fc.weight)
        with torch.no_grad():
            self.fc.bias.copy_(torch.tensor([2.0, 0.0, 0.0]))

    def forward(
        self, backbone: Tensor, low: Tensor, high: Tensor
    ) -> Tuple[Tensor, Tensor]:
        context = torch.cat(
            [self.pool(backbone), self.pool(low), self.pool(high)], dim=1
        )
        weights = F.softmax(self.fc(context).squeeze(-1), dim=-1)
        fused = (
            weights[:, 0, None, None] * backbone
            + weights[:, 1, None, None] * low
            + weights[:, 2, None, None] * high
        )
        return fused, weights


class CorrectFRAE(nn.Module):
    """Correct FRAE implementation with explicit experimental modes."""

    MODES = {"full", "bypass", "fd_only", "mee_only", "no_fusion"}

    def __init__(
        self,
        channels: int,
        mode: str = "full",
        num_experts: int = 4,
        topk: int = 2,
        fixed_rho: float | None = None,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.fd = FeatureFrequencyDecomposer(channels, fixed_rho=fixed_rho)
        self.low_bank = ExpertBank(channels, num_experts, topk)
        self.high_bank = ExpertBank(channels, num_experts, topk)
        self.mee_bank = ExpertBank(channels, num_experts, topk)
        self.fusion = FusionGate(channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, FRAEDiagnostics]:
        if self.mode == "bypass":
            empty = x.new_empty(0)
            return x, FRAEDiagnostics(empty, empty)
        if self.mode == "mee_only":
            out, weights, aux = self.mee_bank(x)
            empty = x.new_empty(0)
            return out, FRAEDiagnostics(
                empty, empty, low_router=weights, aux_loss=aux
            )

        low, high, rho, mask, _ = self.fd(x)
        if self.mode == "fd_only":
            out, fusion_weights = self.fusion(x, low, high)
            return out, FRAEDiagnostics(rho, mask, fusion_weights=fusion_weights)

        low_hat, low_weights, low_aux = self.low_bank(low)
        high_hat, high_weights, high_aux = self.high_bank(high)
        aux = 0.5 * (low_aux + high_aux)
        if self.mode == "no_fusion":
            out = x + low_hat + high_hat
            fusion_weights = None
        else:
            out, fusion_weights = self.fusion(x, low_hat, high_hat)
        return out, FRAEDiagnostics(
            rho,
            mask,
            low_router=low_weights,
            high_router=high_weights,
            fusion_weights=fusion_weights,
            aux_loss=aux,
        )


class NetFRAEAdapter(nn.Module):
    """Adapts ``CorrectFRAE`` to legacy backbones that expect one tensor output."""

    def __init__(self, frae: CorrectFRAE) -> None:
        super().__init__()
        self.frae = frae
        self.last_diagnostics: FRAEDiagnostics | None = None

    def forward(self, x: Tensor) -> Tensor:
        output, diagnostics = self.frae(x)
        self.last_diagnostics = diagnostics
        return output


def collect_router_aux(diagnostics: FRAEDiagnostics) -> Tensor:
    if diagnostics.aux_loss is None:
        if diagnostics.rho.numel():
            return diagnostics.rho.new_zeros(())
        return torch.tensor(0.0)
    return diagnostics.aux_loss
