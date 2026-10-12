"""Correct, lightweight spectrum branch for time/frequency representation tests."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from model.MRAN import NetTime


class LogSpectrumEncoder(nn.Module):
    """Encode the centered complex FFT magnitude without discarding I/Q pre-FFT."""

    def __init__(self, bins: int = 64, embedding_dim: int = 64) -> None:
        super().__init__()
        self.bins = bins
        self.net = nn.Sequential(
            nn.LayerNorm(bins),
            nn.Linear(bins, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"expected [batch, 2, time], got {tuple(x.shape)}")
        iq = torch.complex(x[:, 0], x[:, 1])
        spectrum = torch.fft.fftshift(
            torch.fft.fft(iq, dim=-1, norm="ortho"), dim=-1
        )
        log_magnitude = torch.log1p(spectrum.abs())
        # Retain spectral shape while preventing a single high-energy bin from
        # dominating the small branch. Input dc/RMS normalization is performed
        # by the common model wrapper in build_model.
        log_magnitude = (
            log_magnitude - log_magnitude.mean(dim=-1, keepdim=True)
        ) / log_magnitude.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        pooled = F.adaptive_avg_pool1d(log_magnitude.unsqueeze(1), self.bins).squeeze(1)
        return self.net(pooled)


def log_spectrum_descriptor(x: torch.Tensor, bins: int = 64) -> torch.Tensor:
    """Deterministic standardized log-magnitude descriptor used as an aux target."""
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"expected [batch, 2, time], got {tuple(x.shape)}")
    x = x - x.mean(dim=-1, keepdim=True)
    rms = x.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    x = x / rms
    iq = torch.complex(x[:, 0], x[:, 1])
    spectrum = torch.fft.fftshift(
        torch.fft.fft(iq, dim=-1, norm="ortho"), dim=-1
    )
    descriptor = torch.log1p(spectrum.abs())
    descriptor = (
        descriptor - descriptor.mean(dim=-1, keepdim=True)
    ) / descriptor.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return F.adaptive_avg_pool1d(descriptor.unsqueeze(1), bins).squeeze(1)


class SpectralOnlyNet(nn.Module):
    def __init__(self, classes: int = 9, bins: int = 64) -> None:
        super().__init__()
        self.spectrum = LogSpectrumEncoder(bins=bins, embedding_dim=64)
        self.classifier = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor):
        embedding = self.spectrum(x)
        return embedding, self.classifier(embedding)


class SpectralFusionNet(nn.Module):
    """Fuse the established time embedding with a direct log-spectrum residual."""

    def __init__(self, classes: int = 9, bins: int = 64) -> None:
        super().__init__()
        self.time_model = NetTime(c_in=2, c_out=classes, nf=16)
        self.time_model.fexp = nn.Identity()
        self.spectrum = LogSpectrumEncoder(bins=bins, embedding_dim=64)
        self.time_norm = nn.LayerNorm(64)
        self.spectrum_norm = nn.LayerNorm(64)
        self.gate = nn.Sequential(nn.Linear(128, 64), nn.Sigmoid())
        self.fusion_norm = nn.LayerNorm(64)
        self.classifier = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor):
        time_embedding, _ = self.time_model(x)
        spectral_embedding = self.spectrum(x)
        time_feature = self.time_norm(time_embedding)
        spectral_feature = self.spectrum_norm(spectral_embedding)
        gate = self.gate(torch.cat([time_feature, spectral_feature], dim=1))
        embedding = self.fusion_norm(time_feature + gate * spectral_feature)
        return embedding, self.classifier(embedding)


class SpectralAuxNet(nn.Module):
    """Time backbone with a training-only head that reconstructs spectral shape."""

    def __init__(self, classes: int = 9, bins: int = 64) -> None:
        super().__init__()
        self.time_model = NetTime(c_in=2, c_out=classes, nf=16)
        self.time_model.fexp = nn.Identity()
        self.spectral_head = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, bins),
        )

    def forward(self, x: torch.Tensor):
        return self.time_model(x)

    def predict_spectrum(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.spectral_head(embedding)
