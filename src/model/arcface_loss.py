"""Additive angular-margin classifier used by the Stage-A encoder."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.3, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, value, label):
        value = F.normalize(value, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(value, weight).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=1e-4))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
