"""
https://github.com/Kright/mySmallProjects/blob/master/2023/ml_experiments/superresolution/models/downsampler.py
"""

import torch
import torch.nn as nn


class DownSampler(nn.Module):
    def __init__(self, scale: int, gamma: float = 2.2):
        super(DownSampler, self).__init__()
        self.gamma: float = gamma
        self.avg_pool = nn.AvgPool2d(scale)
        self.upsample = nn.Upsample(scale_factor=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)

    def downsample(self, x: torch.Tensor):
        x = x**self.gamma
        x = self.avg_pool(x)
        x = x ** (1.0 / self.gamma)
        return x
