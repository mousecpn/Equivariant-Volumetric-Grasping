"""Time embedding used by the induced IGD rotation decoder."""

import math
import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=time.device) * -scale)
        embedding = time[:, None] * frequencies[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)
