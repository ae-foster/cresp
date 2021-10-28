# credits: https://github.com/3springs/attentive-neural-processes/blob/master/neural_processes/modules/modules.py

import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[], batch_norm=False, p=0.0, bias=True, start_dim=1, **kwargs):
        super().__init__()
        hidden_dim = hidden_dim if isinstance(hidden_dim, (list, int)) else list(hidden_dim)
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim]
        dims = [in_dim] + hidden_dim + [out_dim]
        layers = []
        layers.append(nn.Flatten(start_dim=start_dim))
        self.start_dim = start_dim
        for i in range(len(dims[:-1])):
            layers.append(nn.Dropout(p=p))
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())

        if not batch_norm:
            layers = layers[:-1]
        else:
            layers = layers[:-2]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPGen(MLP):
    def forward(self, x):
        decoded = self.net(x)
        prediction_dim = decoded.shape[-1] // 2
        means = decoded[..., :prediction_dim]
        scales = decoded[..., prediction_dim:]
        return means, scales
