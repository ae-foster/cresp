import math
import torch
import numpy as np
from torch.distributions import Uniform, Bernoulli
import torch.nn.functional as F


def compute_new_x(x, v, t, sigma):
    l = 1 - 2 * sigma
    x = x - sigma
    unconst = v * t + x

    n_bounce = torch.floor(unconst / l)
    mods = n_bounce % 2
    new_x = mods * (l - (unconst - l * n_bounce)) + (1 - mods) * (unconst - l * n_bounce)
    new_x = new_x + sigma
    return new_x


colours = {
    "rgb": {
        "back": [1.0, 1.0, 1.0],
        0: [12.0 / 100, 47.0 / 100, 71.0 / 100],
        1: [100.0 / 100, 49.8 / 100, 5.5 / 100],
    },
    "bw": {
        "back": [1.0],
        0: [0.0],
        1: [0.0],
    },
}


class Snooker:
    """"""

    def __init__(self, N, w, speed, sigma, inf, col, transform=None, **kwargs):
        self.N = N
        self.w = w
        self.k = 2
        self.sigma = sigma
        self.inf = inf
        self.colours = colours[col]
        self.speed = speed
        self.transform = eval(transform) if transform else None
        self.params = self.sample_functions(N)

    def sample_functions(self, N):
        x = Uniform(low=0, high=1).sample(torch.Size([N, self.k, 2]))
        v = Uniform(low=-1, high=1).sample(torch.Size([N, self.k, 2]))
        v = v * self.speed / v.norm(dim=-1, keepdim=True)
        params = torch.cat([x, v], dim=-1)
        return params

    def make_function(self, coeffs):
        def image(t):
            x0, v0 = coeffs[0][:2], coeffs[0][2:]
            new_x0 = compute_new_x(x0, v0, t, self.sigma)
            x1, v1 = coeffs[1][:2], coeffs[1][2:]
            new_x1 = compute_new_x(x1, v1, t, self.sigma)

            w, h = self.w, self.w

            grid = torch.stack(torch.meshgrid(torch.linspace(0, 1, w), torch.linspace(0, 1, h)), dim=0).view(
                1, 2, 1, w, h
            )
            centres = torch.stack([new_x0, new_x1], axis=-1).view(-1, 2, 2, 1, 1)
            d = (grid - centres).norm(dim=1)
            sigma = torch.tensor([self.sigma])
            overlap = ((new_x0 - new_x1).norm(dim=-1) < 2 * sigma).long()
            weights = (d < sigma).float()
            col_back = torch.tensor(self.colours["back"]).view(1, -1, 1, 1)
            col_0 = torch.tensor(self.colours[0]).view(1, -1, 1, 1)
            col_1 = torch.tensor(self.colours[1]).view(1, -1, 1, 1)
            v = weights[:, [0], ...]
            image0 = v * col_0 + (1 - v) * col_back
            v = weights[:, [1], ...]
            image1 = v * col_1 + (1 - v) * col_back
            image = 0.5 * image0 + 0.5 * image1
            return image, overlap

        return image

    def __len__(self):
        """
        Return number of total models in the loaded dataset.
        """
        return self.N

    def __getitem__(self, index: int):
        """"""
        params = self.params[index] if not self.inf else self.sample_functions(1)[0]
        function = self.make_function(params)
        return SnookerObject(function, params, self.transform)


class SnookerObject:
    def __init__(self, function, params, transform):
        self.transform = transform
        self.function = function
        self.params = params

    def make_views(self, covariates):
        views, labels = self.function(covariates)
        return (views, covariates), labels
