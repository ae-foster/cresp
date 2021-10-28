import torch
from torch import nn
import random
from torchvision import transforms


def ColourDistortion(s=1.0, p_grayscale=0.2, p_jitter=0.8):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=p_jitter)
    rnd_gray = transforms.RandomGrayscale(p=p_grayscale)
    rnd_gamma = GammaDistortion(gamma=0.8 * s)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray, rnd_gamma])
    return color_distort


class GammaDistortion(nn.Module):
    def __init__(self, gamma=0.0):
        super().__init__()
        gamma = min(gamma, 1.0)
        self.range = [1 - gamma, gamma]

    def forward(self, img):
        parameter = random.uniform(*self.range)
        return transforms.functional.adjust_gamma(img, parameter)


class Normalize1d(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        return (tensor - mean) / std
