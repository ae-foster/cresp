import torch
from torch import nn


class ConvEncoder(nn.Module):
    def __init__(self, num_channels, ngf=512, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, ngf // 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf // 8),
            nn.LeakyReLU(),
            nn.Conv2d(ngf // 8, ngf // 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.LeakyReLU(),
            nn.Conv2d(ngf // 4, ngf // 2, 3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.LeakyReLU(),
            nn.Conv2d(ngf // 2, ngf, 3, stride=4, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        z = self.net(x)
        z = z.reshape(z.shape[0], -1)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, nz=512, nc=6, ngf=512, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(nz, ngf // 2, 2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.LeakyReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(ngf // 2, ngf // 4, 2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(ngf // 4, ngf // 8, 2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(ngf // 8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(ngf // 8, nc, 2, stride=2, padding=0),
        )

    def forward(self, z):
        z = z.reshape(z.shape[0], -1, 1, 1)
        # Output has six channels
        output = self.net(z)
        # Scales are post-processed to [0,inf] outside this method
        means = torch.sigmoid(output[:, 0:3, ...])
        scales = output[:, 3:6, ...]
        return means, scales


class ConvEncoder2(nn.Module):
    """ For 28x28 images """

    def __init__(self, num_channels, h_dim, ngf, bias=False, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, ngf, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, 2 * ngf, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(2 * ngf),
            nn.ReLU(True),
            nn.Conv2d(2 * ngf, 4 * ngf, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(4 * ngf),
            nn.ReLU(True),
            nn.Conv2d(4 * ngf, h_dim, kernel_size=2, stride=2),
        )

    def forward(self, x):
        z = self.net(x)
        z = z.reshape(z.shape[0], -1)
        return z
