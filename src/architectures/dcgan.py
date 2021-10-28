import torch
from torch import nn


# custom weights initialization called on netG and netD
def _dcgan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN(nn.Module):
    def __init__(self, nz=512, ngf=64, nc=6):
        super(DCGAN, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            # state size. (nc) x 64 x 64
        )
        self.apply(_dcgan_weights_init)

    def forward(self, z):
        z = z.reshape(z.shape[0], -1, 1, 1)
        # Output has six channels
        output = self.main(z)
        # Scales are post-processed to [0,inf] outside this method
        means = torch.sigmoid(output[:, 0:3, ...])
        scales = output[:, 3:6, ...]
        return means, scales


class DCGANSmall(nn.Module):
    """ For 28x28 images, source: https://github.com/AKASHKADEL/dcgan-mnist/blob/master/networks.py """

    def __init__(self, num_channels, h_dim, ngf, bias=False, **kwargs):
        self.num_channels = num_channels
        super(DCGANSmall, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(h_dim, ngf * 4, 4, 1, 0, bias=bias),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=bias),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 2 * num_channels, 4, 2, 1),
        )
        # self.apply(_dcgan_weights_init)

    def forward(self, z):
        z = z.reshape(z.shape[0], -1, 1, 1)
        # Output has six channels
        output = self.net(z)
        # Scales are post-processed to [0,inf] outside this method
        means = torch.sigmoid(output[:, 0 : self.num_channels, ...])
        scales = output[:, self.num_channels : 2 * self.num_channels, ...]
        return means, scales