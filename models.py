import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        # mxin_channelsx28x28
        self.disc = nn.Sequential(
                # in_channels, out_channels, kernel, stride, padding
                nn.Conv2d(in_channels, 16, 3, 1, 1),
                nn.LeakyReLU(0.2),
                # mx16x28x28
                nn.Conv2d(16, 32, 3, 1, 0),
                nn.LeakyReLU(0.2),
                # mx32x26x26
                nn.Conv2d(32, 64, 3, 2, 0),
                nn.LeakyReLU(),
                # mx64x12x12
                nn.Conv2d(64, 128, 4, 2, 0),
                nn.LeakyReLU(),
                # mx128x5x5
                nn.Conv2d(128, 1, 4, 2, 0),
                nn.Sigmoid()
                # mx1x1x1
            )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, noise_channels, out_channels):
        super(Generator, self).__init__()
        # mx256x1x1
        self.gen = nn.Sequential(
                # in_channels, out_channels, kernel, stride, padding
                nn.ConvTranspose2d(noise_channels, 128, 4, 1, 0),
                nn.ReLU(),
                # mx128x4x4
                nn.ConvTranspose2d(128, 64, 3, 1, 0),
                nn.ReLU(),
                # mx64x6x6
                nn.ConvTranspose2d(64, 32, 4, 2, 0),
                nn.ReLU(),
                # mx32x14x14
                nn.ConvTranspose2d(32, 1, 4, 2, 1),
                nn.Tanh(),
                # mx1x28x28
            )

    def forward(self, x):
        return self.gen(x)
