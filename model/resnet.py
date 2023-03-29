# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
from .unet import BaseUNet

class ResNet(nn.Module):
    def __init__(self, n_channels, reps):
        super(ResNet, self).__init__()
        self.reps = reps
        self.unet0 = BaseUNet(n_channels)
        self.unet1 = BaseUNet(n_channels + 1)

    def forward(self, x):
        y = self.unet0(x)
        y = torch.sigmoid_(torch.add(y, x[:, 0:1, :, :]))
        y = torch.cat((y, x[:, :, :, :]), 1)
        y = self.unet1(y)
        y = torch.sigmoid_(torch.add(y, x[:, 0:1, :, :]))
        return y

