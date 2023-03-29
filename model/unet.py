# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch
from .unet_parts import *


class BaseUNet(nn.Module):
    def __init__(self, n_channels):
        super(BaseUNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 32)
        # self.up4 = up_self(64, 32)
        self.outc = outconv(32, 1)

    def forward(self, x):
        x1 = self.inc(x)  # torch.Size([4, 64, 160, 160])
        x2 = self.down1(x1)  # 64-->128, 160-->80
        x3 = self.down2(x2)  # x3: torch.Size([4, 256, 40, 40])
        x4 = self.down3(x3)  # x4: torch.Size([4, 512, 20, 20])
        y = self.up1(x4, x3)  # x4:
        y = self.up2(y, x2)
        y = self.up3(y, x1)  # self.up3 = up(256, 64)
        # x = self.up4(x)
        y = self.outc(y)
        return y


class UNet(nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.unet = BaseUNet(n_channels=n_channels)

    def forward(self, x):
        return torch.sigmoid_(self.unet(x))