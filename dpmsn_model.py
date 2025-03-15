import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DPMSN(nn.Module):
    def __init__(self, in_channels=3):
        super(DPMSN, self).__init__()
        self.encoder1 = ConvBlock(in_channels, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)

        self.decoder4 = ConvBlock(512, 256)
        self.decoder3 = ConvBlock(256, 128)
        self.decoder2 = ConvBlock(128, 64)
        self.decoder1 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))

        d4 = self.decoder4(F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=True))
        d3 = self.decoder3(F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True))
        d2 = self.decoder2(F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True))
        d1 = self.decoder1(F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True))

        return torch.sigmoid(d1)
