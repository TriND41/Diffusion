import torch
import torch.nn as nn
from model.utils.conv import DoubleConv
from model.utils.conv import UpConv, DownConv

class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, timesteps: int, bilinear: bool = False) -> None:
        super().__init__()
        factor = 2 if bilinear else 1

        # Down Side
        self.in_conv = DoubleConv(in_channels, 64)
        self.down_1 = DownConv(64, 128)
        self.down_2 = DownConv(128, 256)
        self.down_3 = DownConv(256, 512)
        self.down_4 = DownConv(512, 1024 // factor)

        # Bottle Neck
        self.time_embedding = nn.Embedding(timesteps, 1024)
        
        # Up Side
        self.up_1 = UpConv(1024, 512 // factor, bilinear)
        self.up_2 = UpConv(512, 256 // factor, bilinear)
        self.up_3 = UpConv(256, 128 // factor, bilinear)
        self.up_4 = UpConv(128, 64, bilinear)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Down Side
        x1 = self.in_conv(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)

        # Bottle Neck
        x5 += self.time_embedding(t)[:, None, None, None]

        # Up Side
        x = self.up_1(x5, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)
        x = self.out_conv(x)

        return x