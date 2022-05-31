import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lib.resnet_module import Stem, ResLayer
from models.lib.DMAM import DMAM,  ConvBnReLU




class MFM(nn.Module):
    def __init__(self,classes):
        super(MFM, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(
                304, 304, kernel_size=3, stride=1,
                padding=1, groups=304, bias=False),
            nn.BatchNorm2d(304),
            nn.Conv2d(
                304, 304, kernel_size=1, stride=1,
                padding=0, bias=False),
        )

        self.left2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )

        self.right = nn.Sequential(
            nn.Conv2d(
                304, 304, kernel_size=3, stride=1,
                padding=1, groups=304, bias=False),
            nn.BatchNorm2d(304),
            nn.Conv2d(
                304, 304, kernel_size=1, stride=1,
                padding=0, bias=False),
        )

        self.conv = nn.Sequential(
                    ConvBnReLU(608, 256, kernel_size=3, stride=1, padding=1),  # 图片尺寸不变
                    ConvBnReLU(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(256, classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]

        left1 = self.left(x_d)

        left2 = self.left(x_d)

        left2 = self.left2(left2)

        right1 = self.right(x_s)

        right2 = self.right(x_s)

        right1 = F.interpolate(right1, size=dsize, mode='bilinear', align_corners=True)

        left = left1 * torch.sigmoid(right1)

        right = left2 * torch.sigmoid(right2)

        right = F.interpolate(right, size=dsize, mode='bilinear', align_corners=True)

        out = self.conv(torch.cat((right, left), dim=1))

        return out