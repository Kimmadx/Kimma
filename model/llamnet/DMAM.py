import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvBnReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, relu=True):
        super(ConvBnReLU, self).__init__()
        self.add_module(
            'Conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
        )
        self.add_module('BN', nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.999))
        if relu:
            self.add_module('ReLU', nn.ReLU())


class DWConvBnReLU(nn.Sequential):
    def __init__(self,in_channels, out_channels, kernel_size1, kernel_size2, stride, padding1, padding2, dilation=1, groups=1, relu=True):
        super(DWConvBnReLU, self).__init__()
        self.add_module(
            'DWConv', nn.Conv2d(in_channels, in_channels, kernel_size1, stride, padding1, dilation, groups, bias=False),
        )
        self.add_module('BN', nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.999)
        )
        self.add_module('Conv',nn.Conv2d(in_channels,out_channels, kernel_size2, stride, padding2, bias=False)
        )
        if relu:
            self.add_module('ReLU', nn.ReLU()
        )



class ImagePool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.conv = ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        x_size = x.shape
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)
        return x


class DMAM(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(DMAM, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module('c0', ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1))   # channel->out_channels, keep size

        for idx, rate in enumerate(rates):
            self.stages.add_module('c{}'.format(idx+1), DWConvBnReLU(in_channels, out_channels, kernel_size1=3, kernel_size2=1, stride=1, padding1=rate, groups=in_channels, padding2=0, dilation=rate))   # channel->out_channels, keep size
        self.stages.add_module('imagepool', ImagePool(in_channels, out_channels))    # channel->out_channels, keep size
        self.stages.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x):

        x = torch.cat([stage(x) for stage in self.stages.children()], dim=1)

        return x