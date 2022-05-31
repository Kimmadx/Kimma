import torch.nn as nn
import torch.nn.functional as F
from models.lib.aspp import ConvBnReLU





class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation, downsample):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        mid_channels = out_channels // 4

        self.conv3X3_1 = ConvBnReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=dilation,
                                  dilation=dilation, relu=True)
        self.conv3X3_2 = ConvBnReLU(mid_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, relu=True)

        self.shortcut =ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, relu=False)

    def forward(self, x):

        x_ = self.conv3X3_1(x)

        x_ = self.conv3X3_2(x_)

        print('==================================')
        print(x_.shape)
        print(x.shape)

        if self.downsample:
            x_ += self.shortcut(x)
            print('!!!!!!!!!!!!!!!!!!!!!!')
            print(x_.shape)
        else:
            x_ += x
        return F.relu(x_)

class Basicneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation, downsample):
        super(Basicneck, self).__init__()
        self.downsample = downsample
        mid_channels = out_channels // 4
        self.reduce = ConvBnReLU(in_channels, mid_channels, kernel_size=1, stride=stride, padding=0, dilation=1, relu=True)
        self.conv3X3 = ConvBnReLU(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, relu=True)
        self.increase = ConvBnReLU(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, relu=False)
        self.shortcut = ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, relu=False)

    def forward(self, x):
        x_ = self.reduce(x)
        x_ = self.conv3X3(x_)
        x_ = self.increase(x_)
        if self.downsample:
            x_ += self.shortcut(x)
        else:
            x_ += x
        return F.relu(x_)



class ResLayer(nn.Sequential):
    def __init__(self, num_layers, in_channels, out_channels, stride, dialtion, multi_grids=None):
        super(ResLayer, self).__init__()
        if multi_grids is None:
            multi_grids = [1 for _ in range(num_layers)]
        else:
            assert num_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(num_layers):
            self.add_module(
                'block{}'.format(i+1), Basicneck(
                    in_channels=(in_channels if i == 0 else out_channels),
                    out_channels=out_channels,
                    stride=(stride if i == 0 else 1),
                    dilation=dialtion * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


"""
The first conv layer
Note that the max pooling is different from both MSRA and FAIR ResNet.
"""


class Stem(nn.Sequential):
    def __init__(self, out_channels):
        super(Stem, self).__init__()
        self.add_module('conv1', ConvBnReLU(3, out_channels, kernel_size=7, stride=2, padding=3, dilation=1))
        self.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


if __name__ == '__main__':
    multi_grids = [1 for _ in range(5)]
    print(multi_grids)