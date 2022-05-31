from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lib.resnet_module import Stem, ResLayer
from models.lib.DMAM import DMAM,  ConvBnReLU
from LCAM import LCAM
from MFM import MFM



class LLAMNet(nn.Module):
    def __init__(self, classes = 2):
        super(LLAMNet, self).__init__()

        n_blocks = [3, 4, 6, 3]    # backbone=resnet101,num_eachlayer_bottleneck=[3,4,23,3]
        multi_grids = [1, 2, 4]    # 3x3Conv in layer5 each dilation is dilation*[1,2,4]

        self.layer1 = Stem(64)   # 下采样X4
        self.layer2 = ResLayer(n_blocks[0], 64, 256, stride=1, dialtion=1)
        self.layer3 = ResLayer(n_blocks[1], 256, 512, stride=2, dialtion=1)    #
        self.layer4 = ResLayer(n_blocks[2], 512, 1024, stride=1, dialtion=2)   #
        self.layer5 = ResLayer(n_blocks[3], 1024, 2048, stride=1, dialtion=4, multi_grids=multi_grids)

        atrous_rates = [6, 12, 18]  # 3x3Conv in ASPP each dilation is 6,12,18

        self.aspp1 = DMAM(2048, 256, atrous_rates)
        self.fc1 = ConvBnReLU((len(atrous_rates)+3)*256,256,kernel_size=1,stride=1,padding=0)

        # Decoder
        self.reduce1 = ConvBnReLU(256, 48, kernel_size=1, stride=1, padding=0)
        self.reduce2 = ConvBnReLU(512, 96, kernel_size=1, stride=1, padding=0)
        self.conv1_1 = ConvBnReLU(96, 48, kernel_size=1, stride=1, padding=0)


        #self.atten1 = HAttention(64, 64)
        self.atten2 = LCAM(256, 256)
        self.atten1 = LCAM(512, 512)
        self.atten3 = LCAM(1536, 1536)
        self.bga = MFM(classes)



    def forward(self, x):
        h1 = self.layer1(x)   #64*64*64

        h2 = self.layer2(h1)   #256*64*64
        h2 = self.atten2(h2)

        h_1 = self.reduce1(h2)

        h3 = self.layer3(h2)   #512*32*32
        h3 = self.atten1(h3)

        h_2 = self.reduce2(h3) #96*32*32
        h_2 = self.conv1_1(h_2) #48*32*32

        h4 = self.layer4(h3)   #1024*32*32
        h5 = self.layer5(h4)   #2048*32*32
        h6 = self.aspp1(h5)     #1536*32*32
        h6 = self.atten3(h6)
        h7 = self.fc1(h6)

        h8 = F.interpolate(h7, size=h_1.shape[2:], mode="bilinear", align_corners=False)
        h9 = torch.cat((h_1, h8), dim=1)

        h13 = torch.cat((h_2, h7), dim=1)

        h14 = self.bga(h9, h13)

        h11 = F.interpolate(h14, size=x.shape[2:], mode="bilinear", align_corners=False)
        return h11



