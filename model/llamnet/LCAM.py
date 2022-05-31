from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
from h_swish import h_swish

class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True, groups=1):
        super(Conv2dBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias, groups=groups),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class LCAM(nn.Module):
    def __init__(self,in_ch,out_ch,reduction=8):
       super(LCAM, self).__init__()


       self.reduction = reduction

       self.avg = nn.AdaptiveAvgPool2d(1)
       self.max = nn.AdaptiveMaxPool2d(1)
       self.conv1 = nn.Conv1d(in_ch, in_ch//16, kernel_size=1)


       self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # B,C,H,1
       self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # B,C,1,W

       self.query_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch // self.reduction, kernel_size=1)
       self.key_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch // self.reduction, kernel_size=1)
       self.value_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)

       mip = max(8, in_ch // reduction)

       self.conv = nn.Conv2d(2*(in_ch// self.reduction),in_ch,kernel_size=1,stride=1,padding=0)
       self.conv0 = nn.Conv2d(mip, in_ch, kernel_size=1, stride=1, padding=0)
       self.conv1 = nn.Conv2d(in_ch, mip, kernel_size=1, stride=1, padding=0)
       self.conv4 = nn.Conv2d(in_ch//reduction,in_ch,kernel_size=1,stride=1,padding=0)
       self.bn1 = nn.BatchNorm2d(mip)
       self.act = h_swish()
       self.sig = nn.Sigmoid()
       self.relu = nn.ReLU()

       self.softmax = nn.Softmax(-1)
       self.conv2 = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False),
                    nn.BatchNorm2d(in_ch),
                    nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False)
       )
       self.bn2 = nn.BatchNorm2d(in_ch)

    def forward(self,x):
        n_batch, C, h, w = x.size()


        #通道注意力信息
        x_avg = self.avg(x)
        x_max = self.max(x)
        x_avg = self.conv1(x_avg)
        x_max = self.conv1(x_max)
        x_avg = self.relu(x_avg)
        x_max = self.relu(x_max)

        x_ch = torch.cat([x_max,x_avg],dim=1)
        x_ch = self.conv(x_ch)
        x_ch = self.sig(x_ch)
        x_ch = x * x_ch

        # 空间注意力信息
        # B,C,H,W
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y2 = torch.cat([x_h, x_w], dim=2)
        y2 = self.conv1(y2)
        y2 = self.bn1(y2)
        y2 = self.act(y2)

        y2 = self.conv0(y2)

        x_h, x_w = torch.split(y2, [h, w], dim=2)

        x_w = x_w.permute(0, 1, 3, 2)


        #H方向全局注意力
        q_h = self.query_conv(x_h).view(n_batch, C // self.reduction, h, -1)
        k_h = self.key_conv(x_h).view(n_batch, C // self.reduction,h, -1)
        v_h = self.value_conv(x_h).view(n_batch, C , h, -1)

        content_h = torch.matmul(q_h.permute(0, 1, 3, 2), k_h)
        content_h = self.conv4(content_h)
        attention_h = self.softmax(content_h)   # C*1*1

        out_h = torch.matmul(v_h, attention_h.permute(0, 1, 3, 2))  # v_h : c*h*1 out_h: c*h*1
        out_h = out_h + v_h
        out_h = self.bn2(out_h)
        out_h = out_h + x_h

        q_w = self.query_conv(x_w).view(n_batch, C // self.reduction, h, -1)
        k_w = self.key_conv(x_w).view(n_batch, C // self.reduction, h, -1)
        v_w = self.value_conv(x_w).view(n_batch, C, -1, w)


        content_w = torch.matmul(q_w.permute(0, 1, 3, 2), k_w)
        content_w = self.conv4(content_w)
        attention_w = self.softmax(content_w)  # C*1*1

        out_w = torch.matmul(attention_w, v_w)  # v_h : c*1*w out_h: c*1*w
        out_w = out_w + v_w
        out_w = self.bn2(out_w)
        out_w = out_w + x_w

        out_main = torch.matmul(out_h,out_w)
        out_main = out_main + x

        return out_main +x_ch