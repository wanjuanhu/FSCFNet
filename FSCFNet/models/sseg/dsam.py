import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sseg.gsmodule import GumbelSoftmax2D

class ChannelAttentionLayer(nn.Module):
    def __init__(self, C_in, C_out, reduction=16,  affine=True, BN=nn.BatchNorm2d):
        super(ChannelAttentionLayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(C_in, max(1, C_in // reduction), 1, padding=0, bias=False),
                nn.ReLU(),
                nn.Conv2d(max(1, C_in // reduction) , C_out, 1, padding=0, bias=False),
                nn.Sigmoid())
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Adaptive_DSAM(nn.Module):
    def __init__(self, channel):
        super(Adaptive_DSAM, self).__init__()
        self.depth_revise = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Conv2d(16, 3, 1)
        self.GS = GumbelSoftmax2D(hard=True)

        self.channel = channel
        self.conv0 = nn.Conv2d(channel, channel, 1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel, 1, padding=0)
        self.conv2 = nn.Conv2d(channel, channel, 1, padding=0)
        self.conv3 = nn.Conv2d(channel, channel, 1, padding=0)
        self.channel_att = ChannelAttentionLayer(self.channel, self.channel)
        self.conv4 = nn.Conv2d(channel, 4, 1, padding=0)

    def forward(self, x, bins, gumbel=False):
        n, c, h, w = x.size()

        bins = self.depth_revise(bins)
        gate = self.fc(bins)
        bins = self.GS(gate, gumbel=gumbel) * torch.mean(bins, dim=1, keepdim=True)

        x0 = self.conv0(bins[:, 0, :, :].unsqueeze(1) * x)
        x1 = self.conv1(bins[:, 1, :, :].unsqueeze(1) * x)
        x2 = self.conv2(bins[:, 2, :, :].unsqueeze(1) * x)
        # x3 = self.conv3(bins[:, 3, :, :].unsqueeze(1) * x)
        x = (x0 + x1 + x2) + x
        x = self.channel_att(x)
        x = self.conv4(x)

        return x