import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualSubBlock(nn.Module):
    """
    Residual sub-block Ri consisting of 1x1, 3x3, and 1x1 convolutions
    with Instance Normalization and ReLU, plus a residual shortcut.
    Reference: ViNet (Gentilhomme et al. 2023) Section 2.3.1 & Figure 4.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualSubBlock, self).__init__()
        mid_channels = out_channels // 2
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(mid_channels, affine=True)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False)
        self.norm2 = nn.InstanceNorm2d(mid_channels, affine=True)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.norm3 = nn.InstanceNorm2d(out_channels, affine=True)
        
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out = self.relu(out + res)
        return out

class BasicBlock(nn.Module):
    """
    Basic Block Bi composed of two residual sub-blocks: Bi = {R1, R2}
    """
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.r1 = ResidualSubBlock(in_channels, out_channels)
        self.r2 = ResidualSubBlock(out_channels, out_channels)

    def forward(self, x):
        return self.r2(self.r1(x))
