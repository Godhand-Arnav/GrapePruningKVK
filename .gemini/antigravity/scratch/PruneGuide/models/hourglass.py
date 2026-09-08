import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import BasicBlock

class HourglassSubnetwork(nn.Module):
    """
    A 5-level recursive Hourglass Subnetwork as described in ViNet.
    Down-scale pathway with pooling, up-scale pathway with additions,
    and skip connections Si.
    """
    def __init__(self, depth, channels):
        super(HourglassSubnetwork, self).__init__()
        self.depth = depth
        self.channels = channels
        
        # Down pathway block and skip block
        self.b_down = BasicBlock(channels, channels)
        self.skip = BasicBlock(channels, channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if depth > 1:
            self.inner_hg = HourglassSubnetwork(depth - 1, channels)
        else:
            self.inner_hg = BasicBlock(channels, channels)
            
        self.b_up = BasicBlock(channels, channels)

    def forward(self, x):
        skip_feat = self.skip(x)
        down = self.pool(x)
        down = self.b_down(down)
        
        inner = self.inner_hg(down)
        
        up = F.interpolate(inner, scale_factor=2, mode='bilinear', align_corners=False)
        up = self.b_up(up)
        
        return skip_feat + up

class StackedHourglass(nn.Module):
    """
    Stacked Hourglass Network (SHG) for Grapevine Structure Estimation.
    Outputs node heatmaps and branch vector affinity fields (PAFs).
    
    Args:
        num_heatmaps: Total number of output channels (nodes + PAFs)
        in_channels: Input image channels (3 for RGB)
        base_channels: Front feature channels (default 64)
        hg_channels: Channels across residual blocks in Hourglass (default 128/256)
        num_stacks: Number of stacked hourglasses (default 2)
    """
    def __init__(self, num_heatmaps=14, in_channels=3, base_channels=64, hg_channels=128, num_stacks=2):
        super(StackedHourglass, self).__init__()
        self.num_stacks = num_stacks
        self.num_heatmaps = num_heatmaps
        
        # Feature Extraction Module (divides spatial resolution by 4)
        self.front = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            BasicBlock(base_channels, base_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(base_channels, hg_channels)
        )
        
        # Hourglass stacks
        self.hgs = nn.ModuleList([HourglassSubnetwork(depth=4, channels=hg_channels) for _ in range(num_stacks)])
        
        # Intermediate and output heads
        self.heads = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.merge_features = nn.ModuleList()
        self.merge_preds = nn.ModuleList()
        
        for i in range(num_stacks):
            head = nn.Sequential(
                BasicBlock(hg_channels, hg_channels),
                nn.Conv2d(hg_channels, num_heatmaps, kernel_size=1, bias=True)
            )
            self.heads.append(head)
            
            if i < num_stacks - 1:
                self.linear_layers.append(nn.Conv2d(hg_channels, hg_channels, kernel_size=1, bias=False))
                self.merge_features.append(nn.Conv2d(hg_channels, hg_channels, kernel_size=1, bias=False))
                self.merge_preds.append(nn.Conv2d(num_heatmaps, hg_channels, kernel_size=1, bias=False))

    def forward(self, x):
        # Front feature extraction: (B, 3, H, W) -> (B, hg_channels, H/4, W/4)
        feat = self.front(x)
        
        outputs = []
        for i in range(self.num_stacks):
            hg_out = self.hgs[i](feat)
            pred = self.heads[i](hg_out)
            outputs.append(pred)
            
            if i < self.num_stacks - 1:
                feat = feat + self.linear_layers[i](hg_out) + self.merge_preds[i](pred)
                
        return outputs
