import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb

class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.InstanceNorm2d(C_out),
            nn.GELU(),
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.InstanceNorm2d(C_out),
            nn.GELU(),
            )
 
    def forward(self, x):
        return self.layer(x)
 
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.Down(x)
 
class UpSampling(nn.Module):
    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(C, C // 2, 1, 1, 0)

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.Up(up)
        return torch.cat((x, r), 1)
  
class UNet(nn.Module):
    def __init__(self, dim=1, C_in=1, C_out=1):
        super(UNet, self).__init__()
        dim = dim
        self.C1 = Conv(C_in, dim)
        self.D1 = DownSampling(dim)
        self.C2 = Conv(dim, dim * 2)
        self.D2 = DownSampling(dim * 2)
        self.C3 = Conv(dim * 2, dim * 4)
 
        self.U1 = UpSampling(dim * 4)
        self.C4 = Conv(dim * 4, dim * 2)
        self.U2 = UpSampling(dim * 2)
        self.C5 = Conv(dim * 2, dim)
        self.out = nn.Conv2d(dim, C_out, 3, 1, 1)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
  
    def forward(self, x):
        L1 = self.C1(x)
        L2 = self.C2(self.D1(L1))
        M1 = self.C3(self.D2(L2))
 
        R1 = self.C4(self.U1(M1, L2))
        R2 = self.C5(self.U2(R1, L1))
        out = self.out(R2)

        return out
 