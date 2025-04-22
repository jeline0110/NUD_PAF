import torch
import torch.nn as nn
import torch.nn.functional as F
from gauss import GaussKernel
from unet import UNet
from utils import *
import pdb 

class Spectral_Modulate(nn.Module):
    def __init__(self, C, pos_dim=16):        
        super().__init__()
        self.pos_emb = nn.Sequential(
            nn.Conv2d(2, pos_dim, 1, 1, 0, bias=False),
            nn.Tanh(),
            )

        self.modulate = nn.Sequential(
            nn.Conv2d(C + pos_dim, 64, 1, 1, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(64, C, 1, 1, 0, bias=False),
            nn.GELU(),
            )

    def forward(self, x, pos):
        x_emb = self.pos_emb(pos)
        x_cat = torch.cat((x, x_emb), dim=1)
        x_modulate = self.modulate(x_cat)
        out = torch.clamp(x_modulate, 0.0, 1.0)
          
        return out

class Spectral_Degradation(nn.Module):
    def __init__(self, C, c, pos_dim=32):
        super().__init__()
        mid_ch = 64
        self.convs = nn.ModuleList()
        for i in range(c):
            tmp = nn.Sequential(
                nn.Conv2d(C, mid_ch, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(mid_ch),
                nn.GELU(),
                nn.Conv2d(mid_ch, mid_ch//2, 1, 1, 0, bias=False),
                nn.GELU(),
                nn.Conv2d(mid_ch//2, 1, 1, 1, 0, bias=False),
                nn.GELU(),
                )
            self.convs.append(tmp)
        self.spe_modulate = Spectral_Modulate(C, pos_dim)
        self.new_bands = c
       
    def forward(self, x, pos):
        n, bands, h, w = x.shape
        x = self.spe_modulate(x, pos)
        x_speD = torch.zeros((n, self.new_bands, h, w)).cuda()
        for i in range(self.new_bands):
            mid = self.convs[i](x)
            x_speD[:, i:i+1, ...] = mid[:]
        out = torch.clamp(x_speD, 0.0, 1.0)

        return out

class Spatial_Warp(nn.Module):
    def __init__(self, c=4, file=''):        
        super().__init__()
        self.sampler = UNet(dim=16, C_in=c, C_out=2)
        self.file = file

    def forward(self, x, mode='train', warp_map=None):
        if mode == 'train':
            size = x.shape[2:]
            grid_map = get_grid_map(size).cuda()
            sample_map = self.sampler(x)
            thre = 3 
            sample_map[sample_map > thre] = 0
            sample_map[sample_map < -thre] = 0
            warp_map = sample_map + grid_map
            # Need to normalize grid values to [-1, 1] for resampler
            for i in range(2):
                warp_map[:, i, ...] = 2 * (warp_map[:, i, ...] / (size[i] - 1) - 0.5)
            # grid: (batch_size, height, width, 2) 2->(x, y)
            warp_map = warp_map.permute(0, 2, 3, 1)
            warp_map = warp_map[..., [1, 0]]
            torch.save(warp_map.detach(), '../results/deg/%s/warp_map.pt' % self.file)
        elif mode == 'test':
            if warp_map.shape[1] % x.shape[2] == 0:
                scale = warp_map.shape[1] // x.shape[2]
                warp_map = warp_map[:, ::scale, ::scale, :]
            else:
                warp_map = warp_map.permute(0, 3, 1, 2)
                tool = nn.AdaptiveAvgPool2d((x.shape[2:]))
                warp_map = tool(warp_map).permute(0, 2, 3, 1)

        x_warp = F.grid_sample(x, warp_map, mode='bilinear', padding_mode='border', align_corners=True) # nearest bilinear reflection
        out = torch.clamp(x_warp, 0.0, 1.0)
          
        return out

class Spatial_Degradation(nn.Module):
    def __init__(self, scale, c, file='', m_ksize=25):
        super().__init__()
        self.scale = scale
        self.spa_warp = Spatial_Warp(c, file=file)
        self.gk = GaussKernel(m_ksize=m_ksize)
        self.ks = m_ksize

    def forward(self, x, mode='train', warp_map=None):
        bands = x.shape[1]
        x = self.spa_warp(x, mode=mode, warp_map=warp_map)
        self.kernel = self.gk()
        kernel = self.kernel.repeat(bands, 1, 1, 1)
        x_pad, _ = SamePadding(x, (self.ks, self.ks), (1, 1))
        x_spaD = F.conv2d(x_pad, kernel, stride=1, padding=0, bias=None, groups=bands)
        x_spaD = x_spaD[..., ::self.scale, ::self.scale]
        out = torch.clamp(x_spaD, 0.0, 1.0)

        return out

class NUD(nn.Module):
    def __init__(self, C, c, scale, pos=None, file='', pos_dim=32, m_ksize=25):
        super().__init__()
        self.SpeD = Spectral_Degradation(C, c, pos_dim=pos_dim)
        self.SpaD = Spatial_Degradation(scale, c, file=file, m_ksize=m_ksize)
        self.pos = pos

    def forward(self, lrhsi, hrmsi):
        lrhsi_sped = self.SpeD(lrhsi, self.pos)
        hrmsi_spad = self.SpaD(hrmsi)

        return lrhsi_sped, hrmsi_spad
