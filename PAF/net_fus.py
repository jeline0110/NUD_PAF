import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.GELU(),
            )

    def forward(self, x):
        out = self.block(x) + x

        return out

class PosNet(nn.Module):
    def __init__(self, C, pos_dim=4):
        super(PosNet, self).__init__()
        mid_ch = C
        self.pos_emb = nn.Sequential( 
            nn.Conv2d(2, pos_dim, 1, 1, 0, bias=False),
            nn.Tanh(),
            )

        self.mix1 = nn.Sequential(
            nn.Conv2d(C + pos_dim, mid_ch, 1, 1, 0, bias=False),
            nn.GELU(),
            )
        self.mix2 = nn.Sequential(
            nn.Conv2d(mid_ch + pos_dim, C, 1, 1, 0, bias=False),
            nn.GELU(),
            )
       
    def forward(self, x, pos):
        shortcut = x
        emb1 = self.pos_emb(pos)
        x_cat1 = torch.cat((x, emb1), dim=1)
        x = self.mix1(x_cat1)

        x_cat2 = torch.cat((x, emb1), dim=1)
        out = self.mix2(x_cat2) + shortcut

        return out

class PAF(nn.Module):
    def __init__(self, C, est_model):
        super(PAF, self).__init__()
        emb_dim = C
        self.head = nn.Conv2d(C*2, emb_dim, 3, 1, 1)
        # tail
        self.tail = nn.Conv2d(emb_dim, C, 3, 1, 1)
        # body
        self.convblock = ConvBlock(emb_dim)
        # pos
        self.posnet = PosNet(C, 4)
        self.est_model = est_model

    def data_up(self, lrhsi, hrmsi):
        target_size1 = (hrmsi.shape[2], hrmsi.shape[3])
        lrhsi_ups = F.interpolate(lrhsi, size=target_size1, mode='bilinear', align_corners=True)
        target_size2 = (hrmsi.shape[3], lrhsi.shape[1])
        hrmsi = hrmsi.permute(0, 2, 3, 1)
        hrmsi_upc = F.interpolate(hrmsi, size=target_size2, mode='bilinear', align_corners=True)
        hrmsi_upc = hrmsi_upc.permute(0, 3, 1, 2)
        lrhsi_hrmsi_cat = torch.cat((lrhsi_ups, hrmsi_upc), dim=1)

        return lrhsi_hrmsi_cat, lrhsi_ups, hrmsi_upc

    def body(self, hsi, msi, pos):
        x_cat, hsi_ups, _ = self.data_up(hsi, msi)
        x = self.head(x_cat)
        x = self.posnet(x, pos)
        x = self.convblock(x)
        out = self.tail(x) # + hsi_ups

        return out

    def forward(self, lr2hsi=None, lrmsi=None, lrpos=None, lrhsi=None, \
            hrmsi=None, hrpos=None, warp_map=None, mode='train'):
        if mode == 'train':
            out1 = self.body(lr2hsi, lrmsi, lrpos)
            fake = self.body(lrhsi, hrmsi, hrpos)
            out2 = self.est_model.SpaD(fake, mode='test', warp_map=warp_map)
            out3 = self.est_model.SpeD(fake, hrpos)
        else:
            out1 = self.body(lrhsi, hrmsi, hrpos)
            out2, out3 = None, None

        return out1, out2, out3