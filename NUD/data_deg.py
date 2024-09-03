import torch 
import torch.nn.functional as F
import torch.nn as nn 
import scipy.io as scio

class DegDataset():
    def __init__(self, data_path=''):
        # load data
        data = scio.loadmat(data_path)
        hrmsi = data['hrmsi']
        lrhsi = data['lrhsi']
        self.hrmsi_h, self.hrmsi_w, self.hrmsi_bands = hrmsi.shape
        self.lrhsi_h, self.lrhsi_w, self.lrhsi_bands = lrhsi.shape
        assert self.hrmsi_h // self.lrhsi_h == self.hrmsi_w // self.lrhsi_w
        self.scale = self.hrmsi_h // self.lrhsi_h
        hrmsi = torch.tensor(hrmsi, dtype=torch.float32)
        self.hrmsi = hrmsi.permute(2, 0, 1).unsqueeze(0).cuda()
        lrhsi = torch.tensor(lrhsi, dtype=torch.float32)
        self.lrhsi = lrhsi.permute(2, 0, 1).unsqueeze(0).cuda() # b, c, h, w
        
        print('-------------------')
        print('Data info:')
        print('-------------------')
        print('HR-MSI image size:[%d,%d,%d], LR-HSI image size:[%d,%d,%d]' \
            % (self.hrmsi_h, self.hrmsi_w, self.hrmsi_bands, self.lrhsi_h, self.lrhsi_w, self.lrhsi_bands))
        print('Spatial Degradation scale:%d' % self.scale)
        print('Spectral Degradation scale:%d->%d' % (self.lrhsi_bands, self.hrmsi_bands))
