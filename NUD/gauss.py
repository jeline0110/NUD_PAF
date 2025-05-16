import torch
import torch.nn as nn
import math 

class GaussKernel(nn.Module):
    def __init__(self, m_ksize=25, fix_list=[''], init_paras=[3.0, 3.0, 0., 0., 0.]):
        super().__init__()

        sigma_y = torch.ones([1], dtype=torch.float32) * init_paras[0]
        if 'sigma_y' in fix_list:
            self.sigma_y = sigma_y.cuda()
        else:
            self.sigma_y = nn.Parameter(sigma_y)

        sigma_x = torch.ones([1], dtype=torch.float32) * init_paras[1]
        if 'sigma_x' in fix_list:
            self.sigma_x = self.sigma_y
        else:
            self.sigma_x = nn.Parameter(sigma_x)

        offset_y = torch.ones([1], dtype=torch.float32) * init_paras[2]
        if 'offset_y' in fix_list:
            self.offset_y = offset_y.cuda()
        else:
            self.offset_y = nn.Parameter(offset_y)

        offset_x = torch.ones([1], dtype=torch.float32) * init_paras[3]
        if 'offset_x' in fix_list:
            self.offset_x = offset_x.cuda()
        else:
            self.offset_x = nn.Parameter(offset_x)

        theta = torch.ones([1], dtype=torch.float32) * init_paras[4]
        if 'theta' in fix_list:
            self.theta = theta.cuda()
        else:
            self.theta = nn.Parameter(theta)

        self.ksize = m_ksize
        kk = torch.arange(0, self.ksize) - (self.ksize - 1) // 2
        self.yy, self.xx = torch.meshgrid(kk.cuda(), kk.cuda())

    def cal_distance(self):
        theta_bounds = math.pi / 2 - 1e-6
        theta = torch.clamp(self.theta, min=-1.*theta_bounds, max=theta_bounds)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        cos_theta_2 = cos_theta ** 2
        sin_theta_2 = sin_theta ** 2

        sigma_y = torch.clamp(self.sigma_y, min=1e-6)
        sigma_x = torch.clamp(self.sigma_x, min=1e-6)
        sigma_y_2 = 2.0 * (sigma_y ** 2)
        sigma_x_2 = 2.0 * (sigma_x ** 2)

        a = cos_theta_2 / sigma_y_2 + sin_theta_2 / sigma_x_2
        b = sin_theta * cos_theta * (1.0 / sigma_x_2 - 1.0 / sigma_y_2)
        c = sin_theta_2 / sigma_y_2 + cos_theta_2 / sigma_x_2
        for tensor in [a, b, c]:
            tensor.unsqueeze_(-1).unsqueeze_(-1)
        fn = lambda y, x: a * (y ** 2) + 2.0 * b * x * y + c * (x ** 2)

        offset_bounds = (self.ksize - 1) // 2 - 1e-6
        offset_y = torch.clamp(self.offset_y, min=-1.*offset_bounds, max=offset_bounds)
        offset_x = torch.clamp(self.offset_x, min=-1.*offset_bounds, max=offset_bounds)
        yy = self.yy - offset_y
        xx = self.xx - offset_x
        distance = fn(
            yy.view(1, self.ksize, self.ksize),
            xx.view(1, self.ksize, self.ksize)
            )

        return distance

    def get_kernel(self):
        distance = self.cal_distance()
        gauss_kernel = torch.exp(-distance).squeeze(0)
               
        return gauss_kernel

    def forward(self):
        kernel = self.get_kernel()
        kernel = kernel / kernel.sum()

        return kernel
