import sys
import random
import os 
import numpy as np
from numpy import exp, finfo 
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 
import torch 
import torch.nn as nn 
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.metrics import structural_similarity
import pdb

#-----------------------#
# seed function
#-----------------------#
def seed_torch(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

    torch.backends.cudnn.deterministic = True # avoiding nondeterministic algorithms
    torch.backends.cudnn.benchmark = False # select the default convolution algorithm
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8' # control the memory allocation strategy used by cuBLAS
    # torch.set_deterministic(True) # avoiding nondeterministic algorithms

#-----------------------#
# other function
#-----------------------#
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def get_params(net):
    '''Returns parameters that we want to optimize over.
    '''
    params = []
    params += [x for x in net.parameters()]
            
    return params

#-----------------------#
# loss function
#-----------------------#
def train_loss(x, gt, func):
    if x is None:
        loss = 0
    else:
        loss = func(x, gt)

    return loss

#-----------------------#
# evaluate function
#-----------------------#
def ergas(img_fake, img_real, scale=4):
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_)**2)
        return 100 / scale * np.sqrt(mse / (mean_real**2 + np.finfo(np.float64).eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses / (means_real**2 + np.finfo(np.float64).eps)).mean())
    else:
        raise ValueError('Wrong input image dimensions.')

def sam(img1, img2):
    """SAM for 3D image, shape (H, W, C); uint or float[0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    assert img1.ndim == 3 and img1.shape[2] > 1, "image n_channels should be greater than 1"
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    inner_product = (img1_ * img2_).sum(axis=2)
    img1_spectral_norm = np.sqrt((img1_**2).sum(axis=2))
    img2_spectral_norm = np.sqrt((img2_**2).sum(axis=2))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + np.finfo(np.float64).eps)).clip(min=0, max=1)
    
    return np.mean(np.arccos(cos_theta))

def get_error_map(fus, gt, norm=False):
    error_map = (fus - gt) ** 2
    error_map = error_map.mean(-1)
    if norm:
        error_map = error_map - error_map.min()
        error_map = error_map / error_map.max()

    return error_map

def get_psnr_np(img1, img2):
    mse = np.mean((img1 - img2) ** 2, axis=(0, 1))

    psnr = np.zeros_like(mse)
    nonzero_mse = mse > 0
    psnr[nonzero_mse] = 20 * np.log10(1.0 / np.sqrt(mse[nonzero_mse]))
    
    return psnr.mean()

def get_psnr_torch(img1, img2):
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3)) 

    psnr = torch.zeros_like(mse)
    nonzero_mse = mse > 0
    psnr[nonzero_mse] = 20 * torch.log10(1.0 / torch.sqrt(mse[nonzero_mse]))

    return psnr.mean()

def get_mae_np(img1, img2):
    return np.abs(img1 - img2).mean()

def get_mae_torch(img1, img2):
    return torch.abs(img1 - img2).mean()

def get_rmse_np(img1, img2):
    return np.sqrt(((img1 - img2) ** 2).mean())

def get_rmse_torch(img1, img2):
    return torch.sqrt(((img1 - img2) ** 2).mean())

#------------ SSIM ------------ 
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = int(window_size//2), groups = channel)
    mu2 = F.conv2d(img2, window, padding = int(window_size//2), groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = int(window_size//2), groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = int(window_size//2), groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = int(window_size//2), groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

get_ssim_torch = SSIM()
get_ssim_np = structural_similarity

#-----------------------#
# data function
#-----------------------#
def get_pos_matrix(shape):
    h, w = shape
    y_values = torch.linspace(-1, 1, h)
    x_values = torch.linspace(-1, 1, w)
    y_matrix, x_matrix = torch.meshgrid(y_values, x_values)
    pos_matrix = torch.stack((y_matrix, x_matrix), dim=-1)
    pos_matrix = pos_matrix.unsqueeze(0).permute(0, 3, 1, 2)

    return pos_matrix

def uniform_downsample(x, target_h, target_w):
    B = x.shape[0]
    grid_y = torch.linspace(-1, 1, target_h, device=x.device)
    grid_x = torch.linspace(-1, 1, target_w, device=x.device)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    x_down = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return x_down

def get_grid_map(size):
    h, w = size
    vectors = [torch.arange(0, s) for s in (h, w)]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid_map = torch.unsqueeze(grid, 0)

    return grid_map

def SamePadding(images, ksizes, strides, rates=(1, 1)):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = F.pad(images, paddings, mode='replicate') # replicate reflect 
    # images = F.pad(images, paddings, mode='constant', value=0) 

    return images, paddings

#-----------------------#
# lr function
#-----------------------#
def lr_scheduler_1step(optimizer, lrs, epoch, milestones):
    if milestones[0] < epoch:
        optimizer.param_groups[0]['lr'] = lrs[0]

def lr_scheduler_2step(optimizer, lrs, epoch, milestones):
    if milestones[0] < epoch <= milestones[1] and milestones[1] != -1:
        optimizer.param_groups[0]['lr'] = lrs[0]
    if milestones[1] < epoch and milestones[1] != -1:
        optimizer.param_groups[0]['lr'] = lrs[1]

def warm_lr_scheduler(optimizer, lrs, epoch, warm_epoch, deacy_epoch, end_epoch, deacy_power):
    if epoch <= warm_epoch:
        lr = lrs[0] + epoch / warm_epoch * (lrs[1] - lrs[0])
    elif warm_epoch < epoch <= deacy_epoch:
        lr = lrs[1]
    elif deacy_epoch < epoch:
        lr = lrs[1] * (1 - (epoch - warm_epoch) / (end_epoch - warm_epoch)) ** deacy_power
    optimizer.param_groups[0]['lr'] = lr

def get_current_lr(optimizer):
    lr = optimizer.param_groups[0]['lr']

    return lr

def lambda_lr_scheduler(optimizer, lr, epoch, max_epoch):
    ratio = 1.0 - epoch / (max_epoch + 1)
    optimizer.param_groups[0]['lr'] = lr * ratio

#-----------------------#
# init function
#-----------------------#
def xavier_init(layer):
    if isinstance(layer, nn.Conv2d):
        init.xavier_uniform_(layer.weight)

def he_init(layer):
    if isinstance(layer, nn.Conv2d):
        init.kaiming_uniform_(layer.weight)

#-----------------------#
# vis function
#-----------------------#
# save false-color img
def gen_false_color_img(img, save_name='', clist=None):
    h, w, band = img.shape
    img_rgb = np.zeros((h, w, 3))
    img = np.array(img, dtype=np.float64)
    for b in range(band):
        if b not in clist: continue
        img_b = img[..., b]
        img255 = img_b * 255
        # false color
        if b == clist[0]:
            img_rgb[:, :, 0] = img255[:]
        if b == clist[1]:
            img_rgb[:, :, 1] = img255[:]
        if b == clist[2]:
            img_rgb[:, :, 2] = img255[:]
    
    img_rgb = cv2.cvtColor(np.uint8(img_rgb), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(7, 7))  
    plt.axis('off')  
    plt.imshow(img_rgb)
    plt.savefig(save_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    plt.close()

# save heatmap-like image
def vis_errormap(psf, save_name='', annot=False, norm=False, vmax=None):
    if norm:
        psf = psf / psf.sum()
    plt.figure(figsize=(7, 7))
    tmp = sns.heatmap(psf, cmap='turbo', vmax=vmax, annot=annot, xticklabels=False, 
        yticklabels=False, cbar=False, linewidths=0.0, rasterized=True)
    tmp.figure.savefig(save_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    plt.close()


