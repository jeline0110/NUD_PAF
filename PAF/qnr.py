import torch
import torch.nn.functional as F
import math 
from scipy import ndimage
import numpy as np
import cv2
import pdb
from concurrent.futures import ThreadPoolExecutor
from utils import SamePadding
max_workers = 1

def _qindex_lr(info):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    mu1 = info[0]['mu']
    mu2 = info[1]['mu']
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = info[0]['sigma_sq']
    sigma2_sq = info[1]['sigma_sq']

    img1 = info[0]['img']
    img2 = info[1]['img']
    sigma12 =  F.conv2d(img1 * img2, window, stride=1, padding=0, bias=None, groups=1) - mu1_mu2
    assert sigma12.shape == q_map_lr.shape

    idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    q_map_lr[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]

    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
    q_map_lr[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]

    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    q_map_lr[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    
    return torch.mean(q_map_lr)

def _qindex_hr(info):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    mu1 = info[0]['mu']
    mu2 = info[1]['mu']
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = info[0]['sigma_sq']
    sigma2_sq = info[1]['sigma_sq']

    img1 = info[0]['img']
    img2 = info[1]['img']
    sigma12 =  F.conv2d(img1 * img2, window, stride=1, padding=0, bias=None, groups=1) - mu1_mu2

    idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    q_map_hr[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]

    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
    q_map_hr[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]

    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    q_map_hr[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    
    return torch.mean(q_map_hr)

def gaussian2d(N, std):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2)
    t1, t2 = np.meshgrid(t, t)
    std = np.double(std)
    w = np.exp(-0.5 * (t1 / std) ** 2) * np.exp(-0.5 * (t2 / std) ** 2) 

    return w

def kaiser2d(N, beta):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2) / np.double(N - 1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0

    return w

def fir_filter_wind(Hd, w):
    """
    compute fir (finite impulse response) filter with window method
    Hd: desired freqeuncy response (2D)
    w: window (2D)
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = h / np.sum(h)

    return h

def GNyq2win(GNyq, scale=4, N=33):
    """Generate a 2D convolutional window from a given GNyq
    GNyq: Nyquist frequency
    scale: spatial size of PAN / spatial size of MS
    """
    #fir filter with window method
    fcut = 1 / scale
    alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = fir_filter_wind(Hd, w)

    return np.real(h)

def mtf_resize(img, scale=4, N=33):
    # satellite GNyq
    scale = int(scale)
    GNyqPan = 0.15
    # lowpass
    img_ = img.squeeze()
    img_ = img_.astype(np.float32)
    H, W = img_.shape
    lowpass = GNyq2win(GNyqPan, scale, N=N)
    img_ = ndimage.filters.correlate(img_, lowpass, mode='nearest')
    # downsampling
    output_size = (W // scale, H // scale)
    img_ = cv2.resize(img_, dsize=output_size, interpolation=cv2.INTER_NEAREST)

    return img_

def get_lambda_operator(img_info, ishr=False):
    C = len(img_info)
    data = [(img_info[i], img_info[j]) for i in range(C) for j in range(i+1, C)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if ishr: Q = list(executor.map(_qindex_hr, data))
        else: Q = list(executor.map(_qindex_lr, data))
    Q = torch.tensor(Q)

    return Q

def get_lambda_operator_fake(img_info1, img_info2, ishr=False):
    C = len(img_info1)
    data = [(img_info1[i], img_info2[0]) for i in range(C)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if ishr: Q = list(executor.map(_qindex_hr, data))
        else: Q = list(executor.map(_qindex_lr, data))
    Q = torch.tensor(Q)

    return Q

def D_lambda(Q_hr, Q_lr, p=1):
    """
    Spectral distortion
    """
    D_lambda_index = (torch.abs(Q_hr - Q_lr) ** p).mean()

    return D_lambda_index ** (1/p)

def get_s_operator(img_info1, img_info2, ishr=False):
    C = len(img_info1)
    data = [(img_info1[i], img_info2[0]) for i in range(C)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if ishr: Q = list(executor.map(_qindex_hr, data))
        else: Q = list(executor.map(_qindex_lr, data))
    Q = torch.tensor(Q)

    return Q

def D_s(Q_hr, Q_lr, q=1):
    """
    Spatial distortion
    """
    D_s_index = (torch.abs(Q_hr - Q_lr) ** q).mean()

    return D_s_index ** (1/q)

def info_init(x):
    '''
    info=[dict{'img', 'mu', 'sigma_sq'}, ...]
    '''
    bands = x.shape[1]
    x_pad, _ = SamePadding(x, (window.size(2), window.size(3)), (1, 1))
    window_r = window.repeat(bands, 1, 1, 1)
    mu = F.conv2d(x_pad, window_r, stride=1, padding=0, bias=None, groups=bands)
    mu_sq = mu ** 2
    sigma_sq = F.conv2d(x_pad ** 2, window_r, stride=1, padding=0, bias=None, groups=bands) - mu_sq

    info = [{'img':x_pad[:, i:i+1, ...], 'mu':mu[:, i:i+1, ...], \
        'sigma_sq':sigma_sq[:, i:i+1, ...]} for i in range(x_pad.shape[1])]
    
    return info

def qnr_init(hsi, pan, winsize=33, scale=4):
    '''
    hsi: LRHSI
    pan: HRMSI band-mean
    pan_lr: pan with lowpass
    '''
    global window
    window = (torch.ones((1, 1, winsize, winsize)) / (winsize ** 2)).cuda()
    pan_lr = mtf_resize(pan, scale=scale, N=winsize)
    pan = torch.tensor(pan, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    pan_lr = torch.tensor(pan_lr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

    hsi_info = info_init(hsi)
    pan_info = info_init(pan)
    pan_lr_info = info_init(pan_lr)

    global q_map_hr, q_map_lr
    q_map_hr = torch.ones(pan.shape).cuda()
    q_map_lr = torch.ones(hsi[:, :1, ...].shape).cuda()

    Q_lambda_lr = get_lambda_operator(hsi_info, ishr=False)
    Q_s_lr = get_s_operator(hsi_info, pan_lr_info, ishr=False)

    return pan_info, Q_lambda_lr, Q_s_lr

def qnr(fus, pan_info, Q_lambda_lr, Q_s_lr, p=1, q=1, alpha=1, beta=1):
    '''
    fus: generated HRHSI
    '''
    fus_info = info_init(fus)
    """QNR - No reference IQA"""
    Q_lambda_hr = get_lambda_operator(fus_info, ishr=True)
    Q_s_hr = get_s_operator(fus_info, pan_info, ishr=True)
    D_lambda_idx = float(D_lambda(Q_lambda_hr, Q_lambda_lr, p))
    D_s_idx = float(D_s(Q_s_hr, Q_s_lr, q))
    QNR_idx = float((1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta)

    return D_lambda_idx, D_s_idx, QNR_idx

def qnr_fake_init(hsi, fake_mode='mean'):
    hsi_info = info_init(hsi)
    idx = hsi.shape[1]//2
    # single-band hsi_info
    if fake_mode == 'xband':
        hsi_info_sband = info_init(hsi[:,idx:idx+1,:,:])
    elif fake_mode == 'mean':
        hsi_info_sband = info_init(hsi.mean(1).unsqueeze(1))

    return hsi_info, hsi_info_sband

def qnr_fake(fus, hsi_info, hsi_info_sband, pan_info, Q_s_lr, p=1, q=1, alpha=1, beta=1, fake_mode='mean'):
    '''
    fus: generated HRHSI
    '''
    fus_info = info_init(fus)
    idx = fus.shape[1] // 2
    # single-band fus_info
    if fake_mode == 'xband':
        fus_info_sband = info_init(fus[:,idx:idx+1,:,:])
    elif fake_mode == 'mean':
        fus_info_sband = info_init(fus.mean(1).unsqueeze(1))
    """QNR - No reference IQA"""
    Q_lambda_lr = get_lambda_operator_fake(hsi_info, hsi_info_sband, ishr=False)
    Q_lambda_hr = get_lambda_operator_fake(fus_info, fus_info_sband, ishr=True)
    Q_s_hr = get_s_operator(fus_info, pan_info, ishr=True)
    D_lambda_idx = float(D_lambda(Q_lambda_hr, Q_lambda_lr, p))
    D_s_idx = float(D_s(Q_s_hr, Q_s_lr, q))
    QNR_idx = float((1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta)

    return D_lambda_idx, D_s_idx, QNR_idx
