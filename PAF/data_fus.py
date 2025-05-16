from net_deg import NUD
from utils import *
import scipy.io as scio
import torch 
import numpy as np 
import pdb

def Load_NUD(deg_info):
    file, C, c, scale = deg_info['name'], deg_info['C'], deg_info['c'], deg_info['scale']
    est_model = NUD(C, c, scale, pos_dim=4, m_ksize=25).cuda()
    model_dir = os.path.join('../results/deg2', file)
    est_model.load_state_dict(torch.load(model_dir + '/est_model.pkl'))
    est_model.eval()

    map_path = model_dir + '/warp_map.pt'
    warp_map = torch.load(map_path).cuda() # b, c, h, w
    print('Deg components load done !')

    return est_model, warp_map

def Syndata_generator(ffile='paviau', dfile='hypsen'):
    # load NUD
    deg_info = {'name':'hypsen', 'C':89, 'c':4, 'scale':3}
    est_model, warp_map = Load_NUD(deg_info)
    # load data
    scale = deg_info['scale']
    HRHSI = scio.loadmat('../datasets/%s.mat' % ffile)['data']
    HRHSI = HRHSI[0:scale**2*int(HRHSI.shape[0]/scale**2), \
        0:scale**2*int(HRHSI.shape[1]/scale**2), :]
    if ffile == 'paviau' and dfile == 'hypsen':
        HRHSI = HRHSI[:, :, 4:-10] # band match

    assert HRHSI.shape[-1] == deg_info['C']
    hrhsi = torch.tensor(HRHSI, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
    with torch.no_grad():
        lrhsi = est_model.SpaD(hrhsi, mode='test', warp_map=warp_map)
        H, W = hrhsi.shape[2:]
        pos_matrix_hr = get_pos_matrix((H, W)).cuda()
        hrmsi = est_model.SpeD(hrhsi, pos_matrix_hr)

        lr2hsi = est_model.SpaD(lrhsi, mode='test', warp_map=warp_map)
        h, w = lrhsi.shape[2:]
        pos_matrix_lr = uniform_downsample(pos_matrix_hr, h, w)
        lrmsi = est_model.SpeD(lrhsi, pos_matrix_lr)

    return hrhsi, lrhsi, hrmsi, lr2hsi, lrmsi, est_model, warp_map
