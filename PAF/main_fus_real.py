import sys
sys.path.append('..')
sys.path.append('../NUD/')
import torch
import torch.nn as nn
import scipy.io as scio
import numpy as np
from net_fus import PAF
from data_fus import Load_NUD
from utils import *
from qnr import qnr_init, qnr
import pdb
seed_torch() 

class Fuse():
    def __init__(self, file='', max_iter=20, milestones=[-1, -1]):
        super().__init__()
        print('Fus file:%s' % file)
        print('Fuse LR-HSI and HR-MSI ...')
        # data load
        data_path = '../datasets/' + file + '.mat'
        mat = scio.loadmat(data_path)
        HSI = mat['lrhsi'].transpose(2, 0, 1)
        MSI = mat['hrmsi'].transpose(2, 0, 1)
        C, h, w = HSI.shape
        c, H, W = MSI.shape
        scale = H // h
        print('-------------------')
        print('Data info:')
        print('-------------------')
        print('HR-MSI image size:[%d,%d,%d], LR-HSI image size:[%d,%d,%d]' \
            % (H, W, c, h, w, C))
        print('Spatial Fusion scale:%d' % scale)
        print('Spectral Fusion scale:%d->%d' % (c, C))
        # NUD load
        deg_info = {'name': file, 'C': C, 'c': c, 'scale': scale}
        self.est_model, self.warp_map = Load_NUD(deg_info)
        self.save_dir = os.path.join('../results/fus', file)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # data init
        self.pos_matrix_lr = get_pos_matrix((h, w)).cuda() 
        self.pos_matrix_hr = get_pos_matrix((H, W)).cuda()
        self.lrhsi = torch.Tensor(HSI).unsqueeze(0).cuda()
        self.hrmsi = torch.Tensor(MSI).unsqueeze(0).cuda()
        with torch.no_grad():
            self.lr2hsi = self.est_model.SpaD(self.lrhsi, mode='test', warp_map=self.warp_map)
            self.lrmsi = self.est_model.SpeD(self.lrhsi)
        # qnr init
        pan = self.hrmsi.squeeze(0).cpu().numpy().mean(0)
        self.pan_info, self.Q_lambda_lr, self.Q_s_lr = qnr_init(self.lrhsi, pan, winsize=33, scale=scale)
        # training init
        self.fus_model = PAF(C, self.est_model).cuda()
        self.l1 = nn.L1Loss()
        # optim
        self.LR = 5e-4
        self.lrs = [1e-4, 1e-5]
        self.optimizer = torch.optim.Adam(get_params(self.fus_model), lr=self.LR)
        self.max_iter = 135
        self.milestones = [-1, -1]
        self.print_per_iter = 5

    def run(self):
        for iter_ in range(1, self.max_iter + 1):
            lr_scheduler(self.optimizer, self.lrs, iter_, self.milestones)
            # lambda_lr_scheduler(self.optimizer, self.LR, iter_, self.max_iter)
            self.optimizer.zero_grad()
            out1, out2, out3 = self.fus_model(lr2hsi=self.lr2hsi, lrmsi=self.lrmsi, \
                pos=self.pos_matrix_lr, lrhsi=self.lrhsi, hrmsi=self.hrmsi, hrpos=self.pos_matrix_hr, \
                warp_map=self.warp_map, mode='train')
            loss1 = train_loss(out1, self.lrhsi, self.l1)
            loss2 = train_loss(out2, self.lrhsi, self.l1) * 0.5
            loss3 = train_loss(out3, self.hrmsi, self.l1) * 0.1
            loss = loss1 + loss2 + loss3
            loss.backward()
            self.optimizer.step()

            if iter_ % self.print_per_iter == 0:
                lr = get_current_lr(self.optimizer)
                info = 'iter:[%d/%d], lr:%.6f, loss1:%.7f, loss2:%.7f, loss3:%.7f' \
                    % (iter_, self.max_iter, lr, loss1, loss2, loss3)
                print(info)

        with torch.no_grad():
            infer_out, _, _ = self.fus_model(lrhsi=self.lrhsi, hrmsi=self.hrmsi, 
                hrpos=self.pos_matrix_hr, mode='test')

        infer_out = torch.clamp(infer_out, 0.0, 1.0)
        D_lambda, D_s, QNR = qnr(infer_out, self.pan_info, self.Q_lambda_lr, self.Q_s_lr)
        info = 'Dλ:%.5f, Ds:%.5f, QNR:%.5f' % (D_lambda, D_s, QNR)
        print(info)
        with open(self.save_dir + '/info.txt', 'a') as f:
            f.write(info)
        fusion = np.squeeze(infer_out.detach().cpu().numpy()).transpose(1, 2, 0)
        np.save(self.save_dir + '/fusion.npy', fusion)
        gen_false_color_img(fusion, self.save_dir + '/fusion.pdf', clist=[10, 30, 50])
        torch.save(self.fus_model.state_dict(), self.save_dir + '/fus_model.pkl')

if __name__ == '__main__': 
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    file = 'hypsen'
    fuse = Fuse(file=file)
    fuse.run()
    torch.cuda.empty_cache()

