import sys
sys.path.append('..')
sys.path.append('../NUD/')
import torch
import torch.nn as nn
import scipy.io as scio
import numpy as np
from net_fus import PAF
from data_fus import Syndata_generator
from utils import *
from qnr import qnr_init, qnr
import pdb
seed_torch() 

class Fuse():
    def __init__(self, ffile='', dfile='', max_iter=20, milestones=[-1, -1]):
        super().__init__()
        print('Fus file:%s, Deg file:%s' % (ffile, dfile))
        print('Fuse LR-HSI and HR-MSI ...')
        # data load
        self.hrhsi, self.lrhsi, self.hrmsi, self.lr2hsi, self.lrmsi, est_model, \
            self.warp_map = Syndata_generator(ffile=ffile, dfile=dfile)
        C, h, w  = self.lrhsi.shape[1:]
        c, H, W = self.hrmsi.shape[1:]
        scale = H // h
        self.HRHSI = self.hrhsi.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        print('-------------------')
        print('Data info:')
        print('-------------------')
        print('HR-MSI image size:[%d,%d,%d], LR-HSI image size:[%d,%d,%d]' \
            % (H, W, c, h, w, C))
        print('Spatial Fusion scale:%d' % scale)
        print('Spectral Fusion scale:%d->%d' % (c, C))
        self.save_dir = os.path.join('../results/fus', ffile)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # data init
        self.pos_matrix_lr = get_pos_matrix((h, w)).cuda() 
        self.pos_matrix_hr = get_pos_matrix((H, W)).cuda()
        # training init
        self.fus_model = PAF(C, est_model).cuda()
        self.l1 = nn.L1Loss()
        # optim
        LR = 5e-4
        self.lrs = [1e-4, 1e-5]
        self.optimizer = torch.optim.Adam(get_params(self.fus_model), lr=LR)
        self.max_iter = 120000
        self.milestones = [40000, -1]
        self.print_per_iter = 1000
        self.start_val_iter = 80000

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
            # start val
            if iter_ % self.print_per_iter == 0 and iter_ > self.start_val_iter:
                self.fus_model.eval()
                with torch.no_grad():
                    infer_out, _, _ = self.fus_model(lrhsi=self.lrhsi, hrmsi=self.hrmsi, \
                        hrpos=self.pos_matrix_hr, mode='test')
                fusion = np.squeeze(infer_out.detach().cpu().numpy())
                fusion = np.clip(fusion, 0, 1).transpose(1, 2, 0)
                psnr = get_psnr_np(fusion, self.HRHSI)
                info = 'PSNR:%.4f' % psnr
                print(info)
                self.fus_model.train()

        with torch.no_grad():
            infer_out, _, _ = self.fus_model(lrhsi=self.lrhsi, hrmsi=self.hrmsi, \
                hrpos=self.pos_matrix_hr, mode='test')

        fusion = np.squeeze(infer_out.detach().cpu().numpy())
        fusion = np.clip(fusion, 0, 1).transpose(1, 2, 0)
        print('Get Final Results !!!')
        info1 = 'PSNR:%.4f' % get_psnr_np(fusion, self.HRHSI)
        print(info1)
        info2 = 'RMSE:%.4f' % get_rmse_np(fusion, self.HRHSI)
        print(info2)
        info3 = 'ERGAS:%.4f' % ergas(fusion, self.HRHSI)
        print(info3)
        info4 = 'SSIM:%.4f' % get_ssim_np(fusion, self.HRHSI, data_range=1.0, multichannel=True, channel_axis=2)
        print(info4)
        info5 = 'SAM:%.4f' % sam(fusion, self.HRHSI)
        print(info5)
        info = info1 + '\n' + info2 + '\n' + info3 + '\n' + info4 + '\n' + info5 + '\n'
        with open(self.save_dir + '/info.txt', 'a') as f:
            f.write(info)

        error_map = get_error_map(fusion, self.HRHSI, norm=True)
        vis_errormap(error_map, self.save_dir + '/fus_error.pdf', vmax=1.0)
        np.save(self.save_dir + '/fusion.npy', fusion)
        gen_false_color_img(fusion, self.save_dir + '/fusion.pdf', clist=[10, 30, 50])
        torch.save(self.fus_model.state_dict(), self.save_dir + '/fus_model.pkl')

if __name__ == '__main__': 
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    ffile = 'paviau'
    dfile = 'hypsen'
    fuse = Fuse(ffile=ffile, dfile=dfile)
    fuse.run()
    torch.cuda.empty_cache()

