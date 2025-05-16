import sys
sys.path.append('..') # add root path
import torch
from net_deg import NUD
from data_deg import DegDataset
from utils import *
import pdb 
seed_torch()

class Estimate():
    def __init__(self, file=''):
        super().__init__()
        print('Deg file:%s' % file)
        print('Estimate Spectral Degradation and Spatial Degradation ...')
        # data init
        data_path = '../datasets/' + file + '.mat'
        self.dataset = DegDataset(data_path=data_path)
        # net init
        C, h, w = self.dataset.lrhsi.shape[1:]
        c, H, W = self.dataset.hrmsi.shape[1:]
        scale = H // h
        pos_matrix_hr = get_pos_matrix((H, W)).cuda()
        pos_matrix_lr = uniform_downsample(pos_matrix_hr, h, w)
        self.est_model = NUD(C, c, scale, pos=pos_matrix_lr, file=file, pos_dim=4, m_ksize=25).cuda()
        # mkidr
        self.save_dir = os.path.join('../results/deg', file)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # other init
        lr = 5e-4 
        self.max_iter = 3500
        self.print_per_iter = 100
        self.criterion = get_ssim_torch
        self.optimizer = torch.optim.Adam(self.est_model.parameters(), lr=lr)

    def run(self):
        print('-------------------')
        print('Start Estimating !')
        print('-------------------')
        for iter_ in range(self.max_iter):
            self.optimizer.zero_grad()
            lrhsi_sped, hrmsi_spad = self.est_model(self.dataset.lrhsi, self.dataset.hrmsi)
            loss = 1. - self.criterion(lrhsi_sped, hrmsi_spad) 
            loss.backward()
            self.optimizer.step()
            # print info
            if (iter_ + 1) % self.print_per_iter == 0:
                lr = self.optimizer.param_groups[0]['lr']
                info1 = 'iter:[%d/%d], loss:%.4f' % (iter_ + 1, self.max_iter, loss)
                print(info1)
                info2 = 'SSIM:%.4f' % get_ssim_torch(lrhsi_sped, hrmsi_spad)
                print(info2)
                info3 = 'MAE:%.4f' % get_mae_torch(lrhsi_sped, hrmsi_spad)
                print(info3)
                info4 = 'RMSE:%.4f' % get_rmse_torch(lrhsi_sped, hrmsi_spad)
                print(info4)
                info = info1 + '\n' + info2 + '\n' + info3 + '\n' + info4 + '\n'
                with open(self.save_dir + '/info.txt', 'a') as f:
                    f.write(info)

        lrhsi_sped = lrhsi_sped.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        hrmsi_spad = hrmsi_spad.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        gen_false_color_img(lrhsi_sped, self.save_dir + '/lrhsi_sped.pdf', clist=[0, 1, 2])
        gen_false_color_img(hrmsi_spad, self.save_dir + '/hrmsi_spad.pdf', clist=[0, 1, 2])
        # save results
        torch.save(self.est_model.state_dict(), self.save_dir + '/est_model.pkl')
        error_map = get_error_map(lrhsi_sped, hrmsi_spad, norm=True)
        vis_errormap(error_map, self.save_dir + '/degesti_error.pdf')

if __name__ == '__main__': 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    file = 'hypsen'
    estimate = Estimate(file=file)
    estimate.run()
    torch.cuda.empty_cache()