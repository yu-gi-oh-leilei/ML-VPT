import torch
import torch.nn as nn
from torch.nn.functional import multilabel_soft_margin_loss
import torch.nn.functional as F

class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2)
        return loss
    

if __name__ == '__main__':
    import numpy as np
    import os
    import os.path as osp

    device = 'cuda:0'
    path = '/media/data/maleilei/MLIC/MLIC_Partial/CLIP_PartialLabeling_limited/data/kldubug'
    logits_l = osp.join(path, 'logits_l.0.npy')
    logits_t = osp.join(path, 'logits_t.0.npy')

    logits_l = np.load(logits_l)
    logits_t = np.load(logits_t)

    logits_l = torch.from_numpy(logits_l).to(device)
    logits_t = torch.from_numpy(logits_t).to(device)


    # output_0 = np.load('/media/data/maleilei/MLIC/MLIC_Partial/CLIP_PartialLabeling_limited/data/kldubug/coco_output_gather.0.npy')
    # target_0 = np.load('/media/data2/maleilei/MLIC/CDCR/data/coco_target_gather.0.npy')
    # output_0 = torch.from_numpy(output_0).to(device)
    # target_0 = torch.from_numpy(target_0).to(device)

    print(type(logits_l))
    print(type(logits_t))


    criterion = DistillKL(T=4)

    loss = criterion(logits_t, logits_l)
    print(loss)

    # if is_klloss:
    #     debug = '/media/data/maleilei/MLIC/MLIC_Partial/CLIP_PartialLabeling_limited/data/kldubug/'
    #     os.makedirs(debug, exist_ok=True)
    #     np.save(debug+'logits_l.{}.npy'.format(dist.get_rank()), logits_l.cpu().detach().numpy())
    #     np.save(debug+'logits_g.{}.npy'.format(dist.get_rank()), logits_g.cpu().detach().numpy())
    #     np.save(debug+'logits_t.{}.npy'.format(dist.get_rank()), logits_t.cpu().detach().numpy())