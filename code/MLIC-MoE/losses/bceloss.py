import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class  BinaryCrossEntropyLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading, favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(BinaryCrossEntropyLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    if self.disable_torch_grad_focal_loss:
                        torch.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    if self.disable_torch_grad_focal_loss:
                        torch.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)   
                self.loss *= self.asymmetric_w
    
        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000

        return _loss



class BCELoss(nn.Module):

    def __init__(self, margin=0.0, reduce=None, size_average=None):
        super(BCELoss, self).__init__()

        self.margin = margin

        self.reduce = reduce
        self.size_average = size_average

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):

        input, target = input.float(), target.float()

        positive_mask = (target > self.margin).float()
        negative_mask = (target < -self.margin).float()

        positive_loss = self.BCEWithLogitsLoss(input, target)
        negative_loss = self.BCEWithLogitsLoss(-input, -target)

        loss = positive_mask * positive_loss + negative_mask * negative_loss

        if self.reduce:
            if self.size_average:
                return torch.mean(loss[(positive_mask > 0) | (negative_mask > 0)]) if torch.sum(positive_mask + negative_mask) != 0 else torch.mean(loss)
            return torch.sum(loss[(positive_mask > 0) | (negative_mask > 0)]) if torch.sum(positive_mask + negative_mask) != 0 else torch.sum(loss)
        return loss

if __name__ == '__main__':
    import numpy as np

    device = 'cuda:0'

    output_0 = np.load('/media/data2/maleilei/MLIC/CDCR/data/coco_output_gather.0.npy')
    target_0 = np.load('/media/data2/maleilei/MLIC/CDCR/data/coco_target_gather.0.npy')

    output_0 = torch.from_numpy(output_0).to(device)
    target_0 = torch.from_numpy(target_0).to(device)

    print(type(output_0))
    print(type(target_0))


    criterion = BinaryCrossEntropyLossOptimized()

    loss = criterion(output_0, target_0)
    print(loss)

'''
    output_0 = np.load('/media/data2/maleilei/MLIC/CDCR/data/coco_output_gather.0.npy')
    target_0 = np.load('/media/data2/maleilei/MLIC/CDCR/data/coco_target_gather.0.npy')

    output_1 = np.load('/media/data2/maleilei/MLIC/CDCR/data/coco_output_gather.1.npy')
    target_1 = np.load('/media/data2/maleilei/MLIC/CDCR/data/coco_target_gather.1.npy')
    
    output_0 = torch.from_numpy(output_0)
    target_0 = torch.from_numpy(target_0)

    output_1 = torch.from_numpy(output_1)
    target_1 = torch.from_numpy(target_1)

    print((output_1-output_0).sum())
    print((target_1-target_0).sum())
'''

