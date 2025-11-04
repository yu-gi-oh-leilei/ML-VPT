import torch
import torch.nn as nn

class AsymmetricLoss_partial(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, 
                 disable_torch_grad_focal_loss=True,
                 thresh_pos=0.9, thresh_neg=-0.9):
        super(AsymmetricLoss_partial, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        
        self.thresh_pos = thresh_pos
        self.thresh_neg = thresh_neg

        self.margin = 0.0

        self.is_forward_pseudo = False

    def forward_known(self, x, y, is_mean=True):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        y_pos_mask = (y > self.margin).float()
        y_neg_mask = (y < self.margin).float()
        # Basic CE calculation
        los_pos = y_pos_mask * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = y_neg_mask * torch.log(xs_neg.clamp(min=self.eps))

        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y_pos_mask
            pt1 = xs_neg * y_neg_mask  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y_pos_mask + self.gamma_neg * y_neg_mask
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum() / x.shape[0] if is_mean else -loss.mean()



    def forward_pseudo(self, x, y, is_mean=True):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Mask calculation
        y_pos_mask = (y > self.margin).float()
        y_neg_mask = (y < self.margin).float()

        unkown_mask = (y.clone() == 0).float()
        pseudo_mask_pos = (x.clone().detach() >= self.thresh_pos).float() * unkown_mask
        pseudo_mask_neg = (x.clone().detach() < self.thresh_neg).float() * unkown_mask
        
        # Basic CE calculation
        los_pos = y_pos_mask * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = y_neg_mask * torch.log(xs_neg.clamp(min=self.eps))

        pseudo_los_pos = pseudo_mask_pos * torch.log(xs_pos.clamp(min=self.eps))
        pseudo_los_neg = pseudo_mask_neg * torch.log(xs_neg.clamp(min=self.eps))

        loss = los_pos + los_neg
        pseudo_loss =  pseudo_los_pos + pseudo_los_neg


        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            pt0 = xs_pos * y_pos_mask
            pt1 = xs_neg * y_neg_mask  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y_pos_mask + self.gamma_neg * y_neg_mask
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)

            pseudo_pt0 = xs_pos * pseudo_mask_pos
            pseudo_pt1 = xs_neg * pseudo_mask_neg  # pt = p if t > 0 else 1-p
            pseudo_pt = pseudo_pt0 + pseudo_pt1
            pseudo_one_sided_gamma = self.gamma_pos * pseudo_mask_pos + self.gamma_neg * pseudo_mask_neg
            pseudo_one_sided_w = torch.pow(1 - pseudo_pt, pseudo_one_sided_gamma)
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss = loss * one_sided_w
            pseudo_loss = pseudo_loss * pseudo_one_sided_w

        losses = loss + pseudo_loss
        return -losses.sum() / x.shape[0] if is_mean else -losses.mean()


    def forward(self, x, y, is_mean=True):
        if self.is_forward_pseudo is True:
            return self.forward_pseudo(x, y, is_mean)
        else:
            return self.forward_known(x, y, is_mean)


if __name__ == '__main__':
    import numpy as np

    device = 'cuda:0'

    output_0 = np.load('/media/data/maleilei/MLIC/MLIC_Partial/CLIP_PartialLabeling_limited/debug/output.0.npy')
    target_0 = np.load('/media/data/maleilei/MLIC/MLIC_Partial/CLIP_PartialLabeling_limited/debug/target.0.npy')

    output_0 = torch.from_numpy(output_0).to(device)
    target_0 = torch.from_numpy(target_0).to(device)

    print(type(output_0))
    print(type(target_0))
    
    # print(output_0[0])
    # print(target_0[0])

    criterion = AsymmetricLoss_partial(
        gamma_neg=2, 
        gamma_pos=1, 
        clip=0.05, 
        eps=1e-8, 
        disable_torch_grad_focal_loss=True,
        thresh_pos=0.9, 
        thresh_neg=-0.9
    )

    loss = criterion(output_0, target_0)
    print(loss)

        # if is_mean:
        #     if torch.mean(y_pos_mask + y_neg_mask) != 0:
        #         return -torch.mean(loss[(y_pos_mask > 0) | (y_neg_mask > 0)])
        #     else:
        #         return -torch.mean(loss)
        # else:
        #     if torch.sum(y_pos_mask + y_neg_mask) != 0:
        #         return -torch.sum(loss[(y_pos_mask > 0) | (y_neg_mask > 0)])
        #     else:
        #         return -torch.sum(loss)



        # if is_mean:
        #     if torch.mean(y_pos_mask + y_neg_mask + pseudo_mask_pos + pseudo_mask_neg) != 0:
        #         return torch.mean(loss[(y_pos_mask > 0) | (y_neg_mask > 0)]) + torch.mean(pseudo_loss[(pseudo_mask_pos > 0) | (pseudo_mask_neg > 0)])
        #     else:
        #         return torch.mean(loss)
        # else:
        #     if torch.sum(y_pos_mask + y_neg_mask + pseudo_mask_pos + pseudo_mask_neg) != 0:
        #         return torch.sum(loss[(y_pos_mask > 0) | (y_neg_mask > 0)]) + torch.sum(pseudo_loss[(pseudo_mask_pos > 0) | (pseudo_mask_neg > 0)])
        #     else:
        #         return torch.sum(loss)

# def dualcoop_loss(inputs, inputs_g, targets):
#     """
#     using official ASL loss.
#     """
#     loss_fun = AsymmetricLoss_partial(gamma_neg=2, gamma_pos=1, clip=0.05)

#     return loss_fun(inputs, targets) # + loss_fun(inputs_g, targets)

                # losses = loss[(y_pos_mask > 0) | (y_neg_mask > 0)] + pseudo_loss[(pseudo_mask_pos > 0) | (pseudo_mask_neg > 0)]
                # print(loss.shape)
                # print(loss[0])
                # print([(y_pos_mask > 0) | (y_neg_mask > 0)][0])
                # print(loss[(y_pos_mask > 0) | (y_neg_mask > 0)][0])
                # # return torch.mean(losses)
                # # losses = loss[(y_pos_mask > 0) | (y_neg_mask > 0)] + pseudo_loss[(pseudo_mask_pos > 0) | (pseudo_mask_neg > 0)]
                # print(loss[(y_pos_mask > 0) | (y_neg_mask > 0)].shape, pseudo_loss[(pseudo_mask_pos > 0) | (pseudo_mask_neg > 0)].shape)
                # return torch.mean(loss[(y_pos_mask > 0) | (y_neg_mask > 0)]) + torch.mean(pseudo_loss[(pseudo_mask_pos > 0) | (pseudo_mask_neg > 0)])

                            # losses = -loss - pseudo_loss
            # print(-losses.sum() / x.shape[0])

            # print(-loss.sum() / x.shape[0])
            # print(-pseudo_loss.sum() / x.shape[0])

        # y_pos_mask y_neg_mask pseudo_mask_pos pseudo_mask_neg
        # return torch.sum(loss[(y_pos_mask > 0) | (y_neg_mask > 0) | (pseudo_mask_pos > 0) | (pseudo_mask_neg > 0)]) if torch.sum(y_pos_mask + y_neg_mask + pseudo_mask_pos + pseudo_mask_neg) != 0 else torch.sum(loss)
        # print('sum', -loss.sum() / x.shape[0])
        # print('mean', -loss.mean())

        # print('mean', -loss.mean())

        # return -loss.sum() / x.shape[0] if if_partial else -loss.mean()