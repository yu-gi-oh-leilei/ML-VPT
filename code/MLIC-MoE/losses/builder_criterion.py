from copy import deepcopy
import torch
import torch.nn as nn
from .bceloss import BinaryCrossEntropyLossOptimized, BCELoss
from .dualcoop_loss import AsymmetricLoss_partial
from .aslloss import AsymmetricLoss, AsymmetricLossOptimized
from .kl_loss import DistillKL
from models.builder_network import SetCriterion


def build_criterion(cfg, model):

    weight_dict = {'loss_cls': cfg.LOSS.Coef.cls_asl_coef, 
                   'loss_co_moe_cls': cfg.LOSS.Coef.cls_co_moe_coef,
                   'loss_cl_moe_cls': cfg.LOSS.Coef.cls_cl_moe_coef,
                   'loss_co_cls': cfg.LOSS.Coef.cls_co_coef, 
                   'loss_cl_cls': cfg.LOSS.Coef.cls_cl_coef,}
    # losses = ['loss_cls']
    losses = ['loss_co_cls', 'loss_cl_cls', 'loss_co_moe_cls', 'loss_cl_moe_cls']
    losses_test = ['loss_cls']
    criterion = SetCriterion(weight_dict, losses, cfg)

    if cfg.LOSS.loss_mode == 'asl':
        cls_criterion = AsymmetricLossOptimized(
            gamma_neg=cfg.LOSS.ASL.gamma_neg, 
            gamma_pos=cfg.LOSS.ASL.gamma_pos,
            clip=cfg.LOSS.ASL.loss_clip,
            disable_torch_grad_focal_loss=cfg.LOSS.ASL.dtgfl,
            eps=cfg.LOSS.ASL.eps)
        criterion.cls_loss = deepcopy(cls_criterion)
        criterion.co_cls_loss = deepcopy(cls_criterion)
        criterion.cl_cls_loss = deepcopy(cls_criterion)
        criterion.co_moe_cls_loss = deepcopy(cls_criterion)
        criterion.cl_moe_cls_loss = deepcopy(cls_criterion)

        del cls_criterion

    elif cfg.LOSS.loss_mode == 'aslv1':

        cls_criterion = AsymmetricLoss(
            gamma_neg=cfg.LOSS.ASL.gamma_neg, 
            gamma_pos=cfg.LOSS.ASL.gamma_pos,
            clip=cfg.LOSS.ASL.loss_clip,
            disable_torch_grad_focal_loss=cfg.LOSS.ASL.dtgfl,
            eps=cfg.LOSS.ASL.eps)
        criterion.cls_loss = deepcopy(cls_criterion)
        criterion.co_cls_loss = deepcopy(cls_criterion)
        criterion.cl_cls_loss = deepcopy(cls_criterion)
        criterion.moe_cls_loss = deepcopy(cls_criterion)
        del cls_criterion
        
    elif cfg.LOSS.loss_mode == 'bce':
        criterion.cls_loss = BCELoss(reduce=True, size_average=True)
        criterion.co_cls_loss = BCELoss(reduce=True, size_average=True)
        criterion.cl_cls_loss = BCELoss(reduce=True, size_average=True)

    elif cfg.LOSS.loss_mode == 'multi_bce':
        criterion.cls_loss = nn.MultiLabelSoftMarginLoss()
        criterion.co_cls_loss = nn.MultiLabelSoftMarginLoss()
        criterion.cl_cls_loss = nn.MultiLabelSoftMarginLoss()

    device = cal_gpu(model)

    criterion = criterion.to(device)
    
    return criterion


def cal_gpu(module):
    if hasattr(module, 'module') or isinstance(module, torch.nn.DataParallel):
        for submodule in module.module.children():
            if hasattr(submodule, "_parameters"):
                parameters = submodule._parameters
                if "weight" in parameters:
                    return parameters["weight"].device
    else:
        for submodule in module.children():
            if hasattr(submodule, "_parameters"):
                parameters = submodule._parameters
                if "weight" in parameters:
                    return parameters["weight"].device