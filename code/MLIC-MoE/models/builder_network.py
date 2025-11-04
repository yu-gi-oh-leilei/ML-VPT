import os, sys
import os.path as osp

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import math
import json

from typing import Optional, Type, Union

from utils.misc import clean_state_dict
from .factory import register_model, build_backbone
from .utils_model import Element_Wise_Layer, build_position_encoding, TransformerEncoder, \
    LowRankBilinearAttention, GatedGNN
from .classifier import GroupWiseLinear


# DATA.cls_group = 'data/aux_data/'
def build_group_cls(cfg):
    num_class = cfg.DATA.num_class
    aux_tokens_num = cfg.MODEL.PROMPT.aux_tokens
    # co_tokens_num = cl_tokens_num = int(aux_tokens_num / 2)
    group_root_path = os.path.join(cfg.DATA.cls_group, cfg.DATA.dataname)

    if cfg.DATA.dataname in ('coco'):
        co_group_path = os.path.join(group_root_path, 'coco_coocurrence{}_0.json'.format(aux_tokens_num))
        cl_group_path = os.path.join(group_root_path, 'coco_cluster{}_0.json'.format(aux_tokens_num))
    elif cfg.DATA.dataname in ('voc2007'):
        co_group_path = os.path.join(group_root_path, 'voc_coocurrence{}_0.json'.format(aux_tokens_num))
        cl_group_path = os.path.join(group_root_path, 'voc_cluster{}_0.json'.format(aux_tokens_num))
    elif cfg.DATA.dataname in ('nus'):
        co_group_path = os.path.join(group_root_path, 'nus_coocurrence{}_0.json'.format(aux_tokens_num))
        cl_group_path = os.path.join(group_root_path, 'nus_cluster{}_0.json'.format(aux_tokens_num))
    elif cfg.DATA.dataname in ('vg256'):
        co_group_path = os.path.join(group_root_path, 'vg256_coocurrence{}_0.json'.format(aux_tokens_num))
        cl_group_path = os.path.join(group_root_path, 'vg256_cluster{}_0.json'.format(aux_tokens_num))
    
    else:
        assert False, 'not supported dataname: {}'.format(cfg.DATA.dataname)

    with open(co_group_path, 'r') as f:
        co_group = json.load(f)
    with open(cl_group_path, 'r') as f:
        cl_group = json.load(f)

    len_coocurrence = len(co_group)
    co_index = [0 for i in range(num_class)]
    for i in range(len_coocurrence):
        for k, v, in co_group[i].items():
            co_index[int(k)] = int(i)

    len_cluster = len(cl_group)
    cl_index = [0 for i in range(num_class)]
    for i in range(len_cluster):
        for k, v, in cl_group[i].items():
            cl_index[int(k)] = int(i) # + len_coocurrence

    co_index = torch.tensor(co_index)
    cl_index = torch.tensor(cl_index)

    print('co_index:', co_index)
    print('cl_index:', cl_index)


    return co_index, cl_index


class GroupFCWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class Scaler(nn.Module):
    def __init__(self, scale: Optional[float] = None):
        super().__init__()

        if scale is None:
            self.register_parameter("scale", nn.Parameter(torch.tensor(1.0)))
        else:
            self.scale = scale

    def forward(self, input):
        return input * self.scale

    def extra_repr(self):
        learnable = isinstance(self.scale, nn.Parameter)
        return f"scale={self.scale:.4f}, learnable={learnable}"

class Adapt_Experts(nn.Module):
    # Adapt as Experts in Mixture of Experts
    def __init__(self, 
                group_num: int, 
                embed_dim: int, 
                down_sample: Union[float, int] = 5, 
                mode: str = "parallel",
                drop: float = 0.0, 
                act_layer: Type[nn.Module] = nn.GELU,
                scale: Optional[float] = None,
                bias=True):
        super().__init__()
        
        assert mode in ["before", "after", "parallel"], f"Unknown mode {mode}"
        self.mode = mode
        self.group_num = group_num
        self.embed_dim = embed_dim
        self.bias = bias

        self.hidden_dim = down_sample
        if isinstance(down_sample, float):
            self.hidden_dim = int(embed_dim * down_sample)

        # Parameter
        self.W1 = nn.Parameter(torch.Tensor(group_num, embed_dim, self.hidden_dim))
        if bias:
            self.b1 = nn.Parameter(torch.Tensor(group_num, self.hidden_dim))
    
        self.drop = nn.Dropout(drop)
        self.act = act_layer()

        self.W2 = nn.Parameter(torch.Tensor(group_num, self.hidden_dim, embed_dim))
        if bias:
            self.b2 = nn.Parameter(torch.Tensor(group_num, embed_dim))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for i in range(self.W1.size(0)):
                fc1 = torch.nn.Linear(self.embed_dim, self.hidden_dim, bias=self.bias is not None)
                fc2 = torch.nn.Linear(self.hidden_dim, self.embed_dim, bias=self.bias is not None)

                self.W1[i] = fc1.weight.t()
                if self.bias is not None:
                    self.b1[i] = fc1.bias
                self.W2[i] = fc2.weight.t()
                if self.bias is not None:
                    self.b2[i] = fc2.bias

    def forward_adapt(self, x):
        # x: B,K,d
        x = x.unsqueeze(2)  # (bs, len, 1, dim)
        x = torch.matmul(x, self.W1).squeeze(2)
        if self.bias:
            x = x + self.b1
        x = self.act(x)
        x = self.drop(x)        
        x = x.unsqueeze(2)  # (bs, len, 1, dim)
        x = torch.matmul(x, self.W2).squeeze(2)
        if self.bias:
            x = x + self.b2
        return x
    def forward(self, x):
        if self.mode == 'parallel':
            return self.forward_adapt(x) + x
        elif self.mode == 'before' or self.mode == 'after':
            return self.forward_adapt(x)
        else:
            raise NotImplementedError


class Gate_MOE(nn.Module):
    # GateNetwork in Mixture of Experts
    def __init__(self, num_class, num_experts_classes, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.num_experts_classes = num_experts_classes
        self.num_experts = num_experts_classes * num_class

        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, self.num_experts, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, self.num_experts))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_experts):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_experts):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def moe_attention(self, batch_size, logits_pat_1, logits_pat_2):
        # print(logits_pat_1.shape, logits_pat_2.shape)
        split_list1 = torch.split(logits_pat_1, batch_size)              # [64,80] -> 4 * [16,80]
        logits_joint1 = torch.stack(split_list1, dim=1)                  # 4 * [16,80] -> [16, 4, 80]
        # print(logits_joint1.shape)

        logits_sfmx1 = torch.softmax(logits_joint1, dim=1)               # [16, {4}, 80]
        
        split_list2 = torch.split(logits_pat_2, batch_size)              # [64,80] -> 4 * [16,80]
        logits_joint2 = torch.stack(split_list2, dim=1)                  # 4 * [16,80] -> [16, 4, 80]
        
        logits_joint = (logits_sfmx1 * logits_joint2).sum(dim=1)         # [16, 4, 80] -> [16,80]
        return logits_sfmx1, logits_joint

    def forward(self, x, batch_size):

        logit = (self.W * x).sum(dim=2)                                   # [16, 320]    

        # if self.bias:
        #     logit = logit + self.b

        logit = torch.cat(torch.chunk(logit, self.num_experts_classes, dim=1), dim=0)

        logits_atten, logits_joint = self.moe_attention(batch_size=batch_size, logits_pat_1=logit, logits_pat_2=logit)
        
        return logits_atten, logits_joint


class MLIC(nn.Module):
    def __init__(self, backbone, feature_dim, num_class, cfg):
        """[summary]
        Args:
            backbone ([type]): backbone model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        self.img_encoder = backbone
        self.num_class = num_class
        self.cfg = cfg
        self.arch_backbone = cfg.MODEL.BACKBONE.backbone
        self.img_size = cfg.DATA.TRANSFORM.img_size
        self.feature_dim = feature_dim
        # self.aux_tokens = cfg.MODEL.PROMPT.aux_tokens
        # self.scale_tokens = cfg.MODEL.PROMPT.scale_tokens
        # self.co_tokens = self.cl_tokens = int(self.aux_tokens / 2)


        self.aux_tokens = cfg.MODEL.PROMPT.aux_tokens
        self.scale_tokens = cfg.MODEL.PROMPT.scale_tokens


        self.co_tokens = self.aux_tokens
        self.cl_tokens = self.aux_tokens

        # number of tokens
        self.num_aux_tokens = self.aux_tokens * (self.scale_tokens + self.scale_tokens)
        self.num_co_tokens = self.co_tokens * self.scale_tokens
        self.num_cl_tokens = self.cl_tokens * self.scale_tokens

        co_group, cl_group = build_group_cls(cfg)
        self.co_group = co_group
        self.cl_group = cl_group

        if self.scale_tokens == 2:
            self.co_group = torch.cat((co_group+self.co_tokens*0, co_group+self.co_tokens*1), dim=0).cuda()
            self.cl_group = torch.cat((cl_group+self.cl_tokens*2, cl_group+self.cl_tokens*3), dim=0).cuda()
        elif self.scale_tokens == 3:
            self.co_group = torch.cat((co_group+self.co_tokens*0, co_group+self.co_tokens*1, co_group+self.co_tokens*2), dim=0).cuda()
            self.cl_group = torch.cat((cl_group+self.cl_tokens*3, cl_group+self.cl_tokens*4, cl_group+self.cl_tokens*5), dim=0).cuda()
        elif self.scale_tokens == 4:
            self.co_group = torch.cat((co_group+self.co_tokens*0, co_group+self.co_tokens*1, co_group+self.co_tokens*2, co_group+self.co_tokens*3), dim=0).cuda()
            self.cl_group = torch.cat((cl_group+self.cl_tokens*4, cl_group+self.cl_tokens*5, cl_group+self.cl_tokens*6, cl_group+self.cl_tokens*7), dim=0).cuda()
        elif self.scale_tokens == 5:
            self.co_group = torch.cat((co_group+self.co_tokens*0, co_group+self.co_tokens*1, co_group+self.co_tokens*2, co_group+self.co_tokens*3, co_group+self.co_tokens*4), dim=0).cuda()
            self.cl_group = torch.cat((cl_group+self.cl_tokens*5, cl_group+self.cl_tokens*6, cl_group+self.cl_tokens*7, cl_group+self.cl_tokens*8, cl_group+self.cl_tokens*9), dim=0).cuda()
        elif self.scale_tokens == 6:
            self.co_group = torch.cat((co_group+self.co_tokens*0, co_group+self.co_tokens*1, co_group+self.co_tokens*2, co_group+self.co_tokens*3, co_group+self.co_tokens*4, co_group+self.co_tokens*5), dim=0).cuda()
            self.cl_group = torch.cat((cl_group+self.cl_tokens*6, cl_group+self.cl_tokens*7, cl_group+self.cl_tokens*8, cl_group+self.cl_tokens*9, cl_group+self.cl_tokens*10, cl_group+self.cl_tokens*11), dim=0).cuda()
        elif self.scale_tokens == 7:
            self.co_group = torch.cat((co_group+self.co_tokens*0, co_group+self.co_tokens*1, co_group+self.co_tokens*2, co_group+self.co_tokens*3, co_group+self.co_tokens*4, co_group+self.co_tokens*5, co_group+self.co_tokens*6), dim=0).cuda()
            self.cl_group = torch.cat((cl_group+self.cl_tokens*7, cl_group+self.cl_tokens*8, cl_group+self.cl_tokens*9, cl_group+self.cl_tokens*10, cl_group+self.cl_tokens*11, cl_group+self.cl_tokens*12, cl_group+self.cl_tokens*13), dim=0).cuda()
        else:
            assert False, 'not supported scale_tokens: {}'.format(self.scale_tokens)

        if self.arch_backbone in ('resnet50', 'resnet101'):
            self.pool= nn.AdaptiveAvgPool2d((1,1))

        if self.arch_backbone in ('prompted_swin_base_patch4_window7_224_in22k'):
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # self.moe = GroupMOE(num_groups=self.co_tokens+self.cl_tokens, num_experts_group=self.scale_tokens, hidden_dim=self.feature_dim, bias=True)
        
        # self.adapt_experts = Adapt_Experts(group_num=self.num_aux_tokens, embed_dim=self.feature_dim, down_sample=32, mode='parallel', drop=0.0, act_layer=nn.GELU, bias=True)
        # self.adapt_experts = Adapt_Experts(group_num=self.num_aux_tokens, embed_dim=self.feature_dim, down_sample=0.5, mode='parallel', drop=0.0, act_layer=nn.GELU, bias=True)
        self.adapt_experts = Adapt_Experts(group_num=self.num_aux_tokens, embed_dim=self.feature_dim, down_sample=5, mode='parallel', drop=0.0, act_layer=nn.GELU, bias=True)

        self.co_gate = Gate_MOE(num_class=self.num_class, num_experts_classes=self.scale_tokens, hidden_dim=self.feature_dim)
        self.cl_gate = Gate_MOE(num_class=self.num_class, num_experts_classes=self.scale_tokens, hidden_dim=self.feature_dim)
        
        

        self.co_fc = GroupFCWiseLinear(self.num_class, self.feature_dim, bias=True)
        self.cl_fc = GroupFCWiseLinear(self.num_class, self.feature_dim, bias=True)
        # self.group_fc = GroupWiseLinear(num_class=self.num_class, dataname=cfg.DATA.dataname, hidden_dim=self.feature_dim, bias=True)


        if self.arch_backbone in ('prompted_swin_base_patch4_window7_224_in22k'):
            def initial_weight(self):
                self.img_encoder.eval()
                if self.cfg.MODEL.PROMPT.config_PROJECT > -1:
                    self.img_encoder.prompt_proj.eval()
                for name, param in self.img_encoder.named_parameters():
                    if name in ('prompt_embeddings', 
                                'deep_prompt_embeddings_0',
                                'deep_prompt_embeddings_1',
                                'deep_prompt_embeddings_2',
                                'deep_prompt_embeddings_3'):
                        param.requires_grad_(True)
                    elif 'prompt_proj' in name and self.cfg.MODEL.PROMPT.config_PROJECT > -1:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
        else:
            def initial_weight(self):
                self.img_encoder.eval()
                if self.cfg.MODEL.PROMPT.config_PROJECT > -1:
                    self.img_encoder.prompt_proj.eval()
                for name, param in self.img_encoder.named_parameters():
                    if name in ('prompt_embeddings', 'deep_prompt_embeddings'):
                        param.requires_grad_(True)
                    elif 'prompt_proj' in name and self.cfg.MODEL.PROMPT.config_PROJECT > -1:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)

        self.initial_weight = initial_weight.__get__(self)
        self.initial_weight()


        if self.arch_backbone in ('resnet50', 'resnet101'):
            def forward_features(self, image):
                visual_feature = self.img_encoder(image)
                feature = self.pool(visual_feature).view(-1, self.feature_dim)
                return feature

        elif self.arch_backbone in ('vit_base_patch16_mae', 'vit_large_patch16_mae', 'vit_huge_patch14_mae'):
            def forward_features(self, image):
                _, pre_x_norm = self.img_encoder(image)
                feature = pre_x_norm[:, 1:self.num_aux_tokens+1, :]
                return feature
            # def forward_features(self, image):
            #     x_norm, _ = self.img_encoder(image)
            #     feature = x_norm[:, 1:self.num_aux_tokens+1, :]
            #     return feature 

        elif self.arch_backbone in ('vit_small_dinov2', 'vit_base_dinov2'):
            def forward_features(self, image):
                visual_feature = self.img_encoder.forward_features(image)
                feature = visual_feature['x_norm_auxtoken']
                return feature

        elif self.arch_backbone in ('vit_base_moco'):
            def forward_features(self, image):
                _, pre_x_norm = self.img_encoder(image)
                feature = pre_x_norm[:, 1:self.num_aux_tokens+1, :]
                return feature
            # def forward_features(self, image):
            #     x_norm, _ = self.img_encoder(image)
            #     feature = x_norm[:, 1:self.num_aux_tokens+1, :]
            #     return feature 

        elif self.arch_backbone in ('vit_base_patch16_224'):
            # def forward_features(self, image):
            #     _, pre_x_norm = self.img_encoder(image)
            #     feature = pre_x_norm[:, 1:self.num_aux_tokens+1, :]
            #     return feature
            def forward_features(self, image):
                x_norm, _ = self.img_encoder(image)
                feature = x_norm[:, 1:self.num_aux_tokens+1, :]
                return feature  

        elif self.arch_backbone in ('vit_base_patch16_224_in21k'):
            def forward_features(self, image):                      # will be imporved
                _, pre_x_norm = self.img_encoder(image)
                feature = pre_x_norm[:, 1:self.num_aux_tokens+1, :]
                return feature
            # def forward_features(self, image):
            #     x_norm, _ = self.img_encoder(image)
            #     feature = x_norm[:, 1:self.num_aux_tokens+1, :]
            #     return feature    
        
        elif self.arch_backbone in ('prompted_swin_base_patch4_window7_224_in22k'):
            def forward_features(self, image):
                visual_feature = self.img_encoder(image)
                # print(visual_feature.shape)
                visual_feature = self.avgpool(visual_feature.transpose(1, 2))
                visual_feature = torch.flatten(visual_feature, 1)
                # print(visual_feature.shape)
                return visual_feature

        self.forward_features = forward_features.__get__(self)

    def switch_mode_train(self):
        self.train()
        self.initial_weight()
    
    def switch_mode_eval(self):
        self.eval()


    def forward(self, image):
        batch_size = image.size(0)

        feature = self.forward_features(image)


        # Router for GateNetwork
        cat_co_feature = torch.index_select(feature, 1, self.co_group)
        cat_cl_feature = torch.index_select(feature, 1, self.cl_group)

        co_logits_atten, moe_co_logits = self.co_gate(cat_co_feature, batch_size=batch_size)
        cl_logits_atten, moe_cl_logits = self.cl_gate(cat_cl_feature, batch_size=batch_size)

        # Experts
        experts_feature = self.adapt_experts(feature)
        cat_experts_co_feature = torch.index_select(experts_feature, 1, self.co_group) # without cls and patch
        cat_experts_cl_feature = torch.index_select(experts_feature, 1, self.cl_group) # without cls and patch

        stack_experts_co_feature = torch.stack(torch.split(cat_experts_co_feature, self.num_class, dim=1), dim=1)
        stack_experts_cl_feature = torch.stack(torch.split(cat_experts_cl_feature, self.num_class, dim=1), dim=1)

        # Weighted Experts
        co_feature = (stack_experts_co_feature * co_logits_atten.unsqueeze(-1)).sum(dim=1)
        cl_feature = (stack_experts_cl_feature * cl_logits_atten.unsqueeze(-1)).sum(dim=1)


        co_logit = self.co_fc(co_feature)
        cl_logit = self.cl_fc(cl_feature)

        # co_logit, cl_logit = self.group_fc(feature)
        # return {'logit': (co_logit * 0.45 + cl_logit * 0.55 ), #* 0.5, 
        return {'logit': (co_logit + cl_logit) * 0.5, 
                'co_logit': co_logit, 
                'cl_logit': cl_logit,
                'moe_co_logit': moe_co_logits,
                'moe_cl_logit': moe_cl_logits}
        

def build_network(cfg, logger=None):
    backbone, feature_dim = build_backbone(cfg.MODEL.BACKBONE.backbone, cfg=cfg, img_size=cfg.DATA.TRANSFORM.img_size)
    model = MLIC(
        backbone = backbone,
        feature_dim = feature_dim,
        num_class = cfg.MODEL.CLASSIFIER.num_class,
        cfg=cfg
    )

    return model

# @torch.set_grad_enabled(True)
def do_forward_and_criterion_train(cfg, epoch, data_full, model, criterion, is_val):
    output = model(data_full['image'])
    logit = output['logit']

    loss_dict = criterion(output, data_full)
    
    weight_dict = criterion.weight_dict
    
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    
    return logit, losses, loss_dict, weight_dict

@torch.no_grad()
def do_forward_and_criterion_test(cfg, epoch, data_full, model, criterion, is_val):
    output = model(data_full['image'])
    logit = output['logit']
    loss_dict = criterion(output, data_full)
    weight_dict = criterion.weight_dict    
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)


    return logit, losses, loss_dict, weight_dict


def do_forward_and_criterion(cfg, epoch, data_full, model, criterion, is_val=False):
    if not is_val:
        return do_forward_and_criterion_train(cfg, epoch, data_full, model, criterion, is_val)
    else:
        return do_forward_and_criterion_test(cfg, epoch, data_full, model, criterion, is_val)

@register_model
def mlic(cfg):
    backbone, feat_dim = build_backbone(cfg.MODEL.BACKBONE.backbone, img_size=cfg.DATA.TRANSFORM.img_size)
    model = MLIC(backbone, feat_dim, cfg)
    return model


class SetCriterion(nn.Module):

    def __init__(self, weight_dict, losses, cfg):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.losses_test = ['loss_cls']
        self.cls_loss = None
        self.co_cls_loss = None
        self.cl_cls_loss = None
        self.co_moe_cls_loss = None
        self.cl_moe_cls_loss = None

        # self.mse_loss = nn.MSELoss()
        # losses = ['cls', 'sample_en', 'batch_en']
        # losses = ['los_cls', 'loss_sample_en', 'loss_batch_en']

    def loss_cls(self, outputs, targets=None, **kwargs):

        cls = self.cls_loss(outputs['logit'], targets['target'])
        losses = {"loss_cls": cls}

        return losses

    def loss_co_cls(self, outputs, targets=None, **kwargs):

        cls = self.co_cls_loss(outputs['co_logit'], targets['target'])
        losses = {"loss_co_cls": cls}

        return losses

    def loss_cl_cls(self, outputs, targets=None, **kwargs):

        cls = self.cl_cls_loss(outputs['cl_logit'], targets['target'])
        losses = {"loss_cl_cls": cls}

        return losses

    def loss_co_moe_cls(self, outputs, targets=None, **kwargs):
        cls = self.co_moe_cls_loss(outputs['moe_co_logit'], targets['target'])
        losses = {"loss_co_moe_cls": cls}

        return losses

    def loss_cl_moe_cls(self, outputs, targets=None, **kwargs):
        cls = self.cl_moe_cls_loss(outputs['moe_cl_logit'], targets['target'])
        losses = {"loss_cl_moe_cls": cls}

        return losses


    def get_loss_train(self, loss, outputs, targets, **kwargs):
        loss_map = {
            'loss_co_cls': self.loss_co_cls,
            'loss_cl_cls': self.loss_cl_cls,
            'loss_co_moe_cls': self.loss_co_moe_cls,
            'loss_cl_moe_cls': self.loss_cl_moe_cls,    
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def get_loss_test(self, loss, outputs, targets, **kwargs):
        loss_map = {
            'loss_cls': self.loss_cls,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)


    def forward(self, outputs, targets=None, is_val=None, **kwargs):
        losses = {}

        if is_val:
            for loss in self.losses_test:
                losses.update(self.get_loss_test(loss, outputs, targets, **kwargs))
            return losses
        else:
            for loss in self.losses:
                losses.update(self.get_loss_train(loss, outputs, targets, **kwargs))
        
        return losses