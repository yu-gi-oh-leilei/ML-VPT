#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
borrowed from https://github.com/facebookresearch/mae/blob/main/models_vit.py
"""
import math
from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

from ..factory import register_backbone

from functools import reduce
from operator import mul
from ..timm_models.vision_transformer import VisionTransformer
__all__ = [
    'vit_base_patch16_mae',
    'vit_large_patch16_mae',
    'vit_huge_patch14_mae',
]

class VisionTransformer(VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, cfg, global_pool=False, **kwargs):
        kwargs['cfg'] = cfg
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        cfg=kwargs['cfg']
        depth=kwargs['depth']
        patch_size=kwargs['patch_size']

        embed_dim = kwargs['embed_dim']
        self.embed_dim = embed_dim



        # Visual Prompt Tuning
        self.cfg = cfg
        self.scale_tokens = cfg.MODEL.PROMPT.scale_tokens
        self.aux_tokens = cfg.MODEL.PROMPT.aux_tokens 
        self.num_aux_tokens = (self.scale_tokens) * (self.aux_tokens + self.aux_tokens)


        self.aux_tokens = cfg.MODEL.PROMPT.aux_tokens
        self.prompt_config_DROPOUT = cfg.MODEL.PROMPT.config_DROPOUT
        self.prompt_config_INITIATION = cfg.MODEL.PROMPT.config_INITIATION
        self.prompt_config_DEEP = cfg.MODEL.PROMPT.config_DEEP
        self.prompt_config_DEEP_proj = cfg.MODEL.PROMPT.config_DEEP_proj
        self.prompt_config_PROJECT = cfg.MODEL.PROMPT.config_PROJECT
        self.n_blocks = depth

        self.prompt_dropout = nn.Dropout(self.prompt_config_DROPOUT)
        # if project the prompt embeddings
        if self.prompt_config_PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config_PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, embed_dim)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = embed_dim
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.prompt_config_INITIATION == "random":
            # patch_size = 16
            val = math.sqrt(6. / float(3 * reduce(mul, (patch_size, patch_size), 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.num_aux_tokens, self.embed_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config_DEEP:
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    self.n_blocks - 1,
                    self.num_aux_tokens, self.embed_dim
                ))
                # xavier_uniform initialization
                nn.init.uniform_(
                    self.deep_prompt_embeddings.data, -val, val)
        else:
            raise ValueError("Other initiation scheme is not supported")


    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        hidden_states = x
        B = hidden_states.shape[0]

        for i, blk in enumerate(self.blocks):
            if i == 0:
                hidden_states = torch.cat((
                hidden_states[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                hidden_states[:, 1:, :]
                ), dim=1)
                hidden_states = blk(hidden_states)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1+self.num_aux_tokens):, :]
                    ), dim=1)                

                hidden_states = blk(hidden_states)
        
        
        x_norm = self.norm(hidden_states)


        return x_norm, hidden_states

    def forward(self, x):
        x = self.forward_features(x)

        # B = x.shape[0]
        # x = self.patch_embed(x)

        # cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed
        # x = self.pos_drop(x)

        # for blk in self.blocks:
        #     x = blk(x)

        # x = self.norm(x)

        return x

# def build_model(model_type):
#     if "vitb" in model_type:
#         return vit_base_patch16()
#     elif "vitl" in model_type:
#         return vit_large_patch16()
#     elif "vith" in model_type:
#         return vit_huge_patch14()

@register_backbone(feat_dim=768)
def vit_base_patch16_mae(pretrained=False, cfg=None, **kwargs):
    # kwargs['cfg']=cfg
    model = VisionTransformer(
        cfg=cfg,
        drop_path_rate=0.1, global_pool=False,  # using default settings for mae-finetune
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    model = load_pretrained(model, cfg)
    del model.head

    return model

@register_backbone(feat_dim=1024)
def vit_large_patch16_mae(pretrained=False, cfg=None, **kwargs):
    # kwargs['cfg']=cfg
    model = VisionTransformer(
        cfg=cfg,
        drop_path_rate=0.1, global_pool=False,  # using default settings for mae-finetune
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model = load_pretrained(model, cfg)
    del model.head

    return model

@register_backbone(feat_dim=1280)
def vit_huge_patch14_mae(pretrained=False, cfg=None, **kwargs):
    # kwargs['cfg']=cfg
    model = VisionTransformer(
        cfg=cfg,
        drop_path_rate=0.1, global_pool=False,  # using default settings for mae-finetune
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model = load_pretrained(model, cfg)
    del model.head

    return model


def load_pretrained(model, cfg):
    import os
    model_root = cfg.MODEL.pretrain_path
    if cfg.MODEL.BACKBONE.backbone == "vit_base_patch16_mae":
        ckpt = os.path.join(model_root,"mae_pretrain_vit_base.pth")

    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint['model']
    # state_dict = checkpoint


    if hasattr(model, 'module'):
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()
    
    interpolate_pos_embed(model, state_dict)

    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            model_dict[k] = v
        else:
            print('\tMismatched layers: {}, shape is {}, '.format(k, v.shape))
    

    # if cfg.DATA.TRANSFORM.img_size != 224:
    #     v = model_dict['pos_embed']
    #     v = resize_pos_embed(v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    #     model_dict['pos_embed'] = v
    #     print('\tResize position embedding: {} to {}'.format(model.pos_embed.shape, v.shape))

    if hasattr(model, 'module'):
        model.module.load_state_dict(model_dict, strict=False)
    else:
        model.load_state_dict(model_dict, strict=False)

    # interpolate_pos_embed(model, state_dict)

    del checkpoint
    del state_dict
    del model_dict
    torch.cuda.empty_cache() 

    print('\t ==========> load pretrained model ==========> \t')
    
    return model

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
