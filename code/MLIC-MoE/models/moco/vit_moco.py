#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
borrowed from https://github.com/facebookresearch/moco-v3/blob/main/vits.py
"""
import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul

# from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.vision_transformer import _cfg
from timm.layers.helpers import to_2tuple
from timm.layers import PatchEmbed
from timm.models._manipulate import checkpoint_seq
from ..factory import register_backbone

from ..timm_models.vision_transformer import VisionTransformer

__all__ = [
    'vit_small_moco',
    'vit_base_moco',
    'vit_conv_small_moco',
    'vit_conv_base_moco',
]


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, cfg, stop_grad_conv1=False, **kwargs):

        kwargs['cfg'] = cfg

        super().__init__(**kwargs)

        depth=kwargs['depth']
        patch_size=kwargs['patch_size']
        embed_dim = kwargs['embed_dim']
        
        
        # number of tokens, [cls] + patches
        self.num_tokens = 1  
        # self.embed_dim = embed_dim
        
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False


        # Visual Prompt Tuning
        self.cfg = cfg
        self.scale_tokens = cfg.MODEL.PROMPT.scale_tokens
        self.aux_tokens = cfg.MODEL.PROMPT.aux_tokens 
        self.num_aux_tokens = (self.scale_tokens) * (self.aux_tokens + self.aux_tokens)

        # self.aux_tokens = cfg.MODEL.PROMPT.aux_tokens
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


    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False
    
    # def forward_features(self, x):
    #     x = self.patch_embed(x)
    #     x = self.pos_embed(x)
    #     x = self.norm_pre(x)
    #     if self.grad_checkpointing and not torch.jit.is_scripting():
    #         x = checkpoint_seq(self.blocks, x)
    #     else:
    #         # x = self.blocks(x)
    #         hidden_states = x
    #         B = hidden_states.shape[0]

    #         for i, blk in enumerate(self.blocks):
    #             if i == 0:
    #                 hidden_states = torch.cat((
    #                 hidden_states[:, :1, :],
    #                 self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
    #                 hidden_states[:, 1:, :]
    #                 ), dim=1)
    #                 hidden_states = blk(hidden_states)
    #             else:
    #                 if i <= self.deep_prompt_embeddings.shape[0]:
    #                     deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
    #                         self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

    #                     hidden_states = torch.cat((
    #                         hidden_states[:, :1, :],
    #                         deep_prompt_emb,
    #                         hidden_states[:, (1+self.aux_tokens):, :]
    #                     ), dim=1)                

    #                 hidden_states = blk(hidden_states)
    #         x = hidden_states

    #     x_norm = self.norm(x)

    #     return x_norm, hidden_states


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

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.forward_head(x)

        return x


class ConvStem(nn.Module):
    """
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

@register_backbone(feat_dim=384)
def vit_small_moco(pretrained=False, cfg=None, **kwargs):
    model = VisionTransformerMoCo(
        cfg=cfg,
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    model = load_pretrained(model, cfg)
    del model.head
    return model

@register_backbone(feat_dim=768)
def vit_base_moco(pretrained=False, cfg=None, **kwargs):
    model = VisionTransformerMoCo(
        cfg=cfg,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    model = load_pretrained(model, cfg)

    # for key, value in model.state_dict().items():
    #     print(key)
    # for name, param in model.named_parameters():
    #     # param.requires_grad_(False)
    #     print(name)

    del model.head
    return model

@register_backbone(feat_dim=384)
def vit_conv_small_moco(pretrained=False, cfg=None, **kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        cfg=cfg,
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    model = load_pretrained(model, cfg)
    del model.head
    return model

@register_backbone(feat_dim=768)
def vit_conv_base_moco(pretrained=False, cfg=None, **kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        cfg=cfg,
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    model = load_pretrained(model, cfg)
    del model.head
    return model


# def load_pretrained(model, cfg):
#     import os
#     model_root = cfg.MODEL.pretrain_path
#     if cfg.MODEL.BACKBONE.backbone == "vit_base_moco":
#         ckpt = os.path.join(model_root,"mocov3_linear-vit-b-300ep.pth.tar")

#     checkpoint = torch.load(ckpt, map_location="cpu")
#     state_dict = checkpoint['state_dict']

#     for k in list(state_dict.keys()):
#         # retain only base_encoder up to before the embedding layer
#         if k.startswith('module.'):
#             # remove prefix
#             state_dict[k[len("module."):]] = state_dict[k]
#         # delete renamed or unused k
#         del state_dict[k]

#     model.load_state_dict(state_dict, strict=False)

#     return model


def load_pretrained(model, cfg):
    import os

    model_root = cfg.MODEL.pretrain_path
    if cfg.MODEL.BACKBONE.backbone == "vit_base_moco":
        ckpt = os.path.join(model_root,"mocov3_linear-vit-b-300ep.pth.tar")


    if os.path.isfile(ckpt):
        print("=> loading checkpoint '{}'".format(ckpt))
        checkpoint = torch.load(ckpt, map_location='cuda')
        if hasattr(model, 'module'):
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()

        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.'):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        interpolate_pos_embed(model, state_dict)

        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
            else:
                print('\tMismatched layers: {}, shape is {}, '.format(k, v.shape))
                # print(v.shape, model_dict[k].shape)

        if hasattr(model, 'module'):
            model.module.load_state_dict(model_dict, strict=False)
        else:
            model.load_state_dict(model_dict, strict=False)
        
        
        
        del checkpoint
        del state_dict
        del model_dict
        torch.cuda.empty_cache() 
    else:
        print("=> no checkpoint found at '{}'".format(ckpt))

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