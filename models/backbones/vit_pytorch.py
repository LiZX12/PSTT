""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import copy
import json
import logging
import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc as container_abcs
from einops import rearrange

# From PyTorch internals
from einops.layers.torch import Rearrange



logger = logging.getLogger("pit.train")


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., cfg=None,
                 add_norm_size=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cfg = cfg
        if add_norm_size == 1:
            self.proj_1 = nn.Linear(dim, dim)
            self.proj_drop_1 = nn.Dropout(proj_drop)

    def forward(self, x, attention=None, split_attion_type=0, after_attn_fn=0):
        B, N, C = x.shape
        if attention is not None:
            q = x.reshape(B, -1, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
            k, v = attention[0], attention[1]

            if split_attion_type in (1, 5):
                k = rearrange(k, "(B T) H N C -> B H (N T) C", T=self.cfg.DATALOADER.NUM_TEST_IMAGES)
                v = rearrange(v, "(B T) H N C -> B H (N T) C", T=self.cfg.DATALOADER.NUM_TEST_IMAGES)
            elif split_attion_type in (2, 6):
                k = rearrange(k, "(B T) H N C -> B H (N T) C", T=self.cfg.DATALOADER.NUM_TEST_IMAGES // 2)
                v = rearrange(v, "(B T) H N C -> B H (N T) C", T=self.cfg.DATALOADER.NUM_TEST_IMAGES // 2)
            elif split_attion_type in (3, 7):
                k = rearrange(k, "(B T) H N C -> B H (N T) C", T=self.cfg.DATALOADER.NUM_TEST_IMAGES // 4)
                v = rearrange(v, "(B T) H N C -> B H (N T) C", T=self.cfg.DATALOADER.NUM_TEST_IMAGES // 4)
            elif split_attion_type in (4, 8):
                k = rearrange(k, "(B T) H N C -> B H (N T) C", T=self.cfg.DATALOADER.NUM_TEST_IMAGES // 8)
                v = rearrange(v, "(B T) H N C -> B H (N T) C", T=self.cfg.DATALOADER.NUM_TEST_IMAGES // 8)
            elif split_attion_type in (9, 0,):
                pass
            else:
                raise RuntimeError
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        ooo = {1: 1, 2: 2, 3: 4, 4: 8}
        if split_attion_type in (1, 2, 3, 4):
            if self.cfg.RIGHT_STREAM.ALL_ROLL == 1:
                n_list = self.split_h(split_attion_type, q)
                x_list = []
                for q_ in n_list:
                    x = None
                    for i in range(ooo[split_attion_type]):
                        if i > 0:
                            q_ = self._roll(q_, split_attion_type)
                        attn = (q_ @ k.transpose(-2, -1)) * self.scale
                        attn = attn.softmax(dim=-1)
                        attn = self.attn_drop(attn)
                        if x is None:
                            x = (attn @ v).transpose(1, 2).reshape(q_.shape[0], -1, C)
                        else:
                            x = x + (attn @ v).transpose(1, 2).reshape(q_.shape[0], -1, C)
                    if self.cfg.RIGHT_STREAM.AAAA_FLAG == 2:
                        x_list.append(x/ooo[split_attion_type])
                    else:
                        x_list.append(x)
                x = self.n_list_cat(x_list)
            else:
                raise RuntimeError
        elif split_attion_type == 0:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            raise RuntimeError

        if after_attn_fn == 0:
            x = self.proj(x)
            x = self.proj_drop(x)
        elif after_attn_fn == 1:
            x = self.proj_1(x)
            x = self.proj_drop_1(x)
        else:
            raise RuntimeError

        return x, [k, v]

    def _roll(self, x, type):
        if type == 1:
            return x
        elif type == 2:
            T = 2
        elif type == 3:
            T = 4
        elif type == 4:
            T = 8
        else:
            raise RuntimeError
        x = rearrange(x, "(B T) H D C -> B T H D C", T=T)
        x = torch.roll(x, shifts=(1), dims=(1))
        x = rearrange(x, "B T H D C -> (B T) H D C ", T=T)
        return x

    def _roll_kv(self, x, type):
        if type == 1:
            T = 1
        elif type == 2:
            T = 2
        elif type == 3:
            T = 4
        elif type == 4:
            T = 8
        else:
            raise RuntimeError
        T = self.cfg.DATALOADER.NUM_TEST_IMAGES // T

        x = rearrange(x, "(B N) H (D T) C -> (B H D T C", T=T)
        x = torch.roll(x, shifts=(T), dims=(-2))
        x = rearrange(x, "B T H D C -> (B T) H D C ", T=T)
        return x

    def _mask(self, attn, type, i):
        if type == 1 or i == 0:
            return attn
        elif type == 2:
            T = 2
        elif type == 3:
            T = 4
        elif type == 4:
            T = 8
        else:
            raise RuntimeError
        attn = rearrange(attn, "(B T) H D C -> B T H D C", T=T)
        for j in range(i):
            attn[i] += -100
        attn = rearrange(attn, "B T H D C -> (B T) H D C ", T=T)
        return attn

    def split_h(self, type, value):
        if type in (1, 5):
            n_list_k = []
            value = rearrange(value, "(B T) H D C -> B T H D C", T=self.cfg.DATALOADER.NUM_TEST_IMAGES)
            for i in range(self.cfg.DATALOADER.NUM_TEST_IMAGES):
                n_list_k.append(value[:, i])
        elif type in (2, 6):
            n_list_k = []
            value = rearrange(value, "(B T) H D C -> B T H D C", T=self.cfg.DATALOADER.NUM_TEST_IMAGES // 2)
            for i in range(self.cfg.DATALOADER.NUM_TEST_IMAGES // 2):
                n_list_k.append(value[:, i])
        elif type in (3, 7):
            n_list_k = []
            value = rearrange(value, "(B T) H D C -> B T H D C", T=self.cfg.DATALOADER.NUM_TEST_IMAGES // 4)
            for i in range(self.cfg.DATALOADER.NUM_TEST_IMAGES // 4):
                n_list_k.append(value[:, i])
        elif type in (4, 8):
            n_list_k = []
            value = rearrange(value, "(B T) H D C -> B T H D C", T=self.cfg.DATALOADER.NUM_TEST_IMAGES // 8)
            for i in range(self.cfg.DATALOADER.NUM_TEST_IMAGES // 8):
                n_list_k.append(value[:, i])
        else:
            raise RuntimeError
        return n_list_k

    def n_list_cat(self, x_list):
        if self.cfg.RIGHT_STREAM.SPLIT_TYPE in range(10):
            B, D, C = x_list[0].shape
            return torch.cat([x.unsqueeze(1) for x in x_list], dim=1).reshape(-1, D, C)
        else:
            raise RuntimeError


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, cfg=None, add_norm_size=0, init_values=1e-4):
        super().__init__()
        self.groups = 3
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            cfg=cfg,add_norm_size=add_norm_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.cfg = cfg

        if add_norm_size == 0:
            pass
        elif add_norm_size == 1:
            self.norm3 = norm_layer(dim)
        else:
            raise RuntimeError

        # self.dropout_layer = build_dropout(dict(type='DropPath', drop_prob=0.1))
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, return_attention=False, attention=None, split_attion_type=0, type="left"):
        y, attn = self.attn(self.norm1(x), None, 0, after_attn_fn=0)
        x = x + self.drop_path(self.gamma_1 * y)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x

    def diversity_loss_rn596(self, attn):
        B, H, M, M = attn.shape
        attn_ = attn.view(B, H, -1)

        reg = (attn_ @ attn_.transpose(-2, -1))
        I = torch.eye(H)[None].expand(B, H, H).to(attn.device)
        reg = (torch.sum((reg - I) ** 2, dim=(1, 2)) + 1e-10) ** 0.5

        return reg

    def diversity_loss_rn562(self, attn):
        B, H, M, M = attn.shape
        attn = attn ** 0.5  # not converge
        attn_ = attn.view(B, H, -1)

        reg = (attn_ @ attn_.transpose(-2, -1))
        I = torch.eye(H)[None].expand(B, H, H).to(attn.device)
        reg = (torch.sum((reg - I) ** 2, dim=(1, 2)) + 1e-10) ** 0.5
        return reg

    def diversity_loss_rn573(self, attn):
        B, H, M, M = attn.shape
        attn_ = attn.view(B, H, -1)
        attn_ = F.normalize(attn_, dim=2)

        reg = (attn_ @ attn_.transpose(-2, -1))
        I = torch.eye(H)[None].expand(B, H, H).to(attn.device)
        reg = (torch.sum((reg - I) ** 2, dim=(1, 2)) + 1e-10) ** 0.5
        return reg

    def diversity_loss_rn187(self, attn):
        B, H, M, M = attn.shape
        loss = torch.zeros(1).to(attn.device)
        for i in range(H):
            t = attn[:, i].reshape(B, -1)
            for j in range(H):
                if i != j:
                    s = attn.detach()[:, j].reshape(B, -1)
                    loss += self.distill_loss(t, s) / (H - 1)
        return loss

    def distill_loss(self, y_s, y_t, t=4):
        p_s = F.log_softmax(y_s / t, dim=1)
        p_t = F.softmax(y_t / t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (t ** 2) / y_s.shape[0]
        return loss


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768, norm_layer=None, ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.stride_size = stride_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # input:1 3 256 128  out:1 768 21 10
        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]  out:1  210 768
        x = self.norm(x)
        return x


class VIT(nn.Module):
    """ Transformer-based Object Re-Identification
    """

    def __init__(self, img_size=(224,224), patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0,
                 view=0,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, local_feature=False, sie_xishu=1.0,
                 isVideo=False, spatial=False, temporal=False, vis=False, cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.local_feature = local_feature
        self.isVideo = isVideo
        self.spatial = spatial
        self.temporal = temporal
        self.vis = vis
        self.cfg = cfg

        self.patch_embed = PatchEmbed_overlap(
            img_size=img_size, patch_size=patch_size,
            stride_size=stride_size,
            in_chans=in_chans,
            embed_dim=embed_dim)

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        dpr = [drop_path_rate for i in range(depth)] # stochastic depth decay rule

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.cam_num, self.view_num, self.sie_xishu = camera, view, sie_xishu
        self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
        trunc_normal_(self.sie_embed, std=.02)

        print('camera number is : {}'.format(camera))
        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.temporal_pos_drop = nn.Dropout(p=drop_rate)

        self.spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0., attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, cfg=cfg)
            for i in range(depth - 1)])


        self.aggregator = MeanAggregator()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.head_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[-1], norm_layer=norm_layer, cfg=cfg)

        self.norm = norm_layer(embed_dim)


        def create_classfier(c_num_, num_classes_):
            lr = nn.Linear(c_num_, num_classes_, bias=False)
            lr.apply(self.weights_init_classifier)
            bnr = nn.BatchNorm1d(c_num_)
            bnr.bias.requires_grad_(False)
            bnr.apply(self.weights_init_kaiming)
            return nn.Sequential(bnr, lr)

        def create_two_classfier(c_num_):
            lr = nn.Sequential(
                nn.Linear(c_num_, c_num_ // 2, bias=False),
                nn.PReLU(),
                nn.Linear(c_num_ // 2, 1, bias=False)
            )

            lr.apply(self.weights_init_classifier)
            bnr = nn.BatchNorm1d(c_num_)
            bnr.bias.requires_grad_(False)
            bnr.apply(self.weights_init_kaiming)
            return nn.Sequential(bnr, lr)
        # Classifier head
        self.apply(self._init_weights)

        self.classifier = create_classfier(embed_dim, num_classes)
        self.sn_classifier = create_two_classfier(embed_dim)

        self.load_spatiotemporal_param("./deit_3_medium_224_21k.pth")

    def weights_init_kaiming(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)

        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


    def weights_init_classifier(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, camera_id, view_id):
        x = self.patch_embed(x)

        # camera_id = [i for i in camera_id for j in range(x_left.shape[0] // B)]

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = x + self.pos_embed
        # x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.temporal_pos_drop(x)

        for block in self.spatial_blocks:
            x = block(x)

        x = self.head_block(x)
        x = self.norm(x)

        return x[:,0,:]

    def forward(self, x, cam_label=None, view_label=None):
        x = self.forward_features(x, cam_label, view_label)
        return x

    def load_spatiotemporal_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v_ = v.reshape(O, -1, H, W)
                self.state_dict()[k].copy_(v_)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v_ = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
                self.state_dict()[k].copy_(v_)
            try:
                if 'blocks.11' in k:
                    k_ = k.replace('blocks.11', 'head_block')
                elif 'blocks' in k:
                    k_ = 'spatial_' + k
                else:
                    k_ = k
                self.state_dict()[k_].copy_(v)


            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape,
                                                                                                self.state_dict()[
                                                                                                    k].shape))


class MeanAggregator(nn.Module):

    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, x: torch.Tensor):
        return x.mean(dim=1)

    def __call__(self, *args, **kwargs):
        return super(MeanAggregator, self).__call__(*args, **kwargs)


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    # posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    posemb_grid = posemb

    gs_old = int(math.sqrt(posemb_grid.shape[1]))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                posemb_new.shape, hight,
                                                                                                width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = posemb_grid
    return posemb


def vit_base_patch16_224_PiT(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0,
                             drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5,
                             isVideo=False, spatial=False, temporal=False, vis=False, **kwargs):
    model = PiT(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, \
        camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=sie_xishu, local_feature=local_feature, isVideo=isVideo,
        spatial=spatial, temporal=temporal, vis=vis, **kwargs)

    return model


def vit(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, depth=12,embed_dim=768, num_heads=12,patch_size=16,
                                          drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5,
                                          isVideo=False, spatial=False, temporal=False, vis=False,num_classes=None, **kwargs):
    cfg = None
    if "cfg" in kwargs:
        cfg = kwargs["cfg"]
        del kwargs["cfg"]
    model = VIT(
        img_size=img_size, patch_size=patch_size, stride_size=stride_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=4,num_classes=num_classes,
        qkv_bias=True,camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=sie_xishu, local_feature=local_feature, isVideo=isVideo,
        spatial=spatial, temporal=temporal, vis=vis, cfg=cfg, **kwargs)

    return model


def vit_small_patch16_224_PiT(img_size=(256, 128), stride_size=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                              camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = PiT(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,
        qkv_bias=False, drop_path_rate=drop_path_rate, \
        camera=camera, view=view, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)

    return model


def deit_small_patch16_224_PiT(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0,
                               attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    model = PiT(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, camera=camera, view=view,
        sie_xishu=sie_xishu, local_feature=local_feature,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.", )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class SSNet(nn.Module):

    def __init__(self, dim):
        super(SSNet, self).__init__()
        self.glob_pool = nn.Sequential(
            # Rearrange("B H D C -> (B H) D  C"),
            nn.AdaptiveAvgPool2d((1, dim)),
            nn.Linear(in_features=int(dim), out_features=int(dim // 16)),
            nn.Linear(in_features=int(dim // 16), out_features=dim),
            nn.Sigmoid(),
            # Rearrange(" (B H) D  C -> B H D C", H=12),
        )


    def forward(self, x):
        return self.glob_pool(x)