"""
EfficientFormer_v2 - Fixed Version for CTU Task
"""
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict
import itertools

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import to_2tuple

# --- Define CTU Specific Config ---
EfficientFormer_width_ctu_s0 = [32, 48, 96, 176]
EfficientFormer_depth_ctu_s0 = [2, 2, 6, 4]
expansion_ratios_ctu_s0 = {
    '0': [4, 4],
    '1': [4, 4],
    '2': [4, 3, 3, 3, 4, 4],
    '3': [4, 3, 3, 4],
}

# +++ FIX 1: Custom Sequential for QP propagation +++
class QP_Sequential(nn.Module):
    """Sequential container that passes qp parameter to all child modules"""
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
    
    def forward(self, x, qp):
        for module in self.modules_list:
            x = module(x, qp)
        return x


class Attention4D(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 act_layer=nn.ReLU, 
                 stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                nn.BatchNorm2d(dim)
            )
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        self.N = self.resolution * self.resolution 

        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.q = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2d(self.num_heads * self.key_dim)
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2d(self.num_heads * self.key_dim)
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.d, 1),
            nn.BatchNorm2d(self.num_heads * self.d)
        )
        self.v_local = nn.Sequential(
            nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                     kernel_size=3, stride=1, padding=1, groups=self.num_heads * self.d),
            nn.BatchNorm2d(self.num_heads * self.d)
        )
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)

        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, dim, 1),
            nn.BatchNorm2d(dim)
        )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        N_points = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        
        self.register_buffer('attention_bias_idxs',
                           torch.LongTensor(idxs).view(self.N, self.N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            if hasattr(self, 'attention_biases') and hasattr(self, 'attention_bias_idxs'):
                self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)
            H, W = x.shape[2], x.shape[3]

        current_N = H * W
        
        # +++ FIX 2: Better error message +++
        if self.N != current_N:
            raise ValueError(
                f"Attention4D: Expected spatial size {self.resolution}x{self.resolution}={self.N}, "
                f"but got {H}x{W}={current_N}. This indicates a resolution mismatch in the network."
            )

        q = self.q(x).flatten(2).reshape(B, self.num_heads, self.key_dim, self.N).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, self.key_dim, self.N)
        v = self.v(x)
        v_local = self.v_local(v)
        v_glob = v.flatten(2).reshape(B, self.num_heads, self.d, self.N).permute(0, 1, 3, 2)

        attn = (
            (q @ k) * self.scale
            + (self.attention_biases[:, self.attention_bias_idxs]
               if self.training else self.ab)
        )

        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)

        x_attn = (attn @ v_glob)
        out = x_attn.permute(0, 1, 3, 2).reshape(B, self.dh, H, W) + v_local

        if self.upsample is not None:
            out = self.upsample(out)

        out = self.proj(out)
        return out


def stem(in_chs, out_chs, act_layer=nn.GELU):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        act_layer(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        act_layer(),
    )


class Embedding(nn.Module):
    def __init__(self, patch_size=3, stride=2, padding=1,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d,
                 light=False, asub=False, resolution=None, act_layer=nn.GELU, attn_block=None):
        super().__init__()
        self.light = light
        self.asub = asub

        if self.light:
            self.new_proj = nn.Sequential(
                nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=2, padding=1, groups=in_chans),
                nn.BatchNorm2d(in_chans),
                nn.Hardswish(),
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(embed_dim),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(embed_dim)
            )
        elif self.asub:
            if attn_block is None:
                raise ValueError("attn_block must be provided when asub is True")
            self.attn = attn_block(dim=in_chans, out_dim=embed_dim,
                                 resolution=resolution, act_layer=act_layer)
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                stride=stride, padding=padding)
            self.bn = norm_layer(embed_dim) if norm_layer else nn.Identity()
        else:
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                stride=stride, padding=padding)
            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.light:
            out = self.new_proj(x) + self.skip(x)
        elif self.asub:
            out_conv = self.conv(x)
            out_conv = self.bn(out_conv)
            out = self.attn(x) + out_conv
        else:
            x = self.proj(x)
            out = self.norm(x)
        return out


class Mlp_with_QP(nn.Module):
    """MLP with QP injection"""
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv

        self.fc1 = nn.Conv2d(in_features + 1, hidden_features, 1)
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()

        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, 
                               stride=1, padding=1, groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)

        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.norm2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, qp):
        B, C, H, W = x.shape
        
        # +++ FIX 3: Add shape validation +++
        if qp.shape[0] != B:
            raise ValueError(f"QP batch size {qp.shape[0]} doesn't match input batch size {B}")
        
        qp_expanded = qp.view(B, 1, 1, 1).expand(B, 1, H, W)
        x_with_qp = torch.cat([x, qp_expanded], dim=1)

        x = self.fc1(x_with_qp)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return x


class AttnFFN_with_QP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 resolution=7, stride=None):
        super().__init__()

        self.token_mixer = Attention4D(dim, resolution=resolution, 
                                      act_layer=act_layer, stride=stride)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_with_QP(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=act_layer, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), 
                requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), 
                requires_grad=True)

    def forward(self, x, qp):
        attn_out = self.token_mixer(x)
        if self.use_layer_scale:
            mlp_in = x + self.drop_path(self.layer_scale_1 * attn_out)
        else:
            mlp_in = x + self.drop_path(attn_out)
        
        mlp_out = self.mlp(mlp_in, qp)
        
        if self.use_layer_scale:
            x = mlp_in + self.drop_path(self.layer_scale_2 * mlp_out)
        else:
            x = mlp_in + self.drop_path(mlp_out)
        return x


class FFN_with_QP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_with_QP(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=act_layer, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), 
                requires_grad=True)

    def forward(self, x, qp):
        mlp_out = self.mlp(x, qp)
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * mlp_out)
        else:
            x = x + self.drop_path(mlp_out)
        return x


def eformer_block_ctu(dim, index, layers,
                      pool_size=3, mlp_ratio=4.,
                      act_layer=nn.GELU, norm_layer=nn.BatchNorm2d,
                      drop_rate=.0, drop_path_rate=0.,
                      use_layer_scale=True, layer_scale_init_value=1e-5,
                      vit_num=1, resolution=7, e_ratios=None):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)

        if e_ratios is None or str(index) not in e_ratios or block_idx >= len(e_ratios[str(index)]):
            current_mlp_ratio = mlp_ratio
        else:
            current_mlp_ratio = e_ratios[str(index)][block_idx]

        is_attn_block = index >= 2 and block_idx >= layers[index] - vit_num

        if is_attn_block:
            blocks.append(AttnFFN_with_QP(
                dim, mlp_ratio=current_mlp_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                resolution=resolution,
                stride=None,
            ))
        else:
            blocks.append(FFN_with_QP(
                dim, mlp_ratio=current_mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
    
    # +++ FIX 4: Use QP_Sequential instead of nn.Sequential +++
    blocks = QP_Sequential(*blocks)
    return blocks


class EfficientFormerV2(nn.Module):
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=4, downsamples=None,
                 pool_size=3,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 num_classes=21,
                 in_chans=1,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 vit_num=0,
                 distillation=False,
                 resolution=64,
                 e_ratios=None,
                 **kwargs):
        super().__init__()

        if fork_feat:
            raise NotImplementedError("fork_feat=True not adapted.")
        if distillation:
            print("Warning: Distillation not adapted for CTU task.")

        self.num_classes = num_classes
        self.patch_embed = stem(in_chans, embed_dims[0], act_layer=act_layer)
        current_resolution = resolution // 4

        network = []
        for i in range(len(layers)):
            stage = eformer_block_ctu(
                embed_dims[i], i, layers,
                pool_size=pool_size, mlp_ratio=mlp_ratios,
                act_layer=act_layer, norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                resolution=current_resolution,
                vit_num=vit_num if i >= 2 else 0,
                e_ratios=e_ratios
            )
            network.append(stage)
            
            if i >= len(layers) - 1:
                break

            if downsamples is not None and i < len(downsamples) and downsamples[i]:
                network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1],
                        resolution=current_resolution,
                        asub=False,
                        act_layer=act_layer, norm_layer=norm_layer,
                    )
                )
                current_resolution = math.ceil(current_resolution / down_stride)

        self.network = nn.ModuleList(network)
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.dist = False
        self.dist_head = None

        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_tokens(self, x, qp):
        for block_module in self.network:
            # +++ FIX 5: Proper type checking +++
            if isinstance(block_module, QP_Sequential):
                x = block_module(x, qp)
            elif isinstance(block_module, Embedding):
                x = block_module(x)
            else:
                raise TypeError(f"Unexpected module type in network: {type(block_module)}")
        return x

    def forward(self, x, qp):
        # +++ FIX 6: Input validation +++
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got shape {x.shape}")
        if qp.dim() != 1:
            raise ValueError(f"Expected 1D qp tensor (B,), got shape {qp.shape}")
        if x.shape[0] != qp.shape[0]:
            raise ValueError(f"Batch size mismatch: x has {x.shape[0]}, qp has {qp.shape[0]}")
        
        x = self.patch_embed(x)
        x = self.forward_tokens(x, qp)
        x = self.norm(x)
        cls_out = self.head(x.flatten(2).mean(-1))
        return torch.sigmoid(cls_out)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (1, 64, 64), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': (0.5,), 'std': (0.5,),
        'classifier': 'head',
        **kwargs
    }


@register_model
def efficientformerv2_s0_ctu(pretrained=False, **kwargs):
    model = EfficientFormerV2(
        layers=EfficientFormer_depth_ctu_s0,
        embed_dims=EfficientFormer_width_ctu_s0,
        downsamples=[True, True, True, True],
        vit_num=2,
        drop_path_rate=0.0,
        e_ratios=expansion_ratios_ctu_s0,
        resolution=64,
        in_chans=1,
        num_classes=21,
        distillation=False,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def efficientformerv2_1M_ctu(pretrained=False, **kwargs):
    embed_dims_1M = [24, 40, 64, 96]
    depths_1M = [2, 2, 4, 3]
    expansion_ratios_1M = {
        '0': [4, 4],
        '1': [4, 4],
        '2': [4, 4, 4, 4],
        '3': [4, 4, 4],
    }
    
    model = EfficientFormerV2(
        layers=depths_1M,
        embed_dims=embed_dims_1M,
        downsamples=[True, True, True, True],
        vit_num=1,
        drop_path_rate=0.0,
        e_ratios=expansion_ratios_1M,
        resolution=64,
        in_chans=1,
        num_classes=21,
        distillation=False,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model