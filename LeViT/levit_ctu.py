# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# MODIFIED for HEVC CTU Partitioning (QP-Aware) from LeViT
# https://github.com/facebookresearch/LeViT

import torch
import torch.nn as nn
import itertools
# import utils # utils.replace_batchnorm not used here directly

from timm.models.vision_transformer import trunc_normal_
# from timm.models.registry import register_model # Not needed for direct use

# ==============================================================================
# Original Helper Modules (Conv2d_BN, BN_Linear, Residual)
# ==============================================================================

# Note: FLOPS_COUNTER removed for clarity in this adaptation
# global FLOPS_COUNTER
# FLOPS_COUNTER = 0

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        l = torch.nn.Linear(a, b, bias=bias)
        trunc_normal_(l.weight, std=std)
        if bias:
            torch.nn.init.constant_(l.bias, 0)
        self.add_module('l', l)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        # Assumes BNC format, drops tokens
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), x.size(1), 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

# ==============================================================================
# NEW Patch Embedding for 64x64 -> 8x8
# ==============================================================================
def patch_embed_64_8(embed_dim, activation, in_chans=1):
    """
    Creates a patch embedding module for 64x64 input to 8x8 output.
    Uses 3 convolutional layers with stride 2.
    """
    return torch.nn.Sequential(
        Conv2d_BN(in_chans, embed_dim // 4, 3, 2, 1, resolution=64), # 64->32
        activation(),
        Conv2d_BN(embed_dim // 4, embed_dim // 2, 3, 2, 1, resolution=32), # 32->16
        activation(),
        Conv2d_BN(embed_dim // 2, embed_dim, 3, 2, 1, resolution=16) # 16->8
    )

# ==============================================================================
# QP-Aware Modules (Linear_BN + MLP + Residual Wrapper)
# ==============================================================================
class Linear_BN_with_QP(torch.nn.Module):
    """ Linear layer with preceding BN, modified for QP injection (BNC format) """
    def __init__(self, in_features, out_features, bn_weight_init=1):
        super().__init__()
        # Input features = original features + 1 (for QP)
        self.linear = torch.nn.Linear(in_features + 1, out_features, bias=False)
        self.bn = torch.nn.BatchNorm1d(out_features)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # Note: Fusion is more complex with QP injection, maybe skip for now
        # Or adapt carefully if needed for deployment.
        print("Warning: fuse() not implemented for Linear_BN_with_QP")
        return self

    def forward(self, x, qp):
        # x shape: [B, N, C]
        # qp shape: [B]
        B, N, C = x.shape

        # Expand qp: [B] -> [B, 1, 1] -> [B, N, 1]
        qp_expanded = qp.view(B, 1, 1).expand(B, N, 1)

        # Concatenate on channel dimension: [B, N, C+1]
        x_with_qp = torch.cat([x, qp_expanded], dim=2)

        # Linear layer expects [*, in_features]. Reshape BNC -> (B*N)C
        x_flat = x_with_qp.reshape(B * N, C + 1) # Use reshape

        x_flat = self.linear(x_flat)

        # BN expects [N, C] or [N, C, L]. Reshape for BN.
        # Apply BN, output shape [B*N, out_features]
        x_bn = self.bn(x_flat)

        # Reshape back to BNC: [B, N, out_features]
        x_out = x_bn.reshape(B, N, -1) # Use reshape
        return x_out


# Simple Linear -> BN layer (doesn't handle QP)
class Linear_BN_Simple(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        # Assume BNC input, need to flatten/unflatten
        B, N, C = x.shape
        # *** FIX: Use reshape instead of view ***
        x = x.reshape(B * N, C)
        l, bn = self._modules.values()
        x = l(x)
        x = bn(x) # BN operates on [B*N, C_out]
        # *** FIX: Use reshape instead of view ***
        x = x.reshape(B, N, -1) # Reshape back
        return x


class MLP_with_QP(torch.nn.Module):
    """ MLP block using Linear_BN_with_QP and Linear_BN_Simple """
    def __init__(self, in_features, hidden_features, activation, bn_weight_init=0):
        super().__init__()
        self.linear1 = Linear_BN_with_QP(in_features, hidden_features)
        self.act = activation()
        # Use the non-QP version for the second layer
        self.linear2 = Linear_BN_Simple(hidden_features, in_features, bn_weight_init=bn_weight_init)

    def forward(self, x, qp):
        x = self.linear1(x, qp)
        x = self.act(x)
        x = self.linear2(x) # Pass only x to the second layer
        return x


class Residual_with_QP(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x, qp): # Requires QP
        m_out = self.m(x, qp)
        if self.training and self.drop > 0:
            # Assumes BNC format, drops tokens
            return x + m_out * torch.rand(x.size(0), x.size(1), 1,
                                            device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + m_out

# ==============================================================================
# Original Attention Modules (Attention, Subsample, AttentionSubsample)
# ==============================================================================

class Attention(torch.nn.Module):
    # --- UNCHANGED from original levit.py ---
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        # Use Linear_BN_Simple here as it doesn't need QP
        self.qkv = Linear_BN_Simple(dim, h)
        self.proj = torch.nn.Sequential(activation(), Linear_BN_Simple(
            self.dh, dim, bn_weight_init=0))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
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
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x) # B, N, H = (qkv_h)
        # Correct splitting based on Linear_BN_Simple outputting H
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3) # B, h, N, Dk
        k = k.permute(0, 2, 1, 3) # B, h, N, Dk
        v = v.permute(0, 2, 1, 3) # B, h, N, Dv

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class Subsample(torch.nn.Module):
    # --- UNCHANGED from original levit.py ---
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[
            :, ::self.stride, ::self.stride].reshape(B, -1, C)
        return x


class AttentionSubsample(torch.nn.Module):
     # --- UNCHANGED from original levit.py ---
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14, resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_**2
        h = self.dh + nh_kd
        # Use Linear_BN_Simple
        self.kv = Linear_BN_Simple(in_dim, h)

        self.q = torch.nn.Sequential(
            Subsample(stride, resolution),
            Linear_BN_Simple(in_dim, nh_kd))
        self.proj = torch.nn.Sequential(activation(), Linear_BN_Simple(
            self.dh, out_dim))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(
            range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.resolution_2, self.num_heads, self.key_dim).permute(0, 2, 1, 3) # B, h, N_, Dk

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh) # B, N_, Dh
        x = self.proj(x)
        return x


# ==============================================================================
# MODIFIED LeViT Main Class (for CTU Partitioning)
# ==============================================================================

class LeViT_CTU(torch.nn.Module):
    """ LeViT adapted for CTU Partitioning (QP-Aware). """
    def __init__(self, img_size=64,
                 in_chans=1,
                 num_classes=21,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 down_ops=[],
                 attention_activation=torch.nn.Hardswish,
                 mlp_activation=torch.nn.Hardswish,
                 drop_path=0):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim

        # MODIFIED Patch Embedding for 64x64 -> 8x8
        self.patch_embed = patch_embed_64_8(embed_dim[0], activation=mlp_activation, in_chans=in_chans)

        # Calculate initial resolution after patch embedding
        # 64 -> 32 -> 16 -> 8
        resolution = img_size // 8

        # MODIFIED: Use ModuleList and manual iteration
        self.blocks = torch.nn.ModuleList()
        down_ops.append(['']) # Add dummy entry for last stage

        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):

            current_resolution = resolution # Resolution for this stage

            for _ in range(dpth):
                # Attention block (uses standard Residual)
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=current_resolution,
                    ), drop_path))

                # MLP block (uses QP-aware Residual)
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual_with_QP( # MODIFIED: Use QP Residual
                            MLP_with_QP(ed, h, mlp_activation), # MODIFIED: Use QP MLP
                            drop_path)
                    )

            if do[0] == 'Subsample':
                #('Subsample', key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                stride = do[5]
                next_ed = embed_dim[i+1] # Embed dim of the next stage
                next_mr = do[4] # MLP ratio for the block after subsampling

                # Calculate the resolution after subsampling
                resolution_ = (current_resolution - 1) // stride + 1

                # Attention Subsample block (uses standard Residual)
                self.blocks.append(
                    AttentionSubsample(
                        ed, next_ed, # Use current and next embed dim
                        key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=stride,
                        resolution=current_resolution,
                        resolution_=resolution_))

                resolution = resolution_ # Update resolution for the next stage

                # Optional MLP block after subsampling (uses QP-aware Residual)
                if next_mr > 0:
                    h = int(next_ed * next_mr)
                    self.blocks.append(
                        Residual_with_QP( # MODIFIED: Use QP Residual
                            MLP_with_QP(next_ed, h, mlp_activation), # MODIFIED: Use QP MLP
                            drop_path)
                        )

        # Classifier head
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x, qp): # MODIFIED: Added 'qp'
        # Patch Embedding (BCHW -> BCHW)
        x = self.patch_embed(x)

        # Flatten and transpose to BNC format (Batch, Num_Tokens, Channels)
        x = x.flatten(2).transpose(1, 2)

        # Iterate through blocks manually
        for blk in self.blocks:
            # Check if the block is QP-aware (Residual_with_QP or MLP_with_QP directly if not wrapped)
            if isinstance(blk, Residual_with_QP):
                 x = blk(x, qp)
            elif isinstance(blk, MLP_with_QP): # Should be wrapped, but just in case
                 x = blk(x, qp)
            else: # Standard Residual (wrapping Attention) or AttentionSubsample
                 x = blk(x)

        # Average Pooling over tokens (dim=1 for BNC)
        x = x.mean(1)

        # Classifier Head
        x = self.head(x)

        # MODIFIED: Sigmoid activation for multi-label output
        out = torch.sigmoid(x)
        return out

# ==============================================================================
# LeViT Model Configurations (Adapted for 64x64 input -> 8x8 feature map)
# ==============================================================================
# Example: LeViT_128S config adapted
# Original: C='128_256_384', D=16, N='4_6_8', X='2_3_4'
# Resolutions: 14x14 -> 7x7 -> 4x4
# Adapted Resolutions: 8x8 -> 4x4 -> 2x2
LeViT_64T_CTU_config = {
    'embed_dim': [64, 128, 192],   # Reduced C (64, 128, 192)
    'key_dim': [16, 16, 16],        # Kept D=16
    'depth': [1, 1, 1],             # Reduced X (Total 3 blocks)
    'num_heads': [4, 8, 12],         # N: 64/16=4, 128/16=8, 192/16=12
    'attn_ratio': [2, 2, 2],
    'mlp_ratio': [2, 2, 2],
    'down_ops': [
        # Subsample: (key_dim, num_heads, attn_ratio, mlp_ratio, stride)
        ['Subsample', 16, 64 // 16, 4, 2, 2], # Stage 1->2 (8x8 -> 4x4)
        ['Subsample', 16, 128 // 16, 4, 2, 2], # Stage 2->3 (4x4 -> 2x2)
    ],
    'attention_activation': torch.nn.Hardswish,
    'mlp_activation': torch.nn.Hardswish,
    'drop_path': 0
}

LeViT_96S_CTU_config = {
    'embed_dim': [96, 192, 288],   # Reduced C
    'key_dim': [16, 16, 16],        # Kept D=16
    'depth': [2, 2, 3],             # Reduced X
    'num_heads': [3, 6, 6],         # Reduced N (ensure divisibility)
    'attn_ratio': [2, 2, 2],
    'mlp_ratio': [2, 2, 2],
    'down_ops': [
        # Adjust num_heads in down_ops based on new C and D
        ['Subsample', 16, 96 // 16, 4, 2, 2], # 96/16 = 6 heads
        ['Subsample', 16, 192 // 16, 4, 2, 2], # 192/16 = 12 heads
    ],
    'attention_activation': torch.nn.Hardswish,
    'mlp_activation': torch.nn.Hardswish,
    'drop_path': 0
}

# We can define the configs directly in the training script or here
LeViT_128S_CTU_config = {
    'embed_dim': [128, 256, 384],   # C
    'key_dim': [16, 16, 16],        # D
    'depth': [2, 3, 4],             # X
    'num_heads': [4, 6, 8],         # N
    'attn_ratio': [2, 2, 2],        # Default from levit.py model_factory
    'mlp_ratio': [2, 2, 2],         # Default from levit.py model_factory
    'down_ops': [
        # ('Subsample', key_dim, num_heads, attn_ratio, mlp_ratio, stride)
        ['Subsample', 16, 128 // 16, 4, 2, 2], # Stage 1->2 (8x8 -> 4x4)
        ['Subsample', 16, 256 // 16, 4, 2, 2], # Stage 2->3 (4x4 -> 2x2)
    ],
    'attention_activation': torch.nn.Hardswish,
    'mlp_activation': torch.nn.Hardswish,
    'drop_path': 0 # Default for 128S
}

# Add other configs (LeViT_128, LeViT_192 etc.) similarly if needed
# Example: LeViT_256 config adapted
# Original: C='256_384_512', D=32, N='4_6_8', X='4_4_4'
LeViT_256_CTU_config = {
    'embed_dim': [256, 384, 512],   # C
    'key_dim': [32, 32, 32],        # D
    'depth': [4, 4, 4],             # X
    'num_heads': [4, 6, 8],         # N
    'attn_ratio': [2, 2, 2],
    'mlp_ratio': [2, 2, 2],
    'down_ops': [
        ['Subsample', 32, 256 // 32, 4, 2, 2], # Stage 1->2 (8x8 -> 4x4)
        ['Subsample', 32, 384 // 32, 4, 2, 2], # Stage 2->3 (4x4 -> 2x2)
    ],
    'attention_activation': torch.nn.Hardswish,
    'mlp_activation': torch.nn.Hardswish,
    'drop_path': 0 # Default for 256
}