# --------------------------------------------------------
# EfficientViT Model Architecture
# Copyright (c) 2022 Microsoft
#
# MODIFIED for HEVC CTU Partitioning (QP-Aware)
# --------------------------------------------------------
import torch
import itertools
import torch.nn as nn

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite

# ==================================================================================
# Helper Modules (Conv2d_BN, BN_Linear, PatchMerging, Residual)
# ==================================================================================

class Conv2d_BN(torch.nn.Sequential):
    """
    A sequential module combining a 2D Convolution and BatchNorm.
    This structure is convenient and allows for future fusing
    (merging BN parameters into Conv) for faster inference.
    """
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        # Add the 2D convolution layer
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False)) # bias=False because BN has a bias (beta)
        # Add the 2D batch normalization layer
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        # Initialize BN weights to 1 and bias to 0
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the BatchNorm layer's parameters into the Convolution layer's
        weights and bias for faster inference.
        This is a common optimization.
        """
        # Get the conv and bn modules
        c, bn = self._modules.values()
        
        # Calculate the fused weight
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None] # Reshape and multiply
        
        # Calculate the fused bias
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        
        # Create a new Conv2d module with the fused parameters
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    """
    A sequential module combining a 1D BatchNorm and a Linear layer.
    Similar to Conv2d_BN, this allows for future fusing.
    """
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        # Add 1D batch normalization (operates on features)
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        # Add the linear (fully connected) layer
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        # Initialize linear layer weights with truncated normal distribution
        trunc_normal_(self.l.weight, std=std)
        # Initialize bias to 0 if it exists
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the BatchNorm1d parameters into the Linear layer's
        weights and bias for faster inference.
        """
        # Get the bn and linear modules
        bn, l = self._modules.values()
        
        # Calculate fused weight
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        
        # Calculate fused bias
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
            
        # Create a new Linear module with fused parameters
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    """
    Patch Merging layer. This module downsamples the feature map
    by a factor of 2, effectively merging 2x2 patches into one.
    It's used to reduce spatial resolution and increase channel dimension
    between stages.
    """
    def __init__(self, dim, out_dim, input_resolution):
        super().__init__()
        hid_dim = int(dim * 4) # Intermediate hidden dimension
        
        # 1x1 Conv to expand channels
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution)
        self.act = torch.nn.ReLU()
        # 3x3 Depthwise Conv with stride 2 to perform spatial downsampling
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution)
        # Squeeze-and-Excite block for channel-wise attention
        self.se = SqueezeExcite(hid_dim, .25)
        # 1x1 Conv to project back to the desired output dimension
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2)

    def forward(self, x):
        # Apply the layers sequentially
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    """
    Standard Residual block (skip connection).
    Wraps a module 'm' and adds its input 'x' to its output 'm(x)'.
    This version does NOT take a QP argument.
    """
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop # Dropout probability for stochastic depth

    def forward(self, x):
        # Standard residual connection: x + m(x)
        if self.training and self.drop > 0:
            # Apply stochastic depth (dropout for layers)
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            # At inference time, just do x + m(x)
            return x + self.m(x)


# ==================================================================================
# MODIFIED QP-AWARE MODULES
# ==================================================================================

class FFN_with_QP(torch.nn.Module):
    """
    Feed Forward Network (FFN) modified to be "QP-Aware".
    
    It accepts an additional Quantization Parameter (QP) tensor,
    expands it to the spatial dimensions (H, W), and concatenates
    it to the input feature map 'x' along the channel dimension.
    
    Input:  x [B, C, H, W], qp [B]
    Output: [B, C, H, W]
    """
    def __init__(self, ed, h, resolution):
        super().__init__()
        # Pointwise Conv 1: Input channels = embed_dim + 1 (for the QP channel)
        self.pw1 = Conv2d_BN(ed + 1, h, resolution=resolution)
        self.act = torch.nn.ReLU()
        # Pointwise Conv 2: Project back to original embedding dimension
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)

    def forward(self, x, qp):
        B, C, H, W = x.shape
        
        # --- CORE QP-AWARE LOGIC ---
        # qp is [B] (a scalar per batch item)
        # 1. Reshape qp to [B, 1, 1, 1]
        # 2. Expand it to [B, 1, H, W] to match the spatial dimensions of x
        qp_expanded = qp.reshape(B, 1, 1, 1).expand(B, 1, H, W)
        
        # 3. Concatenate x and the expanded qp along the channel dimension (dim=1)
        #    This makes the input [B, C+1, H, W]
        x_with_qp = torch.cat([x, qp_expanded], dim=1)
        # --- END OF CORE LOGIC ---
        
        # Pass the concatenated tensor through the network
        x = self.pw2(self.act(self.pw1(x_with_qp)))
        return x


class Residual_with_QP(torch.nn.Module):
    """
    Residual block modified to pass the 'qp' argument to its child module 'm'.
    
    This is necessary to wrap QP-Aware modules (like FFN_with_QP)
    while maintaining the residual connection.
    """
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m # 'm' is expected to be a module that accepts (x, qp)
        self.drop = drop

    def forward(self, x, qp): # Requires QP argument
        # Pass both x and qp to the wrapped module
        m_out = self.m(x, qp)
        
        if self.training and self.drop > 0:
            # Apply stochastic depth
            return x + m_out * torch.rand(x.size(0), 1, 1, 1,
                                            device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            # Residual connection: x + m(x, qp)
            return x + m_out

# ==================================================================================
# Original FFN (kept for subsampling blocks)
# This module is NOT QP-Aware and is used in parts of the network
# that do not need the QP information (e.g., subsampling layers).
# ==================================================================================
class FFN(torch.nn.Module):
    """
    Standard Feed Forward Network (FFN).
    This version does NOT take a QP argument.
    """
    def __init__(self, ed, h, resolution):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

# ==================================================================================
# Attention Modules (Unchanged from original EfficientViT)
# These modules are NOT QP-Aware.
# ==================================================================================

class CascadedGroupAttention(torch.nn.Module):
    r""" Cascaded Group Attention.
    
    This is a complex attention mechanism from the original EfficientViT paper.
    It splits heads into groups and cascades them, where the output of one
    group is added to the input of the next.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5 # Standard attention scaling factor
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim) # Dimension of Value
        self.attn_ratio = attn_ratio

        qkvs = []        # This list will store the layers responsible for creating the Query, Key, and Value (QKV) tensors for each head.
        dws = []         # This list will store the Depthwise convolution layers that are applied only to the Query (Q) tensor of each head.
        # Create modules for each attention head
        for i in range(num_heads):
            # 1x1 Conv to project to Q, K, and V dimensions
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            # Depthwise Conv for the Query (Q)
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim, resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        
        # Final projection layer to merge head outputs
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        # --- Relative Position Bias ---
        # Pre-calculates relative position biases for all possible
        # offset pairs in the attention window.
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
        
        # Trainable parameter for the biases
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        # A buffer to store the pre-calculated indices for efficient lookup
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        """
        Overrides the .train() method to manage the attention bias buffer.
        """
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            # In train mode, delete the pre-computed buffer
            del self.ab
        else:
            # In eval mode, create the pre-computed buffer for faster inference
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        
        # Get the attention bias: use the pre-computed buffer in eval, or calculate it in train
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        
        # Split input features along channel dim for different heads
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        
        # --- Cascaded Logic ---
        for i, qkv in enumerate(self.qkvs):
            if i > 0: # For all heads after the first one...
                # Add the output of the *previous* head to the input of this head
                feat = feat + feats_in[i]
            
            # Project to Q, K, V
            feat = qkv(feat)
            # Split Q, K, V
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W
            # Apply depthwise conv to Query
            q = self.dws[i](q)
            # Flatten spatial dimensions (H, W) -> N
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N
            
            # --- Standard Attention Calculation ---
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale # (B, N, C/h) @ (B, C/h, N) -> (B, N, N)
                +
                (trainingab[i] if self.training else self.ab[i]) # Add relative position bias
            )
            attn = attn.softmax(dim=-1) # (B, N, N)
            
            # Apply attention to Value
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # (B, d, N) @ (B, N, N) -> (B, d, N) -> (B, d, H, W)
            feats_out.append(feat)
            
        # Concatenate all head outputs and project
        x = self.proj(torch.cat(feats_out, 1))
        return x


class LocalWindowAttention(torch.nn.Module):           # mostly it will not work because out window size=7 and we have resolution 64/16=4
    r""" Local Window Attention.
    
    This module applies the CascadedGroupAttention within local windows
    of the feature map. This is a common technique (like in Swin Transformer)
    to reduce computation from O(N^2) to O(W^2 * N), where W is window size.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution
        
        window_resolution = min(window_resolution, resolution)  # mostly it will not work because out window size=7 and we have resolution 64/16=4
        # Instantiate the core attention module
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                attn_ratio=attn_ratio, 
                                resolution=window_resolution,
                                kernels=kernels,)

    def forward(self, x):
        H = W = self.resolution
        B, C, H_, W_ = x.shape
        assert H == H_ and W == W_, 'input feature has wrong size, expect {}, got {}'.format((H, W), (H_, W_))
               
        # If the feature map is smaller than or equal to the window, just apply attention directly
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            # --- Window Partitioning ---
            # Pad the feature map if its size is not a multiple of the window size
            x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution # Number of windows in height
            nW = pW // self.window_resolution # Number of windows in width
            
            # Reshape to (B * nH * nW, window_res, window_res, C)
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            )
            # (B * nH * nW, C, window_res, window_res)
            x = x.permute(0, 3, 1, 2)
            
            # --- Apply Attention ---
            # Apply attention to all windows in parallel
            x = self.attn(x)
            
            # --- Window Reversal ---
            # (B*nH*nW, C, h, w) -> (B*nH*nW, h, w, C)
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                       C).transpose(2, 3).reshape(B, pH, pW, C)
            
            # Remove padding if it was added
            if padding:
                x = x[:, :H, :W].contiguous()
            
            # (B, H, W, C) -> (B, C, H, W)
            x = x.permute(0, 3, 1, 2)
        return x


# ==================================================================================
# MODIFIED EFFICIENT-VIT BLOCK (QP-Aware)
# ==================================================================================

class EfficientViTBlock(torch.nn.Module):    
    """ 
    A basic EfficientViT building block, modified to be QP-Aware.
    
    The structure is:
    1. Depthwise Conv (Residual)
    2. QP-Aware FFN (Residual_with_QP)
    3. Local Window Attention (Residual)
    4. Depthwise Conv (Residual)
    5. QP-Aware FFN (Residual_with_QP)
    """
    def __init__(self, type,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],
                 ffn_exp_ratio=2.0,): # Added ffn_exp_ratio
        super().__init__()
        
        # Calculate hidden dim for FFNs
        ffn_hidden_dim = int(ed * ffn_exp_ratio)
            
        # 1. Depthwise Conv (standard, no QP)
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        
        # 2. QP-Aware FFN (uses Residual_with_QP wrapper)
        self.ffn0 = Residual_with_QP(FFN_with_QP(ed, ffn_hidden_dim, resolution))

        if type == 's': # 's' type includes the attention block
            # 3. Local Window Attention (standard, no QP)
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar, \
                    resolution=resolution, window_resolution=window_resolution, kernels=kernels))
        
        # 4. Depthwise Conv (standard, no QP)
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        
        # 5. QP-Aware FFN (uses Residual_with_QP wrapper)
        self.ffn1 = Residual_with_QP(FFN_with_QP(ed, ffn_hidden_dim, resolution))

    def forward(self, x, qp):
        """
        The forward pass requires both the feature map 'x' and
        the quantization parameter 'qp'.
        """
        # Pass x to non-QP modules
        x = self.dw0(x)
        # Pass x AND qp to QP-aware modules
        x = self.ffn0(x, qp)
        
        x = self.mixer(x)
        
        # Pass x to non-QP modules
        x = self.dw1(x)
        # Pass x AND qp to QP-aware modules
        x = self.ffn1(x, qp)
        return x

# ==================================================================================
# MODIFIED EFFICIENT-VIT MAIN MODEL (QP-Aware)
# ==================================================================================

class EfficientViT(torch.nn.Module):
    """
    The main EfficientViT model, modified to be QP-Aware.
    The `forward` pass is changed to accept 'qp' and pass it
    through the network.
    """
    def __init__(self, img_size=224,
                patch_size=16,
                in_chans=3,
                num_classes=1000,
                stages=['s', 's', 's'],
                embed_dim=[64, 128, 192],
                key_dim=[16, 16, 16],
                depth=[1, 2, 3],
                num_heads=[4, 4, 4],
                window_size=[7, 7, 7],
                kernels=[5, 5, 5, 5],
                down_ops=[['subsample', 2], ['subsample', 2], ['']],
                ffn_exp_ratio=2.0, # Added ffn_exp_ratio
                ):
        super().__init__()

        resolution = img_size
        # --- Patch Embedding ---
        # Uses a stack of 4 strided convolutions to
        # embed the input image (e.g., 64x64) into a
        # feature map (e.g., 4x4) with 'embed_dim[0]' channels.
        self.patch_embed = torch.nn.Sequential(
                           Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution), # 64->32
                           torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1, resolution=resolution // 2), # 32->16
                           torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1, resolution=resolution // 4), # 16->8
                           torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1, resolution=resolution // 8)) # 8->4

        resolution = img_size // patch_size # 64 // 16 = 4
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        
        # --- CRITICAL MODIFICATION ---
        # We must use nn.ModuleList instead of nn.Sequential.
        # This is because nn.Sequential does not allow passing an
        # extra argument (like 'qp') to its sub-modules' forward methods.
        # We will have to manually iterate through these lists in the forward pass.
        self.blocks1 = torch.nn.ModuleList()
        self.blocks2 = torch.nn.ModuleList()
        self.blocks3 = torch.nn.ModuleList()
        
        # --- Build EfficientViT Stages ---
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            
            # Select the correct ModuleList for the current stage
            current_blocks = eval('self.blocks' + str(i+1))
            
            # Add 'dpth' number of EfficientViTBlocks
            for d in range(dpth):
                current_blocks.append(
                    EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels, 
                                      ffn_exp_ratio=ffn_exp_ratio) # Pass ffn_exp_ratio
                )
            
            # If a downsampling operation is specified
            if do[0] == 'subsample':
                # Get the *next* stage's ModuleList to add subsampling layers to
                next_blocks = eval('self.blocks' + str(i+2))
                
                resolution_ = (resolution - 1) // do[1] + 1
                
                # --- Add Subsampling Layers ---
                # These layers are standard (non-QP-aware)
                # 1. A standard FFN block
                next_blocks.append(torch.nn.Sequential(
                    Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution)),
                ))
                # 2. The PatchMerging layer to downsample
                next_blocks.append(
                    PatchMerging(*embed_dim[i:i + 2], resolution)
                )
                resolution = resolution_ # Update resolution for the next stage
                # 3. Another standard FFN block after downsampling
                next_blocks.append(torch.nn.Sequential(
                    Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1], resolution=resolution)),
                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)),
                ))
        
        # --- Classification Head ---
        # A simple BatchNorm + Linear layer for classification
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        Specifies parameters that should be excluded from weight decay
        (e.g., biases, normalization parameters).
        """
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x, qp): # MODIFIED: Added 'qp' argument
        # 1. Patch Embedding
        x = self.patch_embed(x)
        
        # --- CRITICAL MODIFICATION: Manual Stage Forward ---
        # We manually iterate through each ModuleList and check the
        # type of block to see if we need to pass the 'qp' argument.
        
        # 2. Stage 1
        for blk in self.blocks1:
            if isinstance(blk, EfficientViTBlock):
                # If it's our QP-Aware block, pass 'x' AND 'qp'
                x = blk(x, qp)
            else:
                # Otherwise (e.g., subsampling), just pass 'x'
                x = blk(x)
        
        # 3. Stage 2
        for blk in self.blocks2:
            if isinstance(blk, EfficientViTBlock):
                x = blk(x, qp)
            else:
                x = blk(x) # This stage has subsampling layers
        
        # 4. Stage 3
        for blk in self.blocks3:
            if isinstance(blk, EfficientViTBlock):
                x = blk(x, qp)
            else:
                x = blk(x) # This stage has subsampling layers

        # 5. Global Pooling
        # Average pool the feature map down to 1x1
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        
        # 6. Classification Head
        x = self.head(x)
        
        # 7. --- MODIFIED: Apply sigmoid ---
        # For multi-label binary classification (which this CTU
        # partitioning problem is), we use a sigmoid activation
        # on the final logits.
        out = torch.sigmoid(x)
        return out