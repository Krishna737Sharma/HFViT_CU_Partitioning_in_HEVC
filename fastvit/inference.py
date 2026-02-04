# ----------------------------------------------------------------------------------
# FastViT-Pico (Modified: Stops at 8x8 Resolution)
# Target: Extremely Low Latency, ~0.08M Params
# ----------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import copy
from thop import profile
from ptflops import get_model_complexity_info

# ==================================================================================
# SECTION 1: CONFIGURATION
# ==================================================================================
IMAGE_SIZE = 64
NUM_CHANNELS = 1

# ==================================================================================
# SECTION 2: CPU-OPTIMIZED MODEL ARCHITECTURE (Pico)
# ==================================================================================

class MobileOneBlock(nn.Module):
    """ Standard Reparameterizable Block """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 groups=1, inference_mode=False):
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                          padding, groups=groups, bias=True)
        else:
            self.rbr_skip = nn.BatchNorm2d(in_channels) \
                if out_channels == in_channels and stride == 1 else None
            
            self.rbr_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_scale = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.activation = nn.GELU()

    def forward(self, x):
        if self.inference_mode:
            return self.activation(self.reparam_conv(x))
        
        identity = 0
        if self.rbr_skip is not None:
            identity = self.rbr_skip(x)
        return self.activation(self.rbr_conv(x) + self.rbr_scale(x) + identity)

    def reparameterize(self):
        if self.inference_mode: return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, 
                                      self.stride, self.padding, groups=self.groups, bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
        for attr in ['rbr_conv', 'rbr_scale', 'rbr_skip']:
            if hasattr(self, attr): delattr(self, attr)
        self.inference_mode = True

    def _get_kernel_bias(self):
        # Fuse Conv+BN branches
        k_conv, b_conv = self._fuse_bn(self.rbr_conv[0], self.rbr_conv[1])
        k_scale, b_scale = self._fuse_bn(self.rbr_scale[0], self.rbr_scale[1])
        
        # Pad 1x1 to match 3x3
        pad = self.kernel_size // 2
        k_scale = F.pad(k_scale, [pad, pad, pad, pad])
        
        k_final = k_conv + k_scale
        b_final = b_conv + b_scale

        if self.rbr_skip is not None:
            k_id, b_id = self._fuse_bn(self._identity_kernel(), self.rbr_skip)
            k_final += k_id
            b_final += b_id
        return k_final, b_final

    def _fuse_bn(self, conv, bn):
        if isinstance(conv, torch.Tensor): kernel = conv # Handle identity case
        else: kernel = conv.weight
        
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def _identity_kernel(self):
        k = torch.zeros(self.in_channels, 1, self.kernel_size, self.kernel_size, device=self.rbr_conv[0].weight.device)
        k[range(self.in_channels), 0, self.kernel_size//2, self.kernel_size//2] = 1
        return k


class AdditiveQPFFN(nn.Module):
    """ 
    Optimized FFN for CPU:
    1. Removes concatenation.
    2. Uses 'Additive' QP injection.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # QP Projection: Scalar QP -> Channel bias
        self.qp_proj = nn.Linear(1, dim) 
        
        # FFN: Depthwise -> Pointwise -> Pointwise
        self.conv_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False) 
        self.bn_dw = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x, qp):
        # Additive QP Injection (Zero-copy broadcast)
        # qp: [B, 1] -> [B, C] -> [B, C, 1, 1]
        qp_bias = self.qp_proj(qp).view(x.shape[0], x.shape[1], 1, 1)
        x = x + qp_bias 
        
        # Spatial Mixing
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.act(out)
        
        # Channel Mixing
        out = self.mlp(out)
        return x + out

class FastViT_Pico(nn.Module):
    def __init__(self, inference_mode=False):
        super().__init__()
        
        # ================= PICO CONFIG =================
        # Reduced to 3 stages only (Ends at 8x8 resolution)
        self.embed_dims = [48, 98, 196]  # Removed the 4th dimension (64)
        self.depths = [2, 2, 4]         # Removed the 4th depth
        # ===============================================

        # Stem: 64x64 -> 32x32
        self.stem = MobileOneBlock(1, self.embed_dims[0], 3, 2, 1, inference_mode=inference_mode)
        
        # Stage 1: 32x32 processing
        self.stage1 = self._make_stage(self.embed_dims[0], self.depths[0], inference_mode)
        
        # Down 1: 32x32 -> 16x16
        self.down1 = MobileOneBlock(self.embed_dims[0], self.embed_dims[1], 3, 2, 1, inference_mode=inference_mode)
        
        # Stage 2: 16x16 processing
        self.stage2 = self._make_stage(self.embed_dims[1], self.depths[1], inference_mode)
        
        # Down 2: 16x16 -> 8x8
        self.down2 = MobileOneBlock(self.embed_dims[1], self.embed_dims[2], 3, 2, 1, inference_mode=inference_mode)
        
        # Stage 3: 8x8 processing (FINAL STAGE)
        self.stage3 = self._make_stage(self.embed_dims[2], self.depths[2], inference_mode)
        
        # --- REMOVED down3 and stage4 ---
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Head input is now embed_dims[2] (48) instead of embed_dims[3]
        self.head = nn.Sequential(
            nn.Linear(self.embed_dims[2] + 1, 64), 
            nn.ReLU(),
            nn.Linear(64, 21),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def _make_stage(self, dim, depth, inference_mode):
        blocks = []
        for _ in range(depth):
            blocks.append(AdditiveQPFFN(dim, int(dim * 2))) 
        return nn.Sequential(*blocks)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x, qp):
        # FIX: Ensure QP is [B, 1]
        if qp.dim() == 1:
            qp = qp.view(-1, 1)

        # 1. Stem (64 -> 32)
        x = self.stem(x)
        
        # 2. Stage 1 (32x32)
        for blk in self.stage1: x = blk(x, qp)
        x = self.down1(x) # 32 -> 16
        
        # 3. Stage 2 (16x16)
        for blk in self.stage2: x = blk(x, qp)
        x = self.down2(x) # 16 -> 8
        
        # 4. Stage 3 (8x8) - FINAL PROCESSING
        for blk in self.stage3: x = blk(x, qp)
        
        # --- STOP HERE (No down3, No stage4) ---
        
        # Global Average Pooling (8x8 -> 1x1)
        x = self.gap(x).flatten(1)
        
        # Concatenate QP and Predict
        x = torch.cat([x, qp], dim=1)
        return self.head(x)

    def reparameterize(self):
        for m in self.modules():
            if hasattr(m, 'reparameterize') and m is not self:
                m.reparameterize()

# ==================================================================================
# MAIN EXECUTION (MATCHING SWIN FORMAT)
# ==================================================================================

def main():
    torch.set_num_threads(1) 
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # 1. Initialize Model
    model = FastViT_Pico(inference_mode=False).to(device)
    print(f"Model Parameters (Before Reparam): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 2. Reparameterize
    print("\n--- Reparameterizing Model (Collapsing Branches) ---")
    model.eval()
    model.reparameterize()
    print(f"Model Parameters (After Reparam): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ----------------------------------------------------------------------
    # A. SINGLE INFERENCE TIMING (BS=1)
    # ----------------------------------------------------------------------
    print("\n--- Preparing for Inference Time Measurement on FastViT ---")
    
    dummy_image = torch.randn(1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_qp = torch.tensor([27.0 / 51.0]).to(device)

    print("Performing a warm-up run...")
    with torch.no_grad():
        _ = model(dummy_image, dummy_qp)

    print("\n--- Starting Inference Time Measurement ---")

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_cpu = os.times()
    start_real = time.time()

    with torch.no_grad():
        output = model(dummy_image, dummy_qp)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_real = time.time()
    end_cpu = os.times()

    real_time = (end_real - start_real) * 1000
    user_time = (end_cpu.user - start_cpu.user) * 1000
    system_time = (end_cpu.system - start_cpu.system) * 1000

    print(f"Inference Time (Forward Pass):")
    print(f"  - Real Time:   {real_time:.4f} ms")
    print(f"  - User Time:   {user_time:.4f} ms")
    print(f"  - System Time: {system_time:.4f} ms")
    print("-----------------------------------------\n")

    # ----------------------------------------------------------------------
    # B. FLOPS CALCULATION
    # ----------------------------------------------------------------------
    print("\n--- Creating Model Copy for FLOP Counting ---")
    model_for_flops = copy.deepcopy(model)
    
    print("\n--- Calculating FLOPs using thop ---")
    try:
        macs, params = profile(model_for_flops, inputs=(dummy_image, dummy_qp), verbose=False)
        gflops = (macs * 2) / 1e9
        print(f"Model Parameters: {params:,}")
        print(f"MACs: {macs:,}")
        print(f"GFLOPs (estimated): {gflops:.4f} G")
    except Exception as e:
        print(f"thop profiling failed: {e}")

    print("\n--- Calculating FLOPs using ptflops ---")
    model_for_ptflops = copy.deepcopy(model)
    def input_constructor(input_res):
        B = 1 
        C, H, W = input_res
        dummy_image = torch.randn(B, C, H, W).to(device)
        dummy_qp = torch.tensor([32.0 / 51.0] * B).to(device)
        return {'x': dummy_image, 'qp': dummy_qp}

    try:
        macs, params = get_model_complexity_info(
            model_for_ptflops,
            (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
            input_constructor=input_constructor,
            backend='aten'
        )
        gflops = (macs * 2) / 1e9
        print(f"Model Parameters: {params:,}")
        print(f"MACs: {macs:,}")
        print(f"GFLOPs (estimated): {gflops:.4f} G")
    except Exception as e:
        print(f"ptflops profiling failed: {e}")

    # ----------------------------------------------------------------------
    # C. BATCH INFERENCE TIMING (BS=64)
    # ----------------------------------------------------------------------
    print("\n--- Preparing for Batch Inference Time Measurement ---")

    n_batch_size = 64
    print(f"Using batch size (n): {n_batch_size}")

    dummy_batch_image = torch.randn(n_batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_batch_qp = torch.tensor([32.0 / 51.0] * n_batch_size).to(device) 

    print("Performing a batch warm-up run...")
    with torch.no_grad():
        _ = model(dummy_batch_image, dummy_batch_qp)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    print("\n--- Starting Batch Inference Time Measurement ---")

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_cpu_batch = os.times()
    start_real_batch = time.time()

    with torch.no_grad():
        output_batch = model(dummy_batch_image, dummy_batch_qp)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_real_batch = time.time()
    end_cpu_batch = os.times()

    batch_real_time_ms = (end_real_batch - start_real_batch) * 1000 
    batch_user_time_ms = (end_cpu_batch.user - start_cpu_batch.user) * 1000
    batch_system_time_ms = (end_cpu_batch.system - start_cpu_batch.system) * 1000
    batch_cpu_total_ms = batch_user_time_ms + batch_system_time_ms

    print(f"Total Batch Inference Time (for {n_batch_size} samples):")
    print(f"  - Real Time:            {batch_real_time_ms:.4f} ms")
    print(f"  - CPU (User+System) Time: {batch_cpu_total_ms:.4f} ms")

    avg_real_time_ms = batch_real_time_ms / n_batch_size
    avg_cpu_time_ms = batch_cpu_total_ms / n_batch_size

    print(f"\nAverage Per-Sample Inference Time (in batch of {n_batch_size}):")
    print(f"  - Avg Real Time:            {avg_real_time_ms:.4f} ms/sample")
    print(f"  - Avg CPU (User+System) Time: {avg_cpu_time_ms:.4f} ms/sample")
    print("-----------------------------------------\n")

if __name__ == '__main__':
    main()