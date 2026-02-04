import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import itertools
from torch.utils.data import Dataset
from timm.layers import trunc_normal_, SqueezeExcite

# Optimizations for CPU Latency
torch.set_num_threads(1)

# ==================== Profiling ====================
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

try:
    from ptflops import get_model_complexity_info
    HAS_PTFLOPS = True
except ImportError:
    HAS_PTFLOPS = False

# ==================== Constants ====================
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_CLASSES = 21
SELECT_QP_LIST = [22, 27, 32, 37]
BATCH_SIZE = 512 

# ==================== EfficientViT Components ====================

class Conv2d_BN(nn.Module):
    def __init__(self, in_ch, out_ch, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        
        # [OPTIMIZATION] Use Depthwise Separable Conv if kernel size > 1
        # If 1x1 conv, standard conv is already efficient
        if ks > 1:
            self.c = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, ks, stride, pad, dilation, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
            )
        else:
            self.c = nn.Conv2d(in_ch, out_ch, ks, stride, pad, dilation, groups, bias=False)
            
        self.bn = nn.BatchNorm2d(out_ch)
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        return self.bn(self.c(x))

    @torch.no_grad()
    def fuse(self):
        # This is a helper for manual fusion, but we will use a global fuse_model function
        pass

class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        return x + self.m(x)

class FFN(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.act = nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x

class CascadedGroupAttention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=4, attn_ratio=4, resolution=14, kernels=[5, 5, 5, 5]):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            # Input to QKV is dim // num_heads
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim))
        self.qkvs = nn.ModuleList(qkvs)
        self.dws = nn.ModuleList(dws)
        self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(self.d * num_heads, dim, bn_weight_init=0))

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
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0: 
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            attn = ((q.transpose(-2, -1) @ k) * self.scale + 
                    (trainingab[i] if self.training else self.ab[i]))
            attn = attn.softmax(dim=-1)
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x

class LocalWindowAttention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=4, attn_ratio=4, resolution=14, window_resolution=7, kernels=[5, 5, 5, 5]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        self.window_resolution = window_resolution
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads, attn_ratio=attn_ratio, 
                                           resolution=window_resolution, kernels=kernels)

    def forward(self, x):
        H, W = self.resolution, self.resolution
        B, C, H_, W_ = x.shape
        
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H % self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W % self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            
            x = self.attn(x)
            
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution, C).transpose(2, 3).reshape(B, pH, pW, C)
            if padding:
                x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x

class EfficientViTBlock(nn.Module):    
    def __init__(self, type, ed, kd, nh=4, ar=4, resolution=14, window_resolution=7, kernels=[5, 5, 5, 5]):
        super().__init__()
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn0 = Residual(FFN(ed, int(ed * 2)))

        if type == 's':
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar, 
                    resolution=resolution, window_resolution=window_resolution, kernels=kernels))
                
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn1 = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))

# ==================== EfficientViT for HEVC ====================

class EfficientViT_HEVC(nn.Module):
    def __init__(self):
        super().__init__()
        
        embed_dim = [16, 24, 32]   
        num_heads = [2, 2, 2]     
        window_size = 4
        key_dim = [4, 6, 8]        
        attn_ratio = [max(1, int(embed_dim[i] / (key_dim[i] * num_heads[i]))) for i in range(len(embed_dim))]
        
        self.stem = nn.Sequential(
            Conv2d_BN(NUM_CHANNELS, 8, ks=3, stride=2, pad=1),
            nn.ReLU(),
            Conv2d_BN(8, embed_dim[0], ks=3, stride=2, pad=1)
        )
        
        self.stage1 = nn.Sequential(
            EfficientViTBlock('s', embed_dim[0], key_dim[0], num_heads[0], ar=attn_ratio[0],
                              resolution=16, window_resolution=window_size, kernels=[3, 3, 3, 3])
        )
        self.ds1 = PatchMerging(embed_dim[0], embed_dim[1])
        
        self.stage2 = nn.Sequential(
            EfficientViTBlock('s', embed_dim[1], key_dim[1], num_heads[1], ar=attn_ratio[1],
                              resolution=8, window_resolution=window_size, kernels=[3, 3, 3, 3])
        )
        self.ds2 = PatchMerging(embed_dim[1], embed_dim[2])
        
        self.stage3 = nn.Sequential(
            EfficientViTBlock('s', embed_dim[2], key_dim[2], num_heads[2], ar=attn_ratio[2],
                              resolution=4, window_resolution=window_size, kernels=[3, 3, 3, 3])
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim[2] + 1, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(),
            # nn.Dropout(0.12), # Removed for speed
            nn.Linear(1280, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(0.08), # Removed for speed
            nn.Linear(1024, NUM_CLASSES),
            nn.Sigmoid()
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, qp):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.ds1(x)
        x = self.stage2(x)
        x = self.ds2(x)
        x = self.stage3(x)
        
        feat = self.gap(x).flatten(1)
        
        if qp.dim() == 1:
            qp = qp.unsqueeze(1)
            
        return self.head(torch.cat([feat, qp], dim=1))

# ==================== ADVANCED FUSION LOGIC ====================
def fuse_linear_bn(linear, bn):
    w = linear.weight
    b = linear.bias if linear.bias is not None else torch.zeros_like(bn.running_mean)
    mean, var, eps = bn.running_mean, bn.running_var, bn.eps
    scale, shift = bn.weight, bn.bias
    std = torch.sqrt(var + eps)
    w_fused = w * (scale / std).unsqueeze(1)
    b_fused = (b - mean) / std * scale + shift
    new_linear = nn.Linear(linear.in_features, linear.out_features)
    new_linear.weight.data = w_fused
    new_linear.bias.data = b_fused
    return new_linear

def fuse_model(model):
    model.eval()
    # 1. Fuse Conv2d_BN blocks (Conv + BN)
    # We search recursively because EfficientViT nests layers deeply
    for m in model.modules():
        if isinstance(m, Conv2d_BN):
            # If using Sequential (Depthwise Sep)
            if isinstance(m.c, nn.Sequential):
                # Fuse the Pointwise (second) conv with BN
                m.c[1] = torch.nn.utils.fusion.fuse_conv_bn_eval(m.c[1], m.bn)
                m.bn = nn.Identity()
            # If using standard Conv2d
            elif isinstance(m.c, nn.Conv2d):
                m.c = torch.nn.utils.fusion.fuse_conv_bn_eval(m.c, m.bn)
                m.bn = nn.Identity()

    # 2. Fuse Classification Head (Linear + BN)
    head = model.head
    i = 0
    while i < len(head) - 1:
        if isinstance(head[i], nn.Linear) and isinstance(head[i+1], nn.BatchNorm1d):
            head[i] = fuse_linear_bn(head[i], head[i+1])
            head[i+1] = nn.Identity()
            i += 2
        else:
            i += 1

# ==================== MAIN ====================
if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 1. Instantiate & Optimize Memory
    model = EfficientViT_HEVC().to(device)
    model = model.to(memory_format=torch.channels_last)

    # 2. Fuse
    try:
        fuse_model(model)
        print("Fusion successful (Conv+BN and Linear+BN).")
    except Exception as e:
        print(f"Fusion failed: {e}")

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

    count_parameters(model)

    # ==================== INFERENCE TIMING SECTION (SINGLE) ====================
    print("\n--- Preparing for Inference Time Measurement ---")
    
    print("INFO: Timing will be performed on an untrained model.")

    # 2. Set the model to evaluation mode
    model.eval()

    # 3. Create a dummy input tensor
    # [FIX] Added Channel dimension (1, 1, 64, 64)
    dummy_image = torch.randn(1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device) 
    dummy_qp = torch.tensor([32.0 / 51.0]).to(device) 

    # 4. Perform a "warm-up" run
    print("Performing a warm-up run...")
    with torch.no_grad():
        # [FIX] Correct argument order: (image, qp)
        _ = model(dummy_image, dummy_qp)

    print("\n--- Starting Inference Time Measurement ---")

    # 5. Start timers
    start_cpu = os.times()
    start_real = time.time()

    with torch.no_grad():
        # [FIX] Correct argument order: (image, qp)
        output = model(dummy_image, dummy_qp)

    end_real = time.time()
    end_cpu = os.times()

    # 6. Calculate and print the results
    real_time = (end_real - start_real) * 1000  # in milliseconds
    user_time = (end_cpu.user - start_cpu.user) * 1000 # in milliseconds
    system_time = (end_cpu.system - start_cpu.system) * 1000 # in milliseconds

    print(f"Inference Time (Forward Pass):")
    print(f"- Real Time:{real_time:.4f} ms")
    print(f"- User Time:{user_time:.4f} ms")
    print(f"- System Time:{system_time:.4f} ms")
    print("-----------------------------------------\n")
    
    # ==================== FLOPs Calculation Section ====================
    if HAS_THOP:
        print("\n--- Calculating FLOPs using thop ---")
        # [FIX] Correct inputs order: (image, qp)
        macs, params = profile(model, inputs=(dummy_image, dummy_qp), verbose=False)
        gflops = (macs * 2) / 1e9
        print(f"Model Parameters: {params:,}")
        print(f"MACs: {macs:,}")
        print(f"GFLOPs (estimated): {gflops:.4f} G")
    else:
        print("thop library not found. Skipping thop profiling.")

    if HAS_PTFLOPS:
        print("\n--- Calculating FLOPs using ptflops ---")

        def input_constructor(input_res):
            B = 1 
            dummy_image = torch.randn(B, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
            dummy_qp = torch.tensor([32.0 / 51.0] * B).to(device)
            # [FIX] Changed key 'original_ctu' to 'x' to match forward(self, x, qp)
            return {'x': dummy_image, 'qp': dummy_qp}

        input_res_dummy = (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

        macs, params = get_model_complexity_info(
            model,
            input_res_dummy,
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
    else:
         print("ptflops library not found. Skipping ptflops profiling.")

    # ==================== BATCH INFERENCE TIMING ====================
    print("\n--- Preparing for Batch Inference Time Measurement ---")

    n_batch_size = BATCH_SIZE 
    print(f"Using batch size (n): {n_batch_size}")

    # 1. Create dummy batch inputs
    # [FIX] Added channel dimension
    dummy_batch_image = torch.randn(n_batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_batch_qp = torch.tensor([32.0 / 51.0] * n_batch_size).to(device)

    # 2. Perform a "warm-up" run for the batch
    print("Performing a batch warm-up run...")
    with torch.no_grad():
        _ = model(dummy_batch_image, dummy_batch_qp)

    print("\n--- Starting Batch Inference Time Measurement ---")

    n_iterations = 10
    start_cpu_batch = os.times()
    start_real_batch = time.time()
    
    with torch.no_grad():
        for _ in range(n_iterations):
            output_batch = model(dummy_batch_image, dummy_batch_qp)

    end_real_batch = time.time()
    end_cpu_batch = os.times()

    batch_real_time_ms = (end_real_batch - start_real_batch) * 1000 / n_iterations
    batch_user_time_ms = (end_cpu_batch.user - start_cpu_batch.user) * 1000 / n_iterations
    batch_system_time_ms = (end_cpu_batch.system - start_cpu_batch.system) * 1000 / n_iterations
    batch_cpu_total_ms = batch_user_time_ms + batch_system_time_ms

    print(f"Total Batch Inference Time (for {n_batch_size} samples):")
    print(f"- Real Time:{batch_real_time_ms:.4f} ms")
    print(f"- CPU (User+System) Time: {batch_cpu_total_ms:.4f} ms")

    avg_real_time_ms = batch_real_time_ms / n_batch_size
    avg_cpu_time_ms = batch_cpu_total_ms / n_batch_size

    print(f"\nAverage Per-Sample Inference Time (in batch of {n_batch_size}):")
    print(f"- Avg Real Time:{avg_real_time_ms:.4f} ms/sample")
    print(f"- Avg CPU (User+System) Time: {avg_cpu_time_ms:.4f} ms/sample")
    print("-----------------------------------------\n")
    
    exit()