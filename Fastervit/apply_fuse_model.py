import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from torch.utils.data import Dataset
from timm.layers import trunc_normal_
import torch.nn.utils.fusion # Explicitly import fusion utils

torch.set_num_threads(1) # Usually 1 is best for per-sample latency on CPU

# ==================== Profiling Imports ====================
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
BATCH_SIZE = 64 

# ==================== StreamingDataset (Kept for compatibility) ====================
class StreamingDataset(Dataset):
    def __init__(self, file_path, max_samples):
        self.file_path = file_path
        self.max_samples = max_samples

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        try:
            image = np.random.rand(64, 64).astype(np.float32)
            qp = np.random.choice(SELECT_QP_LIST)
            target = np.random.rand(NUM_CLASSES).astype(np.float32)

            ctu_tensor = torch.tensor(image).unsqueeze(0)
            qp_tensor = torch.tensor(float(qp) / 51.0)
            target = torch.tensor(target)

            return qp_tensor, ctu_tensor, target
        except:
            return torch.zeros(1), torch.zeros(1, 64, 64), torch.zeros(NUM_CLASSES)

# ==================== FasterViT Components ====================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        # Split into Depthwise (groups=in_ch) + Pointwise (1x1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
        

class EfficientResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Use Depthwise Separable here too
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        )
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.bn(self.conv(x)))

class StreamlinedHAT(nn.Module):
    def __init__(self, dim, num_heads=2, window_size=2, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

        self.pos_scale = nn.Parameter(torch.ones(num_heads) * 0.5)

    def forward(self, x, ct):
        B, C, H, W = x.shape

        x_win = x.permute(0, 2, 3, 1).reshape(-1, self.window_size ** 2, C)
        ct = ct.reshape(-1, 1, C)
        tokens = torch.cat([x_win, ct], dim=1)

        shortcut = tokens

        qkv = self.qkv(tokens).reshape(
            tokens.size(0), tokens.size(1), 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn * self.pos_scale.view(1, -1, 1, 1)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(tokens.shape)
        tokens = shortcut + self.proj(out)
        tokens = tokens + self.mlp(self.norm2(tokens))

        x = tokens[:, :-1, :].reshape(B, H, W, C).permute(0, 3, 1, 2)
        ct = tokens[:, -1:, :]

        return x.contiguous(), ct

# ==================== Balanced FasterViT HEVC ====================

class BalancedFasterViT_HEVC(nn.Module):
    def __init__(self):
        super().__init__()

        dims = [8, 16, 24, 32]

        self.stem = ConvBNAct(NUM_CHANNELS, dims[0], 3, 2, 1)

        self.stage1 = nn.Sequential(
            EfficientResBlock(dims[0]),
            ConvBNAct(dims[0], dims[1], 3, 2, 1)
        )

        self.stage2 = nn.Sequential(
            EfficientResBlock(dims[1]),
            ConvBNAct(dims[1], dims[2], 3, 2, 1)
        )

        self.stage3 = nn.Sequential(
            EfficientResBlock(dims[2]),
            ConvBNAct(dims[2], dims[3], 3, 2, 1)
        )

        self.hat = StreamlinedHAT(dims[3], window_size=2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(dims[3] + 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(1024, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(1536, NUM_CLASSES),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, qp):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        ct = F.adaptive_avg_pool2d(x, (2, 2)).flatten(2).transpose(1, 2)
        x, _ = self.hat(x, ct)

        feat = self.gap(x).flatten(1)
        if qp.dim() == 1:
            qp = qp.unsqueeze(1)

        return self.head(torch.cat([feat, qp], dim=1))

# ==================== BN Fusion Logic ====================

def fuse_model(model):
    """
    Fuses Conv+BN and Linear+BN layers for faster inference.
    """
    model.eval()
    
    # PART 1: Fuse Convolutions (Conv2d + BatchNorm2d)
    for m in model.modules():
        if isinstance(m, (ConvBNAct, EfficientResBlock)):
            if len(m.conv) == 2:
                # Fuse the second convolution in the sequence with the following BN
                m.conv[1] = torch.nn.utils.fusion.fuse_conv_bn_eval(m.conv[1], m.bn)
                m.bn = nn.Identity()

    # PART 2: Fuse Linear Layers (Linear + BatchNorm1d)
    head = model.head
    i = 0
    while i < len(head) - 1:
        if isinstance(head[i], nn.Linear) and isinstance(head[i+1], nn.BatchNorm1d):
            
            linear = head[i]
            bn = head[i+1]
            
            # --- Fusion Math for Linear Layers ---
            w = linear.weight
            b = linear.bias if linear.bias is not None else torch.zeros_like(bn.running_mean)
            scale = bn.weight
            shift = bn.bias
            mean = bn.running_mean
            var = bn.running_var
            eps = bn.eps
            
            std = torch.sqrt(var + eps)
            w_fused = w * (scale / std).unsqueeze(1)
            b_fused = (b - mean) / std * scale + shift
            
            # Update Linear layer
            head[i].weight.data = w_fused
            head[i].bias.data = b_fused
            
            # Remove BatchNorm
            head[i+1] = nn.Identity()
            
            i += 2 # Skip next layer
        else:
            i += 1

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    device = torch.device("cpu") # Force CPU for inference timing accuracy
    print(f"Using device: {device}")
    
    INPUT_PATH = 'best_fastervit_hevc_balanced.pth'
    OUTPUT_PATH = 'best_fastervit_model.pth'

    # 1. Instantiate the Model
    print("1. Instantiating model architecture...")
    model = BalancedFasterViT_HEVC().to(device)
    model = model.to(memory_format=torch.channels_last)
    
    # 2. Load Trained Weights
    if os.path.exists(INPUT_PATH):
        print(f"2. Loading weights from {INPUT_PATH}...")
        checkpoint = torch.load(INPUT_PATH, map_location=device)
        
        # Handle dictionary structure from training code
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("   Weights loaded successfully.")
    else:
        print(f"ERROR: Could not find {INPUT_PATH}. Please check file path.")
        exit()

    # 3. Set to Evaluation Mode (Required for Fusion)
    model.eval()

    # 4. Apply Fusion
    print("3. Applying Layer Fusion (Conv+BN and Linear+BN)...")
    try:
        fuse_model(model)
        print("   Fusion successful.")
    except Exception as e:
        print(f"   Fusion FAILED: {e}")
        exit()

    # 5. Save the Fused Model
    print(f"4. Saving fused model to {OUTPUT_PATH}...")
    torch.save(model.state_dict(), OUTPUT_PATH)
    print("   Saved successfully.")

    # 6. Verify Parameter Count
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total Parameters (Fused): {total_params:,}")

    count_parameters(model)

    # ==================== INFERENCE TIMING (SINGLE) ====================
    print("\n" + "="*50)
    print("VERIFYING INFERENCE SPEED (FUSED MODEL)")
    print("="*50)

    # Prepare inputs
    dummy_image = torch.randn(1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device) 
    dummy_qp = torch.tensor([32.0 / 51.0]).to(device) 

    # Warm-up
    print("Performing single-sample warm-up run...")
    with torch.no_grad():
        _ = model(dummy_image, dummy_qp)

    # Timing
    print("Running single-sample inference timing...")
    start_cpu = os.times()
    start_real = time.time()

    with torch.no_grad():
        output = model(dummy_image, dummy_qp)

    end_real = time.time()
    end_cpu = os.times()

    real_time = (end_real - start_real) * 1000
    user_time = (end_cpu.user - start_cpu.user) * 1000
    system_time = (end_cpu.system - start_cpu.system) * 1000

    print(f"Inference Time (Single Sample):")
    print(f"- Real Time:   {real_time:.4f} ms")
    print(f"- CPU Time:    {user_time + system_time:.4f} ms")
    
    # ==================== BATCH INFERENCE TIMING ====================
    print("\n--- Preparing for Batch Inference Time Measurement ---")

    n_batch_size = BATCH_SIZE 
    print(f"Using batch size (n): {n_batch_size}")

    # 1. Create dummy batch inputs
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
    print(f"- Real Time: {batch_real_time_ms:.4f} ms")
    print(f"- CPU Time:  {batch_cpu_total_ms:.4f} ms")

    avg_real_time_ms = batch_real_time_ms / n_batch_size
    avg_cpu_time_ms = batch_cpu_total_ms / n_batch_size

    print(f"\nAverage Per-Sample Inference Time (in batch of {n_batch_size}):")
    print(f"- Avg Real Time: {avg_real_time_ms:.4f} ms/sample")
    print(f"- Avg CPU Time:  {avg_cpu_time_ms:.4f} ms/sample")
    print("-----------------------------------------\n")

    # ==================== FLOPs Calculation ====================
    if HAS_THOP:
        print("\n--- Calculating FLOPs (thop) ---")
        try:
            macs, params = profile(model, inputs=(dummy_image, dummy_qp), verbose=False)
            gflops = (macs * 2) / 1e9
            print(f"MACs: {macs:,}")
            print(f"GFLOPs: {gflops:.4f} G")
        except:
            print("Error calculating FLOPs with thop.")
            
    if HAS_PTFLOPS:
        print("\n--- Calculating FLOPs (ptflops) ---")
        try:
            def input_constructor(input_res):
                return {'x': dummy_image, 'qp': dummy_qp}
            
            macs, params = get_model_complexity_info(
                model, (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
                as_strings=False, print_per_layer_stat=False, verbose=False,
                input_constructor=input_constructor, backend='aten'
            )
            gflops = (macs * 2) / 1e9
            print(f"MACs: {macs:,}")
            print(f"GFLOPs: {gflops:.4f} G")
        except:
            print("Error calculating FLOPs with ptflops.")

    print("\nDone. You can now use 'best_fastervit_model.pth' for deployment.")