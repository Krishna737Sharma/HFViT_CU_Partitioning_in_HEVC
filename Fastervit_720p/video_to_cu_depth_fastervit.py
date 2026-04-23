# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import time

# =========================================================================
# 1. ENVIRONMENT CONFIGURATION
# =========================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU only
DEVICE = torch.device('cpu')

# =========================================================================
# 1.5 MODEL CONSTANTS
# =========================================================================
NUM_CHANNELS = 1   # Grayscale input
NUM_CLASSES  = 21  # Output classes for HEVC partitioning

# =========================================================================
# 2. FASTERVIT MODEL ARCHITECTURE (FUSED VERSION - OPTIMIZED)
# =========================================================================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
        )
        self.bn  = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


class EfficientResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=True)
        )
        self.bn  = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.conv(x))


class StreamlinedHAT(nn.Module):
    def __init__(self, dim, num_heads=2, window_size=2, mlp_ratio=2.0):
        super().__init__()
        self.dim         = dim
        self.num_heads   = num_heads
        self.window_size = window_size
        self.scale       = (dim // num_heads) ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim,     bias=False)

        hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

        self.pos_scale = nn.Parameter(torch.ones(num_heads) * 0.5)

    def forward(self, x, ct):
        B, C, H, W = x.shape

        x_win  = x.permute(0, 2, 3, 1).reshape(-1, self.window_size ** 2, C)
        ct     = ct.reshape(-1, 1, C)
        tokens = torch.cat([x_win, ct], dim=1)

        shortcut = tokens

        qkv = self.qkv(tokens).reshape(
            tokens.size(0), tokens.size(1), 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * (self.pos_scale.view(1, -1, 1, 1) * self.scale)
        out = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        out = out.transpose(1, 2).reshape(tokens.shape)

        tokens = shortcut + self.proj(out)
        tokens = tokens + self.mlp(self.norm2(tokens))

        x  = tokens[:, :-1, :].reshape(B, H, W, C).permute(0, 3, 1, 2)
        ct = tokens[:, -1:, :]

        return x.contiguous(), ct

class CTInteractionLayer(nn.Module):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.dim      = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.ct_qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.ct_proj = nn.Linear(dim, dim,     bias=False)
        self.ct_norm = nn.LayerNorm(dim)

    def forward(self, ct, batch_size):
        num_windows_total    = ct.size(0)
        num_windows_per_image = num_windows_total // batch_size
        dim                  = ct.size(-1)

        ct_grouped = ct.squeeze(1).reshape(batch_size, num_windows_per_image, dim)
        shortcut   = ct_grouped
        ct_normed  = self.ct_norm(ct_grouped)

        qkv = (
            self.ct_qkv(ct_normed)
            .reshape(batch_size, num_windows_per_image, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(batch_size, num_windows_per_image, dim)

        ct_grouped = shortcut + self.ct_proj(out)
        ct         = ct_grouped.reshape(num_windows_total, 1, dim)
        return ct


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

        self.hat1           = StreamlinedHAT(dims[3], window_size=2)
        self.ct_interaction = CTInteractionLayer(dims[3], num_heads=2)
        self.hat2           = StreamlinedHAT(dims[3], window_size=2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(dims[3] + 1, 512, bias=True),
            nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(512, 768, bias=True),
            nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(768, 21),
            nn.Sigmoid()
        )

    def forward(self, x, qp):
        B = x.size(0)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        ct = F.adaptive_avg_pool2d(x, (2, 2)).flatten(2).transpose(1, 2)

        x, ct = self.hat1(x, ct)
        ct     = self.ct_interaction(ct, B)
        x, _  = self.hat2(x, ct)

        feat = self.gap(x).flatten(1)
        if qp.dim() == 1:
            qp = qp.unsqueeze(1)

        return self.head(torch.cat([feat, qp], dim=1))


# =========================================================================
# 3. OPTIMIZED DATA PROCESSING & INFERENCE
# =========================================================================
IMAGE_SIZE = 64
SAVE_FILE  = 'cu_depth.dat'


def get_y_luma_from_frame(f, width, height):
    """Fast YUV frame reading with minimal overhead"""
    y_size = width * height
    y_buf  = f.read(y_size)

    if not y_buf or len(y_buf) < y_size:
        return None

    # Skip UV data (YUV 4:2:0)
    f.read(y_size // 2)

    return np.frombuffer(y_buf, dtype=np.uint8).reshape(height, width)


def extract_ctus_vectorized(luma, padded_height, padded_width):
    """
    Vectorized CTU extraction using NumPy reshape/transpose.
    ~10-50x faster than nested loops.
    """
    h, w = luma.shape
    if h < padded_height or w < padded_width:
        luma = np.pad(luma, ((0, padded_height - h), (0, padded_width - w)), mode='edge')

    num_rows = padded_height // IMAGE_SIZE
    num_cols = padded_width  // IMAGE_SIZE

    ctus = luma.reshape(num_rows, IMAGE_SIZE, num_cols, IMAGE_SIZE)
    ctus = ctus.transpose(0, 2, 1, 3).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)

    return ctus


def process_frame_batch(model, ctus, qp_value, batch_size=256):
    """
    Optimized batch inference:
    - Larger batch size (512) to reduce loop overhead
    - inference_mode (faster than no_grad)
    - Contiguous C-order numpy copy before tensor conversion
    - channels_last memory format for depthwise convs
    """
    num_ctus = ctus.shape[0]
    results  = []

    ctus_normalized = torch.from_numpy(ctus.copy()).float().div_(255.0).unsqueeze(1)
    ctus_normalized = ctus_normalized.contiguous(memory_format=torch.channels_last)

    qp_tensor = torch.full((num_ctus,), qp_value / 51.0, dtype=torch.float32)

    with torch.inference_mode():
        for i in range(0, num_ctus, batch_size):
            end_idx = min(i + batch_size, num_ctus)

            batch_input = ctus_normalized[i:end_idx]
            batch_qp    = qp_tensor[i:end_idx]

            outputs = model(batch_input, batch_qp)
            results.append(outputs.numpy())

    return np.concatenate(results, axis=0)


def process_video(yuv_file, width, height, qp_value, model, save_file):
    """
    Main processing function with optimized pipeline:
    1. Vectorized CTU extraction
    2. Batch inference with optimal batch size
    3. Streaming output (write per frame)
    """
    padded_height      = math.ceil(height / IMAGE_SIZE) * IMAGE_SIZE
    padded_width       = math.ceil(width  / IMAGE_SIZE) * IMAGE_SIZE
    num_ctus_per_frame = (padded_height // IMAGE_SIZE) * (padded_width // IMAGE_SIZE)

    print(f"Processing: {yuv_file}")
    print(f"Resolution: {width}x{height} -> Padded: {padded_width}x{padded_height}")
    print(f"CTUs per frame: {num_ctus_per_frame}")
    print(f"QP: {qp_value}")
    print(f"Device: {DEVICE}")
    print("-" * 70)

    frame_count = 0
    total_start = time.time()

    try:
        with open(yuv_file, 'rb') as f_in, open(save_file, 'wb') as f_out:
            while True:
                luma = get_y_luma_from_frame(f_in, width, height)
                if luma is None:
                    break

                ctus        = extract_ctus_vectorized(luma, padded_height, padded_width)
                predictions = process_frame_batch(model, ctus, qp_value, batch_size=512)

                f_out.write(predictions.astype(np.float32).tobytes())

                frame_count += 1
                print(f"\rFrame {frame_count} processed", end='', flush=True)

        total_time = time.time() - total_start
        fps        = frame_count / total_time if total_time > 0 else 0

        print(f"\n{'-' * 70}")
        print(f"Processing complete!")
        print(f"Total frames:    {frame_count}")
        print(f"Total time:      {total_time:.2f} seconds")
        print(f"Average FPS:     {fps:.2f}")
        print(f"Time per frame:  {total_time / frame_count:.3f} seconds")
        print(f"Output saved to: {save_file}")

    except IOError as e:
        print(f"\nFile error: {e}")
        sys.exit(1)


# =========================================================================
# 4. MAIN EXECUTION
# =========================================================================

def load_model(checkpoint_path):
    """Load model with all inference optimizations applied"""
    print("Initializing FasterViT model...")
    model = BalancedFasterViT_HEVC().to(DEVICE)
    model = model.to(memory_format=torch.channels_last)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Model file '{checkpoint_path}' not found!")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        else:
            state_dict = checkpoint

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        model.eval()

        model.head = torch.jit.script(model.head)
        # mode='reduce-overhead' minimises per-call Python overhead (best for repeated inference)
        try:
            print("Applying torch.compile (reduce-overhead)...")
            model = torch.compile(model, mode='reduce-overhead')
            print("torch.compile applied successfully.")
        except Exception as compile_err:
            print(f"torch.compile not available or failed ({compile_err}), running without it.")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded successfully! Parameters: {total_params:,}")

        return model

    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    print("=" * 70)
    print("FasterViT HEVC CU Partition Prediction")
    print("=" * 70)

    if len(sys.argv) < 5:
        print("\nUsage: python video_to_cu_depth_fastervit.py <yuv_file> <width> <height> <qp>")
        print("\nExample:")
        print("  python video_to_cu_depth_fastervit.py video.yuv 1920 1080 32")
        sys.exit(1)

    yuv_file  = sys.argv[1]
    width     = int(sys.argv[2])
    height    = int(sys.argv[3])
    qp_value  = int(sys.argv[4])

    if not os.path.exists(yuv_file):
        print(f"ERROR: Input file '{yuv_file}' not found!")
        sys.exit(1)

    if not (0 <= qp_value <= 51):
        print(f"ERROR: QP must be between 0 and 51, got {qp_value}")
        sys.exit(1)

    file_bytes  = os.path.getsize(yuv_file)
    frame_bytes = width * height * 3 // 2

    if file_bytes == 0:
        print("ERROR: Input file is empty!")
        sys.exit(1)

    if file_bytes % frame_bytes != 0:
        print(f"WARNING: File size ({file_bytes}) not exact multiple of frame size ({frame_bytes})")

    num_frames = file_bytes // frame_bytes
    print(f"Detected {num_frames} frames in file\n")

    checkpoint_path = 'best_fastervit_model.pth'
    model = load_model(checkpoint_path)

    print("=" * 70)
    process_video(yuv_file, width, height, qp_value, model, SAVE_FILE)
    print("=" * 70)


if __name__ == "__main__":
    main()