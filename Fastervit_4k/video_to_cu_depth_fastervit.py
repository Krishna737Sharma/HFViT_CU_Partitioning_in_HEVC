#coding=utf-8
from __future__ import absolute_import, division, print_function

import math
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# ENVIRONMENT
# =========================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = ''
DEVICE = torch.device('cpu')
print(f'Using device: {DEVICE}')

# =========================================================================
# CONSTANTS (must match training and fusion scripts exactly)
# =========================================================================
IMAGE_SIZE          = 64
NUM_CHANNELS        = 1
NUM_CLASSES         = 21
SAVE_FILE           = 'cu_depth.dat'
INFERENCE_BATCH_SIZE = 256  # Fixed batch size — optimised for CPU throughput.
                            # CTInteractionLayer constraint (4*B % B == 0)
                            # is always satisfied for any constant B.
                            # Last partial batch is padded to exactly 256.

# =========================================================================
# ARCHITECTURE — identical to training; head uses nn.Identity() for fused BN
# =========================================================================
class ConvBNAct(nn.Module):
    """
    Depthwise-separable conv block.
    After BN fusion: conv[1] has bias=True, self.bn is nn.Identity().
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False),  # depthwise
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True),               # pointwise (bias absorbed from BN)
        )
        self.bn  = nn.Identity()  # placeholder — key kept for state-dict compatibility
        self.act = nn.GELU()

    def forward(self, x):
        # bn is Identity after fusion, so self.bn(self.conv(x)) == self.conv(x)
        return self.act(self.bn(self.conv(x)))


class EfficientResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),  # depthwise
            nn.Conv2d(dim, dim, 1, 1, 0, bias=True),               # pointwise
        )
        self.bn  = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.bn(self.conv(x)))


class StreamlinedHAT(nn.Module):
    # num_heads=2 — MUST match training
    def __init__(self, dim, num_heads=2, window_size=2, mlp_ratio=2.0):
        super().__init__()
        self.dim         = dim
        self.num_heads   = num_heads
        self.window_size = window_size
        self.scale       = (dim // num_heads) ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim,     bias=False)

        hidden     = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
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
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn * self.pos_scale.view(1, -1, 1, 1)
        attn = attn.softmax(dim=-1)

        out    = (attn @ v).transpose(1, 2).reshape(tokens.shape)
        tokens = shortcut + self.proj(out)
        tokens = tokens + self.mlp(self.norm2(tokens))

        x  = tokens[:, :-1, :].reshape(B, H, W, C).permute(0, 3, 1, 2)
        ct = tokens[:, -1:, :]
        return x.contiguous(), ct


class CTInteractionLayer(nn.Module):
    # num_heads=2 — MUST match training
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
        num_windows_total     = ct.size(0)
        num_windows_per_image = num_windows_total // batch_size
        dim = ct.size(-1)

        # Safety check — must always hold; caught early during development.
        assert num_windows_total % batch_size == 0, (
            f"CTInteractionLayer: num_windows_total ({num_windows_total}) must be "
            f"divisible by batch_size ({batch_size}). "
            "Use process_frame_batch_safe() which pads the last mini-batch."
        )

        ct_grouped = ct.squeeze(1).reshape(batch_size, num_windows_per_image, dim)
        shortcut   = ct_grouped
        ct_normed  = self.ct_norm(ct_grouped)

        qkv = (
            self.ct_qkv(ct_normed)
            .reshape(batch_size, num_windows_per_image, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out        = (attn @ v).transpose(1, 2).reshape(batch_size, num_windows_per_image, dim)
        ct_grouped = shortcut + self.ct_proj(out)
        ct         = ct_grouped.reshape(num_windows_total, 1, dim)
        return ct


class BalancedFasterViT_HEVC(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [8, 16, 24, 32]

        self.stem   = ConvBNAct(NUM_CHANNELS, dims[0], 3, 2, 1)
        self.stage1 = nn.Sequential(EfficientResBlock(dims[0]), ConvBNAct(dims[0], dims[1], 3, 2, 1))
        self.stage2 = nn.Sequential(EfficientResBlock(dims[1]), ConvBNAct(dims[1], dims[2], 3, 2, 1))
        self.stage3 = nn.Sequential(EfficientResBlock(dims[2]), ConvBNAct(dims[2], dims[3], 3, 2, 1))

        self.hat1           = StreamlinedHAT(dims[3], num_heads=2, window_size=2)
        self.ct_interaction = CTInteractionLayer(dims[3], num_heads=2)
        self.hat2           = StreamlinedHAT(dims[3], num_heads=2, window_size=2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Head: BN slots replaced with Identity (absorbed into preceding Linear).
        # Index layout: 0=Linear, 1=Identity, 2=ReLU, 3=Dropout,
        #               4=Linear, 5=Identity, 6=ReLU, 7=Dropout,
        #               8=Linear, 9=Sigmoid
        self.head = nn.Sequential(
            nn.Linear(dims[3] + 1, 1024, bias=True),
            nn.Identity(),           # was BatchNorm1d(1024) — fused into Linear above
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(1024, 1536, bias=True),
            nn.Identity(),           # was BatchNorm1d(1536) — fused into Linear above
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(1536, NUM_CLASSES),
            nn.Sigmoid(),
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
        x, _   = self.hat2(x, ct)

        feat = self.gap(x).flatten(1)
        if qp.dim() == 1:
            qp = qp.unsqueeze(1)
        return self.head(torch.cat([feat, qp], dim=1))


# =========================================================================
# MODEL LOADING 
# =========================================================================
def load_model(checkpoint_path: str) -> nn.Module:
    print("Initialising FasterViT model (fused architecture) ...")
    model = BalancedFasterViT_HEVC().to(DEVICE)
    model = model.to(memory_format=torch.channels_last)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Model file '{checkpoint_path}' not found!")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)

        # Unwrap if saved inside a dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        else:
            state_dict = checkpoint

        # Strip DataParallel 'module.' prefix if present
        clean_sd = OrderedDict()
        for k, v in state_dict.items():
            clean_sd[k[7:] if k.startswith('module.') else k] = v

        model.load_state_dict(clean_sd, strict=True)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded successfully. Parameters: {total_params:,}")

        # ── INT8 Dynamic Quantization ─────────────────────────────────
        print("Applying INT8 dynamic quantization (Linear layers) ...")
        model = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},          # quantize all Linear layers to int8
            dtype=torch.qint8,
        )
        model = model.to(memory_format=torch.channels_last)
        print("INT8 dynamic quantization complete.")
        # ──────────────────────────────────────────────────────────────

        return model

    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =========================================================================
# DATA PROCESSING
# =========================================================================
def get_y_luma_from_frame(f, width: int, height: int):
    """Read one YUV 4:2:0 frame and return the luma plane as uint8 array."""
    y_size = width * height
    y_buf  = f.read(y_size)
    if not y_buf or len(y_buf) < y_size:
        return None
    f.read(y_size // 2)  # discard UV
    return np.frombuffer(y_buf, dtype=np.uint8).reshape(height, width)


def extract_ctus_vectorized(luma: np.ndarray,
                             padded_height: int,
                             padded_width: int) -> np.ndarray:
    """
    Vectorised CTU extraction.
    Returns float32 array of shape [num_ctus, IMAGE_SIZE, IMAGE_SIZE],
    already normalised to [0, 1].
    """
    h, w = luma.shape
    if h < padded_height or w < padded_width:
        luma = np.pad(luma,
                      ((0, padded_height - h), (0, padded_width - w)),
                      mode='edge')

    num_rows = padded_height // IMAGE_SIZE
    num_cols = padded_width  // IMAGE_SIZE

    # [H, W] → [num_rows, 64, num_cols, 64] → [num_ctus, 64, 64]
    ctus = (luma.reshape(num_rows, IMAGE_SIZE, num_cols, IMAGE_SIZE)
                .transpose(0, 2, 1, 3)
                .reshape(-1, IMAGE_SIZE, IMAGE_SIZE))
    return ctus


def process_frame_batch(model: nn.Module,
                        ctus: np.ndarray,
                        qp_value: int) -> np.ndarray:
    """
    Run inference on all CTUs for ONE frame using INFERENCE_BATCH_SIZE=256.

    CTInteractionLayer divisibility guarantee
    ──────────────────────────────────────────
    Each CTU produces a 2×2 window grid → 4 windows per CTU.
    CTInteractionLayer reshapes its input as [B*4, 1, dim] → [B, 4, dim],
    requiring (B*4) % B == 0, which is trivially true for any constant B.

    The only edge case is the LAST mini-batch when num_ctus % 256 != 0.
    Fix: pad it to exactly 256 by repeating the last CTU, run the forward
    pass (B=256, constraint satisfied), then slice off only the real outputs.

    Args:
        model    : eval-mode fused model.
        ctus     : uint8 array [num_ctus, IMAGE_SIZE, IMAGE_SIZE].
        qp_value : integer QP (0–51).

    Returns:
        float32 numpy array [num_ctus, NUM_CLASSES].
    """
    num_ctus = ctus.shape[0]
    B = INFERENCE_BATCH_SIZE

    # Pre-process entire frame at once — single allocation, no per-batch copies
    ctus_t = (torch.from_numpy(ctus)
              .float()
              .div_(255.0)
              .unsqueeze(1)                          # [N, 1, H, W]
              .to(memory_format=torch.channels_last))

    qp_t = torch.full((num_ctus,), qp_value / 51.0, dtype=torch.float32)

    results = []

    for start in range(0, num_ctus, B):
        end        = min(start + B, num_ctus)
        real_count = end - start

        if real_count == B:
            # ── Full batch: no padding needed ─────────────────────────────
            with torch.inference_mode():
                out = model(ctus_t[start:end], qp_t[start:end])
            results.append(out.numpy())
        else:
            # ── Last partial batch: pad to exactly B ──────────────────────
            # Repeat the last real CTU to fill the remainder.
            pad_count = B - real_count
            img_chunk = torch.cat([
                ctus_t[start:end],
                ctus_t[end - 1 : end].expand(pad_count, -1, -1, -1).contiguous(),
            ], dim=0)                                # shape [B, 1, H, W]
            qp_chunk = torch.cat([
                qp_t[start:end],
                qp_t[end - 1 : end].expand(pad_count),
            ], dim=0)                                # shape [B]

            with torch.inference_mode():
                out = model(img_chunk, qp_chunk)    # B=256, constraint OK

            # Discard padded outputs — keep only real ones
            results.append(out[:real_count].numpy())

    return np.concatenate(results, axis=0)


# =========================================================================
# VIDEO PROCESSING
# =========================================================================
def process_video(yuv_file: str, width: int, height: int,
                  qp_value: int, model: nn.Module,
                  save_file: str) -> None:
    """
    Full video processing pipeline:
      1. Read luma frames from YUV 4:2:0 file.
      2. Vectorised CTU extraction.
      3. Batch inference at INFERENCE_BATCH_SIZE=256 with auto-padding.
      4. Streaming output (float32 binary, one frame at a time).
    """
    padded_height      = math.ceil(height / IMAGE_SIZE) * IMAGE_SIZE
    padded_width       = math.ceil(width  / IMAGE_SIZE) * IMAGE_SIZE
    num_ctus_per_frame = (padded_height // IMAGE_SIZE) * (padded_width // IMAGE_SIZE)

    print(f"Processing : {yuv_file}")
    print(f"Resolution : {width}×{height}  →  padded {padded_width}×{padded_height}")
    print(f"CTUs/frame : {num_ctus_per_frame}")
    print(f"QP         : {qp_value}")
    print(f"Batch size : {INFERENCE_BATCH_SIZE} (last batch padded if needed)")
    print(f"Device     : {DEVICE}")
    print("-" * 70)

    frame_count = 0
    total_start = time.perf_counter()

    try:
        with open(yuv_file, 'rb') as f_in, open(save_file, 'wb') as f_out:
            while True:
                luma = get_y_luma_from_frame(f_in, width, height)
                if luma is None:
                    break

                ctus        = extract_ctus_vectorized(luma, padded_height, padded_width)
                predictions = process_frame_batch(model, ctus, qp_value)

                f_out.write(predictions.astype(np.float32).tobytes())
                frame_count += 1
                print(f"\rFrame {frame_count} processed", end='', flush=True)

        total_time = time.perf_counter() - total_start
        fps        = frame_count / total_time if total_time > 0 else 0.0

        print(f"\n{'-' * 70}")
        print(f"Done!")
        print(f"  Frames processed : {frame_count}")
        print(f"  Total time       : {total_time:.2f} s")
        print(f"  Average FPS      : {fps:.2f}")
        print(f"  Time/frame       : {total_time/max(frame_count,1):.3f} s")
        print(f"  Output saved to  : {save_file}")

    except IOError as e:
        print(f"\nFile error: {e}")
        sys.exit(1)


# =========================================================================
# ENTRY POINT
# =========================================================================
def main():
    print("=" * 70)
    print("FasterViT HEVC CU Partition Prediction (FUSED — INT8 + COMPILED)")
    print("=" * 70)

    usage = (
        "\nUsage:\n"
        "  python inference_script.py <yuv_file> <width> <height> <qp>\n\n"
        "Example:\n"
        "  python inference_script.py video.yuv 1920 1080 32\n\n"
        f"  Inference runs in fixed batches of {INFERENCE_BATCH_SIZE} CTUs.\n"
        "  The last partial batch is automatically padded to 256 and\n"
        "  the padded outputs are discarded — no accuracy impact.\n"
    )

    if len(sys.argv) < 5:
        print(usage)
        sys.exit(1)

    yuv_file  = sys.argv[1]
    width     = int(sys.argv[2])
    height    = int(sys.argv[3])
    qp_value  = int(sys.argv[4])

    # ── Validation ────────────────────────────────────────────────────────
    if not os.path.exists(yuv_file):
        print(f"ERROR: '{yuv_file}' not found.")
        sys.exit(1)
    if not (0 <= qp_value <= 51):
        print(f"ERROR: QP must be 0–51, got {qp_value}.")
        sys.exit(1)

    file_bytes  = os.path.getsize(yuv_file)
    frame_bytes = width * height * 3 // 2
    if file_bytes == 0:
        print("ERROR: Input file is empty.")
        sys.exit(1)
    if file_bytes % frame_bytes != 0:
        print(f"WARNING: File size ({file_bytes} B) is not an exact multiple "
              f"of frame size ({frame_bytes} B). Last frame may be partial.")

    num_frames = file_bytes // frame_bytes
    print(f"Detected {num_frames} frame(s) in file.\n")

    # ── Load model ────────────────────────────────────────────────────────
    checkpoint_path = 'best_fastervit_fused_4k.pth'
    model = load_model(checkpoint_path)

    print("=" * 70)

    # ── Run ───────────────────────────────────────────────────────────────
    process_video(yuv_file, width, height, qp_value, model, SAVE_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()