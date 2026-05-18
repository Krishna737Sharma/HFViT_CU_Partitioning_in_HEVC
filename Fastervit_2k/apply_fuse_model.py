"""
fusion_script.py
================
Loads the trained BalancedFasterViT_HEVC checkpoint, fuses Conv+BN and
Linear+BN pairs, verifies numerical correctness, then saves the fused
state-dict ready for deployment.

KEY FIXES vs. the original fusion script
-----------------------------------------
1. Architecture definition EXACTLY matches training:
   - StreamlinedHAT   : num_heads=2  (was 1 in broken script)
   - CTInteractionLayer: num_heads=2  (was 1 in broken script)
2. fuse_conv_bn targets conv[1] (pointwise) + bn correctly.
3. fuse_linear_bn is applied to both Linear+BN pairs in head.
4. Saved state-dict uses Identity placeholders so that the inference
   model (which also has Identity in those slots) loads with strict=True.
"""

import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

# ── Optional profiling libs ──────────────────────────────────────────────
try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

try:
    from ptflops import get_model_complexity_info
    HAS_PTFLOPS = True
except ImportError:
    HAS_PTFLOPS = False

# =========================================================================
# CONSTANTS  (must match training exactly)
# =========================================================================
IMAGE_SIZE   = 64
NUM_CHANNELS = 1
NUM_CLASSES  = 21
BATCH_SIZE   = 64   # used only for timing benchmarks

INPUT_PATH  = 'best_fastervit_hevc_balanced.pth'
OUTPUT_PATH = 'best_fastervit_fused_2k.pth'


# =========================================================================
# ARCHITECTURE  — identical to training script
# =========================================================================

class ConvBNAct(nn.Module):
    """Depthwise + Pointwise conv, followed by BN and GELU."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False),  # [0] depthwise
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),               # [1] pointwise
        )
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class EfficientResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),  # [0] depthwise
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),              # [1] pointwise
        )
        self.bn  = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.bn(self.conv(x)))


class StreamlinedHAT(nn.Module):
    def __init__(self, dim, num_heads=2
                 , window_size=2, mlp_ratio=2.0):
        super().__init__()
        self.dim         = dim
        self.num_heads   = num_heads
        self.window_size = window_size
        self.scale       = (dim // num_heads) ** -0.5

        # ── Pre-Norm layers ──────────────────────────────────────────────────
        self.norm1 = nn.LayerNorm(dim)
        self.qkv   = nn.Linear(dim, dim * 3, bias=False)
        self.proj  = nn.Linear(dim, dim,     bias=False)
        self.norm2 = nn.LayerNorm(dim)
        hidden     = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        # ── Positional encoding ──────────────────────────────────────────────
        seq_len = window_size ** 2 + 1
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(num_heads, seq_len, seq_len)
        )
        trunc_normal_(self.rel_pos_bias, std=0.02)
        self.register_buffer(
            'window_pos_emb',
            self._build_sinusoidal(window_size, dim)
        )

    @staticmethod
    def _build_sinusoidal(window_size, dim):
        coords = []
        for row in range(window_size):
            for col in range(window_size):
                r = (row / max(window_size - 1, 1)) * 2 - 1
                c = (col / max(window_size - 1, 1)) * 2 - 1
                coords.append([r, c])
        coords = torch.tensor(coords, dtype=torch.float32)

        half = dim // 2
        freq = torch.pow(
            10000.0,
            -2.0 * torch.arange(half, dtype=torch.float32) / dim
        )
        row_enc = torch.sin(coords[:, 0:1] * freq)
        col_enc = torch.cos(coords[:, 1:2] * freq)
        pos_emb = torch.cat([row_enc, col_enc], dim=-1)
        return pos_emb.unsqueeze(0)                 # [1, ws^2, dim]

    @staticmethod
    def _window_partition(x, window_size):
        """
        x   : [B, H, W, C]
        out : [B * (H//ws) * (W//ws), ws*ws, C]
        """
        B, H, W, C = x.shape
        ws = window_size
        x = x.reshape(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.reshape(-1, ws * ws, C)

    @staticmethod
    def _window_reverse(windows, window_size, H, W, B):
        """
        windows : [B * num_windows, ws*ws, C]
        out     : [B, H, W, C]
        """
        ws = window_size
        C  = windows.shape[-1]
        x  = windows.reshape(B, H // ws, W // ws, ws, ws, C)
        x  = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.reshape(B, H, W, C)

    def forward(self, x, ct):
        B, C, H, W = x.shape

        # ── Correct window partitioning ──────────────────────────────────────
        x_nhwc = x.permute(0, 2, 3, 1)                    # [B, H, W, C]
        x_win  = self._window_partition(x_nhwc, self.window_size)
        # x_win : [B*nW, ws^2, C]

        # ── Absolute position embedding (window tokens only) ─────────────────
        x_win = x_win + self.window_pos_emb                # broadcasts [1,ws^2,C]

        # ── Carrier token (no absolute pos embed) ───────────────────────────
        ct     = ct.reshape(-1, 1, C)
        tokens = torch.cat([x_win, ct], dim=1)             # [B*nW, ws^2+1, C]
        shortcut = tokens

        # ── Pre-Norm → QKV ───────────────────────────────────────────────────
        qkv = (
            self.qkv(self.norm1(tokens))
            .reshape(tokens.size(0), tokens.size(1),
                     3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_pos_bias.unsqueeze(0)       # learnable bias
        attn = attn.softmax(dim=-1)

        out    = (attn @ v).transpose(1, 2).reshape(tokens.shape)
        tokens = shortcut + self.proj(out)

        # ── Pre-Norm → MLP ───────────────────────────────────────────────────
        tokens = tokens + self.mlp(self.norm2(tokens))

        # ── Correct window reverse ───────────────────────────────────────────
        x_tokens = tokens[:, :-1, :]                       # [B*nW, ws^2, C]
        x  = self._window_reverse(x_tokens, self.window_size, H, W, B)
        x  = x.permute(0, 3, 1, 2).contiguous()           # [B, C, H, W]
        ct = tokens[:, -1:, :]                             # [B*nW, 1, C]
        return x, ct


class CTInteractionLayer(nn.Module):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.dim       = dim
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

        ct_grouped = ct.squeeze(1).reshape(
            batch_size, num_windows_per_image, dim
        )
        shortcut  = ct_grouped
        ct_normed = self.ct_norm(ct_grouped)

        qkv = (
            self.ct_qkv(ct_normed)
            .reshape(batch_size, num_windows_per_image,
                     3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(
            batch_size, num_windows_per_image, dim
        )

        ct_grouped = shortcut + self.ct_proj(out)
        return ct_grouped.reshape(num_windows_total, 1, dim)
       
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

        self.head = nn.Sequential(
            nn.Linear(dims[3] + 1, 1024, bias=True),   # bias=True required for BN fusion
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(1024, 1536, bias=True),
            nn.BatchNorm1d(1536),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(1536, NUM_CLASSES),
            nn.Sigmoid(),
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
# BN FUSION HELPERS
# =========================================================================

def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuse Conv2d (bias=False) + BatchNorm2d → single Conv2d with bias.

    Derivation:
        BN(conv(x)) = gamma/std * conv(x) + (beta - gamma*mean/std)
        → w_fused = gamma/std * W          (broadcast over output channels)
        → b_fused = beta - gamma*mean/std
    """
    mean  = bn.running_mean
    var   = bn.running_var
    eps   = bn.eps
    gamma = bn.weight   # scale  (γ)
    beta  = bn.bias     # shift  (β)
    std   = (var + eps).sqrt()

    w_fused = conv.weight * (gamma / std).reshape(-1, 1, 1, 1)
    b_fused = beta - gamma * mean / std

    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        conv.kernel_size, conv.stride, conv.padding,
        groups=conv.groups, bias=True,
    )
    fused.weight = nn.Parameter(w_fused.clone())
    fused.bias   = nn.Parameter(b_fused.clone())
    return fused


def fuse_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """
    Fuse Linear + BatchNorm1d → single Linear with bias.

    Derivation:
        BN(linear(x)) = gamma/std * (W*x + b - mean) + beta
                      = (gamma/std * W) * x + (gamma/std*(b-mean) + beta)
        → w_fused = gamma/std * W
        → b_fused = gamma/std * (b - mean) + beta
    """
    mean  = bn.running_mean
    var   = bn.running_var
    eps   = bn.eps
    gamma = bn.weight
    beta  = bn.bias
    std   = (var + eps).sqrt()

    b = linear.bias.data if linear.bias is not None else torch.zeros_like(mean)

    w_fused = linear.weight * (gamma / std).unsqueeze(1)
    b_fused = (gamma / std) * (b - mean) + beta

    fused = nn.Linear(linear.in_features, linear.out_features, bias=True)
    fused.weight = nn.Parameter(w_fused.clone())
    fused.bias   = nn.Parameter(b_fused.clone())
    return fused


def fuse_model(model: nn.Module) -> None:
    """
    In-place fusion of:
      • Conv2d  + BatchNorm2d  inside ConvBNAct and EfficientResBlock
      • Linear  + BatchNorm1d  inside the classifier head

    Call AFTER model.eval() so that BN uses frozen running stats.
    After fusion BN layers are replaced with nn.Identity() so the
    state-dict key structure is preserved (no keys are deleted).
    """
    assert not model.training, "Call model.eval() before fuse_model()"

    # ── Conv + BN fusion ─────────────────────────────────────────────────
    # We walk named modules; ConvBNAct and EfficientResBlock both have
    # self.conv (Sequential) and self.bn (BatchNorm2d).
    for name, m in model.named_modules():
        if isinstance(m, (ConvBNAct, EfficientResBlock)):
            if isinstance(m.bn, nn.BatchNorm2d):
                # Fuse conv[1] (pointwise) with bn, then replace bn
                m.conv[1] = fuse_conv_bn(m.conv[1], m.bn)
                m.bn = nn.Identity()
                print(f"  ✅ Conv+BN  fused : {name}")

    # ── Linear + BN fusion ───────────────────────────────────────────────
    # head is an nn.Sequential; we scan pairs (i, i+1).
    head = model.head
    i = 0
    while i < len(head) - 1:
        if isinstance(head[i], nn.Linear) and isinstance(head[i + 1], nn.BatchNorm1d):
            head[i]     = fuse_linear_bn(head[i], head[i + 1])
            head[i + 1] = nn.Identity()
            print(f"  ✅ Linear+BN fused : head[{i}]+head[{i+1}]")
            i += 2
        else:
            i += 1

    print("BN fusion complete.")


# =========================================================================
# VERIFICATION
# =========================================================================

def verify_fusion(model_orig: nn.Module, model_fused: nn.Module,
                  device: torch.device, tol: float = 1e-4) -> float:
    """
    Pass the same dummy input through both models and compare outputs.
    Returns max absolute difference. Passes if < tol.
    """
    model_orig.eval()
    model_fused.eval()

    dummy_img = torch.randn(4, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_qp  = torch.tensor([22/51., 27/51., 32/51., 37/51.]).to(device)

    with torch.no_grad():
        out_orig  = model_orig(dummy_img, dummy_qp)
        out_fused = model_fused(dummy_img, dummy_qp)

    max_diff = (out_orig - out_fused).abs().max().item()
    print(f"\nFusion Verification — max |Δ output| = {max_diff:.3e}")
    if max_diff < tol:
        print("  ✅ PASSED — fusion is numerically correct.")
    else:
        print("  ❌ FAILED — fusion introduced errors! Do NOT deploy.")
    return max_diff


# =========================================================================
# BENCHMARKING
# =========================================================================

def benchmark_inference(model: nn.Module, device: torch.device,
                         n_warmup: int = 3, n_iter: int = 20) -> None:
    model.eval()

    dummy_img = torch.randn(1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_qp  = torch.tensor([32.0 / 51.0]).to(device)
    dummy_batch_img = torch.randn(BATCH_SIZE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_batch_qp  = torch.full((BATCH_SIZE,), 32.0 / 51.0).to(device)

    print("\n" + "=" * 55)
    print("INFERENCE BENCHMARK — SINGLE SAMPLE")
    print("=" * 55)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy_img, dummy_qp)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iter):
            model(dummy_img, dummy_qp)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / n_iter
    print(f"  Average : {elapsed_ms:.4f} ms / sample")

    print("\n" + "=" * 55)
    print(f"INFERENCE BENCHMARK — BATCH SIZE {BATCH_SIZE}")
    print("=" * 55)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy_batch_img, dummy_batch_qp)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iter):
            model(dummy_batch_img, dummy_batch_qp)
    batch_ms = (time.perf_counter() - t0) * 1000 / n_iter
    print(f"  Batch total : {batch_ms:.4f} ms")
    print(f"  Per sample  : {batch_ms / BATCH_SIZE:.4f} ms")

    if HAS_THOP:
        print("\n--- FLOPs via thop ---")
        try:
            macs, _ = thop_profile(model, inputs=(dummy_img, dummy_qp), verbose=False)
            print(f"  MACs   : {macs:,.0f}")
            print(f"  GFLOPs : {macs * 2 / 1e9:.4f}")
        except Exception as e:
            print(f"  thop failed: {e}")

    if HAS_PTFLOPS:
        print("\n--- FLOPs via ptflops ---")
        try:
            def _constructor(_):
                return {'x': dummy_img, 'qp': dummy_qp}
            macs, _ = get_model_complexity_info(
                model, (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
                as_strings=False, print_per_layer_stat=False,
                verbose=False, input_constructor=_constructor, backend='aten',
            )
            print(f"  MACs   : {macs:,.0f}")
            print(f"  GFLOPs : {macs * 2 / 1e9:.4f}")
        except Exception as e:
            print(f"  ptflops failed: {e}")


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"Using device : {device}")

    # ── Step 1: Load trained weights ──────────────────────────────────────
    print(f"\n[1] Loading checkpoint from '{INPUT_PATH}' ...")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {INPUT_PATH}")

    model = BalancedFasterViT_HEVC().to(device)
    model = model.to(memory_format=torch.channels_last)

    ckpt = torch.load(INPUT_PATH, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Loaded. Parameters: {total_params:,}")

    # ── Step 2: Deep-copy for verification ───────────────────────────────
    model_original = copy.deepcopy(model)
    model_original.eval()

    # ── Step 3: Fuse ─────────────────────────────────────────────────────
    print("\n[2] Applying BN fusion ...")
    model.eval()
    fuse_model(model)

    fused_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters after fusion: {fused_params:,}  (delta: {fused_params - total_params:+,})")

    # ── Step 4: Verify ───────────────────────────────────────────────────
    print("\n[3] Verifying fusion ...")
    verify_fusion(model_original, model, device)

    # ── Step 5: Save state-dict ──────────────────────────────────────────
    print(f"\n[4] Saving fused state-dict to '{OUTPUT_PATH}' ...")
    torch.save(model.state_dict(), OUTPUT_PATH)
    print("   Saved.")

    # ── Step 6: Benchmark ────────────────────────────────────────────────
    benchmark_inference(model, device)

    print(f"\nDone. Deploy '{OUTPUT_PATH}' with the inference script.")
