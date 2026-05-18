import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from torch.utils.data import Dataset
from timm.layers import trunc_normal_

torch.set_num_threads(1)

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

try:
    from torchinfo import summary as torchinfo_summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False

# ==================== Constants ====================
IMAGE_SIZE     = 64
NUM_CHANNELS   = 1
NUM_CLASSES    = 21
SELECT_QP_LIST = [22, 27, 32, 37]
BATCH_SIZE     = 64

# ==================== FasterViT Components ====================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
        )
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class EfficientResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
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

        self.norm1 = nn.LayerNorm(dim)          # Pre-Norm for attention sub-block
        self.qkv   = nn.Linear(dim, dim * 3, bias=False)
        self.proj  = nn.Linear(dim, dim,     bias=False)
        self.norm2 = nn.LayerNorm(dim)          # Pre-Norm for MLP sub-block
        hidden     = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        # ── Positional encoding ──────────────────────────────────────────────
        # seq_len = window_size^2 window tokens + 1 carrier token
        # window_size=2 → 4 + 1 = 5 tokens per window
        seq_len = window_size ** 2 + 1

        # Relative positional bias table: shape [num_heads, seq_len, seq_len]
        # Entry (h, i, j) = bias added to attn[h, i, j] before softmax.
        # Tells the model "how much should token-i attend to token-j"
        # based purely on their positions, independently of their content.
        # Added to attention LOGITS (Q@K scores), not to token features.
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(num_heads, seq_len, seq_len)
        )
        trunc_normal_(self.rel_pos_bias, std=0.02)

        # ── Absolute position embedding for window tokens ────────────────────
        # Fixed sinusoidal 2D embedding: maps each token's (row, col) position
        # inside the window to a dim-vector, added to token features BEFORE
        # attention. Carrier token gets zero embedding (no fixed spatial pos).
        # Registered as buffer: not trained, but moves to correct device.
        self.register_buffer('window_pos_emb', self._build_sinusoidal(window_size, dim))

    @staticmethod
    def _build_sinusoidal(window_size, dim):
        """
        2D sinusoidal position embedding for window_size^2 spatial positions.
        Returns shape [1, window_size^2, dim].

        Row coords use sin, col coords use cos, each projected to dim//2
        frequencies. The carrier token always gets zeros (added separately
        in forward via torch.zeros padding).
        """
        coords = []
        for row in range(window_size):
            for col in range(window_size):
                r = (row / max(window_size - 1, 1)) * 2 - 1   # normalize to [-1, +1]
                c = (col / max(window_size - 1, 1)) * 2 - 1
                coords.append([r, c])
        coords = torch.tensor(coords, dtype=torch.float32)     # [ws^2, 2]

        half = dim // 2
        freq = torch.pow(10000.0, -2.0 * torch.arange(half, dtype=torch.float32) / dim)

        row_enc = torch.sin(coords[:, 0:1] * freq)             # [ws^2, half]
        col_enc = torch.cos(coords[:, 1:2] * freq)             # [ws^2, half]
        pos_emb = torch.cat([row_enc, col_enc], dim=-1)        # [ws^2, dim]
        return pos_emb.unsqueeze(0)                            # [1, ws^2, dim]

    @staticmethod
    def _window_partition(x, window_size):
        """
        x: [B, H, W, C]
        Returns: [B * (H//ws) * (W//ws), ws*ws, C]
        """
        B, H, W, C = x.shape
        ws = window_size
        x = x.reshape(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(-1, ws * ws, C)
        return x

    @staticmethod
    def _window_reverse(windows, window_size, H, W, B):
        """
        windows: [B * num_windows, ws*ws, C]
        Returns: [B, H, W, C]
        """
        ws = window_size
        C = windows.shape[-1]
        num_h, num_w = H // ws, W // ws
        x = windows.reshape(B, num_h, num_w, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, H, W, C)
        return x

    def forward(self, x, ct):
        B, C, H, W = x.shape

        # ── window partitioning ──────────────────────
        x_nhwc = x.permute(0, 2, 3, 1)                        # [B, H, W, C]
        x_win  = self._window_partition(x_nhwc, self.window_size)
        # x_win: [B*num_windows, ws*ws, C]

        # ── Add absolute position embedding ──────────────────
        x_win = x_win + self.window_pos_emb

        ct = ct.reshape(-1, 1, C)
        tokens   = torch.cat([x_win, ct], dim=1)
        shortcut = tokens

        # ── Attention ────────────────────────────────────────
        qkv = (
            self.qkv(self.norm1(tokens))
            .reshape(tokens.size(0), tokens.size(1),
                     3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_pos_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)

        out    = (attn @ v).transpose(1, 2).reshape(tokens.shape)
        tokens = shortcut + self.proj(out)
        tokens = tokens + self.mlp(self.norm2(tokens))

        x_tokens = tokens[:, :-1, :]                           # [B*nW, ws*ws, C]
        x = self._window_reverse(x_tokens, self.window_size,
                                  H, W, B)                     # [B, H, W, C]
        x  = x.permute(0, 3, 1, 2)                            # [B, C, H, W]
        ct = tokens[:, -1:, :]
        return x.contiguous(), ct


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
        out  = (attn @ v).transpose(1, 2).reshape(batch_size, num_windows_per_image, dim)

        ct_grouped = shortcut + self.ct_proj(out)
        ct = ct_grouped.reshape(num_windows_total, 1, dim)
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

        self.head = nn.Sequential(
            nn.Linear(dims[3] + 1, 1024),  
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(1024, 1534),
            nn.BatchNorm1d(1534),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(1534, NUM_CLASSES),
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

        ct     = F.adaptive_avg_pool2d(x, (2, 2)).flatten(2).transpose(1, 2)
        x, ct  = self.hat1(x, ct)
        ct     = self.ct_interaction(ct, B)
        x, _   = self.hat2(x, ct)

        feat = self.gap(x).flatten(1)
        if qp.dim() == 1:
            qp = qp.unsqueeze(1)
        return self.head(torch.cat([feat, qp], dim=1))


# ==================== BN Fusion ====================

def fuse_conv_bn(conv, bn):
    """Fuse Conv2d (bias=False) + BatchNorm2d → new Conv2d with bias."""
    std     = torch.sqrt(bn.running_var + bn.eps)
    w_fused = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
    b_fused = bn.bias - bn.weight * bn.running_mean / std

    fused        = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        conv.kernel_size, conv.stride, conv.padding,
        groups=conv.groups, bias=True
    )
    fused.weight = nn.Parameter(w_fused.clone())
    fused.bias   = nn.Parameter(b_fused.clone())
    return fused


def fuse_linear_bn(linear, bn):
    """Fuse Linear + BatchNorm1d → new Linear with bias."""
    std     = torch.sqrt(bn.running_var + bn.eps)
    b       = linear.bias.data if linear.bias is not None else torch.zeros_like(bn.running_mean)
    w_fused = linear.weight * (bn.weight / std).unsqueeze(1)
    b_fused = (b - bn.running_mean) / std * bn.weight + bn.bias

    fused        = nn.Linear(linear.in_features, linear.out_features, bias=True)
    fused.weight = nn.Parameter(w_fused.clone())
    fused.bias   = nn.Parameter(b_fused.clone())
    return fused


def fuse_model(model):
    """
    Fuse Conv+BN and Linear+BN pairs.
    MUST be called after model.eval() so BN running stats are stable.
    """
    for name, m in model.named_modules():
        if isinstance(m, (ConvBNAct, EfficientResBlock)):
            if isinstance(m.bn, nn.BatchNorm2d):
                m.conv[1] = fuse_conv_bn(m.conv[1], m.bn)
                m.bn      = nn.Identity()

    head = model.head
    i = 0
    while i < len(head) - 1:
        if isinstance(head[i], nn.Linear) and isinstance(head[i + 1], nn.BatchNorm1d):
            head[i]     = fuse_linear_bn(head[i], head[i + 1])
            head[i + 1] = nn.Identity()
            i += 2
        else:
            i += 1


# ==================== MAIN ====================

if __name__ == "__main__":

    device = torch.device("cpu")
    print(f"Using device: {device}")
    print("INFO: Timing on randomly initialized (untrained) model.")

    model = BalancedFasterViT_HEVC().to(device)
    model = model.to(memory_format=torch.channels_last)

    dummy_image     = torch.randn(1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_qp        = torch.tensor([32.0 / 51.0]).to(device)
    dummy_batch_img = torch.randn(BATCH_SIZE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_batch_qp  = torch.tensor([32.0 / 51.0] * BATCH_SIZE).to(device)

    print("\n" + "=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    if HAS_TORCHINFO:
        torchinfo_summary(
            model,
            input_data=[dummy_batch_img, dummy_batch_qp],
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=4,
            verbose=1,
        )
    else:
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters    : {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

    model.eval()
    fuse_model(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters (after fusion): {total_params:,}")

    print("\n" + "=" * 50)
    print("INFERENCE TIMING — SINGLE SAMPLE")
    print("=" * 50)

    with torch.no_grad():
        _ = model(dummy_image, dummy_qp)

    start_cpu  = os.times()
    start_real = time.time()
    with torch.no_grad():
        output = model(dummy_image, dummy_qp)
    end_real = time.time()
    end_cpu  = os.times()

    real_ms = (end_real - start_real) * 1000
    cpu_ms  = (end_cpu.user - start_cpu.user + end_cpu.system - start_cpu.system) * 1000
    print(f"Real Time : {real_ms:.4f} ms")
    print(f"CPU  Time : {cpu_ms:.4f} ms")

    print("\n" + "=" * 50)
    print(f"INFERENCE TIMING — BATCH SIZE {BATCH_SIZE}")
    print("=" * 50)

    with torch.no_grad():
        _ = model(dummy_batch_img, dummy_batch_qp)

    N_ITER = 10
    start_cpu_b  = os.times()
    start_real_b = time.time()
    with torch.no_grad():
        for _ in range(N_ITER):
            _ = model(dummy_batch_img, dummy_batch_qp)
    end_real_b = time.time()
    end_cpu_b  = os.times()

    batch_real_ms = (end_real_b - start_real_b) * 1000 / N_ITER
    batch_cpu_ms  = (end_cpu_b.user - start_cpu_b.user + end_cpu_b.system - start_cpu_b.system) * 1000 / N_ITER

    print(f"Avg Batch Real Time  : {batch_real_ms:.4f} ms")
    print(f"Avg Batch CPU  Time  : {batch_cpu_ms:.4f} ms")
    print(f"Per-Sample Real Time : {batch_real_ms / BATCH_SIZE:.4f} ms/sample")
    print(f"Per-Sample CPU  Time : {batch_cpu_ms  / BATCH_SIZE:.4f} ms/sample")

    if HAS_THOP:
        print("\n--- FLOPs (thop) ---")
        try:
            macs, params = profile(model, inputs=(dummy_image, dummy_qp), verbose=False)
            print(f"Parameters : {params:,}")
            print(f"MACs       : {macs:,}")
            print(f"GFLOPs     : {(macs * 2) / 1e9:.4f} G")
        except Exception as e:
            print(f"thop failed: {e}")
    else:
        print("\nthop not found — skipping. Install with: pip install thop")

    if HAS_PTFLOPS:
        print("\n--- FLOPs (ptflops) ---")
        try:
            def input_constructor(input_res):
                return {"x": dummy_image, "qp": dummy_qp}

            macs, params = get_model_complexity_info(
                model, (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
                as_strings=False, print_per_layer_stat=False,
                verbose=False, input_constructor=input_constructor, backend="aten",
            )
            print(f"Parameters : {params:,}")
            print(f"MACs       : {macs:,}")
            print(f"GFLOPs     : {(macs * 2) / 1e9:.4f} G")
        except Exception as e:
            print(f"ptflops failed: {e}")
    else:
        print("\nptflops not found — skipping. Install with: pip install ptflops")
