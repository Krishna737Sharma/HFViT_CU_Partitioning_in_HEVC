import numpy as np
import scipy.stats as stats
import pandas as pd

# ─────────────────────────────────────────────
# Load CSVs (exclude the "Average" summary row)
# ─────────────────────────────────────────────
cnn_df       = pd.read_csv("/root/myproject/HEVC_Intra_Models-ViT/CNN_BD_Rate_Results.csv")
hfcn_df      = pd.read_csv("/root/myproject/HEVC_Intra_Models-ViT/HFCN_BD_Rate_Results.csv")
fastervit_df = pd.read_csv("/root/myproject/HEVC_Intra_Models-ViT/FasterViT_BD_Rate_Results.csv")

# Drop the "Average" row
cnn_df       = cnn_df[~cnn_df["video name"].str.lower().str.startswith("average")].copy()
hfcn_df      = hfcn_df[~hfcn_df["video name"].str.lower().str.startswith("average")].copy()
fastervit_df = fastervit_df[~fastervit_df["video name"].str.lower().str.startswith("average")].copy()

CONFIDENCE_LEVEL = 0.95

def confidence_interval(data: np.ndarray, cl: float = 0.95):
    """Return (mean, lower, upper) using scipy.stats.t.interval."""
    n    = len(data)
    mean = np.mean(data)
    se   = np.std(data, ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(cl, df=n - 1, loc=mean, scale=se)
    return mean, lo, hi

def report(label: str, df: pd.DataFrame, cols: list[str]):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Confidence Level : {CONFIDENCE_LEVEL*100:.0f}%  |  n = {len(df)} videos")
    print(f"  {'Metric':<30}  {'Mean':>8}  {'Lower CI':>10}  {'Upper CI':>10}")
    print(f"  {'-'*62}")
    for col in cols:
        data = df[col].values.astype(float)
        mean, lo, hi = confidence_interval(data, CONFIDENCE_LEVEL)
        print(f"  {col:<30}  {mean:>8.4f}  {lo:>10.4f}  {hi:>10.4f}")

# ─────────────────────────────────────────────
# CNN
# ─────────────────────────────────────────────
report(
    "Table 1 – CNN Results (vs HM Encoder)",
    cnn_df,
    ["BD-rate PSNR", "BD-Rate VMAF", "BD-Rate MS-SSIM", "speed up wrt HEVC"],
)

# ─────────────────────────────────────────────
# HFCN
# ─────────────────────────────────────────────
report(
    "Table 2 – HFCN Results (vs HM Encoder)",
    hfcn_df,
    ["BD-rate PSNR", "BD-Rate VMAF", "BD-Rate MS-SSIM", "speed up wrt HEVC"],
)

# ─────────────────────────────────────────────
# FasterViT
# ─────────────────────────────────────────────
report(
    "Table 3 – FasterViT Results (vs HM Encoder)",
    fastervit_df,
    ["BD-rate PSNR", "BD-Rate VMAF", "BD-Rate MS-SSIM", "speed up wrt HEVC"],
)

# ─────────────────────────────────────────────
# Combined summary table (all models, HEVC speedup only)
# ─────────────────────────────────────────────
print(f"\n\n{'='*65}")
print("  Summary – Speed-up wrt HEVC across all models")
print(f"{'='*65}")
print(f"  Confidence Level : {CONFIDENCE_LEVEL*100:.0f}%  |  n = 10 videos each")
print(f"  {'Model':<15}  {'Mean':>8}  {'Lower CI':>10}  {'Upper CI':>10}")
print(f"  {'-'*48}")

for label, df in [("CNN", cnn_df), ("HFCN", hfcn_df), ("FasterViT", fastervit_df)]:
    data = df["speed up wrt HEVC"].values.astype(float)
    mean, lo, hi = confidence_interval(data, CONFIDENCE_LEVEL)
    print(f"  {label:<15}  {mean:>8.4f}  {lo:>10.4f}  {hi:>10.4f}")

print()