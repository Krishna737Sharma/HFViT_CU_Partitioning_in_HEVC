import os
import glob
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for nohup/background runs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from siti_tools.siti import SiTiCalculator

# ============================================================
# CONFIGURATION
# ============================================================
VIDEO_DIR = "/root/myproject/HEVC_Intra_Models-ViT/Test_y4m_videos"
OUTPUT_CSV = "./results/siti_results.csv"
OUTPUT_PLOT = "./results/siti_plot.png"
VIDEO_EXTENSIONS = ["*.y4m"]

# ============================================================
# COLLECT ALL VIDEO FILES
# ============================================================
video_files = []
for ext in VIDEO_EXTENSIONS:
    video_files.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))

video_files.sort()

if not video_files:
    print("ERROR: No video files found in", VIDEO_DIR)
    exit(1)

print(f"Found {len(video_files)} video file(s):\n")
for f in video_files:
    print(f"  - {os.path.basename(f)}")
print()

# ============================================================
# LOAD ALREADY PROCESSED RESULTS (SKIP RE-PROCESSING)
# ============================================================
already_done = {}
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split(",")
            if len(parts) == 4:
                already_done[parts[0]] = {
                    "name": parts[0],
                    "SI": float(parts[1]),
                    "TI": float(parts[2]),
                    "frames": int(parts[3]),
                }
    print(f"Found {len(already_done)} already processed videos in CSV (will skip them)\n")

# ============================================================
# CALCULATE SI AND TI FOR EACH VIDEO
# ============================================================
results = list(already_done.values())
total_start = time.time()

for idx, video_path in enumerate(video_files):
    video_name = os.path.basename(video_path)

    # Skip if already processed
    if video_name in already_done:
        print(f"[{idx+1}/{len(video_files)}] SKIPPING (already done): {video_name}")
        continue

    print(f"[{idx+1}/{len(video_files)}] Processing: {video_name} ... ", flush=True)

    try:
        start = time.time()

        # Detect if 10-bit video
        is_10bit = "10bit" in video_name.lower()

        if is_10bit:
            calc = SiTiCalculator(
                color_range='full',
                bit_depth=10,
                legacy=True
            )
        else:
            calc = SiTiCalculator(
                color_range='full',
                bit_depth=8,
                legacy=True
            )

        si_values, ti_values, frame_count = calc.calculate(video_path)

        si_max = max(si_values)
        ti_max = max(ti_values) if ti_values else 0.0

        elapsed = time.time() - start

        results.append({
            "name": video_name,
            "SI": si_max,
            "TI": ti_max,
            "frames": frame_count,
        })

        print(f"  -> SI={si_max:.2f}, TI={ti_max:.2f}, Frames={frame_count}, Time={elapsed:.1f}s")

        # Save after each video (progress not lost if interrupted)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        with open(OUTPUT_CSV, "w") as f:
            f.write("Video,SI,TI,Frames\n")
            for r in results:
                f.write(f"{r['name']},{r['SI']:.4f},{r['TI']:.4f},{r['frames']}\n")
        print(f"  -> Saved to CSV ({len(results)} total)", flush=True)

    except Exception as e:
        print(f"  -> FAILED! Error: {e}")
        import traceback
        traceback.print_exc()

total_elapsed = time.time() - total_start
print(f"\nSuccessfully processed {len(results)} / {len(video_files)} videos in {total_elapsed:.1f}s")

# ============================================================
# SAVE FINAL RESULTS TO CSV
# ============================================================
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

with open(OUTPUT_CSV, "w") as f:
    f.write("Video,SI,TI,Frames\n")
    for r in results:
        f.write(f"{r['name']},{r['SI']:.4f},{r['TI']:.4f},{r['frames']}\n")

print(f"\nResults saved to: {OUTPUT_CSV}")

# ============================================================
# PRINT RESULTS TABLE
# ============================================================
print(f"\n{'='*85}")
print(f"{'Video':<55} {'SI':>8} {'TI':>8} {'Frames':>8}")
print(f"{'='*85}")
for r in results:
    print(f"{r['name']:<55} {r['SI']:>8.2f} {r['TI']:>8.2f} {r['frames']:>8}")
print(f"{'='*85}")

# ============================================================
# PLOT 2D GRAPH: SI (x-axis) vs TI (y-axis)
# ============================================================
if len(results) == 0:
    print("No results to plot!")
    exit(1)

fig, ax = plt.subplots(figsize=(14, 10))

si_list = [r["SI"] for r in results]
ti_list = [r["TI"] for r in results]
names = [r["name"] for r in results]

# Color 8-bit and 10-bit differently
colors = []
for name in names:
    if "10bit" in name.lower():
        colors.append('red')
    else:
        colors.append('steelblue')

scatter = ax.scatter(si_list, ti_list,
                     c=colors, s=150,
                     edgecolors='black', linewidths=0.8,
                     zorder=5, alpha=0.85)

# Annotate each point
for i, name in enumerate(names):
    label = name.replace('.y4m', '').split('_')[0]
    if 'Netflix' in name:
        parts = name.replace('.y4m', '').split('_')
        label = parts[0] + '_' + parts[1] if len(parts) > 1 else parts[0]

    ax.annotate(label,
                (si_list[i], ti_list[i]),
                textcoords="offset points",
                xytext=(10, 8), fontsize=7,
                ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2',
                         facecolor='lightyellow',
                         alpha=0.7, edgecolor='gray'))

ax.set_xlabel("Spatial Information (SI)", fontsize=14, fontweight='bold')
ax.set_ylabel("Temporal Information (TI)", fontsize=14, fontweight='bold')
ax.set_title("SI vs TI of Test Video Sequences (ITU-T P.910)",
             fontsize=16, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.4)

# Legend for 8-bit vs 10-bit
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
           markersize=12, label='8-bit', markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=12, label='10-bit', markeredgecolor='black'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

x_pad = max(si_list) * 0.15
y_pad = max(ti_list) * 0.15
ax.set_xlim(0, max(si_list) + x_pad)
ax.set_ylim(0, max(ti_list) + y_pad)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {OUTPUT_PLOT}")

print("\n========== ALL DONE! ==========")