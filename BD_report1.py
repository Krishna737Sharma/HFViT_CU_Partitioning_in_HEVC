import pandas as pd
import bjontegaard as bd  # pip install bjontegaard
import sys

# --- 1. Data Loading and Parsing ---
filename = '/root/myproject/HEVC_Intra_Models-ViT/Model_performence - Sheet1.csv'

try:
    with open(filename, 'r') as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

parsed_rows = []
current_video = None

# Iterate through the CSV lines to handle the custom structure
for line in lines:
    parts = line.strip().split(',')
    
    # Detect Video Header (e.g., ",,,,,VideoName.yuv,,,,")
    if len(parts) > 5 and parts[5].strip() != '' and parts[0].strip() == '':
        current_video = parts[5].strip()
        continue
        
    # Skip Header rows
    if parts[0].strip() == 'Model':
        continue
        
    # Parse Data Rows
    if parts[0].strip() in ['FasterViT', 'CNN', 'HEVC']:
        try:
            model = parts[0].strip()
            qp = int(parts[1].strip())
            bitrate = float(parts[2].strip())
            psnr = float(parts[3].strip())
            # Index 7 corresponds to "Total time (user+sys) (min)"
            total_time_min = float(parts[7].strip()) 
            ssim = float(parts[8].strip())
            vmaf = float(parts[9].strip())
            
            parsed_rows.append({
                'Video': current_video,
                'Model': model,
                'QP': qp,
                'Bitrate': bitrate,
                'PSNR': psnr,
                'TotalTimeMin': total_time_min,
                'SSIM': ssim,
                'VMAF': vmaf
            })
        except ValueError:
            continue

df = pd.DataFrame(parsed_rows)

# --- 2. Calculate Results ---
videos = df['Video'].unique()
results_cnn = []
results_vit = []

for video in videos:
    df_v = df[df['Video'] == video]
    
    # Get sub-dataframes sorted by QP
    base = df_v[df_v['Model'] == 'HEVC'].sort_values('QP')
    cnn = df_v[df_v['Model'] == 'CNN'].sort_values('QP')
    vit = df_v[df_v['Model'] == 'FasterViT'].sort_values('QP')
    
    # Skip if any model data is missing
    if base.empty or cnn.empty or vit.empty:
        print(f"Skipping {video} due to missing data.")
        continue

    # --- CNN Calculations (vs HEVC) ---
    try:
        bd_psnr_cnn = bd.bd_rate(
            base['Bitrate'].tolist(), base['PSNR'].tolist(),
            cnn['Bitrate'].tolist(), cnn['PSNR'].tolist(),
            method='akima'
        )
        bd_vmaf_cnn = bd.bd_rate(
            base['Bitrate'].tolist(), base['VMAF'].tolist(),
            cnn['Bitrate'].tolist(), cnn['VMAF'].tolist(),
            method='akima'
        )
        bd_ssim_cnn = bd.bd_rate(
            base['Bitrate'].tolist(), base['SSIM'].tolist(),
            cnn['Bitrate'].tolist(), cnn['SSIM'].tolist(),
            method='akima'
        )
    except Exception as e:
        print(f"Error calculating BD-rate for CNN on {video}: {e}")
        bd_psnr_cnn = bd_vmaf_cnn = bd_ssim_cnn = None

    # CNN Speedup: (T_hevc - T_cnn) / T_hevc * 100
    t_hevc = base['TotalTimeMin'].sum()
    t_cnn = cnn['TotalTimeMin'].sum()
    speedup_cnn = (t_hevc - t_cnn) / t_hevc * 100
    
    results_cnn.append({
        'video name': video,
        'BD-rate PSNR': bd_psnr_cnn,
        'BD-Rate VMAF': bd_vmaf_cnn,
        'BD-Rate MS-SSIM': bd_ssim_cnn,
        'speed up': speedup_cnn
    })
    
    # --- FasterViT Calculations (vs HEVC) ---
    try:
        bd_psnr_vit = bd.bd_rate(
            base['Bitrate'].tolist(), base['PSNR'].tolist(),
            vit['Bitrate'].tolist(), vit['PSNR'].tolist(),
            method='akima'
        )
        bd_vmaf_vit = bd.bd_rate(
            base['Bitrate'].tolist(), base['VMAF'].tolist(),
            vit['Bitrate'].tolist(), vit['VMAF'].tolist(),
            method='akima'
        )
        bd_ssim_vit = bd.bd_rate(
            base['Bitrate'].tolist(), base['SSIM'].tolist(),
            vit['Bitrate'].tolist(), vit['SSIM'].tolist(),
            method='akima'
        )
    except Exception as e:
        print(f"Error calculating BD-rate for FasterViT on {video}: {e}")
        bd_psnr_vit = bd_vmaf_vit = bd_ssim_vit = None
    
    # FasterViT Speedup wrt CNN: (T_cnn - T_fastervit) / T_cnn * 100
    t_vit = vit['TotalTimeMin'].sum()
    speedup_vit = (t_cnn - t_vit) / t_cnn * 100
    
    # FasterViT Speedup wrt HEVC: (T_hevc - T_fastervit) / T_hevc * 100
    speedup_vit_wrt_hevc = (t_hevc - t_vit) / t_hevc * 100
    
    results_vit.append({
        'video name': video,
        'BD-rate PSNR': bd_psnr_vit,
        'BD-Rate VMAF': bd_vmaf_vit,
        'BD-Rate MS-SSIM': bd_ssim_vit,
        'speed up wrt CNN': speedup_vit,
        'speed up wrt HEVC': speedup_vit_wrt_hevc
    })

# --- 3. Create Tables and Add Average Row ---
df_cnn_results = pd.DataFrame(results_cnn)
df_vit_results = pd.DataFrame(results_vit)

def add_average_row(df, label_col='video name'):
    if df.empty:
        return df
    # Calculate mean of numeric columns only
    means = df.mean(numeric_only=True)
    # Create the new row as a dictionary
    row = means.to_dict()
    row[label_col] = 'Average'
    # Append to the DataFrame
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

# Add averages
df_cnn_results = add_average_row(df_cnn_results)
df_vit_results = add_average_row(df_vit_results)

# --- 4. Print and Save ---
pd.options.display.float_format = '{:.2f}'.format

print("\n=== Table 1: CNN Results ===")
print(df_cnn_results.to_string(index=False))

print("\n=== Table 2: FasterViT Results ===")
print(df_vit_results.to_string(index=False))

# Optional: Save to CSV
df_cnn_results.to_csv('CNN_BD_Rate_Results.csv', index=False)
df_vit_results.to_csv('FasterViT_BD_Rate_Results.csv', index=False)
print("\nResults saved to 'CNN_BD_Rate_Results.csv' and 'FasterViT_BD_Rate_Results.csv'")
