import pandas as pd
import matplotlib.pyplot as plt
import sys

# --- 1. Data Loading and Parsing ---
# Replace with your actual filename
filename = '/root/myproject/HEVC_Intra_Models-ViT/Model_performence - Sheet1 (1).csv'

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
    # Video name is usually at index 5 in your sheet structure
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
            
            parsed_rows.append({
                'Video': current_video,
                'Model': model,
                'QP': qp,
                'Bitrate': bitrate,
                'PSNR': psnr
            })
        except ValueError:
            continue

df = pd.DataFrame(parsed_rows)

# --- 2. Plotting RD Curves ---
videos = df['Video'].unique()

# Define styles for consistency
styles = {
    'HEVC':      {'color': 'black', 'marker': 'o', 'linestyle': '-'},
    'CNN':       {'color': 'blue',  'marker': 's', 'linestyle': '--'},
    'FasterViT': {'color': 'red',   'marker': '^', 'linestyle': '-.'}
}

for video in videos:
    df_v = df[df['Video'] == video]
    
    # Create a new figure for each video
    plt.figure(figsize=(10, 6))
    
    # Plot each model
    for model in ['HEVC', 'CNN', 'FasterViT']:
        # Sort by Bitrate to ensure the line connects points correctly
        subset = df_v[df_v['Model'] == model].sort_values('Bitrate')
        
        if not subset.empty:
            plt.plot(
                subset['Bitrate'], 
                subset['PSNR'], 
                label=model, 
                color=styles[model]['color'], 
                marker=styles[model]['marker'], 
                linestyle=styles[model]['linestyle'],
                linewidth=2,
                markersize=6
            )
    
    # Graph formatting
    plt.title(f'RD Curve: {video}', fontsize=14)
    plt.xlabel('Bitrate (kbps)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=11)
    
    # Save the plot
    safe_name = "".join([c if c.isalnum() else "_" for c in video])
    output_filename = f'RD_Curve_{safe_name}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_filename}")
    
    # Close figure to free memory
    plt.close()

print("\nAll RD curves have been plotted and saved.")