import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from timm.layers import trunc_normal_ 

# ==========================================
# 1. CONFIGURATION (USER INPUTS)
# ==========================================
# Update these paths to match your actual file locations
YUV_FILE_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/Info&YUV/AI_YUV/IntraValid_4928x3264.yuv'
LABEL_FILE_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/Info&YUV/AI_Info/Info_20170826_151434_AI_IntraValid_4928x3264_qp37_nf25_CUDepth.dat'
MODEL_CHECKPOINT = '/root/myproject/HEVC_Intra_Models-ViT/Fastervit/best_fastervit_hevc_balanced.pth' 

# Parameters
FRAME_IDX = 21       
QP_VALUE = 37       
WIDTH = 4928        
HEIGHT = 3264       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CHANNELS = 1
NUM_CLASSES = 21

# Visualization Settings
LINE_THICKNESS = 2          # Set to 2 as requested
LINE_TYPE = cv2.LINE_4      # Changed to LINE_4 for sharp, pixel-perfect lines
COLOR_GT = (0, 0, 255)      # Red for Ground Truth
COLOR_PRED = (0, 255, 0)    # Green for FasterViT Prediction

# ==========================================
# 2. MODEL DEFINITION (FasterViT)
# ==========================================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
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

class BalancedFasterViT_HEVC(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [8, 16, 24, 32]

        self.stem = ConvBNAct(NUM_CHANNELS, dims[0], 3, 2, 1)
        self.stage1 = nn.Sequential(EfficientResBlock(dims[0]), ConvBNAct(dims[0], dims[1], 3, 2, 1))
        self.stage2 = nn.Sequential(EfficientResBlock(dims[1]), ConvBNAct(dims[1], dims[2], 3, 2, 1))
        self.stage3 = nn.Sequential(EfficientResBlock(dims[2]), ConvBNAct(dims[2], dims[3], 3, 2, 1))

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
            if m.bias is not None: nn.init.zeros_(m.bias)
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
        if qp.dim() == 1: qp = qp.unsqueeze(1)
        return self.head(torch.cat([feat, qp], dim=1))

# ==========================================
# 3. DATA EXTRACTION FUNCTIONS
# ==========================================

def read_specific_yuv_frame(filepath, width, height, frame_idx):
    y_size = width * height
    frame_size = int(width * height * 1.5)
    offset = frame_idx * frame_size
    try:
        with open(filepath, 'rb') as f:
            f.seek(offset)
            y_buf = f.read(y_size)
            if len(y_buf) < y_size: raise ValueError("End of file reached.")
            return np.frombuffer(y_buf, dtype=np.uint8).reshape((height, width))
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        exit()

def read_specific_label_frame(filepath, width, height, frame_idx):
    unit_width = 16
    rows = height // unit_width
    cols = width // unit_width
    bytes_per_frame = rows * cols
    offset = frame_idx * bytes_per_frame
    try:
        with open(filepath, 'rb') as f:
            f.seek(offset)
            buf = f.read(bytes_per_frame)
            if len(buf) < bytes_per_frame: raise ValueError("End of file reached.")
            return np.frombuffer(buf, dtype=np.uint8).reshape((rows, cols))
    except FileNotFoundError:
        print(f"Error: Label file {filepath} not found.")
        exit()

# ==========================================
# 4. DRAWING FUNCTIONS
# ==========================================

def draw_partitions(img_rgb, top_left_y, top_left_x, depth_grid, color, size=64):
    # Using LINE_THICKNESS=2 and LINE_TYPE=cv2.LINE_4 for sharp lines
    cv2.rectangle(img_rgb, (top_left_x, top_left_y), (top_left_x+64, top_left_y+64), color, LINE_THICKNESS, LINE_TYPE)
    max_depth = np.max(depth_grid)
    
    if max_depth >= 1:
        cv2.line(img_rgb, (top_left_x + 32, top_left_y), (top_left_x + 32, top_left_y + 64), color, LINE_THICKNESS, LINE_TYPE)
        cv2.line(img_rgb, (top_left_x, top_left_y + 32), (top_left_x + 64, top_left_y + 32), color, LINE_THICKNESS, LINE_TYPE)
        
        for i in range(2): 
            for j in range(2): 
                sub_grid = depth_grid[i*2:(i+1)*2, j*2:(j+1)*2]
                if np.max(sub_grid) >= 2:
                    y32 = top_left_y + i*32
                    x32 = top_left_x + j*32
                    cv2.line(img_rgb, (x32 + 16, y32), (x32 + 16, y32 + 32), color, LINE_THICKNESS, LINE_TYPE)
                    cv2.line(img_rgb, (x32, y32 + 16), (x32 + 32, y32 + 16), color, LINE_THICKNESS, LINE_TYPE)

                    for m in range(2):
                        for n in range(2):
                            if sub_grid[m, n] >= 3:
                                y16 = y32 + m*16
                                x16 = x32 + n*16
                                cv2.line(img_rgb, (x16 + 8, y16), (x16 + 8, y16 + 16), color, LINE_THICKNESS, LINE_TYPE)
                                cv2.line(img_rgb, (x16, y16 + 8), (x16 + 16, y16 + 8), color, LINE_THICKNESS, LINE_TYPE)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    print(f"1. Loading FasterViT Model from {MODEL_CHECKPOINT}...")
    model = BalancedFasterViT_HEVC().to(DEVICE)
    try:
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("   Weights loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load weights ({e}).")
    model.eval()

    print(f"2. Reading Frame {FRAME_IDX}...")
    Y_frame = read_specific_yuv_frame(YUV_FILE_PATH, WIDTH, HEIGHT, FRAME_IDX)
    
    print(f"3. Reading Labels...")
    label_matrix = read_specific_label_frame(LABEL_FILE_PATH, WIDTH, HEIGHT, FRAME_IDX)

    img_gt = cv2.cvtColor(Y_frame, cv2.COLOR_GRAY2BGR)
    img_pred = cv2.cvtColor(Y_frame, cv2.COLOR_GRAY2BGR)

    print("4. Processing CTUs...")
    rows_ctu = HEIGHT // 64
    cols_ctu = WIDTH // 64
    qp_tensor = torch.tensor([float(QP_VALUE)/51.0], dtype=torch.float32, device=DEVICE).unsqueeze(0) 

    with torch.no_grad():
        for r in range(rows_ctu):
            for c in range(cols_ctu):
                y_pos = r * 64
                x_pos = c * 64
                
                # --- A. GROUND TRUTH (RED) ---
                gt_block = label_matrix[r*4:(r+1)*4, c*4:(c+1)*4]
                draw_partitions(img_gt, y_pos, x_pos, gt_block, color=COLOR_GT)

                # --- B. FASTERVIT PREDICTION (GREEN) ---
                # Preprocessing: [Batch, Channel, Height, Width] -> [1, 1, 64, 64]
                # Normalized to 0-1 range
                ctu = Y_frame[y_pos:y_pos+64, x_pos:x_pos+64]
                ctu_tensor = torch.tensor(ctu, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0) / 255.0
                
                # Run Model
                outputs = model(ctu_tensor, qp_tensor)
                
                # Slice Outputs
                pred_64 = outputs[0, 0]        # Scalar
                pred_32 = outputs[0, 1:5]      # Size 4
                pred_16 = outputs[0, 5:21]     # Size 16
                
                # Logic to fill the grid based on predictions
                pred_grid = np.zeros((4,4), dtype=np.uint8)
                split_64 = pred_64.item() > 0.5
                split_32_arr = (pred_32 > 0.5).cpu().numpy()
                split_16_arr = (pred_16 > 0.5).cpu().numpy()
                
                if split_64:
                    pred_grid[:, :] = 1 
                    idx_32 = 0
                    for i in range(0, 4, 2):
                        for j in range(0, 4, 2):
                            if split_32_arr[idx_32]:
                                pred_grid[i:i+2, j:j+2] = 2 
                                base_row, base_col = i, j
                                sub_indices = [(base_row, base_col), (base_row, base_col+1),
                                               (base_row+1, base_col), (base_row+1, base_col+1)]
                                for (r16, c16) in sub_indices:
                                    # Map to linear index 0-15
                                    linear_idx = r16 * 4 + c16
                                    if split_16_arr[linear_idx]: 
                                        pred_grid[r16, c16] = 3
                            idx_32 += 1
                
                draw_partitions(img_pred, y_pos, x_pos, pred_grid, color=COLOR_PRED)

    print("5. Adding Labels in Header and Saving...")
    header_height = 200
    header_gt = np.ones((header_height, WIDTH, 3), dtype=np.uint8) * 255 
    header_pred = np.ones((header_height, WIDTH, 3), dtype=np.uint8) * 255 

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(header_gt, "Ground Truth", (50, 140), font, 4, COLOR_GT, 8, LINE_TYPE)
    cv2.putText(header_pred, "FasterViT Prediction", (50, 140), font, 4, COLOR_PRED, 8, LINE_TYPE)

    img_gt_full = np.vstack((header_gt, img_gt))
    img_pred_full = np.vstack((header_pred, img_pred))

    combined = np.hstack((img_gt_full, img_pred_full))
    
    # --- NO RESIZING: Saving Full Resolution ---
    output_filename = f"fastervit_partition_vis_frame{FRAME_IDX}_qp{QP_VALUE}_FULLRES.jpg"
    cv2.imwrite(output_filename, combined)
    print(f"Done! Saved full resolution comparison to {output_filename}")

if __name__ == "__main__":
    main()