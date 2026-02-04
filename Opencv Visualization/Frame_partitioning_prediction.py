import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION (USER INPUTS)
# ==========================================
YUV_FILE_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/Info&YUV/AI_YUV/IntraValid_4928x3264.yuv'
LABEL_FILE_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/Info&YUV/AI_Info/Info_20170826_151434_AI_IntraValid_4928x3264_qp37_nf25_CUDepth.dat'
MODEL_CHECKPOINT = '/root/myproject/HEVC_Intra_Models-ViT/Eth_CNN_Pt_full/best_model_4qp_parallel_data_processing_loss_mod.pth'

# Parameters
FRAME_IDX = 0       
QP_VALUE = 37       
WIDTH = 4928        
HEIGHT = 3264       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Visualization Settings
LINE_THICKNESS = 2          # Set to 2 as requested
LINE_TYPE = cv2.LINE_4      # Changed to LINE_4 for sharp, pixel-perfect lines
COLOR_GT = (0, 0, 255)      # Red for Ground Truth
COLOR_PRED = (0, 255, 0)    # Green for CNN Prediction
COLOR_WHITE = (255, 255, 255) # Background for labels

# ==========================================
# 2. MODEL DEFINITION
# ==========================================

def mean_downsample(tensor, scale_factor):
    if tensor.dim() != 3:
         raise ValueError(f"Input tensor must be 3D (batch, height, width). Got {tensor.shape}")
    
    batch_size, h, w = tensor.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    downsampled_tensor = tensor.unfold(1, scale_factor, scale_factor).unfold(2, scale_factor, scale_factor)
    downsampled_tensor = downsampled_tensor.contiguous().view(batch_size, new_h, new_w, -1)
    downsampled_tensor = downsampled_tensor.mean(dim=-1)
    return downsampled_tensor

def norm_batch_ctu(ctu_batch):
    norm_ctu_data_b1 = ctu_batch.clone()
    norm_ctu_data_b2 = ctu_batch.clone()
    norm_ctu_data_b3 = ctu_batch.clone()

    mean_val = torch.mean(ctu_batch, dim=(1, 2), keepdim=True)
    norm_ctu_data_b1 -= mean_val

    for r in range(0, 64, 32):
        for c in range(0, 64, 32):
            m = torch.mean(ctu_batch[:, r:r+32, c:c+32], dim=(1,2), keepdim=True)
            norm_ctu_data_b2[:, r:r+32, c:c+32] -= m

    for r in range(0, 64, 16):
        for c in range(0, 64, 16):
            m = torch.mean(ctu_batch[:, r:r+16, c:c+16], dim=(1,2), keepdim=True)
            norm_ctu_data_b3[:, r:r+16, c:c+16] -= m

    return norm_ctu_data_b1, norm_ctu_data_b2, norm_ctu_data_b3

def downsample_ctu_3_branches(norm_ctu_tuple):
    b1 = mean_downsample(norm_ctu_tuple[0], 4) 
    b2 = mean_downsample(norm_ctu_tuple[1], 2) 
    b3 = mean_downsample(norm_ctu_tuple[2], 1) 
    return (b1, b2, b3)

class ETH_CNN(nn.Module):
    def __init__(self):
        super(ETH_CNN, self).__init__()
        # Branch 1
        self.conv1_b1 = nn.Conv2d(1, 16, 4, 4, 0)
        self.conv2_b1 = nn.Conv2d(16, 24, 2, 2, 0)
        self.conv3_b1 = nn.Conv2d(24, 32, 2, 2, 0)
        # Branch 2
        self.conv1_b2 = nn.Conv2d(1, 16, 4, 4, 0)
        self.conv2_b2 = nn.Conv2d(16, 24, 2, 2, 0)
        self.conv3_b2 = nn.Conv2d(24, 32, 2, 2, 0)
        # Branch 3
        self.conv1_b3 = nn.Conv2d(1, 16, 4, 4, 0)
        self.conv2_b3 = nn.Conv2d(16, 24, 2, 2, 0)
        self.conv3_b3 = nn.Conv2d(24, 32, 2, 2, 0)

        self.fc1_dropout = nn.Dropout(p=0.5)
        self.fc2_dropout = nn.Dropout(p=0.2)

        # FC Layers
        self.fc1_b1 = nn.Linear(2688, 64); self.fc2_b1 = nn.Linear(65, 48); self.fc3_b1 = nn.Linear(49, 1)
        self.fc1_b2 = nn.Linear(2688, 128); self.fc2_b2 = nn.Linear(129, 96); self.fc3_b2 = nn.Linear(97, 4)
        self.fc1_b3 = nn.Linear(2688, 256); self.fc2_b3 = nn.Linear(257, 192); self.fc3_b3 = nn.Linear(193, 16)

    def full_connect_b1(self, x, qp):
        x = self.fc1_dropout(F.leaky_relu(self.fc1_b1(x)))
        x = torch.cat((x, qp), dim=1)
        x = self.fc2_dropout(F.leaky_relu(self.fc2_b1(x)))
        x = torch.cat((x, qp), dim=1)
        return torch.sigmoid(self.fc3_b1(x))

    def full_connect_b2(self, x, qp):
        x = self.fc1_dropout(F.leaky_relu(self.fc1_b2(x)))
        x = torch.cat((x, qp), dim=1)
        x = self.fc2_dropout(F.leaky_relu(self.fc2_b2(x)))
        x = torch.cat((x, qp), dim=1)
        return torch.sigmoid(self.fc3_b2(x))

    def full_connect_b3(self, x, qp):
        x = self.fc1_dropout(F.leaky_relu(self.fc1_b3(x)))
        x = torch.cat((x, qp), dim=1)
        x = self.fc2_dropout(F.leaky_relu(self.fc2_b3(x)))
        x = torch.cat((x, qp), dim=1)
        return torch.sigmoid(self.fc3_b3(x))

    def forward(self, x_in):
        qp = x_in[0]
        original_ctu = x_in[1]
        x = norm_batch_ctu(original_ctu)
        x = downsample_ctu_3_branches(x)

        c1_b1 = F.leaky_relu(self.conv1_b1(x[0].unsqueeze(1))); c2_b1 = F.leaky_relu(self.conv2_b1(c1_b1)); c3_b1 = F.leaky_relu(self.conv3_b1(c2_b1))
        c1_b2 = F.leaky_relu(self.conv1_b2(x[1].unsqueeze(1))); c2_b2 = F.leaky_relu(self.conv2_b2(c1_b2)); c3_b2 = F.leaky_relu(self.conv3_b2(c2_b2))
        c1_b3 = F.leaky_relu(self.conv1_b3(x[2].unsqueeze(1))); c2_b3 = F.leaky_relu(self.conv2_b3(c1_b3)); c3_b3 = F.leaky_relu(self.conv3_b3(c2_b3))

        flat_c3_b1 = c3_b1.view(-1, 32 * 1 * 1); flat_c2_b1 = c2_b1.view(-1, 24 * 2 * 2)
        flat_c3_b2 = c3_b2.view(-1, 32 * 2 * 2); flat_c2_b2 = c2_b2.view(-1, 24 * 4 * 4)
        flat_c3_b3 = c3_b3.view(-1, 32 * 4 * 4); flat_c2_b3 = c2_b3.view(-1, 24 * 8 * 8)

        concat_op = torch.cat((flat_c3_b1, flat_c2_b1, flat_c3_b2, flat_c2_b2, flat_c3_b3, flat_c2_b3), dim=1)
        return self.full_connect_b1(concat_op, qp), self.full_connect_b2(concat_op, qp), self.full_connect_b3(concat_op, qp)

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
    """
    Draws partitions using the specified color.
    """
    # Draw the 64x64 boundary
    cv2.rectangle(img_rgb, (top_left_x, top_left_y), (top_left_x+64, top_left_y+64), color, LINE_THICKNESS, LINE_TYPE)

    max_depth = np.max(depth_grid)
    
    if max_depth >= 1:
        # Split into 4 32x32 blocks
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
                            val = sub_grid[m, n]
                            if val >= 3:
                                y16 = y32 + m*16
                                x16 = x32 + n*16
                                cv2.line(img_rgb, (x16 + 8, y16), (x16 + 8, y16 + 16), color, LINE_THICKNESS, LINE_TYPE)
                                cv2.line(img_rgb, (x16, y16 + 8), (x16 + 16, y16 + 8), color, LINE_THICKNESS, LINE_TYPE)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    print(f"1. Loading Model from {MODEL_CHECKPOINT}...")
    model = ETH_CNN().to(DEVICE)
    try:
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Warning: Could not load weights exactly ({e}).")
    model.eval()

    print(f"2. Reading Frame {FRAME_IDX}...")
    Y_frame = read_specific_yuv_frame(YUV_FILE_PATH, WIDTH, HEIGHT, FRAME_IDX)
    
    print(f"3. Reading Labels...")
    label_matrix = read_specific_label_frame(LABEL_FILE_PATH, WIDTH, HEIGHT, FRAME_IDX)

    # Prepare Visualization Images
    img_gt = cv2.cvtColor(Y_frame, cv2.COLOR_GRAY2BGR)
    img_pred = cv2.cvtColor(Y_frame, cv2.COLOR_GRAY2BGR)

    print("4. Processing CTUs...")
    
    rows_ctu = HEIGHT // 64
    cols_ctu = WIDTH // 64
    qp_tensor = torch.tensor([float(QP_VALUE)/51.0], device=DEVICE).unsqueeze(0) 

    with torch.no_grad():
        for r in range(rows_ctu):
            for c in range(cols_ctu):
                y_pos = r * 64
                x_pos = c * 64
                
                # --- A. GROUND TRUTH (RED) ---
                gt_block = label_matrix[r*4:(r+1)*4, c*4:(c+1)*4]
                draw_partitions(img_gt, y_pos, x_pos, gt_block, color=COLOR_GT)

                # --- B. MODEL PREDICTION (GREEN) ---
                ctu = Y_frame[y_pos:y_pos+64, x_pos:x_pos+64]
                ctu_tensor = torch.tensor(ctu, dtype=torch.float32, device=DEVICE).unsqueeze(0) / 255.0 
                
                b1, b2, b3 = model((qp_tensor, ctu_tensor)) 
                
                pred_grid = np.zeros((4,4), dtype=np.uint8)
                split_64 = b1.item() > 0.5
                split_32 = (b2.squeeze() > 0.5).cpu().numpy()
                split_16 = (b3.squeeze() > 0.5).cpu().numpy()
                
                if split_64:
                    pred_grid[:, :] = 1 
                    idx_32 = 0
                    for i in range(0, 4, 2):
                        for j in range(0, 4, 2):
                            if split_32[idx_32]:
                                pred_grid[i:i+2, j:j+2] = 2 
                                base_row, base_col = i, j
                                sub_indices = [(base_row, base_col), (base_row, base_col+1),
                                               (base_row+1, base_col), (base_row+1, base_col+1)]
                                for (r16, c16) in sub_indices:
                                    if split_16[r16 * 4 + c16]: pred_grid[r16, c16] = 3
                            idx_32 += 1
                
                draw_partitions(img_pred, y_pos, x_pos, pred_grid, color=COLOR_PRED)

    print("5. Adding Labels in Header and Saving...")
    
    # Define Header
    header_height = 200
    header_gt = np.ones((header_height, WIDTH, 3), dtype=np.uint8) * 255 # White background
    header_pred = np.ones((header_height, WIDTH, 3), dtype=np.uint8) * 255 # White background

    # Add Text to Headers with LINE_TYPE
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(header_gt, "Ground Truth", (50, 140), font, 4, COLOR_GT, 8, LINE_TYPE)
    cv2.putText(header_pred, "ETH-CNN Prediction", (50, 140), font, 4, COLOR_PRED, 8, LINE_TYPE)

    # Stack Header on top of Image
    img_gt_full = np.vstack((header_gt, img_gt))
    img_pred_full = np.vstack((header_pred, img_pred))

    # Stack Side by Side
    combined = np.hstack((img_gt_full, img_pred_full))
    
    # --- NO RESIZING: Saving Full Resolution ---
    output_filename = f"partition_vis_frame{FRAME_IDX}_qp{QP_VALUE}_FULLRES.jpg"
    cv2.imwrite(output_filename, combined)
    print(f"Done! Saved full resolution comparison to {output_filename}")

if __name__ == "__main__":
    main()