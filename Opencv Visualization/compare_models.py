import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from timm.layers import trunc_normal_

# ==========================================
# 1. CONFIGURATION
# ==========================================
YUV_FILE_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/Info&YUV/AI_YUV/IntraValid_4928x3264.yuv'
LABEL_FILE_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/Info&YUV/AI_Info/Info_20170826_151434_AI_IntraValid_4928x3264_qp37_nf25_CUDepth.dat'

# Checkpoints
CNN_CHECKPOINT = '/root/myproject/HEVC_Intra_Models-ViT/Eth_CNN_Pt_full/best_model_4qp_parallel_data_processing_loss_mod.pth'
VIT_CHECKPOINT = '/root/myproject/HEVC_Intra_Models-ViT/Fastervit/best_fastervit_hevc_balanced.pth'

# Parameters
FRAME_IDX = 21       
QP_VALUE = 37       
WIDTH = 4928        
HEIGHT = 3264       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- VISUALIZATION SETTINGS ---
LINE_THICKNESS = 2          # Set to 2 as requested
LINE_TYPE = cv2.LINE_4      # Changed to LINE_4 for sharp, non-shadowy lines
COLOR_GT = (0, 0, 255)      # Red
COLOR_CNN = (0, 255, 0)     # Green
COLOR_VIT = (255, 0, 0)     # Blue

# ==========================================
# 2. MODEL DEFINITIONS
# ==========================================

# --- A. ETH-CNN ---
def mean_downsample(tensor, scale_factor):
    batch_size, h, w = tensor.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    downsampled_tensor = tensor.unfold(1, scale_factor, scale_factor).unfold(2, scale_factor, scale_factor)
    downsampled_tensor = downsampled_tensor.contiguous().view(batch_size, new_h, new_w, -1)
    downsampled_tensor = downsampled_tensor.mean(dim=-1)
    return downsampled_tensor

def norm_batch_ctu(ctu_batch):
    norm_ctu_data_b1 = ctu_batch.clone(); norm_ctu_data_b2 = ctu_batch.clone(); norm_ctu_data_b3 = ctu_batch.clone()
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
    return (mean_downsample(norm_ctu_tuple[0], 4), mean_downsample(norm_ctu_tuple[1], 2), mean_downsample(norm_ctu_tuple[2], 1))

class ETH_CNN(nn.Module):
    def __init__(self):
        super(ETH_CNN, self).__init__()
        self.conv1_b1 = nn.Conv2d(1, 16, 4, 4, 0); self.conv2_b1 = nn.Conv2d(16, 24, 2, 2, 0); self.conv3_b1 = nn.Conv2d(24, 32, 2, 2, 0)
        self.conv1_b2 = nn.Conv2d(1, 16, 4, 4, 0); self.conv2_b2 = nn.Conv2d(16, 24, 2, 2, 0); self.conv3_b2 = nn.Conv2d(24, 32, 2, 2, 0)
        self.conv1_b3 = nn.Conv2d(1, 16, 4, 4, 0); self.conv2_b3 = nn.Conv2d(16, 24, 2, 2, 0); self.conv3_b3 = nn.Conv2d(24, 32, 2, 2, 0)
        self.fc1_dropout = nn.Dropout(p=0.5); self.fc2_dropout = nn.Dropout(p=0.2)
        self.fc1_b1 = nn.Linear(2688, 64); self.fc2_b1 = nn.Linear(65, 48); self.fc3_b1 = nn.Linear(49, 1)
        self.fc1_b2 = nn.Linear(2688, 128); self.fc2_b2 = nn.Linear(129, 96); self.fc3_b2 = nn.Linear(97, 4)
        self.fc1_b3 = nn.Linear(2688, 256); self.fc2_b3 = nn.Linear(257, 192); self.fc3_b3 = nn.Linear(193, 16)
    def full_connect_b1(self, x, qp):
        x = self.fc1_dropout(F.leaky_relu(self.fc1_b1(x))); x = torch.cat((x, qp), dim=1)
        x = self.fc2_dropout(F.leaky_relu(self.fc2_b1(x))); x = torch.cat((x, qp), dim=1); return torch.sigmoid(self.fc3_b1(x))
    def full_connect_b2(self, x, qp):
        x = self.fc1_dropout(F.leaky_relu(self.fc1_b2(x))); x = torch.cat((x, qp), dim=1)
        x = self.fc2_dropout(F.leaky_relu(self.fc2_b2(x))); x = torch.cat((x, qp), dim=1); return torch.sigmoid(self.fc3_b2(x))
    def full_connect_b3(self, x, qp):
        x = self.fc1_dropout(F.leaky_relu(self.fc1_b3(x))); x = torch.cat((x, qp), dim=1)
        x = self.fc2_dropout(F.leaky_relu(self.fc2_b3(x))); x = torch.cat((x, qp), dim=1); return torch.sigmoid(self.fc3_b3(x))
    def forward(self, x_in):
        qp = x_in[0]; x = norm_batch_ctu(x_in[1]); x = downsample_ctu_3_branches(x)
        c1_b1 = F.leaky_relu(self.conv1_b1(x[0].unsqueeze(1))); c2_b1 = F.leaky_relu(self.conv2_b1(c1_b1)); c3_b1 = F.leaky_relu(self.conv3_b1(c2_b1))
        c1_b2 = F.leaky_relu(self.conv1_b2(x[1].unsqueeze(1))); c2_b2 = F.leaky_relu(self.conv2_b2(c1_b2)); c3_b2 = F.leaky_relu(self.conv3_b2(c2_b2))
        c1_b3 = F.leaky_relu(self.conv1_b3(x[2].unsqueeze(1))); c2_b3 = F.leaky_relu(self.conv2_b3(c1_b3)); c3_b3 = F.leaky_relu(self.conv3_b3(c2_b3))
        flat = torch.cat((c3_b1.view(-1,32), c2_b1.view(-1,96), c3_b2.view(-1,128), c2_b2.view(-1,384), c3_b3.view(-1,512), c2_b3.view(-1,1536)), dim=1)
        return self.full_connect_b1(flat, qp), self.full_connect_b2(flat, qp), self.full_connect_b3(flat, qp)

# --- B. FasterViT ---
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False), nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False))
        self.bn = nn.BatchNorm2d(out_ch); self.act = nn.GELU()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class EfficientResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False), nn.Conv2d(dim, dim, 1, 1, 0, bias=False))
        self.bn = nn.BatchNorm2d(dim); self.act = nn.GELU()
    def forward(self, x): return x + self.act(self.bn(self.conv(x)))

class StreamlinedHAT(nn.Module):
    def __init__(self, dim, num_heads=2, window_size=2, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim; self.num_heads = num_heads; self.window_size = window_size; self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False); self.proj = nn.Linear(dim, dim, bias=False)
        hidden = int(dim * mlp_ratio); self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.pos_scale = nn.Parameter(torch.ones(num_heads) * 0.5)
    def forward(self, x, ct):
        B, C, H, W = x.shape
        x_win = x.permute(0, 2, 3, 1).reshape(-1, self.window_size ** 2, C); ct = ct.reshape(-1, 1, C)
        tokens = torch.cat([x_win, ct], dim=1); shortcut = tokens
        qkv = self.qkv(tokens).reshape(tokens.size(0), tokens.size(1), 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]; attn = (q @ k.transpose(-2, -1)) * self.scale; attn = attn * self.pos_scale.view(1, -1, 1, 1); attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(tokens.shape); tokens = shortcut + self.proj(out); tokens = tokens + self.mlp(self.norm2(tokens))
        x = tokens[:, :-1, :].reshape(B, H, W, C).permute(0, 3, 1, 2); ct = tokens[:, -1:, :]
        return x.contiguous(), ct

class BalancedFasterViT_HEVC(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [8, 16, 24, 32]
        self.stem = ConvBNAct(1, dims[0], 3, 2, 1)
        self.stage1 = nn.Sequential(EfficientResBlock(dims[0]), ConvBNAct(dims[0], dims[1], 3, 2, 1))
        self.stage2 = nn.Sequential(EfficientResBlock(dims[1]), ConvBNAct(dims[1], dims[2], 3, 2, 1))
        self.stage3 = nn.Sequential(EfficientResBlock(dims[2]), ConvBNAct(dims[2], dims[3], 3, 2, 1))
        self.hat = StreamlinedHAT(dims[3], window_size=2); self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(dims[3] + 1, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.12), nn.Linear(1024, 1536), nn.BatchNorm1d(1536), nn.ReLU(), nn.Dropout(0.08), nn.Linear(1536, 21), nn.Sigmoid())
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x, qp):
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x); x = self.stage3(x)
        ct = F.adaptive_avg_pool2d(x, (2, 2)).flatten(2).transpose(1, 2); x, _ = self.hat(x, ct)
        feat = self.gap(x).flatten(1); 
        if qp.dim() == 1: qp = qp.unsqueeze(1)
        return self.head(torch.cat([feat, qp], dim=1))

# ==========================================
# 3. UTILS
# ==========================================
def read_specific_yuv_frame(filepath, width, height, frame_idx):
    y_size = width * height; frame_size = int(width * height * 1.5); offset = frame_idx * frame_size
    with open(filepath, 'rb') as f:
        f.seek(offset); y_buf = f.read(y_size)
        if len(y_buf) < y_size: raise ValueError("End of file reached.")
        return np.frombuffer(y_buf, dtype=np.uint8).reshape((height, width))

def read_specific_label_frame(filepath, width, height, frame_idx):
    unit_width = 16; rows = height // unit_width; cols = width // unit_width
    bytes_per_frame = rows * cols; offset = frame_idx * bytes_per_frame
    with open(filepath, 'rb') as f:
        f.seek(offset); buf = f.read(bytes_per_frame)
        if len(buf) < bytes_per_frame: raise ValueError("End of file reached.")
        return np.frombuffer(buf, dtype=np.uint8).reshape((rows, cols))

def draw_partitions(img_rgb, top_left_y, top_left_x, depth_grid, color):
    # Using LINE_TYPE for sharpness
    cv2.rectangle(img_rgb, (top_left_x, top_left_y), (top_left_x+64, top_left_y+64), color, LINE_THICKNESS, LINE_TYPE)
    max_depth = np.max(depth_grid)

    if max_depth >= 1:
        cv2.line(img_rgb, (top_left_x + 32, top_left_y), (top_left_x + 32, top_left_y + 64), color, LINE_THICKNESS, LINE_TYPE)
        cv2.line(img_rgb, (top_left_x, top_left_y + 32), (top_left_x + 64, top_left_y + 32), color, LINE_THICKNESS, LINE_TYPE)
        
        for i in range(2): 
            for j in range(2): 
                sub_grid = depth_grid[i*2:(i+1)*2, j*2:(j+1)*2]
                if np.max(sub_grid) >= 2:
                    y32 = top_left_y + i*32; x32 = top_left_x + j*32
                    cv2.line(img_rgb, (x32 + 16, y32), (x32 + 16, y32 + 32), color, LINE_THICKNESS, LINE_TYPE)
                    cv2.line(img_rgb, (x32, y32 + 16), (x32 + 32, y32 + 16), color, LINE_THICKNESS, LINE_TYPE)
                    
                    for m in range(2):
                        for n in range(2):
                            if sub_grid[m, n] >= 3:
                                y16 = y32 + m*16; x16 = x32 + n*16
                                cv2.line(img_rgb, (x16 + 8, y16), (x16 + 8, y16 + 16), color, LINE_THICKNESS, LINE_TYPE)
                                cv2.line(img_rgb, (x16, y16 + 8), (x16 + 16, y16 + 8), color, LINE_THICKNESS, LINE_TYPE)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    if "IntraTrain" in YUV_FILE_PATH and "IntraValid" in LABEL_FILE_PATH:
        print("\n" + "!"*60)
        print("WARNING: Mixed Train/Valid files detected.")
        print("!"*60 + "\n")

    print("1. Loading ETH-CNN...")
    model_cnn = ETH_CNN().to(DEVICE)
    try:
        ckpt_cnn = torch.load(CNN_CHECKPOINT, map_location=DEVICE, weights_only=False)
        model_cnn.load_state_dict(ckpt_cnn['model_state_dict'] if 'model_state_dict' in ckpt_cnn else ckpt_cnn)
    except: print("Warning: CNN Weights issue.")
    model_cnn.eval()

    print("2. Loading FasterViT...")
    model_vit = BalancedFasterViT_HEVC().to(DEVICE)
    try:
        ckpt_vit = torch.load(VIT_CHECKPOINT, map_location=DEVICE, weights_only=False)
        model_vit.load_state_dict(ckpt_vit['model_state_dict'] if 'model_state_dict' in ckpt_vit else ckpt_vit)
    except: print("Warning: ViT Weights issue.")
    model_vit.eval()

    print(f"3. Reading Data...")
    Y_frame = read_specific_yuv_frame(YUV_FILE_PATH, WIDTH, HEIGHT, FRAME_IDX)
    label_matrix = read_specific_label_frame(LABEL_FILE_PATH, WIDTH, HEIGHT, FRAME_IDX)

    img_gt = cv2.cvtColor(Y_frame, cv2.COLOR_GRAY2BGR)
    img_cnn = cv2.cvtColor(Y_frame, cv2.COLOR_GRAY2BGR)
    img_vit = cv2.cvtColor(Y_frame, cv2.COLOR_GRAY2BGR)

    rows_ctu = HEIGHT // 64; cols_ctu = WIDTH // 64
    qp_val = float(QP_VALUE)/51.0
    qp_tensor = torch.tensor([qp_val], device=DEVICE).unsqueeze(0)

    print("4. Running Inference...")
    with torch.no_grad():
        for r in range(rows_ctu):
            for c in range(cols_ctu):
                y_pos = r * 64; x_pos = c * 64
                
                # --- A. Ground Truth ---
                gt_block = label_matrix[r*4:(r+1)*4, c*4:(c+1)*4]
                draw_partitions(img_gt, y_pos, x_pos, gt_block, COLOR_GT)

                # --- B. ETH-CNN ---
                ctu_block = Y_frame[y_pos:y_pos+64, x_pos:x_pos+64]
                ctu_tensor = torch.tensor(ctu_block, dtype=torch.float32, device=DEVICE).unsqueeze(0) / 255.0
                b1, b2, b3 = model_cnn((qp_tensor, ctu_tensor))
                
                pred_grid_cnn = np.zeros((4,4), dtype=np.uint8)
                if b1.item() > 0.5:
                    pred_grid_cnn[:, :] = 1
                    s32 = (b2.squeeze() > 0.5).cpu().numpy(); s16 = (b3.squeeze() > 0.5).cpu().numpy()
                    idx32 = 0
                    for i in range(0, 4, 2):
                        for j in range(0, 4, 2):
                            if s32[idx32]:
                                pred_grid_cnn[i:i+2, j:j+2] = 2
                                base_sub = [(i,j),(i,j+1),(i+1,j),(i+1,j+1)]
                                for (r16, c16) in base_sub:
                                    if s16[r16*4+c16]: pred_grid_cnn[r16, c16] = 3
                            idx32 += 1
                draw_partitions(img_cnn, y_pos, x_pos, pred_grid_cnn, COLOR_CNN)

                # --- C. FasterViT ---
                ctu_vit = ctu_tensor.unsqueeze(0) # [1, 1, 64, 64]
                out_vit = model_vit(ctu_vit, qp_tensor)
                out_np = (out_vit > 0.5).cpu().numpy()[0]
                
                pred_grid_vit = np.zeros((4,4), dtype=np.uint8)
                if out_np[0]: # Split 64
                    pred_grid_vit[:, :] = 1
                    s32_vit = out_np[1:5]; s16_vit = out_np[5:21]
                    idx32 = 0
                    for i in range(0, 4, 2):
                        for j in range(0, 4, 2):
                            if s32_vit[idx32]:
                                pred_grid_vit[i:i+2, j:j+2] = 2
                                base_sub = [(i,j),(i,j+1),(i+1,j),(i+1,j+1)]
                                for (r16, c16) in base_sub:
                                    if s16_vit[r16*4+c16]: pred_grid_vit[r16, c16] = 3
                            idx32 += 1
                draw_partitions(img_vit, y_pos, x_pos, pred_grid_vit, COLOR_VIT)

    print("5. Saving FULL RESOLUTION Output...")
    header_h = 200
    h_gt = np.ones((header_h, WIDTH, 3), dtype=np.uint8) * 255
    h_cnn = np.ones((header_h, WIDTH, 3), dtype=np.uint8) * 255
    h_vit = np.ones((header_h, WIDTH, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(h_gt, "Ground Truth", (50, 140), font, 4, COLOR_GT, 8, cv2.LINE_AA)
    cv2.putText(h_cnn, "ETH-CNN", (50, 140), font, 4, COLOR_CNN, 8, cv2.LINE_AA)
    cv2.putText(h_vit, "FasterViT", (50, 140), font, 4, COLOR_VIT, 8, cv2.LINE_AA)

    final = np.hstack((np.vstack((h_gt, img_gt)), np.vstack((h_cnn, img_cnn)), np.vstack((h_vit, img_vit))))
    
    # REMOVED RESIZING! Saving full resolution.
    cv2.imwrite(f"comparison_F{FRAME_IDX}_QP{QP_VALUE}_FULL.jpg", final)
    print("Done. Saved full resolution image.")

if __name__ == "__main__":
    main()