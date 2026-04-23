#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import time

# =========================================================================
# 1. ENVIRONMENT CONFIGURATION
# =========================================================================

# Set device (CPU or GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set to '' for CPU, '0' for GPU
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# =========================================================================
# 2. HFCN MODEL DEFINITION
# =========================================================================

class HFCN(nn.Module):
    def __init__(self):
        super(HFCN, self).__init__()

        # Block 1
        self.block1_conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.block1_bn1 = nn.BatchNorm2d(8)
        self.block1_conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.block1_bn2 = nn.BatchNorm2d(8)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.block2_conv1 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.block2_bn1 = nn.BatchNorm2d(8)
        self.block2_conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.block2_bn2 = nn.BatchNorm2d(8)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Branch 2 (16x16 output)
        self.br2_conv1 = nn.Conv2d(8, 32, kernel_size=4, stride=4, padding=0)
        self.br2_bn1 = nn.BatchNorm2d(32)
        self.br2_conv2 = nn.Conv2d(33, 16, kernel_size=1, padding=0)
        self.br2_bn2 = nn.BatchNorm2d(16)
        self.br2_conv3 = nn.Conv2d(16, 1, kernel_size=1, padding=0)

        # Block 3
        self.block3_conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.block3_bn1 = nn.BatchNorm2d(16)
        self.block3_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.block3_bn2 = nn.BatchNorm2d(16)
        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Branch 3 (32x32 output)
        self.br3_conv1 = nn.Conv2d(16, 8, kernel_size=4, stride=4, padding=0)
        self.br3_bn1 = nn.BatchNorm2d(8)
        self.br3_conv2 = nn.Conv2d(9, 4, kernel_size=1, padding=0)
        self.br3_bn2 = nn.BatchNorm2d(4)
        self.br3_conv3 = nn.Conv2d(4, 1, kernel_size=1, padding=0)

        # Block 4
        self.block4_conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.block4_bn1 = nn.BatchNorm2d(16)
        self.block4_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.block4_bn2 = nn.BatchNorm2d(16)
        self.block4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Branch 4 (64x64 output)
        self.br4_conv1 = nn.Conv2d(16, 8, kernel_size=4, stride=4, padding=0)
        self.br4_bn1 = nn.BatchNorm2d(8)
        self.br4_conv2 = nn.Conv2d(9, 4, kernel_size=1, padding=0)
        self.br4_bn2 = nn.BatchNorm2d(4)
        self.br4_conv3 = nn.Conv2d(4, 1, kernel_size=1, padding=0)

        self._init_weights()

    def _init_weights(self):
        """Apply He uniform initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        qp_scalar = x[0]
        img = x[1].unsqueeze(1)
        batch_size = img.size(0)

        # Block 1
        x = self.block1_pool(self.block1_bn2(F.relu(self.block1_conv2(self.block1_bn1(F.relu(self.block1_conv1(img)))))))
        
        # Block 2
        x = self.block2_pool(self.block2_bn2(F.relu(self.block2_conv2(self.block2_bn1(F.relu(self.block2_conv1(x)))))))
        
        # Branch 2 Prediction (16x16)
        b2 = self.br2_bn1(F.relu(self.br2_conv1(x)))
        qp_plane_b2 = qp_scalar.view(batch_size, 1, 1, 1).expand(batch_size, 1, 4, 4)
        b2 = torch.cat([b2, qp_plane_b2], dim=1)
        b2 = torch.sigmoid(self.br2_conv3(self.br2_bn2(F.relu(self.br2_conv2(b2)))))
        pred_16 = b2.view(batch_size, -1)

        # Block 3
        x = self.block3_pool(self.block3_bn2(F.relu(self.block3_conv2(self.block3_bn1(F.relu(self.block3_conv1(x)))))))

        # Branch 3 Prediction (32x32)
        b3 = self.br3_bn1(F.relu(self.br3_conv1(x)))
        qp_plane_b3 = qp_scalar.view(batch_size, 1, 1, 1).expand(batch_size, 1, 2, 2)
        b3 = torch.cat([b3, qp_plane_b3], dim=1)
        b3 = torch.sigmoid(self.br3_conv3(self.br3_bn2(F.relu(self.br3_conv2(b3)))))
        pred_32 = b3.view(batch_size, -1)

        # Block 4
        x = self.block4_pool(self.block4_bn2(F.relu(self.block4_conv2(self.block4_bn1(F.relu(self.block4_conv1(x)))))))

        # Branch 4 Prediction (64x64)
        b4 = self.br4_bn1(F.relu(self.br4_conv1(x)))
        qp_plane_b4 = qp_scalar.view(batch_size, 1, 1, 1).expand(batch_size, 1, 1, 1)
        b4 = torch.cat([b4, qp_plane_b4], dim=1)
        b4 = torch.sigmoid(self.br4_conv3(self.br4_bn2(F.relu(self.br4_conv2(b4)))))
        pred_64 = b4.view(batch_size, -1).squeeze(1)

        return pred_64, pred_32, pred_16


# =========================================================================
# 3. DATA PROCESSING & INFERENCE
# =========================================================================

IMAGE_SIZE = 64
SAVE_FILE = 'cu_depth.dat'

def print_current_line(str_val):
    """Print progress on same line"""
    print('\r' + str_val, end='')
    sys.stdout.flush()


def get_file_size(path):
    """Get file size in bytes"""
    try:
        return os.path.getsize(path)
    except Exception as err:
        print(f"Error accessing file: {err}")
        return 0


def get_Y_for_one_frame(f, frame_width, frame_height, image_size):
    """
    Reads one Y-frame from YUV file and pads to multiples of image_size
    """
    y_size = frame_width * frame_height
    y_buf = f.read(y_size)
    
    if not y_buf or len(y_buf) < y_size:
        return None
    
    # Skip UV data (YUV 4:2:0 format)
    f.read(y_size // 2)
    
    data = np.frombuffer(y_buf, dtype=np.uint8).reshape(frame_height, frame_width)
    
    # Calculate padded dimensions
    valid_height = math.ceil(frame_height / image_size) * image_size
    valid_width = math.ceil(frame_width / image_size) * image_size
    
    # Pad if necessary
    if valid_height > frame_height or valid_width > frame_width:
        padded_data = np.zeros((valid_height, valid_width), dtype=np.uint8)
        padded_data[:frame_height, :frame_width] = data
        
        # Edge replication padding
        if valid_width > frame_width:
            padded_data[:frame_height, frame_width:] = data[:, -1:]
        if valid_height > frame_height:
            padded_data[frame_height:, :frame_width] = data[-1:, :]
        if valid_height > frame_height and valid_width > frame_width:
            padded_data[frame_height:, frame_width:] = data[-1, -1]
        
        data = padded_data
    
    return data


def get_y_conv_on_large_data(model, input_image, qp_seq, sub_batch_size=512):
    """
    Runs inference on batch of CTUs with sub-batching for memory efficiency
    Input: input_image as float32 [0.0, 1.0] (already normalized)
    Output: [batch_size, 21] predictions (1 + 4 + 16)
    """
    batch_size = input_image.shape[0]
    y_conv_out = np.zeros((batch_size, 21), dtype=np.float32)
    
    model.eval()
    
    with torch.no_grad():
        for i in range(math.ceil(batch_size / sub_batch_size)):
            index_start = i * sub_batch_size
            index_end = min((i + 1) * sub_batch_size, batch_size)
            
            # Input is already normalized float32 [0.0, 1.0]
            batch_imgs_np = input_image[index_start:index_end, :, :]
            ctu_tensor = torch.from_numpy(batch_imgs_np).to(DEVICE)
            
            # Normalize QP: [0,51] -> [0.0,1.0]
            batch_qp_np = np.ones((index_end - index_start), dtype=np.float32) * qp_seq / 51.0
            qp_tensor = torch.from_numpy(batch_qp_np).to(DEVICE)
            
            # Forward pass: returns (pred_64, pred_32, pred_16)
            pred_64, pred_32, pred_16 = model((qp_tensor, ctu_tensor))
            
            # Collect results: [batch, 1] + [batch, 4] + [batch, 16] = [batch, 21]
            out_64 = pred_64.cpu().numpy().reshape(-1, 1)  # [batch, 1]
            out_32 = pred_32.cpu().numpy()  # [batch, 4]
            out_16 = pred_16.cpu().numpy()  # [batch, 16]
            
            y_conv_out[index_start:index_end, :] = np.concatenate([out_64, out_32, out_16], axis=1)
    
    return y_conv_out


def get_prob(yuv_file, image_size, save_file, qp_seq, n_frames_start, n_frames_end, 
             frame_width, frame_height, model):
    """
    Main function to process YUV video and generate CU depth probabilities
    OPTIMIZED: Writes per-frame to reduce memory usage
    """
    try:
        f = open(yuv_file, 'rb')
        f_out = open(save_file, 'wb')
    except IOError as e:
        print(f"File Error: {e}")
        return
    
    # Skip to start frame
    frame_bytes = frame_width * frame_height * 3 // 2
    f.seek(n_frames_start * frame_bytes)
    
    print(f"Processing {yuv_file}... QP={qp_seq}")
    print(f"Device: {DEVICE}")
    print(f"Writing to {save_file}")
    
    n_frames_to_process = n_frames_end - n_frames_start
    
    # Calculate grid dimensions
    valid_height = math.ceil(frame_height / image_size) * image_size
    valid_width = math.ceil(frame_width / image_size) * image_size
    blocks_per_frame = (valid_height // image_size) * (valid_width // image_size)
    
    total_start = time.time()
    
    for k in range(n_frames_to_process):
        valid_luma = get_Y_for_one_frame(f, frame_width, frame_height, image_size)
        if valid_luma is None:
            break
        
        # Extract CTUs from frame (store as uint8 for memory efficiency)
        input_batch = np.zeros((blocks_per_frame, image_size, image_size), dtype=np.uint8)
        
        index = 0
        # Raster scan order
        for ystart in range(0, valid_height, image_size):
            for xstart in range(0, valid_width, image_size):
                input_batch[index] = valid_luma[ystart:ystart + image_size, xstart:xstart + image_size]
                index += 1
        
        # Convert to float32 and normalize to [0.0, 1.0]
        input_batch = input_batch.astype(np.float32) / 255.0

        # Run inference
        y_conv_out = get_y_conv_on_large_data(model, input_batch, qp_seq)
        
        # Write immediately to reduce memory footprint
        prob_arr = y_conv_out.flatten().astype(np.float32)
        f_out.write(prob_arr.tobytes())
        
        print_current_line('Frame %d/%d' % (k + 1, n_frames_to_process))
    
    total_end = time.time()
    elapsed = total_end - total_start
    fps = n_frames_to_process / elapsed if elapsed > 0 else 0
    
    print(f"\n\nProcessing complete!")
    print(f"Total time: {elapsed:.2f} sec")
    print(f"Average FPS: {fps:.2f}")
    
    f.close()
    f_out.close()


# =========================================================================
# 4. MAIN EXECUTION
# =========================================================================

def main():
    if len(sys.argv) < 5:
        print("Usage: python video_to_cu_depth_hfcn.py <yuv_file> <width> <height> <qp>")
        sys.exit(1)
    
    yuv_file = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    qp_seq = int(sys.argv[4])
    
    # Model checkpoint path
    checkpoint_path = 'best_model_HFCN_pyt.pth'
    
    # Initialize model
    model = HFCN().to(DEVICE)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            # Load state dict directly (saved as model.state_dict() in training code)
            state_dict = torch.load(checkpoint_path, map_location=DEVICE)
            
            # Remove 'module.' prefix if saved from DataParallel
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict)
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    else:
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        sys.exit(1)
    
    # Verify file
    file_bytes = get_file_size(yuv_file)
    frame_bytes = width * height * 3 // 2
    
    if file_bytes == 0:
        print("Error: Input file size is 0 or file not found.")
        sys.exit(1)
    
    if file_bytes % frame_bytes != 0:
        print(f"Warning: File size ({file_bytes}) is not exact multiple of frame size ({frame_bytes})")
    
    n_frames_total = file_bytes // frame_bytes
    print(f"Total frames in file: {n_frames_total}")
    
    # Process all frames
    get_prob(yuv_file, IMAGE_SIZE, SAVE_FILE, qp_seq, 0, n_frames_total, width, height, model)


if __name__ == "__main__":
    main()