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
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Set to '' for CPU, '0' for GPU
DEVICE = torch.device('cpu')

# =========================================================================
# 2. NORMALIZATION AND DOWNSAMPLING FUNCTIONS
# =========================================================================

def norm_batch_ctu(ctu_batch):
    """
    Normalizes CTU batch for the 3 separate branches.
    CRITICAL: Calculates mean per-instance, not globally across batch.
    """
    ctu_data = ctu_batch.clone().detach().float()
    
    if ctu_data.dim() == 2:
        ctu_data = ctu_data.unsqueeze(0)
    
    norm_ctu_data_b1 = ctu_data.clone()
    norm_ctu_data_b2 = ctu_data.clone()
    norm_ctu_data_b3 = ctu_data.clone()

    # Branch B1: Mean removal at 64x64 level (global)
    mean_value_level1 = torch.mean(ctu_data[:, 0:64, 0:64], dim=(1, 2), keepdim=True)
    norm_ctu_data_b1 -= mean_value_level1
    
    # Branch B2: Mean removal at 32x32 level (4 quadrants)
    mean_value_level2_1 = torch.mean(ctu_data[:, 0:32, 0:32], dim=(1, 2), keepdim=True)
    mean_value_level2_2 = torch.mean(ctu_data[:, 0:32, 32:64], dim=(1, 2), keepdim=True)
    mean_value_level2_3 = torch.mean(ctu_data[:, 32:64, 0:32], dim=(1, 2), keepdim=True)
    mean_value_level2_4 = torch.mean(ctu_data[:, 32:64, 32:64], dim=(1, 2), keepdim=True)
    
    norm_ctu_data_b2[:, 0:32, 0:32] -= mean_value_level2_1
    norm_ctu_data_b2[:, 0:32, 32:64] -= mean_value_level2_2
    norm_ctu_data_b2[:, 32:64, 0:32] -= mean_value_level2_3
    norm_ctu_data_b2[:, 32:64, 32:64] -= mean_value_level2_4

    # Branch B3: Mean removal at 16x16 level (16 blocks)
    # NOTE: Training code uses per-row loop with 4 column blocks (matching training exactly)
    for i in range(0, 64, 16):
        mean_value_level3_1 = torch.mean(ctu_data[:, i:i+16, 0:16])
        mean_value_level3_2 = torch.mean(ctu_data[:, i:i+16, 16:32])
        mean_value_level3_3 = torch.mean(ctu_data[:, i:i+16, 32:48])
        mean_value_level3_4 = torch.mean(ctu_data[:, i:i+16, 48:64])

        norm_ctu_data_b3[:, i:i+16, 0:16]  -= mean_value_level3_1
        norm_ctu_data_b3[:, i:i+16, 16:32] -= mean_value_level3_2
        norm_ctu_data_b3[:, i:i+16, 32:48] -= mean_value_level3_3
        norm_ctu_data_b3[:, i:i+16, 48:64] -= mean_value_level3_4
    
    return norm_ctu_data_b1, norm_ctu_data_b2, norm_ctu_data_b3


def mean_downsample(tensor, scale_factor):
    """Downsamples tensor using mean pooling via unfold (matches training code)"""
    if tensor.dim() != 3:
        raise ValueError("Input tensor must be 3D (batch, height, width).")
    
    batch_size, h, w = tensor.shape
    new_h, new_w = h // scale_factor, w // scale_factor

    downsampled_tensor = tensor.unfold(1, scale_factor, scale_factor).unfold(2, scale_factor, scale_factor)
    downsampled_tensor = downsampled_tensor.contiguous().view(batch_size, new_h, new_w, -1)
    downsampled_tensor = downsampled_tensor.mean(dim=-1)
    
    return downsampled_tensor


def downsample_ctu_3_branches(norm_ctu_tuple):
    """Downsample normalized CTUs for 3 branches"""
    branch1_ctu = norm_ctu_tuple[0]
    branch2_ctu = norm_ctu_tuple[1]
    branch3_ctu = norm_ctu_tuple[2]

    # Branch 1: 64x64 -> 16x16 (downsample by 4)
    downsampled_ctu_16_16 = mean_downsample(branch1_ctu, 4)
    
    # Branch 2: 64x64 -> 32x32 (downsample by 2)
    downsampled_ctu_32_32 = mean_downsample(branch2_ctu, 2)
    
    # Branch 3: 64x64 -> 64x64 (no downsampling, scale_factor=1)
    downsampled_ctu_64_64 = mean_downsample(branch3_ctu, 1)

    return (downsampled_ctu_16_16, downsampled_ctu_32_32, downsampled_ctu_64_64)


# =========================================================================
# 3. ETH-CNN MODEL DEFINITION (MATCHES TRAINING CODE EXACTLY)
# =========================================================================

class ETH_CNN(nn.Module):
    def __init__(self):
        super(ETH_CNN, self).__init__()
        
        # Branch 1 convolution layers
        self.conv1_b1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=4, padding=0)
        self.conv2_b1 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=2, stride=2, padding=0)
        self.conv3_b1 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=2, stride=2, padding=0)

        # Branch 2 convolution layers
        self.conv1_b2 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=4, padding=0)
        self.conv2_b2 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=2, stride=2, padding=0)
        self.conv3_b2 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=2, stride=2, padding=0)

        # Branch 3 convolution layers
        self.conv1_b3 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=4, padding=0)
        self.conv2_b3 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=2, stride=2, padding=0)
        self.conv3_b3 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=2, stride=2, padding=0)

        # Dropout layers
        self.fc1_dropout = nn.Dropout(p=0.5)
        self.fc2_dropout = nn.Dropout(p=0.25)

        # Branch 1 fully connected layers (64x64 prediction -> 1 output)
        self.fc1_b1 = nn.Linear(in_features=1344, out_features=48)
        self.fc2_b1 = nn.Linear(in_features=49, out_features=24)
        self.fc3_b1 = nn.Linear(in_features=25, out_features=1)

        # Branch 2 fully connected layers (32x32 prediction -> 4 outputs)
        self.fc1_b2 = nn.Linear(in_features=1344, out_features=96)
        self.fc2_b2 = nn.Linear(in_features=97, out_features=48)
        self.fc3_b2 = nn.Linear(in_features=49, out_features=4)

        # Branch 3 fully connected layers (16x16 prediction -> 16 outputs)
        self.fc1_b3 = nn.Linear(in_features=1344, out_features=136)
        self.fc2_b3 = nn.Linear(in_features=137, out_features=96)
        self.fc3_b3 = nn.Linear(in_features=97, out_features=16)

    def full_connect_b1(self, x, qp):
        qp_tensor = qp.unsqueeze(1)
        
        fc1_activation_op = F.leaky_relu(self.fc1_b1(x))
        fc1_activation_op = self.fc1_dropout(fc1_activation_op)
        
        qp_fc1_activation_op = torch.cat((fc1_activation_op, qp_tensor), dim=1)
        fc2_activation_op = F.leaky_relu(self.fc2_b1(qp_fc1_activation_op))
        fc2_activation_op = self.fc2_dropout(fc2_activation_op)

        qp_fc2_activation_op = torch.cat((fc2_activation_op, qp_tensor), dim=1)
        fc3_activation_op = torch.sigmoid(self.fc3_b1(qp_fc2_activation_op))
        return fc3_activation_op

    def full_connect_b2(self, x, qp):
        qp_tensor = qp.unsqueeze(1)

        fc1_activation_op = F.leaky_relu(self.fc1_b2(x))
        fc1_activation_op = self.fc1_dropout(fc1_activation_op)

        qp_fc1_activation_op = torch.cat((fc1_activation_op, qp_tensor), dim=1)
        fc2_activation_op = F.leaky_relu(self.fc2_b2(qp_fc1_activation_op))
        fc2_activation_op = self.fc2_dropout(fc2_activation_op)

        qp_fc2_activation_op = torch.cat((fc2_activation_op, qp_tensor), dim=1)
        fc3_activation_op = torch.sigmoid(self.fc3_b2(qp_fc2_activation_op))
        return fc3_activation_op
    
    def full_connect_b3(self, x, qp):
        qp_tensor = qp.unsqueeze(1)

        fc1_activation_op = F.leaky_relu(self.fc1_b3(x))
        fc1_activation_op = self.fc1_dropout(fc1_activation_op)
        
        qp_fc1_activation_op = torch.cat((fc1_activation_op, qp_tensor), dim=1)
        fc2_activation_op = F.leaky_relu(self.fc2_b3(qp_fc1_activation_op))
        fc2_activation_op = self.fc2_dropout(fc2_activation_op)

        qp_fc2_activation_op = torch.cat((fc2_activation_op, qp_tensor), dim=1)
        fc3_activation_op = torch.sigmoid(self.fc3_b3(qp_fc2_activation_op))
        return fc3_activation_op
    
    def forward(self, x):
        qp = x[0]
        original_ctu = x[1]
        
        x = norm_batch_ctu(original_ctu)
        x = downsample_ctu_3_branches(x)

        # Branch 1
        h_conv1_b1_op = F.leaky_relu(self.conv1_b1(x[0].unsqueeze(1)))
        h_conv2_b1_op = F.leaky_relu(self.conv2_b1(h_conv1_b1_op))
        h_conv3_b1_op = F.leaky_relu(self.conv3_b1(h_conv2_b1_op))

        # Branch 2
        h_conv1_b2_op = F.leaky_relu(self.conv1_b2(x[1].unsqueeze(1)))
        h_conv2_b2_op = F.leaky_relu(self.conv2_b2(h_conv1_b2_op))
        h_conv3_b2_op = F.leaky_relu(self.conv3_b2(h_conv2_b2_op))

        # Branch 3
        h_conv1_b3_op = F.leaky_relu(self.conv1_b3(x[2].unsqueeze(1)))
        h_conv2_b3_op = F.leaky_relu(self.conv2_b3(h_conv1_b3_op))
        h_conv3_b3_op = F.leaky_relu(self.conv3_b3(h_conv2_b3_op))

        # Flatten and concatenate outputs
        reshaped_conv3_b3_op = h_conv3_b3_op.view(-1, 16 * 4 * 4)
        reshaped_conv3_b2_op = h_conv3_b2_op.view(-1, 16 * 2 * 2)
        reshaped_conv3_b1_op = h_conv3_b1_op.view(-1, 16 * 1 * 1)
        reshaped_conv2_b3_op = h_conv2_b3_op.view(-1, 12 * 8 * 8)
        reshaped_conv2_b2_op = h_conv2_b2_op.view(-1, 12 * 4 * 4)
        reshaped_conv2_b1_op = h_conv2_b1_op.view(-1, 12 * 2 * 2)

        concatenated_output = torch.cat((
            reshaped_conv3_b1_op, reshaped_conv2_b1_op, 
            reshaped_conv3_b2_op, reshaped_conv2_b2_op, 
            reshaped_conv3_b3_op, reshaped_conv2_b3_op
        ), dim=1)

        b1_op = self.full_connect_b1(concatenated_output, qp)
        b2_op = self.full_connect_b2(concatenated_output, qp)
        b3_op = self.full_connect_b3(concatenated_output, qp)

        return (b1_op.squeeze(dim=0), b2_op.squeeze(dim=0), b3_op.squeeze(dim=0))


# =========================================================================
# 4. DATA PROCESSING & INFERENCE
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
    CRITICAL: Normalization happens inside model forward pass
    """
    batch_size = input_image.shape[0]
    y_conv_out = np.zeros((batch_size, 21), dtype=np.float32)
    
    model.eval()
    
    with torch.no_grad():
        for i in range(math.ceil(batch_size / sub_batch_size)):
            index_start = i * sub_batch_size
            index_end = min((i + 1) * sub_batch_size, batch_size)
            
            # Normalize: uint8 [0,255] -> float32 [0.0,1.0]
            batch_imgs_np = input_image[index_start:index_end, :, :].astype(np.float32) / 255.0
            ctu_tensor = torch.from_numpy(batch_imgs_np).to(DEVICE)
            
            # Normalize QP: [0,51] -> [0.0,1.0]
            batch_qp_np = np.ones((index_end - index_start), dtype=np.float32) * qp_seq / 51.0
            qp_tensor = torch.from_numpy(batch_qp_np).to(DEVICE)
            
            # Forward pass
            b1_op, b2_op, b3_op = model((qp_tensor, ctu_tensor))
            
            # Collect results: [batch, 1] + [batch, 4] + [batch, 16] = [batch, 21]
            out_64 = b1_op.cpu().numpy()
            out_32 = b2_op.cpu().numpy()
            out_16 = b3_op.cpu().numpy()
            
            # Ensure proper 2D shape for concatenation
            if out_64.ndim == 1:
                out_64 = out_64.reshape(-1, 1)
            if out_32.ndim == 1:
                out_32 = out_32.reshape(-1, 4)
            if out_16.ndim == 1:
                out_16 = out_16.reshape(-1, 16)
            
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
        
        # Convert to Float32 BEFORE inference (Matches training logic)
        input_batch = input_batch.astype(np.float32)

        # Run inference (normalization to [0,1] happens inside get_y_conv_on_large_data)
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
# 5. MAIN EXECUTION
# =========================================================================

def main():
    if len(sys.argv) < 5:
        print("Usage: python video_to_cu_depth_pytorch.py <yuv_file> <width> <height> <qp>")
        sys.exit(1)
    
    yuv_file = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    qp_seq = int(sys.argv[4])
    
    # Model checkpoint path
    checkpoint_path = 'best_model_ETH_CNN.pth'
    
    # Initialize model
    model = ETH_CNN().to(DEVICE)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict):
                # Check if it looks like a raw state_dict
                state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if saved from DataParallel
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict)
            print("Model loaded successfully!")
            
            # Print parameter count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
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