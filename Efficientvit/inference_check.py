import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
import os
import wandb
import matplotlib.pyplot as plt
import time
# FLOPs/Inference libraries
from thop import profile # For MACs/Params calculation
from ptflops import get_model_complexity_info # Alternative for MACs/Params

# --- EFFICIENTVIT MODEL IMPORTS ---
# Import our modified QP-Aware model and its configurations
from efficientvit_ctu import EfficientViT
from build_configs import EfficientViT_m0_TINY

# Global debug flag
DEBUG = False

# ==================== Constants ====================
IMAGE_SIZE = 64        # Input image size (64x64)
NUM_CHANNELS = 1       # Input channels 
NUM_LABEL_BYTES = 16   # Original label size (4x4 grid)
# Calculated length of a single sample in the binary data file
NUM_SAMPLE_LENGTH = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + 64 + (51 + 1) * NUM_LABEL_BYTES
SELECT_QP_LIST = [22, 27, 32, 37] # QPs used in the dataset
# Total output classes for the model
NUM_CLASSES = 21 # 1 (for 64x64 split) + 4 (for 32x32 splits) + 16 (for 16x16 splits)
BATCH_SIZE = 512     # Batch size for batch inference test

# ==================== StreamingDataset Class (Not used in this script) ====================
# This class is copied from the training script for completeness,
# but this inference check script uses dummy (random) data instead.
class StreamingDataset(Dataset):
    def __init__(self, file_path, max_samples):
        self.file_path = file_path
        self.max_samples = max_samples

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        with open(self.file_path, 'rb') as file_reader:
            offset = idx * NUM_SAMPLE_LENGTH
            file_reader.seek(offset)
            data = np.frombuffer(file_reader.read(NUM_SAMPLE_LENGTH), dtype=np.uint8)
            image = data[:4096].astype(np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
            qp = np.random.choice(SELECT_QP_LIST, size=1)[0]
            label = np.zeros((NUM_LABEL_BYTES,))
            qp_index = int(qp)
            label[:] = data[4160 + qp_index * NUM_LABEL_BYTES: 4160 + (qp_index + 1) * NUM_LABEL_BYTES]
            ctu_tensor = torch.from_numpy(image).float().permute(2, 0, 1) # [C, H, W]
            qp_tensor = torch.tensor(float(qp), dtype=torch.float32)
            # Normalize
            ctu_tensor /= 255.0
            qp_tensor /= 51.0
            # --- Label Processing ---
            y_image = torch.tensor(label, dtype=torch.float32).view(1, 4, 4)
            y_image_16 = F.relu(y_image - 2)
            avg_pool_result = F.avg_pool2d(y_image, kernel_size=2)
            y_image_32 = F.relu(avg_pool_result - 1) - F.relu(avg_pool_result - 2)
            avg_pool_result_4 = F.avg_pool2d(y_image, kernel_size=4)
            y_image_64 = F.relu(avg_pool_result_4 - 0) - F.relu(avg_pool_result_4 - 1)
            y_image_valid_32 = F.relu(avg_pool_result - 0) - F.relu(avg_pool_result - 1)
            y_image_valid_16 = F.relu(y_image - 1) - F.relu(y_image - 2)
            y_flat_16 = y_image_16.view(-1)
            y_flat_32 = y_image_32.view(-1)
            y_flat_64 = y_image_64.view(-1)
            y_flat_valid_32 = y_image_valid_32.view(-1)
            y_flat_valid_16 = y_image_valid_16.view(-1)
            target = torch.cat((y_flat_64, y_flat_32, y_flat_16), dim=0)
            return qp_tensor, ctu_tensor, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target

# ==================== Device Setup ====================
device = torch.device("cpu")
print(f"Using device: {device}")

# ==================== Model Instantiation ====================

# We will test the smallest model, EfficientViT_m0_TINY
model_cfg = EfficientViT_m0_TINY

# Get ffn_exp_ratio, default to 2.0 if not in config
ffn_ratio = model_cfg.get('ffn_exp_ratio', 2.0)

print(f"Instantiating model: EfficientViT-M0-TINY (QP-Aware)")
model = EfficientViT(
    img_size=IMAGE_SIZE,          
    patch_size=16,                
    in_chans=NUM_CHANNELS,      
    num_classes=NUM_CLASSES,    
    embed_dim=model_cfg['embed_dim'],
    key_dim=[16, 16, 16], # Default key dim
    depth=model_cfg['depth'],
    num_heads=model_cfg['num_heads'],
    window_size=model_cfg['window_size'],
    kernels=model_cfg['kernels'],
    ffn_exp_ratio=ffn_ratio, # Pass the expansion ratio
).to(device)


def count_parameters(model):
    """Helper function to count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

count_parameters(model)

# ==================== INFERENCE TIMING SECTION (SINGLE) ====================
print("\n--- Preparing for Single Sample Inference Time Measurement ---")
print("INFO: Timing will be performed on an untrained model.")

# 1. Set the model to evaluation mode
# This is crucial as it disables dropout and uses running stats for BatchNorm.
model.eval()

# 2. Create dummy input tensors to simulate a single prediction (Batch Size = 1)
#    The model's forward pass is: forward(self, x, qp)
dummy_image = torch.randn(1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
# Simulate a QP of 32, normalized by 51.0 (as done in training)
dummy_qp = torch.tensor([32.0 / 51.0]).to(device) 

# 3. Perform a "warm-up" run.
# This initializes CUDA kernels and ensures that the first
# timed run is not slower due to one-time setup costs.
print("Performing a warm-up run...")
with torch.no_grad(): # Disable gradient calculation for inference
    _ = model(dummy_image, dummy_qp)

# If using GPU, synchronize to ensure the warm-up run is complete
if device.type == 'cuda':
    torch.cuda.synchronize()

print("\n--- Starting Single Sample Inference Time Measurement ---")

# Synchronize before starting the timer for accurate measurement
if device.type == 'cuda':
    torch.cuda.synchronize()

# 4. Record start times
start_cpu = os.times()   # CPU time (user + system)
start_real = time.time() # Real "wall clock" time

# 5. Run inference
with torch.no_grad():
    output = model(dummy_image, dummy_qp)

# 6. Synchronize again to ensure the forward pass is complete before stopping the timer
if device.type == 'cuda':
    torch.cuda.synchronize()

# 7. Record end times
end_real = time.time()
end_cpu = os.times()

# 8. Calculate and print the results
real_time = (end_real - start_real) * 1000   # in milliseconds
user_time = (end_cpu.user - start_cpu.user) * 1000 # in milliseconds
system_time = (end_cpu.system - start_cpu.system) * 1000 # in milliseconds

print(f"Inference Time (Forward Pass, B=1):")
print(f"   - Real Time:      {real_time:.4f} ms")
print(f"   - User Time:      {user_time:.4f} ms")
print(f"   - System Time: {system_time:.4f} ms")
print("-----------------------------------------\n")

# ==================== FLOPs Calculation Section (thop) ====================
print("\n--- Calculating FLOPs using 'thop' ---")
# 'thop' requires inputs to be passed as a tuple or list
# Our model's forward signature is (x, qp), so we pass (dummy_image, dummy_qp)
inputs_tuple = (dummy_image, dummy_qp) 
macs, params = profile(model, inputs=inputs_tuple, verbose=False)

# Convert MACs (Multiply-Accumulate Operations) to GFLOPs
# FLOPs are roughly 2 * MACs
gflops = (macs * 2) / 1e9

print(f"Model Parameters (from thop): {params:,.0f}")
print(f"MACs (from thop): {macs:,.0f}")
print(f"GFLOPs (estimated, 2*MACs): {gflops:.4f} G")

# ==================== FLOPs Calculation Section (ptflops) ====================
print("\n--- Calculating FLOPs using 'ptflops' ---")

# 'ptflops' is often more robust for complex models (like transformers)
# and can require a custom 'input_constructor' to handle
# models with non-standard input signatures.

def input_constructor(input_res):
    """
    Creates dummy inputs for ptflops calculation.
    Must match the model's forward(self, x, qp) signature.
    The inputs must be returned as a dictionary.
    """
    B = 1 # Batch size of 1 for calculation
    C, H, W = input_res
    # Create the 'x' tensor
    dummy_image = torch.randn(B, C, H, W).to(device)
    # Create the 'qp' tensor
    dummy_qp = torch.tensor([32.0 / 51.0] * B).to(device) # Example: QP 32
    # Return as a dictionary mapping argument names to tensors
    return {'x': dummy_image, 'qp': dummy_qp}

# Define the resolution of the main 'x' tensor
input_res = (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE) # (1, 64, 64)

macs, params = get_model_complexity_info(
    model,
    input_res, # Pass the resolution of the 'x' tensor
    as_strings=False,
    print_per_layer_stat=False,
    verbose=False,
    input_constructor=input_constructor, # Use our custom constructor
    backend='aten' # 'aten' backend is generally needed for transformer ops
)

gflops = (macs * 2) / 1e9

print(f"Model Parameters (from ptflops): {params:,.0f}")
print(f"MACs (from ptflops): {macs:,.0f}")
print(f"GFLOPs (estimated, 2*MACs): {gflops:.4f} G")

# ==================== BATCH INFERENCE TIMING (AVERAGE PER SAMPLE) ====================
print("\n--- Preparing for Batch Inference Time Measurement ---")

n_batch_size = BATCH_SIZE 
print(f"Using batch size (n): {n_batch_size}")

# 1. Create dummy batch inputs (Batch Size = n_batch_size)
dummy_batch_image = torch.randn(n_batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
dummy_batch_qp = torch.tensor([32.0 / 51.0] * n_batch_size).to(device)

# 2. Perform a "warm-up" run for the batch
print("Performing a batch warm-up run...")
with torch.no_grad():
    _ = model(dummy_batch_image, dummy_batch_qp)

# 3. Synchronize
if device.type == 'cuda':
    torch.cuda.synchronize()

print("\n--- Starting Batch Inference Time Measurement ---")

# 4. Start timers
if device.type == 'cuda':
    torch.cuda.synchronize()

start_cpu_batch = os.times()
start_real_batch = time.time()

# 5. Run batch inference
with torch.no_grad():
    output_batch = model(dummy_batch_image, dummy_batch_qp)

# 6. Synchronize
if device.type == 'cuda':
    torch.cuda.synchronize()

end_real_batch = time.time()
end_cpu_batch = os.times()

# 7. Calculate and print the results

# Total time for the entire batch
batch_real_time_ms = (end_real_batch - start_real_batch) * 1000 
batch_user_time_ms = (end_cpu_batch.user - start_cpu_batch.user) * 1000
batch_system_time_ms = (end_cpu_batch.system - start_cpu_batch.system) * 1000
batch_cpu_total_ms = batch_user_time_ms + batch_system_time_ms

print(f"Total Batch Inference Time (for {n_batch_size} samples):")
print(f"   - Real Time:              {batch_real_time_ms:.4f} ms")
print(f"   - CPU (User+System) Time: {batch_cpu_total_ms:.4f} ms")

# Average time per sample (Total Time / Batch Size)
# This is a more realistic measure of throughput.
avg_real_time_ms = batch_real_time_ms / n_batch_size
avg_cpu_time_ms = batch_cpu_total_ms / n_batch_size

print(f"\nAverage Per-Sample Inference Time (in batch of {n_batch_size}):")
print(f"   - Avg Real Time:              {avg_real_time_ms:.4f} ms/sample")
print(f"   - Avg CPU (User+System) Time: {avg_cpu_time_ms:.4f} ms/sample")
print("-----------------------------------------\n")

# Exit the script
exit()