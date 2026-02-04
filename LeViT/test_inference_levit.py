import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
import os
import time

torch.set_num_threads(1)
# FLOPs/Inference libraries
from thop import profile
from ptflops import get_model_complexity_info

# --- LeViT MODEL IMPORTS ---
# Humara naya model aur uski config import karein
from levit_ctu import LeViT_CTU, LeViT_64T_CTU_config, LeViT_96S_CTU_config, LeViT_128S_CTU_config # Using 128S as default

# Global debug flag
DEBUG = False

# ==================== Constants ====================
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_CLASSES = 21 # 1 + 4 + 16 splits
BATCH_SIZE = 512

# --- Dummy Constants (Not used but kept for context) ---
NUM_LABEL_BYTES = 16
NUM_SAMPLE_LENGTH = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + 64 + (51 + 1) * NUM_LABEL_BYTES
SELECT_QP_LIST = [22, 27, 32, 37]


# ==================== Device Setup ====================
device = torch.device("cpu")
print(f"Using device: {device}")

# ==================== Model Instantiation ====================

# Hum M0 model (sabse chota) ko test karenge
model_cfg = LeViT_64T_CTU_config
arch_name = "LeViT-64T-CTU"

print(f"Creating model: {arch_name}")
model = LeViT_CTU(
    img_size=IMAGE_SIZE,
    in_chans=NUM_CHANNELS,
    num_classes=NUM_CLASSES,
    **model_cfg # Pass the chosen config dictionary
).to(device)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

count_parameters(model)

# ==================== INFERENCE TIMING SECTION (SINGLE) ====================
print("\n--- Preparing for Inference Time Measurement ---")

print("INFO: Timing will be performed on an untrained model.")

# 1. Set the model to evaluation mode
model.eval()

# 2. Create a dummy input tensor and QP to simulate a single prediction
dummy_image = torch.randn(1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
dummy_qp = torch.tensor([32.0 / 51.0]).to(device) # Example: QP 32, normalized

# 3. Perform a "warm-up" run to initialize operations
print("Performing a warm-up run...")
with torch.no_grad():
    _ = model(dummy_image, dummy_qp)

if device.type == 'cuda':
    torch.cuda.synchronize()

print("\n--- Starting Inference Time Measurement ---")

# Synchronize before starting the timer
if device.type == 'cuda':
    torch.cuda.synchronize()

start_cpu = os.times()
start_real = time.time()

# 4. Run inference
with torch.no_grad():
    output = model(dummy_image, dummy_qp)

# Synchronize again to make sure the model forward pass is complete
if device.type == 'cuda':
    torch.cuda.synchronize()

end_real = time.time()
end_cpu = os.times()

# 5. Calculate and print the results
real_time = (end_real - start_real) * 1000   # in milliseconds
user_time = (end_cpu.user - start_cpu.user) * 1000 # in milliseconds
system_time = (end_cpu.system - start_cpu.system) * 1000 # in milliseconds

print(f"Inference Time (Forward Pass):")
print(f"   - Real Time:      {real_time:.4f} ms")
print(f"   - User Time:      {user_time:.4f} ms")
print(f"   - System Time: {system_time:.4f} ms")
print("-----------------------------------------\n")

# ==================== FLOPs Calculation Section ====================
print("\n--- Calculating FLOPs using thop ---")
# thop ko inputs (tuple) mein pass karna hota hai
try:
    macs_thop, params_thop = profile(model, inputs=(dummy_image, dummy_qp), verbose=False)
    gflops_thop = (macs_thop * 2) / 1e9
    print(f"(thop) Model Parameters: {params_thop:,}")
    print(f"(thop) MACs: {macs_thop:,}")
    print(f"(thop) GFLOPs (estimated): {gflops_thop:.4f} G")
except Exception as e:
    print(f"(thop) Error calculating FLOPs: {e}")


print("\n--- Calculating FLOPs using ptflops ---")

# ptflops ke liye input_constructor
def input_constructor(input_res):
    B = 1 # Batch size of 1 for calculation
    C, H, W = input_res
    # forward(self, x, qp) signature ke liye inputs
    img = torch.randn(B, C, H, W).to(device)
    qp_val = torch.tensor([32.0 / 51.0] * B).to(device) # Example: QP 32
    return {'x': img, 'qp': qp_val}

# 'x' tensor ka resolution
input_res_ptflops = (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE) # (1, 64, 64)

try:
    macs_pt, params_pt = get_model_complexity_info(
        model,
        input_res_ptflops,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
        input_constructor=input_constructor,
        # backend='aten' # LeViT might work better without forcing aten sometimes
    )
    gflops_pt = (macs_pt * 2) / 1e9
    print(f"(ptflops) Model Parameters: {params_pt:,}")
    print(f"(ptflops) MACs: {macs_pt:,}")
    print(f"(ptflops) GFLOPs (estimated): {gflops_pt:.4f} G")
except Exception as e:
    print(f"(ptflops) Error calculating FLOPs: {e}")


# ==================== BATCH INFERENCE TIMING (AVERAGE PER SAMPLE) ====================
print("\n--- Preparing for Batch Inference Time Measurement ---")

n_batch_size = BATCH_SIZE
print(f"Using batch size (n): {n_batch_size}")

# 1. Create dummy batch inputs
dummy_batch_image = torch.randn(n_batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
dummy_batch_qp = torch.tensor([32.0 / 51.0] * n_batch_size).to(device)

# 2. Perform a "warm-up" run for the batch
print("Performing a batch warm-up run...")
with torch.no_grad():
    _ = model(dummy_batch_image, dummy_batch_qp)

print("\n--- Starting Batch Inference Time Measurement ---")

n_iterations = 10
start_cpu_batch = os.times()
start_real_batch = time.time()

with torch.no_grad():
    for _ in range(n_iterations):
        output_batch = model(dummy_batch_image, dummy_batch_qp)

end_real_batch = time.time()
end_cpu_batch = os.times()

batch_real_time_ms = (end_real_batch - start_real_batch) * 1000 / n_iterations
batch_user_time_ms = (end_cpu_batch.user - start_cpu_batch.user) * 1000 / n_iterations
batch_system_time_ms = (end_cpu_batch.system - start_cpu_batch.system) * 1000 / n_iterations
batch_cpu_total_ms = batch_user_time_ms + batch_system_time_ms

print(f"Total Batch Inference Time (for {n_batch_size} samples):")
print(f"- Real Time:{batch_real_time_ms:.4f} ms")
print(f"- CPU (User+System) Time: {batch_cpu_total_ms:.4f} ms")

avg_real_time_ms = batch_real_time_ms / n_batch_size
avg_cpu_time_ms = batch_cpu_total_ms / n_batch_size

print(f"\nAverage Per-Sample Inference Time (in batch of {n_batch_size}):")
print(f"- Avg Real Time:{avg_real_time_ms:.4f} ms/sample")
print(f"- Avg CPU (User+System) Time: {avg_cpu_time_ms:.4f} ms/sample")
print("-----------------------------------------\n")

exit()