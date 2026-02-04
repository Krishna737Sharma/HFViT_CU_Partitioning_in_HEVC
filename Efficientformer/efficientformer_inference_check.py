import torch
import torch.nn as nn
import time
import os
import numpy as np
import math # Needed for resolution calculation if used

torch.set_num_threads(1)
# Make sure timm, thop, ptflops are installed: pip install timm thop ptflops
from thop import profile
from ptflops import get_model_complexity_info

# Import your modified EfficientFormerV2 model for CTU
from efficientformer_v2_ctu import efficientformerv2_1M_ctu # Choose the variant

# ==================== Constants ====================
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_CLASSES = 21 # Number of outputs for your CTU task
BATCH_SIZE = 512 # Match your training batch size or choose another

# ==================== Model Setup ====================
device = torch.device("cpu") # Use CPU for consistent timing
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model configured for the CTU task
print("Instantiating EfficientFormerV2 CTU model...")
# Note: Defaults for in_chans, num_classes, resolution are already set in efficientformer_v2_ctu.py
model = efficientformerv2_1M_ctu(pretrained=False).to(device)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

count_parameters(model)

# ==================== INFERENCE TIMING SECTION ====================
print("\n--- Preparing for Inference Time Measurement ---")

# Optional: Load checkpoint if needed
# checkpoint_path = 'best_efficientformerv2_s0_ctu_model.pth'
# if os.path.exists(checkpoint_path):
#     print(f"Loading checkpoint from {checkpoint_path}...")
#     # ... checkpoint loading logic ...
# else:
#     print("INFO: Timing will be performed on an untrained model.")

print("INFO: Timing will be performed on an untrained model.")

# Set model to eval mode
model.eval()

# Create dummy inputs for a single prediction
dummy_image = torch.randn(1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
dummy_qp = torch.tensor([32.0 / 51.0]).to(device) # Example: QP 32, normalized

# Warm-up run
print("Performing a warm-up run...")
with torch.no_grad():
    _ = model(dummy_image, dummy_qp)

# Synchronize if using CUDA
if device.type == 'cuda':
    torch.cuda.synchronize()

print("\n--- Starting Single Inference Time Measurement ---")

# Synchronize before timing
if device.type == 'cuda':
    torch.cuda.synchronize()

start_cpu = os.times()
start_real = time.time()

# Perform inference
with torch.no_grad():
    output = model(dummy_image, dummy_qp)

# Synchronize after inference
if device.type == 'cuda':
    torch.cuda.synchronize()

end_real = time.time()
end_cpu = os.times()

# Calculate and print results
real_time_ms = (end_real - start_real) * 1000
user_time_ms = (end_cpu.user - start_cpu.user) * 1000
system_time_ms = (end_cpu.system - start_cpu.system) * 1000

print(f"Single Inference Time (Forward Pass):")
print(f"  - Real Time:    {real_time_ms:.4f} ms")
print(f"  - User Time:    {user_time_ms:.4f} ms")
print(f"  - System Time: {system_time_ms:.4f} ms")
print("-----------------------------------------\n")

# ==================== FLOPs Calculation Section ====================
print("\n--- Calculating FLOPs/MACs ---")

# Using thop
print("Calculating with thop...")
try:
    macs_thop, params_thop = profile(model, inputs=(dummy_image, dummy_qp), verbose=False)
    gflops_thop = (macs_thop * 2) / 1e9
    print(f"  thop:")
    print(f"    - Params: {params_thop:,.0f}")
    print(f"    - MACs:   {macs_thop:,.0f}")
    print(f"    - GFLOPs (est.): {gflops_thop:.4f} G")
except Exception as e:
    print(f"  thop failed: {e}")
    print("    - thop might not support all custom operations or tensor manipulations.")


# Using ptflops
print("\nCalculating with ptflops...")

def input_constructor_ptflops(input_res):
    """ Required for ptflops with non-standard inputs """
    B = 1 # Batch size 1 for calculation
    C, H, W = input_res
    dummy_image_pt = torch.randn(B, C, H, W).to(device)
    dummy_qp_pt = torch.tensor([32.0 / 51.0] * B).to(device)
    # Must return a dictionary matching the forward signature's argument names: 'x', 'qp'
    return {'x': dummy_image_pt, 'qp': dummy_qp_pt}

input_resolution_ptflops = (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

try:
    macs_pt, params_pt = get_model_complexity_info(
        model,
        input_resolution_ptflops,
        input_constructor=input_constructor_ptflops,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
        # EfficientFormerV2 uses primarily Conv and standard ops, 'fx' might work better if 'aten' fails
        backend='aten' # or try 'fx'
    )
    gflops_pt = (macs_pt * 2) / 1e9
    print(f"  ptflops:")
    print(f"    - Params: {params_pt:,.0f}")
    print(f"    - MACs:   {macs_pt:,.0f}")
    print(f"    - GFLOPs (est.): {gflops_pt:.4f} G")
except Exception as e:
    print(f"  ptflops failed: {e}")
    print("    - ptflops might not support all operations in this model.")

print("-----------------------------------------\n")


# ==================== BATCH INFERENCE TIMING (AVERAGE PER SAMPLE) ====================
print("\n--- Preparing for Batch Inference Time Measurement ---")

n_batch_size = BATCH_SIZE
print(f"Using batch size (n): {n_batch_size}")

# Create dummy batch inputs
dummy_batch_image = torch.randn(n_batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
dummy_batch_qp = torch.tensor([32.0 / 51.0] * n_batch_size).to(device)

# Batch warm-up run
print("Performing a batch warm-up run...")
with torch.no_grad():
    _ = model(dummy_batch_image, dummy_batch_qp)

# Synchronize
if device.type == 'cuda':
    torch.cuda.synchronize()

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
