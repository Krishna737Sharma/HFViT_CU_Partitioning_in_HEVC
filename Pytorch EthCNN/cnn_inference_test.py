import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from thop import profile
from ptflops import get_model_complexity_info

#torch.set_num_threads(1)
# ==================== Constants ====================
IMAGE_SIZE = 64
NUM_CHANNELS = 1
BATCH_SIZE = 64

# ==================== Helper Functions (from ETH_CNN.py) ====================

def norm_batch_ctu(ctu_batch):
    """
    Function to normalize ctu at all 3 branches.
    Takes a batch of CTUs [B, H, W]
    """
    # Convert the input to a tensor with float32 dtype
    ctu_data = ctu_batch.clone().detach().float()

    # Check the number of dimensions
    if ctu_data.dim() == 2:
        # If the tensor is 2D (single sample), add a batch dimension
        ctu_data = ctu_data.unsqueeze(0)  # Shape: [1, 64, 64]
        batch_size = 1
    else:
        # If the tensor is already 3D, extract batch size
        batch_size = ctu_data.size(0)  # Shape: [batch_size, 64, 64]

    # Clone the CTU data for different branches
    norm_ctu_data_b1 = ctu_data.clone()
    norm_ctu_data_b2 = ctu_data.clone()
    norm_ctu_data_b3 = ctu_data.clone()

    # Branch B1: Mean removal at the level of the whole CTU (64x64)
    mean_value_level1 = torch.mean(ctu_data[:, 0:64, 0:64], dim=(1, 2), keepdim=True)
    norm_ctu_data_b1 -= mean_value_level1

    # Branch B2: Mean removal at the level of 32x32 blocks
    mean_value_level2_1 = torch.mean(ctu_data[:, 0:32, 0:32], dim=(1, 2), keepdim=True)
    mean_value_level2_2 = torch.mean(ctu_data[:, 0:32, 32:64], dim=(1, 2), keepdim=True)
    mean_value_level2_3 = torch.mean(ctu_data[:, 32:64, 0:32], dim=(1, 2), keepdim=True)
    mean_value_level2_4 = torch.mean(ctu_data[:, 32:64, 32:64], dim=(1, 2), keepdim=True)
    norm_ctu_data_b2[:, 0:32, 0:32]   -= mean_value_level2_1
    norm_ctu_data_b2[:, 0:32, 32:64]  -= mean_value_level2_2
    norm_ctu_data_b2[:, 32:64, 0:32]  -= mean_value_level2_3
    norm_ctu_data_b2[:, 32:64, 32:64] -= mean_value_level2_4

    # Branch B3: Mean removal at the level of 16x16 blocks
    # Note: This loop is not ideal for batch processing but matches the original script.
    # For performance, this could be vectorized.
    for i in range(0, 64, 16):
        mean_value_level3_1 = torch.mean(ctu_data[:, i:i+16, 0:16], dim=(1, 2), keepdim=True)
        mean_value_level3_2 = torch.mean(ctu_data[:, i:i+16, 16:32], dim=(1, 2), keepdim=True)
        mean_value_level3_3 = torch.mean(ctu_data[:, i:i+16, 32:48], dim=(1, 2), keepdim=True)
        mean_value_level3_4 = torch.mean(ctu_data[:, i:i+16, 48:64], dim=(1, 2), keepdim=True)

        norm_ctu_data_b3[:, i:i+16, 0:16]  -= mean_value_level3_1
        norm_ctu_data_b3[:, i:i+16, 16:32] -= mean_value_level3_2
        norm_ctu_data_b3[:, i:i+16, 32:48] -= mean_value_level3_3
        norm_ctu_data_b3[:, i:i+16, 48:64] -= mean_value_level3_4

    # If the input was originally 2D, remove the batch dimension from the output
    if batch_size == 1 and ctu_batch.dim() == 2:
        norm_ctu_data_b1 = norm_ctu_data_b1.squeeze(0)
        norm_ctu_data_b2 = norm_ctu_data_b2.squeeze(0)
        norm_ctu_data_b3 = norm_ctu_data_b3.squeeze(0)

    return norm_ctu_data_b1, norm_ctu_data_b2, norm_ctu_data_b3

def mean_downsample(tensor, scale_factor):
    """
    Downsamples a batch of 3D tensors [B, H, W]
    """
    if tensor.dim() != 3:
        # Add batch dim if a single 2D tensor is passed
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        else:
            raise ValueError(f"Input tensor must be 3D (batch, height, width) or 2D (height, width), but got {tensor.dim()}D")

    batch_size, h, w = tensor.shape
    new_h, new_w = h // scale_factor, w // scale_factor

    # Unfolding and downsampling applied to each tensor in the batch
    downsampled_tensor = tensor.unfold(1, scale_factor, scale_factor).unfold(2, scale_factor, scale_factor)
    downsampled_tensor = downsampled_tensor.contiguous().view(batch_size, new_h, new_w, -1)
    downsampled_tensor = downsampled_tensor.mean(dim=-1)

    return downsampled_tensor

def downsample_ctu_3_branches(norm_ctu_tuple):
    # extracting ctu from normalized ctu tuple
    branch1_ctu = norm_ctu_tuple[0]  # Tensor with shape [batch_size, 64, 64]
    branch2_ctu = norm_ctu_tuple[1]  # Tensor with shape [batch_size, 64, 64]
    branch3_ctu = norm_ctu_tuple[2]  # Tensor with shape [batch_size, 64, 64]

    # Branch 1 downsampling
    downsampled_ctu_16_16 = mean_downsample(branch1_ctu, 4)  # Downsampling to [batch_size, 16, 16]
    downsampled_ctu_32_32 = mean_downsample(branch2_ctu, 2)  # Downsampling to [batch_size, 32, 32]
    downsampled_ctu_64_64 = mean_downsample(branch3_ctu, 1)  # No downsampling (same size)

    return (downsampled_ctu_16_16, downsampled_ctu_32_32, downsampled_ctu_64_64)

# ==================== ETH_CNN Model Definition ====================

class ETH_CNN(nn.Module):
    def __init__(self):
        super(ETH_CNN, self).__init__()
        
        # Branch 1 convolution layers using nn.Conv2d
        self.conv1_b1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=4, padding=0)
        self.conv2_b1 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=2, stride=2, padding=0)
        self.conv3_b1 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=2, stride=2, padding=0)

         # Branch 2 convolution layers using nn.Conv2d
        self.conv1_b2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=4, padding=0)
        self.conv2_b2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=2, stride=2, padding=0)
        self.conv3_b2 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=2, stride=2, padding=0)

        # Branch 3 convolution layers using nn.Conv2d
        self.conv1_b3 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=4, padding=0)
        self.conv2_b3 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=2, stride=2, padding=0)
        self.conv3_b3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=2, stride=2, padding=0)


        # Fully connected Layers
        self.fc1_dropout = nn.Dropout(p=0.5)  # Set the dropout rate to the desired value
        self.fc2_dropout = nn.Dropout(p=0.2)  # Set the dropout rate to the desired value

        # branch 1
        self.fc1_b1 = nn.Linear(in_features=2688, out_features=64)
        self.fc2_b1 = nn.Linear(in_features=65, out_features=48)
        self.fc3_b1 = nn.Linear(in_features=49, out_features=1)

        # branch 2
        self.fc1_b2 = nn.Linear(in_features=2688, out_features=128)
        self.fc2_b2 = nn.Linear(in_features=129, out_features=96) 
        self.fc3_b2 = nn.Linear(in_features=97, out_features=4)  

        # branch 3
        self.fc1_b3 = nn.Linear(in_features=2688, out_features=256) 
        self.fc2_b3 = nn.Linear(in_features=257, out_features=192) 
        self.fc3_b3 = nn.Linear(in_features=193, out_features=16)  

    # Fully Connected Layer
    # Branch 1
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
    
    # Branch 2
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
    
    # Branch 3
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
    
    def forward(self, qp, original_ctu):
        """
        Modified forward pass to accept qp and ctu as separate arguments
        for easier profiling.
        qp: [B] QP scalar per sample (normalized)
        original_ctu: [B, H, W] image tensor (normalized)
        """
        
        x = norm_batch_ctu(original_ctu)

        x = downsample_ctu_3_branches(x)
        
        # x[0] shape: [B, 16, 16]
        # x[1] shape: [B, 32, 32]
        # x[2] shape: [B, 64, 64]
        
        # Add channel dimension for conv layers
        x0 = x[0].unsqueeze(1) # [B, 1, 16, 16]
        x1 = x[1].unsqueeze(1) # [B, 1, 32, 32]
        x2 = x[2].unsqueeze(1) # [B, 1, 64, 64]

        # Branch 1
        h_conv1_b1_op = F.leaky_relu(self.conv1_b1(x0))
        h_conv2_b1_op = F.leaky_relu(self.conv2_b1(h_conv1_b1_op))
        h_conv3_b1_op = F.leaky_relu(self.conv3_b1(h_conv2_b1_op))

        # Branch 2
        h_conv1_b2_op = F.leaky_relu(self.conv1_b2(x1))
        h_conv2_b2_op = F.leaky_relu(self.conv2_b2(h_conv1_b2_op))
        h_conv3_b2_op = F.leaky_relu(self.conv3_b2(h_conv2_b2_op))

        # Branch 3
        h_conv1_b3_op = F.leaky_relu(self.conv1_b3(x2))
        h_conv2_b3_op = F.leaky_relu(self.conv2_b3(h_conv1_b3_op))
        h_conv3_b3_op = F.leaky_relu(self.conv3_b3(h_conv2_b3_op))
       

        # Flatten and concatenate outputs
        reshaped_conv3_b3_op = h_conv3_b3_op.view(-1, 32 * 4 * 4) # 512
        reshaped_conv3_b2_op = h_conv3_b2_op.view(-1, 32 * 2 * 2) # 128
        reshaped_conv3_b1_op = h_conv3_b1_op.view(-1, 32 * 1 * 1) # 32
        reshaped_conv2_b3_op = h_conv2_b3_op.view(-1, 24 * 8 * 8) # 1536
        reshaped_conv2_b2_op = h_conv2_b2_op.view(-1, 24 * 4 * 4) # 384
        reshaped_conv2_b1_op = h_conv2_b1_op.view(-1, 24 * 2 * 2) # 96
        
        # Total features = 512 + 128 + 32 + 1536 + 384 + 96 = 2688
        concatenated_output = torch.cat((
            reshaped_conv3_b1_op, reshaped_conv2_b1_op, 
            reshaped_conv3_b2_op, reshaped_conv2_b2_op, 
            reshaped_conv3_b3_op, reshaped_conv2_b3_op
        ), dim=1)

        b1_op = self.full_connect_b1(concatenated_output, qp)
        b2_op = self.full_connect_b2(concatenated_output, qp)
        b3_op = self.full_connect_b3(concatenated_output, qp)
        
        # Squeeze outputs to match original script's output shape [B, 1], [B, 4], [B, 16]
        return (b1_op.squeeze(dim=1), b2_op, b3_op)


# ==================== Main Execution ====================

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Instantiate the model
    model = ETH_CNN().to(device)
    
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

    count_parameters(model)

    # ==================== INFERENCE TIMING SECTION (SINGLE) ====================
    print("\n--- Preparing for Inference Time Measurement ---")
    
    print("INFO: Timing will be performed on an untrained model.")

    # 2. Set the model to evaluation mode
    model.eval()

    # 3. Create a dummy input tensor and QP to simulate a single prediction
    # Input for ETH_CNN.forward(qp, original_ctu)
    # qp: [B], original_ctu: [B, H, W]
    dummy_image = torch.randn(1, IMAGE_SIZE, IMAGE_SIZE).to(device) # [1, 64, 64]
    dummy_qp = torch.tensor([32.0 / 51.0]).to(device) # Example: QP 32, normalized [1]

    # 4. Perform a "warm-up" run
    print("Performing a warm-up run...")
    with torch.no_grad():
        _ = model(dummy_qp, dummy_image)

    print("\n--- Starting Inference Time Measurement ---")

    # 5. Start timers
    start_cpu = os.times()
    start_real = time.time()

    with torch.no_grad():
        output = model(dummy_qp, dummy_image)

    end_real = time.time()
    end_cpu = os.times()

    # 6. Calculate and print the results
    real_time = (end_real - start_real) * 1000  # in milliseconds
    user_time = (end_cpu.user - start_cpu.user) * 1000 # in milliseconds
    system_time = (end_cpu.system - start_cpu.system) * 1000 # in milliseconds

    print(f"Inference Time (Forward Pass):")
    print(f"- Real Time:{real_time:.4f} ms")
    print(f"- User Time:{user_time:.4f} ms")
    print(f"- System Time:{system_time:.4f} ms")
    print("-----------------------------------------\n")
    
    # ==================== FLOPs Calculation Section ====================
    print("\n--- Calculating FLOPs using thop ---")
    # thop inputs must be a tuple matching the args to forward()
    macs, params = profile(model, inputs=(dummy_qp, dummy_image), verbose=False)

    # Convert MACs to GFLOPs (FLOPs ≈ 2 * MACs)
    gflops = (macs * 2) / 1e9

    print(f"Model Parameters: {params:,}")
    print(f"MACs: {macs:,}")
    print(f"GFLOPs (estimated): {gflops:.4f} G")

    print("\n--- Calculating FLOPs using ptflops ---")

    def input_constructor(input_res):
        # input_res is ignored, but required by the function signature.
        # We create all inputs needed by the model's forward pass.
        B = 1 # Batch size of 1 for calculation
        # Create dummy inputs on the correct device, matching forward(self, qp, original_ctu)
        dummy_image = torch.randn(B, IMAGE_SIZE, IMAGE_SIZE).to(device)
        dummy_qp = torch.tensor([32.0 / 51.0] * B).to(device) # Example: QP 32, normalized
        # Return as a dictionary of keyword arguments
        return {'qp': dummy_qp, 'original_ctu': dummy_image}

    # Define a dummy input resolution. It will be passed to input_constructor.
    # We'll use (C, H, W) like the ViT script for consistency, 
    # even though our constructor will ignore C.
    input_res_dummy = (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE) # (1, 64, 64)

    macs, params = get_model_complexity_info(
        model,
        input_res_dummy,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
        input_constructor=input_constructor,
        backend='aten' # Use 'aten' backend for custom ops / control flow
    )

    # Convert MACs to GFLOPs (FLOPs ≈ 2 * MACs)
    gflops = (macs * 2) / 1e9

    print(f"Model Parameters: {params:,}")
    print(f"MACs: {macs:,}")
    print(f"GFLOPs (estimated): {gflops:.4f} G")

    # ==================== BATCH INFERENCE TIMING (AVERAGE PER SAMPLE) ====================
    print("\n--- Preparing for Batch Inference Time Measurement ---")

    n_batch_size = BATCH_SIZE 
    print(f"Using batch size (n): {n_batch_size}")

    # 1. Create dummy batch inputs
    dummy_batch_image = torch.randn(n_batch_size, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_batch_qp = torch.tensor([32.0 / 51.0] * n_batch_size).to(device)

    # 2. Perform a "warm-up" run for the batch
    print("Performing a batch warm-up run...")
    with torch.no_grad():
        _ = model(dummy_batch_qp, dummy_batch_image)

    print("\n--- Starting Batch Inference Time Measurement ---")

    # 4. Start timers
    start_cpu_batch = os.times()
    start_real_batch = time.time()
    n_iterations = 10

    # 5. Run inference
    with torch.no_grad():
        for _ in range(n_iterations):
            output_batch = model(dummy_batch_qp, dummy_batch_image)

    end_real_batch = time.time()
    end_cpu_batch = os.times()

    # 7. Calculate and print the results

    # Total time for the batch
    batch_real_time_ms = (end_real_batch - start_real_batch) * 1000 / n_iterations
    batch_user_time_ms = (end_cpu_batch.user - start_cpu_batch.user) * 1000 / n_iterations
    batch_system_time_ms = (end_cpu_batch.system - start_cpu_batch.system) * 1000 / n_iterations
    batch_cpu_total_ms = batch_user_time_ms + batch_system_time_ms

    print(f"Total Batch Inference Time (for {n_batch_size} samples):")
    print(f"- Real Time:{batch_real_time_ms:.4f} ms")
    print(f"- CPU (User+System) Time: {batch_cpu_total_ms:.4f} ms")

    # Average time per sample in the batch
    avg_real_time_ms = batch_real_time_ms / n_batch_size
    avg_cpu_time_ms = batch_cpu_total_ms / n_batch_size # This is (user+system) / n

    print(f"\nAverage Per-Sample Inference Time (in batch of {n_batch_size}):")
    print(f"- Avg Real Time:{avg_real_time_ms:.4f} ms/sample")
    print(f"- Avg CPU (User+System) Time: {avg_cpu_time_ms:.4f} ms/sample")
    print("-----------------------------------------\n")
    
    # Exit the script after the timing measurement is complete
    exit()
