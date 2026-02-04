# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchsummary import summary # Commented out as it might not be installed, but kept import line as requested
from torch.utils.data import Dataset, DataLoader, Subset
from math import log10
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import torch.onnx
import onnxruntime
# import wandb  <-- Removed
import sys
import os
import random
from torch.utils.data import DataLoader, Subset, Dataset

# Constants
overall_least_loss = float('inf')
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_LABEL_BYTES = 16
NUM_SAMPLE_LENGTH = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + 64 + (51 + 1) * NUM_LABEL_BYTES
SELECT_QP_LIST = [22, 27, 32, 37]

# StreamingDataset class
class StreamingDataset(Dataset):
    def __init__(self, file_path, max_samples):
        self.file_path = file_path
        self.max_samples = max_samples

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        # Ensure each worker opens its own file reader
        with open(self.file_path, 'rb') as file_reader:
            # Seek to the offset for the sample
            offset = idx * NUM_SAMPLE_LENGTH
            file_reader.seek(offset)

            # Read the data for a single sample
            data = np.frombuffer(file_reader.read(NUM_SAMPLE_LENGTH), dtype=np.uint8)

            # Process image
            image = data[:4096].astype(np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

            # Process QP
            qp = np.random.choice(SELECT_QP_LIST, size=1)[0]

            # Process label
            label = np.zeros((NUM_LABEL_BYTES,))
            qp_index = int(qp)
            label[:] = data[4160 + qp_index * NUM_LABEL_BYTES: 4160 + (qp_index + 1) * NUM_LABEL_BYTES]

            # Convert image to PyTorch tensor
            ctu_tensor = torch.tensor(image, dtype=torch.float32).squeeze(2)
            qp_tensor = torch.tensor(float(qp), dtype=torch.float32)

            # Convert label to hierarchical output
            y_image = torch.tensor(label, dtype=torch.float32).view(1, 4, 4)

            # Perform hierarchical pooling
            y_image_16 = F.relu(y_image - 2)
            y_image_32 = F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 1) - \
                         F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 2)
            y_image_64 = F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=4) - 0) - \
                         F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=4) - 1)
            y_image_valid_32 = F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 0) - \
                               F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 1)
            y_image_valid_16 = F.relu(y_image - 1) - F.relu(y_image - 2)

            # Flatten the resulting tensors
            y_flat_16 = y_image_16.view(-1)
            y_flat_32 = y_image_32.view(-1)
            y_flat_64 = y_image_64.view(-1)
            y_flat_valid_32 = y_image_valid_32.view(-1)
            y_flat_valid_16 = y_image_valid_16.view(-1)

            # Normalize the image and QP tensor
            ctu_tensor /= 255.0
            qp_tensor /= 51.0

            # Return a single sample
            return qp_tensor, ctu_tensor, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, torch.tensor(label, dtype=torch.float32)


# Function to create new DataLoader with randomly selected samples
def create_subset_dataloader(file_path, total_samples, subset_size, batch_size, device, shuffle=True):
    # Use worker_init_fn to ensure random seeds are set for each worker
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    full_dataset = StreamingDataset(file_path, total_samples)
    # Ensure subset size does not exceed total samples
    real_subset_size = min(subset_size, total_samples)
    subset_indices = random.sample(range(total_samples), real_subset_size)
    
    return DataLoader(
        Subset(full_dataset, subset_indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    ), subset_indices


# File paths
train_file_path = "/raid/somdyutiai/Krishna_24AI60R38/PycharmProjects/HEVC_Intra_Models-ETH-CNN_Pt/Data/AI_Train_1668975.dat_shuffled"
validation_file_path = "/raid/somdyutiai/Krishna_24AI60R38/PycharmProjects/HEVC_Intra_Models-ETH-CNN_Pt/Data/AI_Valid_98175.dat_shuffled"
test_file_path = "/raid/somdyutiai/Krishna_24AI60R38/PycharmProjects/HEVC_Intra_Models-ETH-CNN_Pt/Data/AI_Test_196350.dat_shuffled"

# Dataset sizes
TRAINSET_MAXSIZE = 1668975
VALIDSET_MAXSIZE = 98175
TESTSET_MAXSIZE = 196350

# Batch size
BATCH_SIZE = 64

# Initial DataLoaders and indices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CHANGED: Use Full MaxSize for one-shot training ---
train_loader, train_indices = create_subset_dataloader(train_file_path, TRAINSET_MAXSIZE, TRAINSET_MAXSIZE, BATCH_SIZE, device=device, shuffle=True)
validation_loader, validation_indices = create_subset_dataloader(validation_file_path, VALIDSET_MAXSIZE, VALIDSET_MAXSIZE, BATCH_SIZE, device=device, shuffle=False)
test_loader, test_indices = create_subset_dataloader(test_file_path, TESTSET_MAXSIZE, TESTSET_MAXSIZE, BATCH_SIZE, device=device, shuffle=False)
# -------------------------------------------------------


"""
Function to normalize ctu at all 3 branches and return tuple of 3 64*64 ctu
"""

# %%
def norm_batch_ctu(ctu_batch):
    # Convert the input to a tensor with float32 dtype
    ctu_data = ctu_batch.clone().detach().float()
    
    # Check the number of dimensions
    if ctu_data.dim() == 2:
        # If the tensor is 2D, add a batch dimension
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
    mean_value_level1 = torch.mean(ctu_data[:, 0:64, 0:64], dim=(1, 2), keepdim=True)  # Compute mean for each CTU in the batch
    norm_ctu_data_b1 -= mean_value_level1  # Subtract the mean for branch B1
    
    # Branch B2: Mean removal at the level of 32x32 blocks
    mean_value_level2_1 = torch.mean(ctu_data[:, 0:32, 0:32], dim=(1, 2), keepdim=True)
    mean_value_level2_2 = torch.mean(ctu_data[:, 0:32, 32:64], dim=(1, 2), keepdim=True)
    mean_value_level2_3 = torch.mean(ctu_data[:, 32:64, 0:32], dim=(1, 2), keepdim=True)
    mean_value_level2_4 = torch.mean(ctu_data[:, 32:64, 32:64], dim=(1, 2), keepdim=True)
    # Mean removal for branch B2 (vectorized for the entire batch)
    norm_ctu_data_b2[:, 0:32, 0:32]   -= mean_value_level2_1
    norm_ctu_data_b2[:, 0:32, 32:64]  -= mean_value_level2_2
    norm_ctu_data_b2[:, 32:64, 0:32]  -= mean_value_level2_3
    norm_ctu_data_b2[:, 32:64, 32:64] -= mean_value_level2_4

    # Branch B3: Mean removal at the level of 16x16 blocks
    for i in range(0, 64, 16):
        mean_value_level3_1 = torch.mean(ctu_data[:, i:i+16, 0:16])
        mean_value_level3_2 = torch.mean(ctu_data[:, i:i+16, 16:32])
        mean_value_level3_3 = torch.mean(ctu_data[:, i:i+16, 32:48])
        mean_value_level3_4 = torch.mean(ctu_data[:, i:i+16, 48:64])

        # Mean removal for branch B3 (vectorized for the entire batch)
        norm_ctu_data_b3[:, i:i+16, 0:16]  -= mean_value_level3_1
        norm_ctu_data_b3[:, i:i+16, 16:32] -= mean_value_level3_2
        norm_ctu_data_b3[:, i:i+16, 32:48] -= mean_value_level3_3
        norm_ctu_data_b3[:, i:i+16, 48:64] -= mean_value_level3_4

    # If the input was originally 2D, remove the batch dimension from the output
    if batch_size == 1:
        norm_ctu_data_b1 = norm_ctu_data_b1.squeeze(0)  # Shape: [64, 64]
        norm_ctu_data_b2 = norm_ctu_data_b2.squeeze(0)  # Shape: [64, 64]
        norm_ctu_data_b3 = norm_ctu_data_b3.squeeze(0)  # Shape: [64, 64]

        

    # Return the normalized CTU data for all three branches
    return norm_ctu_data_b1, norm_ctu_data_b2, norm_ctu_data_b3



# %%
"""
Function to downsample ctu's after normalization
"""

# %%
# Modified mean_downsample to handle batched data
def mean_downsample(tensor, scale_factor):
    if tensor.dim() != 3:
        raise ValueError("Input tensor must be 3D (batch, height, width).")
    
    batch_size, h, w = tensor.shape
    new_h, new_w = h // scale_factor, w // scale_factor

    # Unfolding and downsampling applied to each tensor in the batch
    downsampled_tensor = tensor.unfold(1, scale_factor, scale_factor).unfold(2, scale_factor, scale_factor)
    downsampled_tensor = downsampled_tensor.contiguous().view(batch_size, new_h, new_w, -1)
    downsampled_tensor = downsampled_tensor.mean(dim=-1)
    
    return downsampled_tensor

# Modified downsample_ctu_3_branches to work with batched input
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



# %%
"""
### ETH-CNN model Architecture
"""


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
        fc3_activation_op = F.sigmoid(self.fc3_b1(qp_fc2_activation_op))
        # print("Branch 1 Fc op: ", fc3_activation_op)
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
        fc3_activation_op = F.sigmoid(self.fc3_b2(qp_fc2_activation_op))
        # print("Branch 2 Fc op: ", fc3_activation_op)
        return fc3_activation_op
    
    # Branch 3
    def full_connect_b3(self, x, qp):
        qp_tensor = qp.unsqueeze(1)

        fc1_activation_op = F.leaky_relu(self.fc1_b3(x))
        fc1_activation_op = self.fc1_dropout(fc1_activation_op)
        # print("full connected layer B3 fc1 shape:", fc1_activation_op.shape)
        
        qp_fc1_activation_op = torch.cat((fc1_activation_op, qp_tensor), dim=1)
        fc2_activation_op = F.leaky_relu(self.fc2_b3(qp_fc1_activation_op))
        fc2_activation_op = self.fc2_dropout(fc2_activation_op)
        # print("full connected layer B3 fc2 shape:", fc2_activation_op.shape)

        qp_fc2_activation_op = torch.cat((fc2_activation_op, qp_tensor), dim=1)
        fc3_activation_op = F.sigmoid(self.fc3_b3(qp_fc2_activation_op))
        # print("Branch 3 Fc op: ", fc3_activation_op)
        return fc3_activation_op
    
    def forward(self, x):
        # qp = torch.tensor([x[0]])
        qp = x[0]
        original_ctu = x[1]
        
        x = norm_batch_ctu(original_ctu)

        x = downsample_ctu_3_branches(x)

        # Branch 1
        h_conv1_b1_op = F.leaky_relu(self.conv1_b1(x[0].unsqueeze(1)))
        # print("Conv 1 Branch 1: ", h_conv1_b1_op.shape)
        h_conv2_b1_op = F.leaky_relu(self.conv2_b1(h_conv1_b1_op))
        # print("Branch 1 Conv 2 op shape: ", h_conv2_b1_op.shape)
        h_conv3_b1_op = F.leaky_relu(self.conv3_b1(h_conv2_b1_op))
        # print("Branch 1 Conv 3 op shape: ", h_conv3_b1_op.shape)

        # Branch 2
        h_conv1_b2_op = F.leaky_relu(self.conv1_b2(x[1].unsqueeze(1)))
        # print("Conv 1 Branch 2: ", h_conv1_b2_op.shape)
        h_conv2_b2_op = F.leaky_relu(self.conv2_b2(h_conv1_b2_op))
        # print("Branch 2 Conv 2 op shape: ", h_conv2_b2_op.shape)
        h_conv3_b2_op = F.leaky_relu(self.conv3_b2(h_conv2_b2_op))
        # print("Branch 2 Conv 3 op shape: ", h_conv3_b2_op.shape)

        # Branch 3
        h_conv1_b3_op = F.leaky_relu(self.conv1_b3(x[2].unsqueeze(1)))
        # print("Conv 1 Branch 3: ", h_conv1_b3_op.shape)
        h_conv2_b3_op = F.leaky_relu(self.conv2_b3(h_conv1_b3_op))
        # print("Branch 3 Conv 2 op shape: ", h_conv2_b3_op.shape)
        h_conv3_b3_op = F.leaky_relu(self.conv3_b3(h_conv2_b3_op))
        # print("Branch 3 Conv 3 op shape: ", h_conv3_b3_op.shape)
        

        # Flatten and concatenate outputs
        reshaped_conv3_b3_op = h_conv3_b3_op.view(-1, 32 * 4 * 4)
        reshaped_conv3_b2_op = h_conv3_b2_op.view(-1, 32 * 2 * 2)
        reshaped_conv3_b1_op = h_conv3_b1_op.view(-1, 32 * 1 * 1)
        reshaped_conv2_b3_op = h_conv2_b3_op.view(-1, 24 * 8 * 8)
        reshaped_conv2_b2_op = h_conv2_b2_op.view(-1, 24 * 4 * 4)
        reshaped_conv2_b1_op = h_conv2_b1_op.view(-1, 24 * 2 * 2)

        concatenated_output = torch.cat((
            reshaped_conv3_b1_op, reshaped_conv2_b1_op, 
            reshaped_conv3_b2_op, reshaped_conv2_b2_op, 
            reshaped_conv3_b3_op, reshaped_conv2_b3_op
        ), dim=1)

        # print("concatenated op after conv shape: ", concatenated_output.shape)

        b1_op = self.full_connect_b1(concatenated_output, qp)
        b2_op = self.full_connect_b2(concatenated_output, qp)
        b3_op = self.full_connect_b3(concatenated_output, qp)

        return (b1_op.squeeze(dim=0), b2_op.squeeze(dim=0), b3_op.squeeze(dim=0))      
        


# %%
"""
Declare Model
"""

# %%

def custom_repo_loss(y_flat_64, y_conv_flat_64, y_flat_32, y_conv_flat_32, y_flat_valid_32,
                     y_flat_16, y_conv_flat_16, y_flat_valid_16):
    # Ensure all tensors are on the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    y_flat_64 = y_flat_64.to(device, non_blocking=True)
    y_conv_flat_64 = y_conv_flat_64.to(device, non_blocking=True)
    y_flat_32 = y_flat_32.to(device, non_blocking=True)
    y_conv_flat_32 = y_conv_flat_32.to(device, non_blocking=True)
    y_flat_valid_32 = y_flat_valid_32.to(device, non_blocking=True)
    y_flat_16 = y_flat_16.to(device, non_blocking=True)
    y_conv_flat_16 = y_conv_flat_16.to(device, non_blocking=True)
    y_flat_valid_16 = y_flat_valid_16.to(device, non_blocking=True)

    # Avoid division by zero
    epsilon = 1e-12

    # Loss for 64
    loss_64_mean_pos = torch.sum(-y_flat_64 * torch.log(y_conv_flat_64 + epsilon)) / \
                        (torch.count_nonzero(y_flat_64) + epsilon)
    loss_64_mean_neg = torch.sum(-(1 - y_flat_64) * torch.log(1 - y_conv_flat_64 + epsilon)) / \
                        (torch.count_nonzero(1 - y_flat_64) + epsilon)
    loss_64 = (loss_64_mean_pos + loss_64_mean_neg) / 2

    # Loss for 32
    pos_mask_32 = y_flat_32 * y_flat_valid_32
    neg_mask_32 = (1 - y_flat_32) * y_flat_valid_32
    loss_32_mean_pos = torch.sum(-pos_mask_32 * torch.log(y_conv_flat_32 + epsilon)) / \
                        (torch.count_nonzero(pos_mask_32) + epsilon)
    loss_32_mean_neg = torch.sum(-neg_mask_32 * torch.log(1 - y_conv_flat_32 + epsilon)) / \
                        (torch.count_nonzero(neg_mask_32) + epsilon)
    loss_32 = (loss_32_mean_pos + loss_32_mean_neg) / 2

    # Loss for 16
    pos_mask_16 = y_flat_16 * y_flat_valid_16
    neg_mask_16 = (1 - y_flat_16) * y_flat_valid_16
    loss_16_mean_pos = torch.sum(-pos_mask_16 * torch.log(y_conv_flat_16 + epsilon)) / \
                        (torch.count_nonzero(pos_mask_16) + epsilon)
    loss_16_mean_neg = torch.sum(-neg_mask_16 * torch.log(1 - y_conv_flat_16 + epsilon)) / \
                        (torch.count_nonzero(neg_mask_16) + epsilon)
    loss_16 = (loss_16_mean_pos + loss_16_mean_neg) / 2

    # Total loss
    total_loss = loss_64 + loss_32 + loss_16

    return loss_64, loss_32, loss_16, total_loss


# %%
"""
Train the model
"""

# %%
# Initialize config variables
config_learning_rate = 0.01
config_momentum = 0.9
config_epochs = 100000
config_exponential_decay_ratio = 0.3163

# %%
def calculate_accuracy_repo(y_flat_64, y_conv_flat_64, y_flat_32, y_conv_flat_32, y_flat_valid_32,
                        y_flat_16, y_conv_flat_16, y_flat_valid_16):
    # Ensure all tensors are on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    y_flat_64 = y_flat_64.to(device, non_blocking=True)
    y_conv_flat_64 = y_conv_flat_64.to(device, non_blocking=True)
    y_flat_32 = y_flat_32.to(device, non_blocking=True)
    y_conv_flat_32 = y_conv_flat_32.to(device, non_blocking=True)
    y_flat_valid_32 = y_flat_valid_32.to(device, non_blocking=True)
    y_flat_16 = y_flat_16.to(device, non_blocking=True)
    y_conv_flat_16 = y_conv_flat_16.to(device, non_blocking=True)
    y_flat_valid_16 = y_flat_valid_16.to(device, non_blocking=True)

    epsilon = 1e-12

    # Correct predictions for 64
    correct_prediction_64 = torch.round(y_conv_flat_64) == torch.round(y_flat_64)
    accuracy_64 = torch.mean(correct_prediction_64.float())*100

    # Correct predictions for 32
    correct_prediction_valid_32 = y_flat_valid_32 * (torch.round(y_conv_flat_32) == torch.round(y_flat_32)).float()
    accuracy_32 = torch.sum(y_flat_valid_32 * correct_prediction_valid_32) / (torch.sum(y_flat_valid_32) + epsilon)*100

    # Correct predictions for 16
    correct_prediction_valid_16 = y_flat_valid_16 * (torch.round(y_conv_flat_16) == torch.round(y_flat_16)).float()
    accuracy_16 = torch.sum(y_flat_valid_16 * correct_prediction_valid_16) / (torch.sum(y_flat_valid_16) + epsilon)*100

    # Stack accuracies
    accuracy_list = torch.stack([accuracy_64, accuracy_32, accuracy_16])
    avg_acc = (accuracy_64+accuracy_32+accuracy_16)/3

    return avg_acc, accuracy_list[0], accuracy_list[1], accuracy_list[2]


"""
Training Loop
"""


# %%
# Initialize the device and model
num_classes = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ETH_CNN().to(device, non_blocking=True)

def to_numpy(data):
    if isinstance(data, list):  # If it's a list, process each element
        return np.array([x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in data])
    elif isinstance(data, torch.Tensor):  # If it's a single tensor
        return data.detach().cpu().numpy()
    return np.array(data)  # For NumPy arrays or other compatible types

# Function to save plots during training
def save_plots(epoch, train_acc, val_acc, train_l1, train_l2, train_l3, 
               val_l1, val_l2, val_l3, train_loss, train_loss_l1, train_loss_l2, 
               train_loss_l3, val_loss, val_loss_l1, val_loss_l2, val_loss_l3):
    
    # Create a directory for plots if it doesn't exist
    if not os.path.exists('training_plots'):
        os.makedirs('training_plots')

    # Convert lists to numpy for plotting
    t_acc = to_numpy(train_acc)
    v_acc = to_numpy(val_acc)
    
    # 1. Combined Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(t_acc) + 1), t_acc, label='Train Combined Accuracy')
    plt.plot(range(1, len(v_acc) + 1), v_acc, label='Val Combined Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Train vs Validation Accuracy (Epoch {epoch})')
    plt.legend()
    plt.grid()
    plt.savefig(f'training_plots/combined_accuracy_epoch_{epoch}.png') 
    plt.close()

    # 2. Training Level-wise Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_l1) + 1), to_numpy(train_l1), label='Train L1 Acc')
    plt.plot(range(1, len(train_l2) + 1), to_numpy(train_l2), label='Train L2 Acc')
    plt.plot(range(1, len(train_l3) + 1), to_numpy(train_l3), label='Train L3 Acc')
    plt.title(f'Training Level-wise Accuracy (Epoch {epoch})')
    plt.legend()
    plt.grid()
    plt.savefig(f'training_plots/train_level_acc_epoch_{epoch}.png')
    plt.close()

    # 3. Validation Level-wise Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_l1) + 1), to_numpy(val_l1), label='Val L1 Acc')
    plt.plot(range(1, len(val_l2) + 1), to_numpy(val_l2), label='Val L2 Acc')
    plt.plot(range(1, len(val_l3) + 1), to_numpy(val_l3), label='Val L3 Acc')
    plt.title(f'Validation Level-wise Accuracy (Epoch {epoch})')
    plt.legend()
    plt.grid()
    plt.savefig(f'training_plots/val_level_acc_epoch_{epoch}.png')
    plt.close()

    # 4. Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), to_numpy(train_loss), label='Train Total Loss')
    plt.plot(range(1, len(train_loss_l1) + 1), to_numpy(train_loss_l1), label='Train L1 Loss')
    plt.plot(range(1, len(train_loss_l2) + 1), to_numpy(train_loss_l2), label='Train L2 Loss')
    plt.plot(range(1, len(train_loss_l3) + 1), to_numpy(train_loss_l3), label='Train L3 Loss')
    plt.title(f'Training Loss (Epoch {epoch})')
    plt.legend()
    plt.grid()
    plt.savefig(f'training_plots/train_loss_epoch_{epoch}.png')
    plt.close()

    # 5. Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_loss) + 1), to_numpy(val_loss), label='Val Total Loss')
    plt.plot(range(1, len(val_loss_l1) + 1), to_numpy(val_loss_l1), label='Val L1 Loss')
    plt.plot(range(1, len(val_loss_l2) + 1), to_numpy(val_loss_l2), label='Val L2 Loss')
    plt.plot(range(1, len(val_loss_l3) + 1), to_numpy(val_loss_l3), label='Val L3 Loss')
    plt.title(f'Validation Loss (Epoch {epoch})')
    plt.legend()
    plt.grid()
    plt.savefig(f'training_plots/val_loss_epoch_{epoch}.png')
    plt.close()
    
    print(f"Plots saved to 'training_plots' folder for epoch {epoch}")

# Optimizer and scheduler
initial_lr = config_learning_rate
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=config_momentum)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_exponential_decay_ratio)

num_epochs = config_epochs

# Lists to store accuracy & loss for each epoch
accuracy_64_list = []
accuracy_32_list = []
accuracy_16_list = []

train_loss_list = []
train_loss_l1_list = []
train_loss_l2_list = []
train_loss_l3_list = []

val_loss_list = []
val_loss_l1_list = []
val_loss_l2_list = []
val_loss_l3_list = []


train_acc_3_level_list = []
train_acc_level1_list = []
train_acc_level2_list = []
train_acc_level3_list = []

val_acc_3_level_list = []
val_acc_level1_list = []
val_acc_level2_list = []
val_acc_level3_list = []

# Load checkpoint if available
start_epoch = 0  # Default to start training from the beginning
best_loss = float('inf')  # Initialize best_loss for comparison
patience = 10  # Number of epochs to wait for improvement
patience_counter = 0  # Counter for tracking no-improvement epochs
num_patience_counter_changed = 0
validation_shuffle_count = 0 # NEW: Counter for termination

checkpoint_path = 'checkpoint_100_200_1000.pth'
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    # Load checkpoint
    checkpoint = torch.load('checkpoint_100_200_1000.pth', weights_only=False)

    # Restore model, optimizer, scheduler, and epoch
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    overall_least_loss = checkpoint['overall_least_loss']

    print(f"Resumed training from epoch {start_epoch} with best loss {best_loss}.")
else:
    print("No checkpoint found, starting fresh training.")


list_checkpoint_path = 'checkpoint_saved_500_epoch.pth'
if os.path.exists(list_checkpoint_path):
    print(f"Loading checkpoint for restoring lists from {list_checkpoint_path}...")
    # Load checkpoint
    list_checkpoint = torch.load(list_checkpoint_path, weights_only=False)

    # Restore lists
    train_acc_3_level_list = list_checkpoint['train_acc_3_level_list'].tolist()
    val_acc_3_level_list = list_checkpoint['val_acc_3_level_list'].tolist()
    train_acc_level1_list = list_checkpoint['train_acc_level1_list'].tolist()
    train_acc_level2_list = list_checkpoint['train_acc_level2_list'].tolist()
    train_acc_level3_list = list_checkpoint['train_acc_level3_list'].tolist()
    train_loss_list = list_checkpoint['train_loss_list'].tolist()
    train_loss_l1_list = list_checkpoint['train_loss_l1_list'].tolist()
    train_loss_l2_list = list_checkpoint['train_loss_l2_list'].tolist()
    train_loss_l3_list = list_checkpoint['train_loss_l3_list'].tolist()
    val_acc_level1_list = list_checkpoint['val_acc_level1_list'].tolist()
    val_acc_level2_list = list_checkpoint['val_acc_level2_list'].tolist()
    val_acc_level3_list = list_checkpoint['val_acc_level3_list'].tolist()
    val_loss_list = list_checkpoint['val_loss_list'].tolist()
    val_loss_l1_list = list_checkpoint['val_loss_l1_list'].tolist()
    val_loss_l2_list = list_checkpoint['val_loss_l2_list'].tolist()
    val_loss_l3_list = list_checkpoint['val_loss_l3_list'].tolist()
    
    print(f"lists have been reloaded with length {len(train_acc_3_level_list)}.")
else:
    print("No list checkpoint found, no lists restored.")


# Epoch run
for epoch in range(start_epoch, num_epochs):
    # Change dataset after `epochs_per_dataset_change` epochs
    if patience_counter >= patience:
        num_patience_counter_changed += 1
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            # Load checkpoint
            checkpoint = torch.load('checkpoint_100_200_1000.pth', weights_only=False)

            # Restore model, optimizer, scheduler, and epoch
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            overall_least_loss = checkpoint['overall_least_loss']
            print(f"Resumed training from epoch {start_epoch} with best loss {best_loss}.")
        else:
            print("No checkpoint found, starting fresh training.")
            
        patience_counter=0

        best_loss = float('inf')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Changing training datasets...")
        # --- CHANGED: Use Full MaxSize for one-shot training ---
        new_train_loader, new_train_indices = create_subset_dataloader(train_file_path, TRAINSET_MAXSIZE, TRAINSET_MAXSIZE, BATCH_SIZE, device=device, shuffle=True)
        # Log dataset changes
        print("Train indices changed:", new_train_indices != train_indices)

        # Update loaders and indices
        train_loader, train_indices = new_train_loader, new_train_indices

    # Change dataset after `epochs_per_dataset_change` epochs
    if num_patience_counter_changed >= 3:
        num_patience_counter_changed = 0

        # --- NEW: Termination Logic ---
        validation_shuffle_count += 1
        if validation_shuffle_count > 2:
            print("\n" + "=" * 60)
            print(f"TERMINATION: Validation dataset shuffled {validation_shuffle_count} times without loss improvement.")
            print("=" * 60)
            break
        # ------------------------------

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Changing validation datasets...")
        # --- CHANGED: Use Full MaxSize for one-shot training ---
        new_validation_loader, new_validation_indices = create_subset_dataloader(validation_file_path, VALIDSET_MAXSIZE, VALIDSET_MAXSIZE, BATCH_SIZE, device=device, shuffle=False)
        # Log dataset changes
        print("Validation indices changed:", new_validation_indices != validation_indices)
        # Update loaders and indices
        validation_loader, validation_indices = new_validation_loader, new_validation_indices

    # Training phase
    model.train()
    running_loss = 0.0
    running_loss_l1, running_loss_l2, running_loss_l3 = 0.0, 0.0, 0.0
    overall_running_acc = 0.0
    acc1_epoch, acc2_epoch, acc3_epoch = 0.0, 0.0, 0.0

    for i, (qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, label_batch) in enumerate(train_loader):
        # Prepare inputs and true labels
        qp_batch = qp_batch.to(device, non_blocking=True)
        ctu_batch = ctu_batch.to(device, non_blocking=True)
        true_labels = [
            y_flat_64.to(device, non_blocking=True),
            y_flat_32.to(device, non_blocking=True),
            y_flat_16.to(device, non_blocking=True)
        ]
        true_valid_labels = [
            y_flat_valid_32.to(device, non_blocking=True),
            y_flat_valid_16.to(device, non_blocking=True)
        ]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model((qp_batch, ctu_batch))

        # Compute loss
        l1_loss, l2_loss, l3_loss, combined_avg_loss = custom_repo_loss(
            true_labels[0], outputs[0],
            true_labels[1], outputs[1], true_valid_labels[0],
            true_labels[2], outputs[2], true_valid_labels[1]
        )

        # Backward pass and optimization step
        combined_avg_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            # Accumulate loss
            running_loss += combined_avg_loss.item()
            running_loss_l1 += l1_loss.item()
            running_loss_l2 += l2_loss.item()
            running_loss_l3 += l3_loss.item()

            # Calculate accuracies
            train_avg_acc, acc1, acc2, acc3 = calculate_accuracy_repo(
                true_labels[0], outputs[0],
                true_labels[1], outputs[1], true_valid_labels[0],
                true_labels[2], outputs[2], true_valid_labels[1]
            )

            overall_running_acc += train_avg_acc
            acc1_epoch += acc1
            acc2_epoch += acc2
            acc3_epoch += acc3

    if (epoch + 1) % 10 == 0:
        # Average metrics over all batches
        avg_loss = running_loss / len(train_loader)  # Average training loss
        train_epoch_loss_l1 = running_loss_l1 / len(train_loader)
        train_epoch_loss_l2 = running_loss_l2 / len(train_loader)
        train_epoch_loss_l3 = running_loss_l3 / len(train_loader)

        overall_running_acc /= len(train_loader)  # Overall training accuracy
        acc1_epoch /= len(train_loader)  # Accuracy for level 1
        acc2_epoch /= len(train_loader)  # Accuracy for level 2
        acc3_epoch /= len(train_loader)  # Accuracy for level 3

        train_loss_list.append(avg_loss)
        train_loss_l1_list.append(train_epoch_loss_l1)
        train_loss_l2_list.append(train_epoch_loss_l2)
        train_loss_l3_list.append(train_epoch_loss_l3)

        train_acc_3_level_list.append(overall_running_acc)
        train_acc_level1_list.append(acc1_epoch)
        train_acc_level2_list.append(acc2_epoch)
        train_acc_level3_list.append(acc3_epoch)

        # Log training results
        print(f"Epoch [{epoch + 1}/{num_epochs}] Training Results:")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy Level 1: {acc1_epoch:.4f}% Accuracy Level 2: {acc2_epoch:.4f}% Accuracy Level 3: {acc3_epoch:.4f}%")

    # Adjust learning rate with the scheduler
    if (epoch+1) % 25000 == 0:
        scheduler.step()

    if (epoch + 1) % 50 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch + 1,
            'best_loss': best_loss,
            'overall_least_loss': overall_least_loss,
            'train_acc_3_level_list': to_numpy(train_acc_3_level_list),
            'val_acc_3_level_list': to_numpy(val_acc_3_level_list),
            'train_acc_level1_list': to_numpy(train_acc_level1_list),
            'train_acc_level2_list': to_numpy(train_acc_level2_list),
            'train_acc_level3_list': to_numpy(train_acc_level3_list),
            'train_loss_list': to_numpy(train_loss_list),
            'train_loss_l1_list': to_numpy(train_loss_l1_list),
            'train_loss_l2_list': to_numpy(train_loss_l2_list),
            'train_loss_l3_list': to_numpy(train_loss_l3_list),
            'val_acc_level1_list': to_numpy(val_acc_level1_list),
            'val_acc_level2_list': to_numpy(val_acc_level2_list),
            'val_acc_level3_list': to_numpy(val_acc_level3_list),
            'val_loss_list': to_numpy(val_loss_list),
            'val_loss_l1_list': to_numpy(val_loss_l1_list),
            'val_loss_l2_list': to_numpy(val_loss_l2_list),
            'val_loss_l3_list': to_numpy(val_loss_l3_list),
        }, 'checkpoint_saved_500_epoch.pth')

        print(f"Checkpoint saved for epoch {epoch + 1} with name checkpoint_saved_500_epoch.pth")

    
    # Validation phase
    if (epoch + 1) % 10 == 0:
        model.eval()  # Set model to evaluation mode
        val_running_loss, val_running_l1_loss, val_running_l2_loss, val_running_l3_loss = 0.0, 0.0, 0.0, 0.0
        val_overall_acc_epoch = 0.0
        val_acc1_epoch = 0.0
        val_acc2_epoch = 0.0
        val_acc3_epoch = 0.0

        with torch.no_grad():
            for qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, label_batch in validation_loader:
                qp_batch = qp_batch.to(device, non_blocking=True)
                ctu_batch = ctu_batch.to(device, non_blocking=True)
                true_labels = [
                    y_flat_64.to(device, non_blocking=True),
                    y_flat_32.to(device, non_blocking=True),
                    y_flat_16.to(device, non_blocking=True)
                ]
                true_valid_labels = [
                    y_flat_valid_32.to(device, non_blocking=True),
                    y_flat_valid_16.to(device, non_blocking=True)
                ]

                outputs = model((qp_batch, ctu_batch))

                l1_loss, l2_loss, l3_loss, combined_avg_loss = custom_repo_loss(
                    true_labels[0], outputs[0],
                    true_labels[1], outputs[1], true_valid_labels[0],
                    true_labels[2], outputs[2], true_valid_labels[1]
                )

                val_running_loss += combined_avg_loss.item()
                val_running_l1_loss += l1_loss.item()
                val_running_l2_loss += l2_loss.item()
                val_running_l3_loss += l3_loss.item()

                overall_acc, acc1, acc2, acc3 = calculate_accuracy_repo(
                    true_labels[0], outputs[0],
                    true_labels[1], outputs[1], true_valid_labels[0],
                    true_labels[2], outputs[2], true_valid_labels[1]
                )

                val_overall_acc_epoch += overall_acc
                val_acc1_epoch += acc1
                val_acc2_epoch += acc2
                val_acc3_epoch += acc3

        # Average validation loss and accuracy metrics
        avg_val_loss = val_running_loss / len(validation_loader)
        val_epoch_loss_l1 = val_running_l1_loss / len(validation_loader)
        val_epoch_loss_l2 = val_running_l2_loss / len(validation_loader)
        val_epoch_loss_l3 = val_running_l3_loss / len(validation_loader)

        val_loss_list.append(avg_val_loss)
        val_loss_l1_list.append(val_epoch_loss_l1)
        val_loss_l2_list.append(val_epoch_loss_l2)
        val_loss_l3_list.append(val_epoch_loss_l3)

        avg_val_overall_acc_epoch = val_overall_acc_epoch / len(validation_loader)
        avg_val_acc1_epoch = val_acc1_epoch / len(validation_loader)
        avg_val_acc2_epoch = val_acc2_epoch / len(validation_loader)
        avg_val_acc3_epoch = val_acc3_epoch / len(validation_loader)

        val_acc_3_level_list.append(avg_val_overall_acc_epoch)
        val_acc_level1_list.append(avg_val_acc1_epoch)
        val_acc_level2_list.append(avg_val_acc2_epoch)
        val_acc_level3_list.append(avg_val_acc3_epoch)

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Accuracy 64x64: {avg_val_acc1_epoch:.2f}%, 32x32: {avg_val_acc2_epoch:.2f}%, 16x16: {avg_val_acc3_epoch:.2f}%")
        print()

        # --- NEW CODE: Save plots every 100 epochs ---
        if (epoch + 1) % 100 == 0:
            save_plots(
                epoch + 1,
                train_acc_3_level_list, val_acc_3_level_list,
                train_acc_level1_list, train_acc_level2_list, train_acc_level3_list,
                val_acc_level1_list, val_acc_level2_list, val_acc_level3_list,
                train_loss_list, train_loss_l1_list, train_loss_l2_list, train_loss_l3_list,
                val_loss_list, val_loss_l1_list, val_loss_l2_list, val_loss_l3_list
            )
        # ---------------------------------------------
            
        if avg_val_loss <= best_loss:
            best_loss = avg_val_loss
            patience_counter = 0  # Reset patience counter
            validation_shuffle_count = 0 # NEW: Reset shuffle count
            # Save the model if validation loss improves
            if avg_val_loss < overall_least_loss:
                torch.save(model.state_dict(), 'best_model_4qp_parallel_data_processing_loss_mod.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'best_loss': best_loss,
                    'overall_least_loss': overall_least_loss,
                    'train_acc_3_level_list': to_numpy(train_acc_3_level_list),
                    'val_acc_3_level_list': to_numpy(val_acc_3_level_list),
                    'train_acc_level1_list': to_numpy(train_acc_level1_list),
                    'train_acc_level2_list': to_numpy(train_acc_level2_list),
                    'train_acc_level3_list': to_numpy(train_acc_level3_list),
                    'train_loss_list': to_numpy(train_loss_list),
                    'train_loss_l1_list': to_numpy(train_loss_l1_list),
                    'train_loss_l2_list': to_numpy(train_loss_l2_list),
                    'train_loss_l3_list': to_numpy(train_loss_l3_list),
                    'val_acc_level1_list': to_numpy(val_acc_level1_list),
                    'val_acc_level2_list': to_numpy(val_acc_level2_list),
                    'val_acc_level3_list': to_numpy(val_acc_level3_list),
                    'val_loss_list': to_numpy(val_loss_list),
                    'val_loss_l1_list': to_numpy(val_loss_l1_list),
                    'val_loss_l2_list': to_numpy(val_loss_l2_list),
                    'val_loss_l3_list': to_numpy(val_loss_l3_list),
                }, 'checkpoint_100_200_1000.pth')

                print(f"Checkpoint saved for epoch {epoch + 1}")
                overall_least_loss = avg_val_loss
        else:
            patience_counter += 1  # Increment patience counter


# Final plot generation after loop completes (as a backup)
print("Training complete. Generating final plots...")
save_plots(
    num_epochs,
    train_acc_3_level_list, val_acc_3_level_list,
    train_acc_level1_list, train_acc_level2_list, train_acc_level3_list,
    val_acc_level1_list, val_acc_level2_list, val_acc_level3_list,
    train_loss_list, train_loss_l1_list, train_loss_l2_list, train_loss_l3_list,
    val_loss_list, val_loss_l1_list, val_loss_l2_list, val_loss_l3_list
)