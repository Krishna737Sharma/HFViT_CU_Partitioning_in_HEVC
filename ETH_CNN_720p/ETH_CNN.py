# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
import sys
import os
import random
from torch.utils.data import DataLoader, Subset, Dataset

# Configuration
CONFIG_LEARNING_RATE = 0.0001
CONFIG_MOMENTUM = 0.9
CONFIG_EPOCHS = 10000 
CONFIG_EXPONENTIAL_DECAY_RATIO = 0.3163
BATCH_SIZE = 64

# Scheduler Configuration
SCHEDULER_STEP_FREQUENCY = 1000  # Execute scheduler every 1000 epochs

# Constants for Data Loading
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_LABEL_BYTES = 16
NUM_SAMPLE_LENGTH = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + 64 + (51 + 1) * NUM_LABEL_BYTES
SELECT_QP_LIST = [22, 27, 32, 37]

# --- 1. Data Loading ---
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

            ctu_tensor = torch.tensor(image, dtype=torch.float32).squeeze(2)
            qp_tensor = torch.tensor(float(qp), dtype=torch.float32)

            # Generate hierarchical labels
            y_image = torch.tensor(label, dtype=torch.float32).view(1, 4, 4)
            
            y_image_16 = F.relu(y_image - 2)
            y_image_32 = F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 1) - \
                         F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 2)
            y_image_64 = F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=4) - 0) - \
                         F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=4) - 1)
            y_image_valid_32 = F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 0) - \
                               F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 1)
            y_image_valid_16 = F.relu(y_image - 1) - F.relu(y_image - 2)

            y_flat_16 = y_image_16.view(-1)
            y_flat_32 = y_image_32.view(-1)
            y_flat_64 = y_image_64.view(-1)
            y_flat_valid_32 = y_image_valid_32.view(-1)
            y_flat_valid_16 = y_image_valid_16.view(-1)

            ctu_tensor /= 255.0
            qp_tensor /= 51.0

            return qp_tensor, ctu_tensor, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, torch.tensor(label, dtype=torch.float32)


def create_subset_dataloader(file_path, total_samples, subset_size, batch_size, device, shuffle=True):
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    full_dataset = StreamingDataset(file_path, total_samples)
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


# --- 2. Normalization and Downsampling Functions ---
def norm_batch_ctu(ctu_batch):
    ctu_data = ctu_batch.clone().detach().float()
    
    if ctu_data.dim() == 2:
        ctu_data = ctu_data.unsqueeze(0)
        batch_size = 1
    else:
        batch_size = ctu_data.size(0)

    norm_ctu_data_b1 = ctu_data.clone()
    norm_ctu_data_b2 = ctu_data.clone()
    norm_ctu_data_b3 = ctu_data.clone()

    # Branch B1: Mean removal at 64x64 level
    mean_value_level1 = torch.mean(ctu_data[:, 0:64, 0:64], dim=(1, 2), keepdim=True)
    norm_ctu_data_b1 -= mean_value_level1
    
    # Branch B2: Mean removal at 32x32 level
    mean_value_level2_1 = torch.mean(ctu_data[:, 0:32, 0:32], dim=(1, 2), keepdim=True)
    mean_value_level2_2 = torch.mean(ctu_data[:, 0:32, 32:64], dim=(1, 2), keepdim=True)
    mean_value_level2_3 = torch.mean(ctu_data[:, 32:64, 0:32], dim=(1, 2), keepdim=True)
    mean_value_level2_4 = torch.mean(ctu_data[:, 32:64, 32:64], dim=(1, 2), keepdim=True)
    
    norm_ctu_data_b2[:, 0:32, 0:32]   -= mean_value_level2_1
    norm_ctu_data_b2[:, 0:32, 32:64]  -= mean_value_level2_2
    norm_ctu_data_b2[:, 32:64, 0:32]  -= mean_value_level2_3
    norm_ctu_data_b2[:, 32:64, 32:64] -= mean_value_level2_4

    # Branch B3: Mean removal at 16x16 level
    for i in range(0, 64, 16):
        mean_value_level3_1 = torch.mean(ctu_data[:, i:i+16, 0:16])
        mean_value_level3_2 = torch.mean(ctu_data[:, i:i+16, 16:32])
        mean_value_level3_3 = torch.mean(ctu_data[:, i:i+16, 32:48])
        mean_value_level3_4 = torch.mean(ctu_data[:, i:i+16, 48:64])

        norm_ctu_data_b3[:, i:i+16, 0:16]  -= mean_value_level3_1
        norm_ctu_data_b3[:, i:i+16, 16:32] -= mean_value_level3_2
        norm_ctu_data_b3[:, i:i+16, 32:48] -= mean_value_level3_3
        norm_ctu_data_b3[:, i:i+16, 48:64] -= mean_value_level3_4

    if batch_size == 1:
        norm_ctu_data_b1 = norm_ctu_data_b1.squeeze(0)
        norm_ctu_data_b2 = norm_ctu_data_b2.squeeze(0)
        norm_ctu_data_b3 = norm_ctu_data_b3.squeeze(0)

    return norm_ctu_data_b1, norm_ctu_data_b2, norm_ctu_data_b3


def mean_downsample(tensor, scale_factor):
    if tensor.dim() != 3:
        raise ValueError("Input tensor must be 3D (batch, height, width).")
    
    batch_size, h, w = tensor.shape
    new_h, new_w = h // scale_factor, w // scale_factor

    downsampled_tensor = tensor.unfold(1, scale_factor, scale_factor).unfold(2, scale_factor, scale_factor)
    downsampled_tensor = downsampled_tensor.contiguous().view(batch_size, new_h, new_w, -1)
    downsampled_tensor = downsampled_tensor.mean(dim=-1)
    
    return downsampled_tensor


def downsample_ctu_3_branches(norm_ctu_tuple):
    branch1_ctu = norm_ctu_tuple[0]
    branch2_ctu = norm_ctu_tuple[1]
    branch3_ctu = norm_ctu_tuple[2]

    downsampled_ctu_16_16 = mean_downsample(branch1_ctu, 4)
    downsampled_ctu_32_32 = mean_downsample(branch2_ctu, 2)
    downsampled_ctu_64_64 = mean_downsample(branch3_ctu, 1)

    return (downsampled_ctu_16_16, downsampled_ctu_32_32, downsampled_ctu_64_64)


# --- 3. ETH-CNN Model Architecture ---
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

        # Fully connected Layers
        self.fc1_dropout = nn.Dropout(p=0.5)
        self.fc2_dropout = nn.Dropout(p=0.25)

        # Branch 1
        self.fc1_b1 = nn.Linear(in_features=1344, out_features=48)
        self.fc2_b1 = nn.Linear(in_features=49, out_features=24)
        self.fc3_b1 = nn.Linear(in_features=25, out_features=1)

        # Branch 2
        self.fc1_b2 = nn.Linear(in_features=1344, out_features=96)
        self.fc2_b2 = nn.Linear(in_features=97, out_features=48)
        self.fc3_b2 = nn.Linear(in_features=49, out_features=4)

        # Branch 3
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
        fc3_activation_op = F.sigmoid(self.fc3_b1(qp_fc2_activation_op))
        return fc3_activation_op

    def full_connect_b2(self, x, qp):
        qp_tensor = qp.unsqueeze(1)

        fc1_activation_op = F.leaky_relu(self.fc1_b2(x))
        fc1_activation_op = self.fc1_dropout(fc1_activation_op)

        qp_fc1_activation_op = torch.cat((fc1_activation_op, qp_tensor), dim=1)
        fc2_activation_op = F.leaky_relu(self.fc2_b2(qp_fc1_activation_op))
        fc2_activation_op = self.fc2_dropout(fc2_activation_op)

        qp_fc2_activation_op = torch.cat((fc2_activation_op, qp_tensor), dim=1)
        fc3_activation_op = F.sigmoid(self.fc3_b2(qp_fc2_activation_op))
        return fc3_activation_op
    
    def full_connect_b3(self, x, qp):
        qp_tensor = qp.unsqueeze(1)

        fc1_activation_op = F.leaky_relu(self.fc1_b3(x))
        fc1_activation_op = self.fc1_dropout(fc1_activation_op)
        
        qp_fc1_activation_op = torch.cat((fc1_activation_op, qp_tensor), dim=1)
        fc2_activation_op = F.leaky_relu(self.fc2_b3(qp_fc1_activation_op))
        fc2_activation_op = self.fc2_dropout(fc2_activation_op)

        qp_fc2_activation_op = torch.cat((fc2_activation_op, qp_tensor), dim=1)
        fc3_activation_op = F.sigmoid(self.fc3_b3(qp_fc2_activation_op))
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

def count_parameters(model, model_name="Model"):
    """Count and print model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print("\n" + "=" * 70)
    print(f"{model_name} - Parameter Summary")
    print("=" * 70)
    print(f"Total Parameters:        {total_params:,}")
    print(f"Trainable Parameters:    {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    print("=" * 70)
        
    return total_params, trainable_params


# --- 4. Loss and Accuracy Functions ---
def custom_repo_loss(y_flat_64, y_conv_flat_64, y_flat_32, y_conv_flat_32, y_flat_valid_32,
                     y_flat_16, y_conv_flat_16, y_flat_valid_16):
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

    total_loss = loss_64 + loss_32 + loss_16

    return loss_64, loss_32, loss_16, total_loss


def calculate_accuracy_repo(y_flat_64, y_conv_flat_64, y_flat_32, y_conv_flat_32, y_flat_valid_32,
                        y_flat_16, y_conv_flat_16, y_flat_valid_16):
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
    accuracy_64 = torch.mean(correct_prediction_64.float()) * 100

    # Correct predictions for 32
    correct_prediction_valid_32 = y_flat_valid_32 * (torch.round(y_conv_flat_32) == torch.round(y_flat_32)).float()
    accuracy_32 = torch.sum(y_flat_valid_32 * correct_prediction_valid_32) / (torch.sum(y_flat_valid_32) + epsilon) * 100

    # Correct predictions for 16
    correct_prediction_valid_16 = y_flat_valid_16 * (torch.round(y_conv_flat_16) == torch.round(y_flat_16)).float()
    accuracy_16 = torch.sum(y_flat_valid_16 * correct_prediction_valid_16) / (torch.sum(y_flat_valid_16) + epsilon) * 100

    accuracy_list = torch.stack([accuracy_64, accuracy_32, accuracy_16])
    avg_acc = (accuracy_64 + accuracy_32 + accuracy_16) / 3

    return avg_acc, accuracy_list[0], accuracy_list[1], accuracy_list[2]


# --- 5. Plotting Function ---
def save_plots(epoch, history):
    if not os.path.exists('training_plots'):
        os.makedirs('training_plots')
    
    steps = len(history['train_loss'])
    epochs_range = range(2, (steps * 2) + 1, 2)

    # 1. Training vs Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title(f'Training and Validation Loss (Epoch {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_plots/loss_plot_epoch_{epoch}.png')
    plt.close()

    # 2. Training vs Validation Accuracy (Total)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy', linestyle='--')
    plt.title(f'Training and Validation Accuracy (Epoch {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_plots/accuracy_plot_epoch_{epoch}.png')
    plt.close()

    # 3. L1 Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['train_acc_l1'], label='Train L1 Acc')
    plt.plot(epochs_range, history['val_acc_l1'], label='Val L1 Acc', linestyle='--')
    plt.title(f'L1 Accuracy (64x64 Split) (Epoch {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_plots/l1_accuracy_plot_epoch_{epoch}.png')
    plt.close()

    # 4. L2 Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['train_acc_l2'], label='Train L2 Acc')
    plt.plot(epochs_range, history['val_acc_l2'], label='Val L2 Acc', linestyle='--')
    plt.title(f'L2 Accuracy (32x32 Split) (Epoch {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_plots/l2_accuracy_plot_epoch_{epoch}.png')
    plt.close()

    # 5. L3 Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['train_acc_l3'], label='Train L3 Acc')
    plt.plot(epochs_range, history['val_acc_l3'], label='Val L3 Acc', linestyle='--')
    plt.title(f'L3 Accuracy (16x16 Split) (Epoch {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_plots/l3_accuracy_plot_epoch_{epoch}.png')
    plt.close()
    
    print(f"Plots saved to 'training_plots' folder for epoch {epoch}")


# --- 6. Main Execution ---

# File paths
train_file_path = "/raid/somdyutiai/Krishna_24AI60R38/PycharmProjects/HEVC_Intra_Models-ETH-CNN_Pt/Data/720p_dataset/AI_Train_40800.dat_shuffled"
validation_file_path = "/raid/somdyutiai/Krishna_24AI60R38/PycharmProjects/HEVC_Intra_Models-ETH-CNN_Pt/Data/720p_dataset/AI_Valid_2400.dat_shuffled"
test_file_path = "/raid/somdyutiai/Krishna_24AI60R38/PycharmProjects/HEVC_Intra_Models-ETH-CNN_Pt/Data/720p_dataset/AI_Test_4800.dat_shuffled"

TRAINSET_MAXSIZE = 40800
VALIDSET_MAXSIZE = 2400
TESTSET_MAXSIZE = 4800

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Model and Optimizer with Weight Decay
model = ETH_CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=CONFIG_LEARNING_RATE, momentum=CONFIG_MOMENTUM, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=CONFIG_EXPONENTIAL_DECAY_RATIO)

# COUNT AND PRINT PARAMETERS
total_params, trainable_params = count_parameters(model, "ETH-CNN")

# Checkpoint paths
checkpoint_path = 'checkpoint_eth_cnn.pth'
best_checkpoint_path = 'checkpoint_best_eth_cnn.pth'

# --- TRAINING VARIABLES INITIALIZATION ---
start_epoch = 1
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': [],
    'train_acc_l1': [], 'val_acc_l1': [],
    'train_acc_l2': [], 'val_acc_l2': [],
    'train_acc_l3': [], 'val_acc_l3': []
}
overall_least_loss = float('inf')
best_loss = float('inf')
patience = 10
patience_counter = 0
num_patience_counter_changed = 0
validation_shuffle_count = 0

# --- RECOVERY LOGIC ---
if os.path.exists(checkpoint_path):
    print(f"Found checkpoint: {checkpoint_path}. Resuming training...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        overall_least_loss = checkpoint.get('overall_least_loss', float('inf'))
        best_loss = checkpoint.get('best_loss', overall_least_loss)
        patience_counter = checkpoint.get('patience_counter', 0)
        num_patience_counter_changed = checkpoint.get('num_patience_counter_changed', 0)
        validation_shuffle_count = checkpoint.get('validation_shuffle_count', 0)
        
        print(f"Successfully resumed from Epoch {start_epoch}")
        print(f"Best Loss: {best_loss:.4f}")
        print(f"Overall Least Loss: {overall_least_loss:.4f}")
        print(f"Patience Counter: {patience_counter}/{patience}")
        print(f"Training Dataset Changes: {num_patience_counter_changed}/3")
        print(f"Validation Shuffles: {validation_shuffle_count}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        best_loss = float('inf')
        patience_counter = 0
        num_patience_counter_changed = 0
        validation_shuffle_count = 0
else:
    print("No checkpoint found. Starting training from scratch.")

# Initialize Loaders
train_loader, train_indices = create_subset_dataloader(train_file_path, TRAINSET_MAXSIZE, TRAINSET_MAXSIZE, BATCH_SIZE, device=device, shuffle=True)
validation_loader, validation_indices = create_subset_dataloader(validation_file_path, VALIDSET_MAXSIZE, VALIDSET_MAXSIZE, BATCH_SIZE, device=device, shuffle=False)

# --- Training Loop ---
print(f"\nStarting Training from Epoch {start_epoch}...")
print(f"Scheduler will execute every {SCHEDULER_STEP_FREQUENCY} epochs")
print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
print()

for epoch in range(start_epoch, CONFIG_EPOCHS + 1):
    
    # =================================================================
    # 1. FIXED: Patience check -> Reload best model + Shuffle training
    #    (Mirrors reference code logic exactly)
    # =================================================================
    if patience_counter >= patience:
        num_patience_counter_changed += 1
        
        # Reload best model weights (like reference code)
        if os.path.exists(best_checkpoint_path):
            print(f"\n>>> Patience reached ({patience_counter}). Reloading best model from {best_checkpoint_path}...")
            best_ckpt = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(best_ckpt['model_state_dict'])
            optimizer.load_state_dict(best_ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(best_ckpt['scheduler_state_dict'])
            best_loss = best_ckpt['best_loss']
            print(f"    Restored best_loss to: {best_loss:.4f}")
        
        patience_counter = 0  # Reset counter
        
        print(f">>> Refreshing Training Data... (change #{num_patience_counter_changed}/3)")
        train_loader, train_indices = create_subset_dataloader(
            train_file_path, TRAINSET_MAXSIZE, TRAINSET_MAXSIZE, 
            BATCH_SIZE, device=device, shuffle=True
        )
    
    # =================================================================
    # 2. FIXED: After 3 training shuffles -> Shuffle validation
    #    (Separate check, NOT nested inside patience check)
    # =================================================================
    if num_patience_counter_changed >= 3:
        num_patience_counter_changed = 0
        validation_shuffle_count += 1
        
        # Check termination BEFORE shuffling
        if validation_shuffle_count > 1:
            print("\n" + "=" * 60)
            print(f"TERMINATION: Validation dataset shuffled {validation_shuffle_count} times without sufficient improvement.")
            print(f"Best overall validation loss achieved: {overall_least_loss:.4f}")
            print("=" * 60)
            break
        
        print(f"\n>>> 3 training shuffles exhausted. Refreshing VALIDATION dataset... (shuffle #{validation_shuffle_count})")
        validation_loader, validation_indices = create_subset_dataloader(
            validation_file_path, VALIDSET_MAXSIZE, VALIDSET_MAXSIZE, 
            BATCH_SIZE, device=device, shuffle=False
        )

    # =================================================================
    # 3. Training Phase
    # =================================================================
    model.train()
    r_loss = 0.0
    r_acc = 0.0
    r_acc1, r_acc2, r_acc3 = 0.0, 0.0, 0.0
    
    for i, (qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, _) in enumerate(train_loader):
        qp_batch, ctu_batch = qp_batch.to(device), ctu_batch.to(device)
        labels = [y_flat_64.to(device), y_flat_32.to(device), y_flat_16.to(device)]
        valid_masks = [y_flat_valid_32.to(device), y_flat_valid_16.to(device)]

        optimizer.zero_grad()
        outputs = model((qp_batch, ctu_batch))

        _, _, _, total_loss = custom_repo_loss(
            labels[0], outputs[0],
            labels[1], outputs[1], valid_masks[0],
            labels[2], outputs[2], valid_masks[1]
        )
        
        total_loss.backward()
        optimizer.step()

        avg_acc, acc1, acc2, acc3 = calculate_accuracy_repo(
            labels[0], outputs[0],
            labels[1], outputs[1], valid_masks[0],
            labels[2], outputs[2], valid_masks[1]
        )

        r_loss += total_loss.item()
        r_acc += avg_acc.item()
        r_acc1 += acc1.item()
        r_acc2 += acc2.item()
        r_acc3 += acc3.item()

    # =================================================================
    # 4. Learning Rate Scheduling
    # =================================================================
    if epoch % SCHEDULER_STEP_FREQUENCY == 0:
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        print(f"\n>>> Scheduler Step at Epoch {epoch}: LR changed from {old_lr:.6f} to {new_lr:.6f}\n")

    # =================================================================
    # 5. Validation Phase (Every 2 epochs)
    # =================================================================
    if epoch % 2 == 0:
        model.eval()
        v_loss = 0.0
        v_acc = 0.0
        v_acc1, v_acc2, v_acc3 = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, _ in validation_loader:
                qp_batch, ctu_batch = qp_batch.to(device), ctu_batch.to(device)
                labels = [y_flat_64.to(device), y_flat_32.to(device), y_flat_16.to(device)]
                valid_masks = [y_flat_valid_32.to(device), y_flat_valid_16.to(device)]

                outputs = model((qp_batch, ctu_batch))

                _, _, _, total_loss = custom_repo_loss(
                    labels[0], outputs[0],
                    labels[1], outputs[1], valid_masks[0],
                    labels[2], outputs[2], valid_masks[1]
                )
                
                avg_acc, acc1, acc2, acc3 = calculate_accuracy_repo(
                    labels[0], outputs[0],
                    labels[1], outputs[1], valid_masks[0],
                    labels[2], outputs[2], valid_masks[1]
                )

                v_loss += total_loss.item()
                v_acc += avg_acc.item()
                v_acc1 += acc1.item()
                v_acc2 += acc2.item()
                v_acc3 += acc3.item()

        # Normalize metrics
        train_len = len(train_loader)
        val_len = len(validation_loader)
        
        # Append to history
        history['train_loss'].append(r_loss / train_len)
        history['val_loss'].append(v_loss / val_len)
        history['train_acc'].append(r_acc / train_len)
        history['val_acc'].append(v_acc / val_len)
        history['train_acc_l1'].append(r_acc1 / train_len)
        history['val_acc_l1'].append(v_acc1 / val_len)
        history['train_acc_l2'].append(r_acc2 / train_len)
        history['val_acc_l2'].append(v_acc2 / val_len)
        history['train_acc_l3'].append(r_acc3 / train_len)
        history['val_acc_l3'].append(v_acc3 / val_len)

        curr_val_loss = history['val_loss'][-1]

        print(f"Epoch [{epoch}/{CONFIG_EPOCHS}] Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {curr_val_loss:.4f}")
        print(f"  Train Acc: {history['train_acc'][-1]:.2f}% | Val Acc: {history['val_acc'][-1]:.2f}%")
        print(f"  Train L1: {history['train_acc_l1'][-1]:.2f}% | Train L2: {history['train_acc_l2'][-1]:.2f}% | Train L3: {history['train_acc_l3'][-1]:.2f}%")
        print(f"  Val L1:   {history['val_acc_l1'][-1]:.2f}% | Val L2:   {history['val_acc_l2'][-1]:.2f}% | Val L3:   {history['val_acc_l3'][-1]:.2f}%")
        print(f"  [Patience: {patience_counter}/{patience} | Train Shuffles: {num_patience_counter_changed}/3 | Val Shuffles: {validation_shuffle_count}]")
        
        # Save Plots (Every 100 epochs)
        if epoch % 100 == 0:
            save_plots(epoch, history)

        if curr_val_loss <= best_loss:
            best_loss = curr_val_loss
            overall_least_loss = curr_val_loss
            patience_counter = 0
            validation_shuffle_count = 0
            
            print(f"  ✓ New Best Model saved! (Loss: {best_loss:.4f})")
            
            # Save best model weights
            torch.save(model.state_dict(), 'best_model_ETH_CNN.pth')
            
            # Save best complete checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': curr_val_loss,
                'best_loss': best_loss,
                'overall_least_loss': overall_least_loss,
                'history': history,
                'current_lr': optimizer.param_groups[0]['lr'],
                'patience_counter': patience_counter,
                'num_patience_counter_changed': num_patience_counter_changed,
                'validation_shuffle_count': validation_shuffle_count
            }, best_checkpoint_path)
        else:
            patience_counter += 1
        
        # ALWAYS save regular checkpoint after validation (enables exact resume)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': curr_val_loss,
            'best_loss': best_loss,
            'overall_least_loss': overall_least_loss,
            'history': history,
            'current_lr': optimizer.param_groups[0]['lr'],
            'patience_counter': patience_counter,
            'num_patience_counter_changed': num_patience_counter_changed,
            'validation_shuffle_count': validation_shuffle_count
        }, checkpoint_path)
        
        # Visual confirmation (every 50 epochs)
        if epoch % 50 == 0:
            print(f"━━━ Checkpoint auto-saved at epoch {epoch} ━━━")

print("\n" + "=" * 70)
print("Training Complete.")
print(f"Best validation loss achieved: {overall_least_loss:.4f}")
print(f"Best model saved as: best_model_ETH_CNN.pth")
print("=" * 70)

# Final plot generation
if epoch <= CONFIG_EPOCHS:
    save_plots(epoch, history)
else:
    save_plots(CONFIG_EPOCHS, history)