import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
import os
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Constants
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
        with open(self.file_path, 'rb') as file_reader:
            offset = idx * NUM_SAMPLE_LENGTH
            file_reader.seek(offset)
            data = np.frombuffer(file_reader.read(NUM_SAMPLE_LENGTH), dtype=np.uint8)

            # Process image
            image = data[:4096].astype(np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

            # Process QP
            qp = np.random.choice(SELECT_QP_LIST, size=1)[0]

            # Process label
            label = np.zeros((NUM_LABEL_BYTES,))
            qp_index = int(qp)
            label[:] = data[4160 + qp_index * NUM_LABEL_BYTES: 4160 + (qp_index + 1) * NUM_LABEL_BYTES]

            # Convert to tensors
            ctu_tensor = torch.tensor(image, dtype=torch.float32).squeeze(2)
            qp_tensor = torch.tensor(float(qp), dtype=torch.float32)

            # Hierarchical output
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

            # Flatten
            y_flat_16 = y_image_16.view(-1)
            y_flat_32 = y_image_32.view(-1)
            y_flat_64 = y_image_64.view(-1)
            y_flat_valid_32 = y_image_valid_32.view(-1)
            y_flat_valid_16 = y_image_valid_16.view(-1)

            # Normalize
            ctu_tensor /= 255.0
            qp_tensor /= 51.0

            return qp_tensor, ctu_tensor, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, torch.tensor(label, dtype=torch.float32)


def create_dataloader(file_path, total_samples, batch_size, device, shuffle=False):
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    full_dataset = StreamingDataset(file_path, total_samples)
    
    return DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )


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

    # Branch B1: Mean removal at 64x64
    mean_value_level1 = torch.mean(ctu_data[:, 0:64, 0:64], dim=(1, 2), keepdim=True)
    norm_ctu_data_b1 -= mean_value_level1
    
    # Branch B2: Mean removal at 32x32
    mean_value_level2_1 = torch.mean(ctu_data[:, 0:32, 0:32], dim=(1, 2), keepdim=True)
    mean_value_level2_2 = torch.mean(ctu_data[:, 0:32, 32:64], dim=(1, 2), keepdim=True)
    mean_value_level2_3 = torch.mean(ctu_data[:, 32:64, 0:32], dim=(1, 2), keepdim=True)
    mean_value_level2_4 = torch.mean(ctu_data[:, 32:64, 32:64], dim=(1, 2), keepdim=True)
    
    norm_ctu_data_b2[:, 0:32, 0:32]   -= mean_value_level2_1
    norm_ctu_data_b2[:, 0:32, 32:64]  -= mean_value_level2_2
    norm_ctu_data_b2[:, 32:64, 0:32]  -= mean_value_level2_3
    norm_ctu_data_b2[:, 32:64, 32:64] -= mean_value_level2_4

    # Branch B3: Mean removal at 16x16
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


# ETH-CNN Model Architecture
class ETH_CNN(nn.Module):
    def __init__(self):
        super(ETH_CNN, self).__init__()
        
        # Branch 1 convolution layers
        self.conv1_b1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=4, padding=0)
        self.conv2_b1 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=2, stride=2, padding=0)
        self.conv3_b1 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=2, stride=2, padding=0)

        # Branch 2 convolution layers
        self.conv1_b2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=4, padding=0)
        self.conv2_b2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=2, stride=2, padding=0)
        self.conv3_b2 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=2, stride=2, padding=0)

        # Branch 3 convolution layers
        self.conv1_b3 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=4, padding=0)
        self.conv2_b3 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=2, stride=2, padding=0)
        self.conv3_b3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=2, stride=2, padding=0)

        # Dropout layers
        self.fc1_dropout = nn.Dropout(p=0.5)
        self.fc2_dropout = nn.Dropout(p=0.2)

        # Branch 1 FC layers
        self.fc1_b1 = nn.Linear(in_features=2688, out_features=64)
        self.fc2_b1 = nn.Linear(in_features=65, out_features=48)
        self.fc3_b1 = nn.Linear(in_features=49, out_features=1)

        # Branch 2 FC layers
        self.fc1_b2 = nn.Linear(in_features=2688, out_features=128)
        self.fc2_b2 = nn.Linear(in_features=129, out_features=96)
        self.fc3_b2 = nn.Linear(in_features=97, out_features=4)

        # Branch 3 FC layers
        self.fc1_b3 = nn.Linear(in_features=2688, out_features=256)
        self.fc2_b3 = nn.Linear(in_features=257, out_features=192)
        self.fc3_b3 = nn.Linear(in_features=193, out_features=16)

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

        # Flatten and concatenate
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

        b1_op = self.full_connect_b1(concatenated_output, qp)
        b2_op = self.full_connect_b2(concatenated_output, qp)
        b3_op = self.full_connect_b3(concatenated_output, qp)

        return (b1_op.squeeze(dim=0), b2_op.squeeze(dim=0), b3_op.squeeze(dim=0))


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

    # Accuracy for 64x64
    correct_prediction_64 = torch.round(y_conv_flat_64) == torch.round(y_flat_64)
    accuracy_64 = torch.mean(correct_prediction_64.float()) * 100

    # Accuracy for 32x32
    correct_prediction_valid_32 = y_flat_valid_32 * (torch.round(y_conv_flat_32) == torch.round(y_flat_32)).float()
    accuracy_32 = torch.sum(y_flat_valid_32 * correct_prediction_valid_32) / (torch.sum(y_flat_valid_32) + epsilon) * 100

    # Accuracy for 16x16
    correct_prediction_valid_16 = y_flat_valid_16 * (torch.round(y_conv_flat_16) == torch.round(y_flat_16)).float()
    accuracy_16 = torch.sum(y_flat_valid_16 * correct_prediction_valid_16) / (torch.sum(y_flat_valid_16) + epsilon) * 100

    avg_acc = (accuracy_64 + accuracy_32 + accuracy_16) / 3

    return avg_acc, accuracy_64, accuracy_32, accuracy_16


def evaluate_model(model, data_loader, device, dataset_name="Validation"):
    """
    Comprehensive evaluation function
    """
    model.eval()
    
    total_loss = 0.0
    total_loss_l1 = 0.0
    total_loss_l2 = 0.0
    total_loss_l3 = 0.0
    
    total_acc = 0.0
    total_acc_64 = 0.0
    total_acc_32 = 0.0
    total_acc_16 = 0.0
    
    num_batches = 0
    
    # For collecting predictions and ground truth
    all_predictions_64 = []
    all_ground_truth_64 = []
    all_predictions_32 = []
    all_ground_truth_32 = []
    all_predictions_16 = []
    all_ground_truth_16 = []
    
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name} Dataset")
    print(f"{'='*60}\n")
    
    with torch.no_grad():
        for batch_idx, (qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, 
                       y_flat_valid_32, y_flat_valid_16, label_batch) in enumerate(data_loader):
            
            # Move data to device
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
            
            # Forward pass
            outputs = model((qp_batch, ctu_batch))
            
            # Calculate loss
            l1_loss, l2_loss, l3_loss, combined_loss = custom_repo_loss(
                true_labels[0], outputs[0],
                true_labels[1], outputs[1], true_valid_labels[0],
                true_labels[2], outputs[2], true_valid_labels[1]
            )
            
            # Calculate accuracy
            avg_acc, acc_64, acc_32, acc_16 = calculate_accuracy_repo(
                true_labels[0], outputs[0],
                true_labels[1], outputs[1], true_valid_labels[0],
                true_labels[2], outputs[2], true_valid_labels[1]
            )
            
            # Accumulate metrics
            total_loss += combined_loss.item()
            total_loss_l1 += l1_loss.item()
            total_loss_l2 += l2_loss.item()
            total_loss_l3 += l3_loss.item()
            
            total_acc += avg_acc.item()
            total_acc_64 += acc_64.item()
            total_acc_32 += acc_32.item()
            total_acc_16 += acc_16.item()
            
            num_batches += 1
            
            # Collect predictions for confusion matrix
            pred_64 = torch.round(outputs[0]).cpu().numpy().flatten()
            true_64 = torch.round(true_labels[0]).cpu().numpy().flatten()
            all_predictions_64.extend(pred_64)
            all_ground_truth_64.extend(true_64)
            
            pred_32 = torch.round(outputs[1]).cpu().numpy().flatten()
            true_32 = torch.round(true_labels[1]).cpu().numpy().flatten()
            all_predictions_32.extend(pred_32)
            all_ground_truth_32.extend(true_32)
            
            pred_16 = torch.round(outputs[2]).cpu().numpy().flatten()
            true_16 = torch.round(true_labels[2]).cpu().numpy().flatten()
            all_predictions_16.extend(pred_16)
            all_ground_truth_16.extend(true_16)
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f"Progress: {batch_idx + 1}/{len(data_loader)} batches processed")
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_loss_l1 = total_loss_l1 / num_batches
    avg_loss_l2 = total_loss_l2 / num_batches
    avg_loss_l3 = total_loss_l3 / num_batches
    
    avg_acc = total_acc / num_batches
    avg_acc_64 = total_acc_64 / num_batches
    avg_acc_32 = total_acc_32 / num_batches
    avg_acc_16 = total_acc_16 / num_batches
    
    # Print results
    print(f"\n{'-'*60}")
    print(f"EVALUATION RESULTS - {dataset_name} Dataset")
    print(f"{'-'*60}")
    print(f"\nLoss Metrics:")
    print(f"  Total Loss:     {avg_loss:.6f}")
    print(f"  Loss L1 (64x64): {avg_loss_l1:.6f}")
    print(f"  Loss L2 (32x32): {avg_loss_l2:.6f}")
    print(f"  Loss L3 (16x16): {avg_loss_l3:.6f}")
    print(f"\nAccuracy Metrics:")
    print(f"  Overall Accuracy: {avg_acc:.2f}%")
    print(f"  Accuracy L1 (64x64): {avg_acc_64:.2f}%")
    print(f"  Accuracy L2 (32x32): {avg_acc_32:.2f}%")
    print(f"  Accuracy L3 (16x16): {avg_acc_16:.2f}%")
    print(f"{'-'*60}\n")
    
    # Create results dictionary
    results = {
        'total_loss': avg_loss,
        'loss_l1': avg_loss_l1,
        'loss_l2': avg_loss_l2,
        'loss_l3': avg_loss_l3,
        'overall_accuracy': avg_acc,
        'accuracy_64': avg_acc_64,
        'accuracy_32': avg_acc_32,
        'accuracy_16': avg_acc_16,
        'predictions_64': all_predictions_64,
        'ground_truth_64': all_ground_truth_64,
        'predictions_32': all_predictions_32,
        'ground_truth_32': all_ground_truth_32,
        'predictions_16': all_predictions_16,
        'ground_truth_16': all_ground_truth_16
    }
    
    return results


def plot_confusion_matrices(results, save_dir='evaluation_results'):
    """
    Plot confusion matrices for all three levels
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    levels = ['64x64', '32x32', '16x16']
    pred_keys = ['predictions_64', 'predictions_32', 'predictions_16']
    truth_keys = ['ground_truth_64', 'ground_truth_32', 'ground_truth_16']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (level, pred_key, truth_key) in enumerate(zip(levels, pred_keys, truth_keys)):
        cm = confusion_matrix(results[truth_key], results[pred_key])
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                   ax=axes[idx], cbar_kws={'label': 'Percentage'})
        axes[idx].set_title(f'Confusion Matrix - {level}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to {save_dir}/confusion_matrices.png")
    plt.close()


def plot_accuracy_comparison(results, save_dir='evaluation_results'):
    """
    Plot accuracy comparison across different levels
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    levels = ['64x64\n(L1)', '32x32\n(L2)', '16x16\n(L3)', 'Overall']
    accuracies = [
        results['accuracy_64'],
        results['accuracy_32'],
        results['accuracy_16'],
        results['overall_accuracy']
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(levels, accuracies, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Accuracy Across Different CU Partition Levels', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Accuracy comparison saved to {save_dir}/accuracy_comparison.png")
    plt.close()


def plot_loss_comparison(results, save_dir='evaluation_results'):
    """
    Plot loss comparison across different levels
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    levels = ['64x64\n(L1)', '32x32\n(L2)', '16x16\n(L3)', 'Total']
    losses = [
        results['loss_l1'],
        results['loss_l2'],
        results['loss_l3'],
        results['total_loss']
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(levels, losses, color=['#9b59b6', '#e67e22', '#1abc9c', '#34495e'])
    
    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('Loss', fontsize=12)
    plt.title('Model Loss Across Different CU Partition Levels', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Loss comparison saved to {save_dir}/loss_comparison.png")
    plt.close()


def save_detailed_report(results, dataset_name, save_dir='evaluation_results'):
    """
    Save detailed evaluation report to text file
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    report_path = f'{save_dir}/evaluation_report_{dataset_name}.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"ETH-CNN MODEL EVALUATION REPORT - {dataset_name} Dataset\n")
        f.write("="*80 + "\n\n")
        
        f.write("LOSS METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Loss:          {results['total_loss']:.6f}\n")
        f.write(f"Loss L1 (64x64):     {results['loss_l1']:.6f}\n")
        f.write(f"Loss L2 (32x32):     {results['loss_l2']:.6f}\n")
        f.write(f"Loss L3 (16x16):     {results['loss_l3']:.6f}\n\n")
        
        f.write("ACCURACY METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Overall Accuracy:    {results['overall_accuracy']:.4f}%\n")
        f.write(f"Accuracy L1 (64x64): {results['accuracy_64']:.4f}%\n")
        f.write(f"Accuracy L2 (32x32): {results['accuracy_32']:.4f}%\n")
        f.write(f"Accuracy L3 (16x16): {results['accuracy_16']:.4f}%\n\n")
        
        # Classification reports
        for level, pred_key, truth_key in [('64x64', 'predictions_64', 'ground_truth_64'),
                                           ('32x32', 'predictions_32', 'ground_truth_32'),
                                           ('16x16', 'predictions_16', 'ground_truth_16')]:
            f.write(f"\nCLASSIFICATION REPORT - {level}\n")
            f.write("-"*80 + "\n")
            report = classification_report(results[truth_key], results[pred_key], 
                                          target_names=['Not Split', 'Split'], zero_division=0)
            f.write(report + "\n")
    
    print(f"Detailed report saved to {report_path}")


def main():
    # Configuration
    BATCH_SIZE = 64
    MODEL_PATH = 'best_model_4qp_parallel_data_processing_loss_mod.pth'
    
    # Dataset paths
    validation_file_path = "/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/Extract_Data/AI_Valid_98175.dat_shuffled"
    test_file_path = "/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/Extract_Data/AI_Test_196350.dat_shuffled"
    
    VALIDSET_MAXSIZE = 98175
    TESTSET_MAXSIZE = 196350
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = ETH_CNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    print("Model loaded successfully!")
    
    # Create data loaders
    print("\nCreating validation data loader...")
    validation_loader = create_dataloader(validation_file_path, VALIDSET_MAXSIZE, 
                                         BATCH_SIZE, device, shuffle=False)
    
    print("Creating test data loader...")
    test_loader = create_dataloader(test_file_path, TESTSET_MAXSIZE, 
                                   BATCH_SIZE, device, shuffle=False)
    
    # Evaluate on validation set
    print("\n" + "="*80)
    print("STARTING VALIDATION SET EVALUATION")
    print("="*80)
    val_results = evaluate_model(model, validation_loader, device, "Validation")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("STARTING TEST SET EVALUATION")
    print("="*80)
    test_results = evaluate_model(model, test_loader, device, "Test")
    
    # Generate visualizations and reports for validation set
    print("\nGenerating validation set visualizations...")
    plot_confusion_matrices(val_results, save_dir='evaluation_results/validation')
    plot_accuracy_comparison(val_results, save_dir='evaluation_results/validation')
    plot_loss_comparison(val_results, save_dir='evaluation_results/validation')
    save_detailed_report(val_results, "Validation", save_dir='evaluation_results/validation')
    
    # Generate visualizations and reports for test set
    print("\nGenerating test set visualizations...")
    plot_confusion_matrices(test_results, save_dir='evaluation_results/test')
    plot_accuracy_comparison(test_results, save_dir='evaluation_results/test')
    plot_loss_comparison(test_results, save_dir='evaluation_results/test')
    save_detailed_report(test_results, "Test", save_dir='evaluation_results/test')
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\nResults saved to 'evaluation_results/' directory")
    print("  - validation/: Validation set results")
    print("  - test/: Test set results")
    print("\nGenerated files:")
    print("  - confusion_matrices.png: Confusion matrices for all levels")
    print("  - accuracy_comparison.png: Accuracy comparison chart")
    print("  - loss_comparison.png: Loss comparison chart")
    print("  - evaluation_report_*.txt: Detailed text report")


if __name__ == "__main__":
    main()