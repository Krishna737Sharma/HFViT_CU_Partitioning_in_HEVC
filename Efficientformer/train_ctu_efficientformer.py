# ----------------------------------------------------------------------------------
# Training script for HEVC CTU Partition Prediction
# Using the modified QP-Aware EfficientFormerV2 Model
# ----------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
import os
import wandb
import time

# Import the modified EfficientFormerV2 model
from efficientformer_v2_ctu import efficientformerv2_s0_ctu # Use the CTU variant

# ==================================================================================
# SECTION 1: DATA LOADING (Copied from your previous scripts)
# ==================================================================================
DEBUG = False
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_LABEL_BYTES = 16
NUM_SAMPLE_LENGTH = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + 64 + (51 + 1) * NUM_LABEL_BYTES
SELECT_QP_LIST = [22, 27, 32, 37]
# 21 total outputs: 1 (64x64) + 4 (32x32) + 16 (16x16)
NUM_CLASSES = 21

class StreamingDataset(Dataset):
    """ Efficiently loads HEVC CU data samples from a large binary file """
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

            ctu_tensor = torch.from_numpy(image).float().permute(2, 0, 1) # C, H, W
            qp_tensor = torch.tensor(float(qp), dtype=torch.float32)

            ctu_tensor /= 255.0
            qp_tensor /= 51.0 # Normalize QP

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

def create_subset_dataloader(file_path, total_samples, subset_size, batch_size, shuffle=True):
    """ Creates a DataLoader from a random subset of the full dataset. """
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % (2**32)
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    full_dataset = StreamingDataset(file_path, total_samples)
    subset_indices = random.sample(range(total_samples), subset_size)

    return DataLoader(
        Subset(full_dataset, subset_indices),
        batch_size=batch_size, shuffle=shuffle, num_workers=2,
        pin_memory=True, worker_init_fn=worker_init_fn
    ), subset_indices

# ==================================================================================
# SECTION 2: ACCURACY CALCULATION (Copied from your previous scripts)
# ==================================================================================
def calculate_accuracy_repo(y_flat_64, y_conv_flat_64,
                            y_flat_32, y_conv_flat_32, y_flat_valid_32,
                            y_flat_16, y_conv_flat_16, y_flat_valid_16):
    device = y_flat_64.device
    y_conv_flat_64 = y_conv_flat_64.to(device, non_blocking=True)
    y_flat_32 = y_flat_32.to(device, non_blocking=True)
    y_conv_flat_32 = y_conv_flat_32.to(device, non_blocking=True)
    y_flat_valid_32 = y_flat_valid_32.to(device, non_blocking=True)
    y_flat_16 = y_flat_16.to(device, non_blocking=True)
    y_conv_flat_16 = y_conv_flat_16.to(device, non_blocking=True)
    y_flat_valid_16 = y_flat_valid_16.to(device, non_blocking=True)

    epsilon = 1e-12
    correct_prediction_64 = torch.round(y_conv_flat_64) == torch.round(y_flat_64)
    accuracy_64 = torch.mean(correct_prediction_64.float()) * 100

    comparison_32 = (torch.round(y_conv_flat_32) == torch.round(y_flat_32)).float()
    correct_prediction_valid_32 = y_flat_valid_32 * comparison_32
    accuracy_32 = torch.sum(correct_prediction_valid_32) / (torch.sum(y_flat_valid_32) + epsilon) * 100

    comparison_16 = (torch.round(y_conv_flat_16) == torch.round(y_flat_16)).float()
    correct_prediction_valid_16 = y_flat_valid_16 * comparison_16
    accuracy_16 = torch.sum(correct_prediction_valid_16) / (torch.sum(y_flat_valid_16) + epsilon) * 100

    count = 1.0
    valid_count_32 = torch.sum(y_flat_valid_32) > 0
    valid_count_16 = torch.sum(y_flat_valid_16) > 0
    if valid_count_32: count += 1.0
    if valid_count_16: count += 1.0

    accuracy_32 = torch.nan_to_num(accuracy_32, nan=0.0)
    accuracy_16 = torch.nan_to_num(accuracy_16, nan=0.0)

    avg_acc = (accuracy_64 + accuracy_32 + accuracy_16) / count

    if not isinstance(avg_acc, torch.Tensor):
         avg_acc = torch.tensor(avg_acc, device=device)

    return avg_acc, accuracy_64, accuracy_32, accuracy_16

# ==================================================================================
# SECTION 3: TRAINING & VALIDATION
# ==================================================================================
def main():
    # --- Configuration ---
    train_file_path = "/root/myproject/HEVC_Intra_Models-ViT/Data/AI_Train_1668975.dat_shuffled"
    validation_file_path = "/root/myproject/HEVC_Intra_Models-ViT/Data/AI_Valid_98175.dat_shuffled"
    TRAINSET_MAXSIZE = 1668975
    VALIDSET_MAXSIZE = 98175
    BATCH_SIZE = 64
    MODEL_VARIANT = 'efficientformerv2_s0_ctu' # Ensure this matches the imported model

    # --- Initialize wandb ---
    wandb.login(key="5c560f0045b5a49dcf8caa862e58469329427192") # Replace with your key

    config = {
        "learning_rate": 0.001,
        "optimizer": "AdamW",
        "weight_decay": 0.05,
        "epochs": 300, # Start with fewer epochs for testing
        "architecture": MODEL_VARIANT,
        "batch_size": BATCH_SIZE,
    }

    wandb.init(project="EfficientFormerV2-HEVC-Partition", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DataLoaders ---
    train_loader, _ = create_subset_dataloader(train_file_path, TRAINSET_MAXSIZE, 500, BATCH_SIZE, shuffle=True)
    validation_loader, _ = create_subset_dataloader(validation_file_path, VALIDSET_MAXSIZE, 100, BATCH_SIZE, shuffle=False)

    # --- Model, Optimizer, Loss ---
    print(f"Creating model: {MODEL_VARIANT}")
    model = efficientformerv2_s0_ctu(pretrained=False).to(device) # Call the specific function

    wandb.watch(model)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {param_count:,}")
    wandb.config.update({"parameters": param_count}) # Log params to wandb

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.BCELoss()

    # --- Checkpoint and Patience Variables ---
    best_val_acc = 0.0
    start_epoch = 0
    patience = 50
    patience_counter = 0
    num_patience_counter_changed = 0
    checkpoint_path = f'best_{MODEL_VARIANT}_model.pth'

    # --- Load Checkpoint ---
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1 # Start from next epoch
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0; best_val_acc = 0.0

    # --- Training Loop ---
    print("--- Starting Training ---")
    for epoch in range(start_epoch, config['epochs']):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        # --- Dataset Refresh ---
        if num_patience_counter_changed >= 5:
             print("Patience threshold met multiple times. Refreshing datasets.")
             num_patience_counter_changed = 0; best_val_acc = 0.0 # Reset
             train_loader, _ = create_subset_dataloader(train_file_path, TRAINSET_MAXSIZE, 80000, BATCH_SIZE, shuffle=True)
             validation_loader, _ = create_subset_dataloader(validation_file_path, VALIDSET_MAXSIZE, 10000, BATCH_SIZE, shuffle=False)
             print("Datasets refreshed.")

        # --- Train Epoch ---
        for batch_idx, batch in enumerate(train_loader):
            qp_batch, ctu_batch, _, _, _, _, _, target = batch
            inputs, qp_tensor, target = ctu_batch.to(device), qp_batch.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, qp_tensor) # Pass both inputs
            loss = criterion(outputs, target)

            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping."); continue

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # --- Validation Epoch ---
        model.eval()
        val_loss, val_acc, val_count = 0.0, 0.0, 0
        with torch.no_grad():
            for val_batch in validation_loader:
                qp_batch, ctu_batch, y64, y32, y16, v32, v16, target = val_batch
                inputs, qp_tensor, target = ctu_batch.to(device), qp_batch.to(device), target.to(device)
                y64d, y32d, y16d = y64.to(device), y32.to(device), y16.to(device)
                v32d, v16d = v32.to(device), v16.to(device)

                outputs = model(inputs, qp_tensor)
                loss = criterion(outputs, target)

                if torch.isnan(loss): print(f"NaN validation loss @ epoch {epoch+1}. Skipping."); continue

                val_loss += loss.item(); val_count += 1
                avg_acc, _, _, _ = calculate_accuracy_repo(y64d, outputs[:, 0:1], y32d, outputs[:, 1:5], v32d, y16d, outputs[:, 5:21], v16d)
                if not torch.isnan(avg_acc): val_acc += avg_acc.item()
                else: print(f"NaN accuracy @ epoch {epoch+1}.")

        avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
        avg_val_acc = val_acc / val_count if val_count > 0 else 0.0
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{config['epochs']} | T: {epoch_time:.2f}s | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")

        wandb.log({
            "epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
            "val_accuracy": avg_val_acc, "learning_rate": current_lr
        })

        # --- Save Checkpoint & Patience ---
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc; patience_counter = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, checkpoint_path)
            print(f"---> New best model saved @ {avg_val_acc:.2f}%")
        else:
             patience_counter += 1
             if patience_counter >= patience:
                print(f"Patience hit. Reloading best model.")
                patience_counter = 0; num_patience_counter_changed += 1
                if os.path.exists(checkpoint_path):
                    try:
                        checkpoint = torch.load(checkpoint_path, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        best_val_acc = checkpoint.get('best_val_acc', 0.0)
                        print(f"Best model restored (Acc: {best_val_acc:.2f}%).")
                    except Exception as e: print(f"Error reload: {e}.")
                else: print("WARNING: No best model checkpoint found.")

        scheduler.step()

if __name__ == '__main__':
    main()

