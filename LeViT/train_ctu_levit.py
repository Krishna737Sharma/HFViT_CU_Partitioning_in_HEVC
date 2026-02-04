# ----------------------------------------------------------------------------------
# Training script for HEVC CTU Partition Prediction
# Using the modified QP-Aware LeViT Model
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

# Import the modified LeViT model and example config
from levit_ctu import LeViT_CTU, LeViT_128S_CTU_config # Using 128S as default

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
    """
    Efficiently loads HEVC CU data samples from a large binary file
    without loading the entire file into memory.
    """
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

            # Extract and normalize image (CTU) and QP
            image = data[:4096].astype(np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
            qp = np.random.choice(SELECT_QP_LIST, size=1)[0]
            
            # Extract label based on the selected QP
            label = np.zeros((NUM_LABEL_BYTES,))
            qp_index = int(qp)
            label[:] = data[4160 + qp_index * NUM_LABEL_BYTES: 4160 + (qp_index + 1) * NUM_LABEL_BYTES]

            # [H, W, C] -> [C, H, W]
            ctu_tensor = torch.from_numpy(image).float().permute(2, 0, 1) 
            qp_tensor = torch.tensor(float(qp), dtype=torch.float32)

            # Normalize
            ctu_tensor /= 255.0
            qp_tensor /= 51.0

            # Process labels for multi-level loss calculation
            y_image = torch.tensor(label, dtype=torch.float32).view(1, 4, 4)
            y_image_16 = F.relu(y_image - 2)
            
            # Use avg_pool2d on (1, 4, 4) tensor
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

            # Final target tensor for the model (21 outputs)
            target = torch.cat((y_flat_64, y_flat_32, y_flat_16), dim=0)

            return qp_tensor, ctu_tensor, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target

def create_subset_dataloader(file_path, total_samples, subset_size, batch_size, shuffle=True):
    """Creates a DataLoader from a random subset of the full dataset."""
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % (2**32)
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
    
    full_dataset = StreamingDataset(file_path, total_samples)
    subset_indices = random.sample(range(total_samples), subset_size)
    
    return DataLoader(
        Subset(full_dataset, subset_indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    ), subset_indices

# ==================================================================================
# SECTION 2: ACCURACY CALCULATION (Copied from your previous scripts)
# ==================================================================================

def calculate_accuracy_repo(y_flat_64, y_conv_flat_64,
                            y_flat_32, y_conv_flat_32, y_flat_valid_32,
                            y_flat_16, y_conv_flat_16, y_flat_valid_16):
    device = y_flat_64.device
    # Ensure all tensors are on the same device before comparison
    y_conv_flat_64 = y_conv_flat_64.to(device, non_blocking=True)
    y_flat_32 = y_flat_32.to(device, non_blocking=True)
    y_conv_flat_32 = y_conv_flat_32.to(device, non_blocking=True)
    y_flat_valid_32 = y_flat_valid_32.to(device, non_blocking=True)
    y_flat_16 = y_flat_16.to(device, non_blocking=True)
    y_conv_flat_16 = y_conv_flat_16.to(device, non_blocking=True)
    y_flat_valid_16 = y_flat_valid_16.to(device, non_blocking=True)
    
    epsilon = 1e-12
    
    # Accuracy for 64x64 (1 output)
    correct_prediction_64 = torch.round(y_conv_flat_64) == torch.round(y_flat_64)
    accuracy_64 = torch.mean(correct_prediction_64.float()) * 100
    
    # Accuracy for 32x32 (4 outputs)
    # Ensure multiplication happens only where y_flat_valid_32 is 1
    comparison_32 = (torch.round(y_conv_flat_32) == torch.round(y_flat_32)).float()
    correct_prediction_valid_32 = y_flat_valid_32 * comparison_32 
    accuracy_32 = torch.sum(correct_prediction_valid_32) / (torch.sum(y_flat_valid_32) + epsilon) * 100
    
    # Accuracy for 16x16 (16 outputs)
    # Ensure multiplication happens only where y_flat_valid_16 is 1
    comparison_16 = (torch.round(y_conv_flat_16) == torch.round(y_flat_16)).float()
    correct_prediction_valid_16 = y_flat_valid_16 * comparison_16
    accuracy_16 = torch.sum(correct_prediction_valid_16) / (torch.sum(y_flat_valid_16) + epsilon) * 100
    
    # Average accuracy - handle potential NaN if sums are zero
    valid_sums = torch.sum(y_flat_valid_32) + torch.sum(y_flat_valid_16)
    count = 1.0 + (1.0 if torch.sum(y_flat_valid_32) > 0 else 0.0) + (1.0 if torch.sum(y_flat_valid_16) > 0 else 0.0)
    
    # Ensure accuracies are tensors before checking isfinite
    accuracy_32 = torch.nan_to_num(accuracy_32, nan=0.0) # Replace NaN with 0 if sum was 0
    accuracy_16 = torch.nan_to_num(accuracy_16, nan=0.0) # Replace NaN with 0 if sum was 0

    avg_acc = (accuracy_64 + accuracy_32 + accuracy_16) / count

    if DEBUG:
        print("DEBUG: Accuracy per branch:")
        print(f"       64-part: {accuracy_64.item():.2f}%")
        print(f"       32-part: {accuracy_32.item():.2f}% (Count: {torch.sum(y_flat_valid_32).item()})")
        print(f"       16-part: {accuracy_16.item():.2f}% (Count: {torch.sum(y_flat_valid_16).item()})")
        print(f"       Average ({count}): {avg_acc.item():.2f}%")
        
    # Ensure avg_acc is a tensor before returning
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
    MODEL_VARIANT = '128S' # Choose '128S', '256', etc.
    
    # --- Initialize wandb ---
    wandb.login(key="5c560f0045b5a49dcf8caa862e58469329427192") 
    
    # Get model config based on variant choice
    if MODEL_VARIANT == '128S':
        model_cfg = LeViT_128S_CTU_config
        arch_name = "LeViT-128S-CTU"
    elif MODEL_VARIANT == '256':
         model_cfg = LeViT_256_CTU_config
         arch_name = "LeViT-256-CTU"
    # Add other variants if needed
    else:
        raise ValueError(f"Unsupported LeViT variant: {MODEL_VARIANT}")

    
    config = {
        "learning_rate": 0.001, # LeViT paper uses 5e-4 * batch_size * world_size / 512
        "optimizer": "AdamW",
        "epochs": 1000, # LeViT paper uses 1000
        "architecture": arch_name,
        "batch_size": BATCH_SIZE,
        "embed_dim": model_cfg['embed_dim'],
        "depth": model_cfg['depth'],
        "num_heads": model_cfg['num_heads'],
    }

    wandb.init(
        project="LeViT-HEVC-Partition", 
        config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DataLoaders ---
    train_loader, train_indices = create_subset_dataloader(train_file_path, TRAINSET_MAXSIZE, 500, BATCH_SIZE, shuffle=True)
    validation_loader, validation_indices = create_subset_dataloader(validation_file_path, VALIDSET_MAXSIZE, 100, BATCH_SIZE, shuffle=False)
    
    # --- Model, Optimizer, Loss ---
    print(f"Creating model: {arch_name}")
    model = LeViT_CTU(
        img_size=IMAGE_SIZE,
        in_chans=NUM_CHANNELS,
        num_classes=NUM_CLASSES,
        **model_cfg # Pass the chosen config dictionary
    ).to(device)
    
    wandb.watch(model) 
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.025) # LeViT uses 0.025
    # LeViT uses Cosine scheduler with 5 warmup epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs']-5) # Adjust T_max for warmup
    
    # Use BCELoss because our model's output is passed through sigmoid
    criterion = nn.BCELoss()
    
    # --- Checkpoint and Patience Variables ---
    best_val_loss = 0.0
    start_epoch = 0
    patience = 50 # Or use LeViT paper's schedule details
    patience_counter = 0
    checkpoint_path = f'best_levit_{MODEL_VARIANT.lower()}_ctu_model.pth' # Variant specific name

    # --- Load Checkpoint if exists ---
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', 0.0) # Handle older checkpoints
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            best_val_loss = 0.0


    # --- Training Loop ---
    print("--- Starting Training ---")
    for epoch in range(start_epoch, config['epochs']):
        # Manual Warmup (LeViT paper uses 5 epochs)
        if epoch < 5:
             lr_scale = min(1., float(epoch + 1) / 5.)
             for pg in optimizer.param_groups:
                 pg['lr'] = config['learning_rate'] * lr_scale

        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Get all data from the loader
            qp_batch, ctu_batch, y64, y32, y16, v32, v16, target = batch
            
            inputs = ctu_batch.to(device)
            qp_tensor = qp_batch.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            
            # Pass both inputs and qp_tensor to the model
            outputs = model(inputs, qp_tensor)
            
            loss = criterion(outputs, target)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                continue # Skip optimizer step and backward if loss is NaN

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        
        # --- Validation Loop ---
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        val_count = 0
        with torch.no_grad():
            for val_batch in validation_loader:
                qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target = val_batch
                
                # Move ground truth tensors needed for accuracy calc to device *inside* the loop
                y_flat_64_dev = y_flat_64.to(device, non_blocking=True)
                y_flat_32_dev = y_flat_32.to(device, non_blocking=True)
                y_flat_16_dev = y_flat_16.to(device, non_blocking=True)
                y_flat_valid_32_dev = y_flat_valid_32.to(device, non_blocking=True)
                y_flat_valid_16_dev = y_flat_valid_16.to(device, non_blocking=True)

                inputs = ctu_batch.to(device)
                qp_tensor = qp_batch.to(device)
                target = target.to(device)
                
                outputs = model(inputs, qp_tensor)
                
                loss = criterion(outputs, target)

                if torch.isnan(loss):
                    print(f"NaN validation loss detected at epoch {epoch+1}. Skipping batch.")
                    continue
                    
                val_loss += loss.item()
                val_count += 1
                
                # Use the custom accuracy function with tensors on the correct device
                avg_acc, _, _, _ = calculate_accuracy_repo(
                    y_flat_64_dev, outputs[:, 0:1],       # 64x64 split (1 output)
                    y_flat_32_dev, outputs[:, 1:5], y_flat_valid_32_dev, # 32x32 splits (4 outputs)
                    y_flat_16_dev, outputs[:, 5:21], y_flat_valid_16_dev # 16x16 splits (16 outputs)
                )
                if not torch.isnan(avg_acc): # Check if accuracy is valid
                     val_acc += avg_acc.item()
                else:
                     print(f"NaN accuracy detected at epoch {epoch+1}.")


        avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
        avg_val_acc = val_acc / val_count if val_count > 0 else 0.0
        epoch_time = time.time() - start_time
        
        current_lr = optimizer.param_groups[0]['lr'] # Get current LR

        print(f"Epoch {epoch+1}/{config['epochs']} | Time: {epoch_time:.2f}s | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")
        
        wandb.log({
            "epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
            "val_accuracy": avg_val_acc, "learning_rate": current_lr
        })
            
        # --- Model Saving and Patience ---
        # Save based on validation accuracy improvement
        if avg_val_acc > best_val_loss: # Rename best_val_loss to best_val_acc
            best_val_acc = avg_val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc, # Save best accuracy
            }, checkpoint_path)
            print(f"----> New best model saved with validation accuracy: {avg_val_acc:.2f}%")
        else:
             patience_counter += 1
        
             if patience_counter >= patience:
                print(f"Patience of {patience} epochs reached with no accuracy improvement. Reloading best model and changing training data.")
                patience_counter = 0 # Reset counter after action
                 
                # Load the best model weights
                if os.path.exists(checkpoint_path):
                    print(f"Loading best model from {checkpoint_path}...")
                    try:
                        checkpoint = torch.load(checkpoint_path, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        # Optional: Restore optimizer/scheduler if needed, or reset them
                        # optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
                        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        best_val_acc = checkpoint.get('best_val_acc', 0.0) # Restore best acc
                        print(f"Best model weights restored (Val Acc: {best_val_acc:.2f}%).")
                    except Exception as e:
                        print(f"Error reloading best model: {e}. Continuing with current model.")
                else:
                    print("WARNING: No checkpoint found to reload. Continuing with current model.")

                print("Changing training dataset...")
                new_train_loader, new_train_indices = create_subset_dataloader(train_file_path, TRAINSET_MAXSIZE, 80000, BATCH_SIZE, shuffle=True)
                
                # Update loaders and indices
                train_loader, train_indices = new_train_loader, new_train_indices
                print("Training dataset has been refreshed.")

        # Step the scheduler *after* the warmup phase
        if epoch >= 5:
            scheduler.step()

if __name__ == '__main__':
    main()