# ----------------------------------------------------------------------------------
# Training script for HEVC CTU Partition Prediction
# Using the modified QP-Aware EfficientViT Model
# ----------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
import os
import wandb # For logging metrics
import time

# Import the modified model and configs
from efficientvit_ctu import EfficientViT
from build_configs import EfficientViT_m0, EfficientViT_m0_TINY # Import configs

# ==================================================================================
# SECTION 1: DATA LOADING
# ==================================================================================
DEBUG = False
IMAGE_SIZE = 64        # Input image size (64x64)
NUM_CHANNELS = 1       # Input channels (1 for luma)
NUM_LABEL_BYTES = 16   # The label for one QP is a 4x4 grid (16 bytes)
# Calculated length of a single sample in the binary data file
NUM_SAMPLE_LENGTH = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + 64 + (51 + 1) * NUM_LABEL_BYTES
SELECT_QP_LIST = [22, 27, 32, 37] # QPs to train on
# 21 total outputs: 1 (64x64) + 4 (32x32) + 16 (16x16)
NUM_CLASSES = 21 

class StreamingDataset(Dataset):
    """
    Efficiently loads HEVC CU data samples from a large binary file
    without loading the entire file into memory.
    
    It reads only the required bytes for a single sample at a time
    using file.seek().
    """
    def __init__(self, file_path, max_samples):
        self.file_path = file_path      # Path to the large .dat file
        self.max_samples = max_samples  # Total number of samples in the file

    def __len__(self):
        # Return the total number of samples
        return self.max_samples

    def __getitem__(self, idx):
        """
        Fetches and processes a single sample by its index.
        """
        # 'with open' ensures the file is closed even if errors occur
        with open(self.file_path, 'rb') as file_reader:
            # Calculate the byte offset for the desired sample
            offset = idx * NUM_SAMPLE_LENGTH
            # Seek to that position in the file
            file_reader.seek(offset)
            # Read the exact number of bytes for one sample
            data = np.frombuffer(file_reader.read(NUM_SAMPLE_LENGTH), dtype=np.uint8)

            # --- Extract Image (CTU) ---
            # First 4096 bytes are the 64x64 luma plane
            image = data[:4096].astype(np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
            
            # --- Extract QP ---
            # Randomly select one QP from the available list for this sample
            # This acts as a form of data augmentation
            qp = np.random.choice(SELECT_QP_LIST, size=1)[0]
            
            # --- Extract Label (based on selected QP) ---
            # The data file contains labels for all QPs (0-51).
            # We must extract the specific 16-byte label corresponding
            # to the randomly selected QP.
            label = np.zeros((NUM_LABEL_BYTES,))
            qp_index = int(qp)
            # 4160 is the offset where labels start (4096 (image) + 64 (padding?))
            label_start_index = 4160 + qp_index * NUM_LABEL_BYTES
            label_end_index = 4160 + (qp_index + 1) * NUM_LABEL_BYTES
            label[:] = data[label_start_index : label_end_index]

            # --- Convert to Tensors ---
            # [H, W, C] -> [C, H, W] for PyTorch
            ctu_tensor = torch.from_numpy(image).float().permute(2, 0, 1) 
            qp_tensor = torch.tensor(float(qp), dtype=torch.float32)

            # --- Normalize Inputs ---
            # Normalize image pixels from [0, 255] to [0.0, 1.0]
            ctu_tensor /= 255.0
            # Normalize QP from [0, 51] to [0.0, 1.0]
            qp_tensor /= 51.0

            # --- Process Labels ---
            # The 16-byte (4x4) label encodes the split decisions at all levels.
            # We need to decode this into a 21-element target vector
            # corresponding to the model's 21 outputs.
            
            # Label values: 0=NoSplit, 1=Split(32), 2=Split(16), 3=Split(8)
            y_image = torch.tensor(label, dtype=torch.float32).view(1, 4, 4)
            
            # 1. 16x16 splits (16 outputs)
            # A 16x16 block is split if its label is 3 (F.relu(3-2)=1)
            y_image_16 = F.relu(y_image - 2)
            
            # Use avg_pool to check splits at higher levels
            avg_pool_result = F.avg_pool2d(y_image, kernel_size=2) # 4x4 -> 2x2
            
            # 2. 32x32 splits (4 outputs)
            # A 32x32 block is split if its label is 2 (F.relu(2-1)-F.relu(2-2)=1)
            y_image_32 = F.relu(avg_pool_result - 1) - F.relu(avg_pool_result - 2)
            
            avg_pool_result_4 = F.avg_pool2d(y_image, kernel_size=4) # 4x4 -> 1x1
            
            # 3. 64x64 split (1 output)
            # The 64x64 block is split if its label is 1 (F.relu(1-0)-F.relu(1-1)=1)
            y_image_64 = F.relu(avg_pool_result_4 - 0) - F.relu(avg_pool_result_4 - 1)
            
            # --- Create Validation Masks (for accuracy calculation) ---
            # We only want to measure 32x32 accuracy if the parent 64x64 block *was* split.
            # We only want to measure 16x16 accuracy if the parent 32x32 block *was* split.
            
            # Valid 32x32 block: label is 1, 2, or 3 (i.e., >= 1)
            y_image_valid_32 = F.relu(avg_pool_result - 0) - F.relu(avg_pool_result - 1)
            # Valid 16x16 block: label is 2 or 3 (i.e., >= 2)
            y_image_valid_16 = F.relu(y_image - 1) - F.relu(y_image - 2)

            # Flatten all tensors
            y_flat_16 = y_image_16.view(-1)         # [16]
            y_flat_32 = y_image_32.view(-1)         # [4]
            y_flat_64 = y_image_64.view(-1)         # [1]
            y_flat_valid_32 = y_image_valid_32.view(-1) # [4]
            y_flat_valid_16 = y_image_valid_16.view(-1) # [16]

            # Concatenate into the final 21-element target vector
            target = torch.cat((y_flat_64, y_flat_32, y_flat_16), dim=0) # [1, 4, 16] -> [21]

            # Return all parts for training and validation
            return (qp_tensor, ctu_tensor, 
                    y_flat_64, y_flat_32, y_flat_16, 
                    y_flat_valid_32, y_flat_valid_16, 
                    target)

def create_subset_dataloader(file_path, total_samples, subset_size, batch_size, shuffle=True):
    """
    Creates a DataLoader from a random subset of the full dataset.
    This is essential for the large StreamingDataset, as shuffling
    the entire dataset is not feasible.
    """
    def worker_init_fn(worker_id):
        """Ensures different workers have different random seeds."""
        seed = torch.initial_seed() % (2**32)
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
    
    # Create an instance of the full dataset
    full_dataset = StreamingDataset(file_path, total_samples)
    # Generate a list of random indices to form the subset
    subset_indices = random.sample(range(total_samples), subset_size)
    
    # Create a DataLoader using torch.utils.data.Subset
    return DataLoader(
        Subset(full_dataset, subset_indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,      # Use multiple workers for data loading
        pin_memory=True,    # Speeds up CPU-to-GPU data transfer
        worker_init_fn=worker_init_fn
    ), subset_indices

# ==================================================================================
# SECTION 2: ACCURACY CALCULATION
# ==================================================================================

def calculate_accuracy_repo(y_flat_64, y_conv_flat_64,
                            y_flat_32, y_conv_flat_32, y_flat_valid_32,
                            y_flat_16, y_conv_flat_16, y_flat_valid_16):
    """
    Calculates the accuracy for each partition level (64, 32, 16)
    using the validation masks.
    """
    device = y_flat_64.device
    # Ensure all tensors are on the same device
    y_conv_flat_64 = y_conv_flat_64.to(device, non_blocking=True)
    y_flat_32 = y_flat_32.to(device, non_blocking=True)
    y_conv_flat_32 = y_conv_flat_32.to(device, non_blocking=True)
    y_flat_valid_32 = y_flat_valid_32.to(device, non_blocking=True)
    y_flat_16 = y_flat_16.to(device, non_blocking=True)
    y_conv_flat_16 = y_conv_flat_16.to(device, non_blocking=True)
    y_flat_valid_16 = y_flat_valid_16.to(device, non_blocking=True)
    
    epsilon = 1e-12 # To prevent division by zero
    
    # --- Accuracy for 64x64 (1 output) ---
    # Round model output (which is post-sigmoid, 0-1) to 0 or 1
    correct_prediction_64 = torch.round(y_conv_flat_64) == torch.round(y_flat_64)
    accuracy_64 = torch.mean(correct_prediction_64.float()) * 100
    
    # --- Accuracy for 32x32 (4 outputs) ---
    # 1. Find correct predictions: (round(output) == label)
    # 2. Mask with 'y_flat_valid_32': Only count predictions where the parent 64x64 was split.
    correct_prediction_valid_32 = y_flat_valid_32 * (torch.round(y_conv_flat_32) == torch.round(y_flat_32)).float()
    # 3. Accuracy = (Sum of valid correct predictions) / (Total number of valid blocks)
    accuracy_32 = torch.sum(correct_prediction_valid_32) / (torch.sum(y_flat_valid_32) + epsilon) * 100
    
    # --- Accuracy for 16x16 (16 outputs) ---
    # 1. Find correct predictions
    # 2. Mask with 'y_flat_valid_16': Only count predictions where the parent 32x32 was split.
    correct_prediction_valid_16 = y_flat_valid_16 * (torch.round(y_conv_flat_16) == torch.round(y_flat_16)).float()
    # 3. Accuracy = (Sum of valid correct predictions) / (Total number of valid blocks)
    accuracy_16 = torch.sum(correct_prediction_valid_16) / (torch.sum(y_flat_valid_16) + epsilon) * 100
    
    # --- Average Accuracy ---
    avg_acc = (accuracy_64 + accuracy_32 + accuracy_16) / 3
    
    if DEBUG:
        print("DEBUG: Accuracy per branch:")
        print(f"       64-part: {accuracy_64.item():.2f}%")
        print(f"       32-part: {accuracy_32.item():.2f}%")
        print(f"       16-part: {accuracy_16.item():.2f}%")
        print(f"       Average: {avg_acc.item():.2f}%")
        
    return avg_acc, accuracy_64, accuracy_32, accuracy_16

# ==================================================================================
# SECTION 3: TRAINING & VALIDATION
# ==================================================================================

def main():
    # --- Configuration ---
    # !!! UPDATE THESE PATHS to your local dataset locations !!!
    train_file_path = "/root/myproject/HEVC_Intra_Models-ViT/Data/AI_Train_1668975.dat_shuffled"
    validation_file_path = "/root/myproject/HEVC_Intra_Models-ViT/Data/AI_Valid_98175.dat_shuffled"
    TRAINSET_MAXSIZE = 1668975 # Total samples in train file
    VALIDSET_MAXSIZE = 98175   # Total samples in valid file
    BATCH_SIZE = 64
    
    # --- Initialize wandb ---
    # Make sure to log in with your API key
    # wandb.login(key="YOUR_WANDB_API_KEY") 
    
    # Get model config
    model_cfg = EfficientViT_m0_TINY # Use the tiny model config
    
    # Config dictionary for wandb logging
    config = {
        "learning_rate": 0.001,
        "optimizer": "AdamW",
        "epochs": 10000, # Set a high number, will be stopped by patience
        "architecture": "EfficientViT-M0-TINY-CTU", # Updated model name
        "batch_size": BATCH_SIZE,
        "embed_dim": model_cfg['embed_dim'],
        "depth": model_cfg['depth'],
        "num_heads": model_cfg['num_heads'],
        "window_size": model_cfg['window_size'],
        "ffn_exp_ratio": model_cfg.get('ffn_exp_ratio', 2.0) # Get ratio
    }

    wandb.init(
        project="EfficientViT-HEVC-Partition", 
        config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DataLoaders ---
    # Create initial small dataloaders for testing
    train_loader, train_indices = create_subset_dataloader(train_file_path, TRAINSET_MAXSIZE, 80000, BATCH_SIZE, shuffle=True)
    validation_loader, validation_indices = create_subset_dataloader(validation_file_path, VALIDSET_MAXSIZE, 60000, BATCH_SIZE, shuffle=False)
    
    # --- Model, Optimizer, Loss ---
    
    # Get the ffn_exp_ratio from the config, default to 2.0
    ffn_ratio = model_cfg.get('ffn_exp_ratio', 2.0)
    
    print(f"Instantiating model: {config['architecture']}")
    model = EfficientViT(
        img_size=IMAGE_SIZE,
        patch_size=16, # 64 / 16 = 4x4 feature map
        in_chans=NUM_CHANNELS,
        num_classes=NUM_CLASSES,
        embed_dim=model_cfg['embed_dim'],
        key_dim=[16, 16, 16], # Default key dim
        depth=model_cfg['depth'],
        num_heads=model_cfg['num_heads'],
        window_size=model_cfg['window_size'],
        kernels=model_cfg['kernels'],
        ffn_exp_ratio=ffn_ratio, # Pass the ratio to the model
    ).to(device)
    
    wandb.watch(model, log='all', log_freq=100) # Track model gradients
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # --- LOSS FUNCTION ---
    # Use Binary Cross Entropy (BCE) Loss.
    # This is the correct choice because the model's output is
    # passed through a sigmoid, and we are treating each of the 21
    # outputs as an independent binary classification (Split / No Split).
    criterion = nn.BCELoss()
    
    # --- Checkpoint and Patience Variables ---
    best_val_loss = float('inf')
    start_epoch = 0
    patience = 50 # Number of epochs to wait for improvement before reloading
    patience_counter = 0
    checkpoint_path = 'best_efficientvit_ctu_model_tiny.pth' # Checkpoint path

    # --- Load Checkpoint if exists ---
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming training from epoch {start_epoch}")

    # --- Training Loop ---
    print("--- Starting Training ---")
    for epoch in range(start_epoch, config['epochs']):
        start_time = time.time()
        
        # --- TRAINING ---
        model.train() # Set model to training mode (enables dropout, etc.)
        running_loss = 0.0
        
        for batch in train_loader:
            # Get all data from the loader
            qp_batch, ctu_batch, _, _, _, _, _, target = batch
            
            # Move data to the device (GPU/CPU)
            inputs = ctu_batch.to(device)
            qp_tensor = qp_batch.to(device)
            target = target.to(device)

            optimizer.zero_grad() # Clear previous gradients
            
            # --- FORWARD PASS ---
            # Pass *both* the image (inputs) and the qp_tensor
            # to the model's forward method.
            outputs = model(inputs, qp_tensor)
            
            # --- LOSS & BACKWARD PASS ---
            loss = criterion(outputs, target) # Calculate loss
            loss.backward() # Backpropagate gradients
            optimizer.step() # Update model weights
            
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval() # Set model to evaluation mode
        val_loss, val_acc = 0.0, 0.0
        
        # Disable gradient calculations during validation
        with torch.no_grad():
            for val_batch in validation_loader:
                # Get all data, including label components for accuracy
                qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target = val_batch
                
                inputs = ctu_batch.to(device)
                qp_tensor = qp_batch.to(device)
                target = target.to(device)
                
                # --- FORWARD PASS (Validation) ---
                outputs = model(inputs, qp_tensor)
                
                loss = criterion(outputs, target)
                val_loss += loss.item()
                
                # --- ACCURACY CALCULATION ---
                # Slice the 21-element output tensor to match the labels
                # for each level.
                avg_acc, _, _, _ = calculate_accuracy_repo(
                    y_flat_64, outputs[:, 0:1],       # 64x64 split (1 output)
                    y_flat_32, outputs[:, 1:5], y_flat_valid_32, # 32x32 splits (4 outputs)
                    y_flat_16, outputs[:, 5:21], y_flat_valid_16 # 16x16 splits (16 outputs)
                )
                val_acc += avg_acc.item()
        
        avg_val_loss = val_loss / len(validation_loader)
        avg_val_acc = val_acc / len(validation_loader)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{config['epochs']} | Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")
        
        # --- Log metrics to wandb ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": avg_val_acc,
            "learning_rate": scheduler.get_last_lr()[0]
        })
            
        # --- Model Saving and Patience Logic ---
        if avg_val_loss < best_val_loss:
            # If validation loss improved, save the model
            best_val_loss = avg_val_loss
            patience_counter = 0 # Reset patience
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"----> New best model saved with validation loss: {avg_val_loss:.4f}")
        else:
            # If validation loss did not improve, increment patience counter
            patience_counter += 1
        
            if patience_counter >= patience:
                # --- PATIENCE REACHED ---
                print(f"Patience of {patience} epochs reached. Reloading best model and changing training data.")
                patience_counter = 0
                
                # 1. Load the best model weights
                if os.path.exists(checkpoint_path):
                    print(f"Loading best model from {checkpoint_path}...")
                    checkpoint = torch.load(checkpoint_path)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("Best model weights restored.")
                else:
                    print("WARNING: No checkpoint found to reload. Continuing with current model.")

                # 2. Create a NEW training dataloader
                # This logic re-samples the training dataset,
                # effectively creating a new "epoch" on a different
                # subset of the data.
                print("Changing training dataset...")
                # Create a larger subset for the next "super-epoch"
                new_train_loader, new_train_indices = create_subset_dataloader(
                    train_file_path, TRAINSET_MAXSIZE, 80000, BATCH_SIZE, shuffle=True
                )
                
                # Update the loader for the next loop
                train_loader, train_indices = new_train_loader, new_train_indices
                print("Training dataset has been refreshed.")

        # Step the learning rate scheduler
        scheduler.step()

if __name__ == '__main__':
    main()