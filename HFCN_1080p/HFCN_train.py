# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# --- Configuration ---
CONFIG_LEARNING_RATE = 0.0001  
CONFIG_EPOCHS = 15000         
BATCH_SIZE = 64              

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
        num_workers=4, 
        prefetch_factor=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    ), subset_indices

# --- 2. Model Architecture (HFCN) ---
class HFCN(nn.Module):
    def __init__(self):
        super(HFCN, self).__init__()

        # Block 1
        self.block1_conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.block1_bn1 = nn.BatchNorm2d(8)
        self.block1_conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.block1_bn2 = nn.BatchNorm2d(8)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.block2_conv1 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.block2_bn1 = nn.BatchNorm2d(8)
        self.block2_conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.block2_bn2 = nn.BatchNorm2d(8)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Branch 2 (16x16 output)
        self.br2_conv1 = nn.Conv2d(8, 32, kernel_size=4, stride=4, padding=0)
        self.br2_bn1 = nn.BatchNorm2d(32)
        self.br2_conv2 = nn.Conv2d(33, 16, kernel_size=1, padding=0)
        self.br2_bn2 = nn.BatchNorm2d(16)
        self.br2_conv3 = nn.Conv2d(16, 1, kernel_size=1, padding=0)

        # Block 3
        self.block3_conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.block3_bn1 = nn.BatchNorm2d(16)
        self.block3_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.block3_bn2 = nn.BatchNorm2d(16)
        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Branch 3 (32x32 output)
        self.br3_conv1 = nn.Conv2d(16, 8, kernel_size=4, stride=4, padding=0)
        self.br3_bn1 = nn.BatchNorm2d(8)
        self.br3_conv2 = nn.Conv2d(9, 4, kernel_size=1, padding=0)
        self.br3_bn2 = nn.BatchNorm2d(4)
        self.br3_conv3 = nn.Conv2d(4, 1, kernel_size=1, padding=0)

        # Block 4
        self.block4_conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.block4_bn1 = nn.BatchNorm2d(16)
        self.block4_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.block4_bn2 = nn.BatchNorm2d(16)
        self.block4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Branch 4 (64x64 output)
        self.br4_conv1 = nn.Conv2d(16, 8, kernel_size=4, stride=4, padding=0)
        self.br4_bn1 = nn.BatchNorm2d(8)
        self.br4_conv2 = nn.Conv2d(9, 4, kernel_size=1, padding=0)
        self.br4_bn2 = nn.BatchNorm2d(4)
        self.br4_conv3 = nn.Conv2d(4, 1, kernel_size=1, padding=0)

        # ── CHANGE 1: Apply He uniform initialization ──
        self._init_weights()

    def _init_weights(self):
        """Match the he_uniform initialization from the reference code."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        qp_scalar = x[0]
        img = x[1].unsqueeze(1)
        batch_size = img.size(0)

        # Block 1
        x = self.block1_pool(self.block1_bn2(F.relu(self.block1_conv2(self.block1_bn1(F.relu(self.block1_conv1(img)))))))
        
        # Block 2
        x = self.block2_pool(self.block2_bn2(F.relu(self.block2_conv2(self.block2_bn1(F.relu(self.block2_conv1(x)))))))
        
        # Branch 2 Prediction
        b2 = self.br2_bn1(F.relu(self.br2_conv1(x)))
        qp_plane_b2 = qp_scalar.view(batch_size, 1, 1, 1).expand(batch_size, 1, 4, 4)
        b2 = torch.cat([b2, qp_plane_b2], dim=1)
        b2 = torch.sigmoid(self.br2_conv3(self.br2_bn2(F.relu(self.br2_conv2(b2)))))
        pred_16 = b2.view(batch_size, -1)

        # Block 3
        x = self.block3_pool(self.block3_bn2(F.relu(self.block3_conv2(self.block3_bn1(F.relu(self.block3_conv1(x)))))))

        # Branch 3 Prediction
        b3 = self.br3_bn1(F.relu(self.br3_conv1(x)))
        qp_plane_b3 = qp_scalar.view(batch_size, 1, 1, 1).expand(batch_size, 1, 2, 2)
        b3 = torch.cat([b3, qp_plane_b3], dim=1)
        b3 = torch.sigmoid(self.br3_conv3(self.br3_bn2(F.relu(self.br3_conv2(b3)))))
        pred_32 = b3.view(batch_size, -1)

        # Block 4
        x = self.block4_pool(self.block4_bn2(F.relu(self.block4_conv2(self.block4_bn1(F.relu(self.block4_conv1(x)))))))

        # Branch 4 Prediction
        b4 = self.br4_bn1(F.relu(self.br4_conv1(x)))
        qp_plane_b4 = qp_scalar.view(batch_size, 1, 1, 1).expand(batch_size, 1, 1, 1)
        b4 = torch.cat([b4, qp_plane_b4], dim=1)
        b4 = torch.sigmoid(self.br4_conv3(self.br4_bn2(F.relu(self.br4_conv2(b4)))))
        pred_64 = b4.view(batch_size, -1).squeeze(1)

        return pred_64, pred_32, pred_16

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

# --- 3. Loss and Accuracy Functions ---
def custom_repo_loss(y_flat_64, y_conv_flat_64, y_flat_32, y_conv_flat_32, y_flat_valid_32,
                     y_flat_16, y_conv_flat_16, y_flat_valid_16):
    device = y_conv_flat_64.device
    epsilon = 1e-12

    # Loss for 64
    loss_64 = -torch.mean(y_flat_64 * torch.log(y_conv_flat_64 + epsilon) + 
                          (1 - y_flat_64) * torch.log(1 - y_conv_flat_64 + epsilon))

    # Loss for 32 (Masked)
    pos_mask_32 = y_flat_32 * y_flat_valid_32
    neg_mask_32 = (1 - y_flat_32) * y_flat_valid_32
    loss_32 = -torch.sum(pos_mask_32 * torch.log(y_conv_flat_32 + epsilon) + 
                         neg_mask_32 * torch.log(1 - y_conv_flat_32 + epsilon)) / (torch.sum(y_flat_valid_32) + epsilon)

    # Loss for 16 (Masked)
    pos_mask_16 = y_flat_16 * y_flat_valid_16
    neg_mask_16 = (1 - y_flat_16) * y_flat_valid_16
    loss_16 = -torch.sum(pos_mask_16 * torch.log(y_conv_flat_16 + epsilon) + 
                         neg_mask_16 * torch.log(1 - y_conv_flat_16 + epsilon)) / (torch.sum(y_flat_valid_16) + epsilon)

    return loss_64, loss_32, loss_16, (loss_64 + loss_32 + loss_16)

def calculate_accuracy_repo(y_flat_64, y_conv_flat_64, y_flat_32, y_conv_flat_32, y_flat_valid_32,
                        y_flat_16, y_conv_flat_16, y_flat_valid_16):
    epsilon = 1e-12
    # 64
    acc_64 = torch.mean((torch.round(y_conv_flat_64) == torch.round(y_flat_64)).float()) * 100
    # 32
    valid_pred_32 = (torch.round(y_conv_flat_32) == torch.round(y_flat_32)).float()
    acc_32 = torch.sum(valid_pred_32 * y_flat_valid_32) / (torch.sum(y_flat_valid_32) + epsilon) * 100
    # 16
    valid_pred_16 = (torch.round(y_conv_flat_16) == torch.round(y_flat_16)).float()
    acc_16 = torch.sum(valid_pred_16 * y_flat_valid_16) / (torch.sum(y_flat_valid_16) + epsilon) * 100
    
    avg_acc = (acc_64 + acc_32 + acc_16) / 3
    return avg_acc, acc_64, acc_32, acc_16

# --- 4. Plotting Function ---
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

# --- 5. Main Execution ---

# File paths
train_file_path = "/home/krishna/my_project/Data/1080p_dataset/AI_Train_163200.dat_shuffled"
validation_file_path = "/home/krishna/my_project/Data/1080p_dataset/AI_Valid_9600.dat_shuffled"
test_file_path = "/home/krishna/my_project/Data/1080p_dataset/AI_Test_19200.dat_shuffled"

TRAINSET_MAXSIZE = 163200
VALIDSET_MAXSIZE = 9600
TESTSET_MAXSIZE = 19200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Model and Optimizer
model = HFCN().to(device)
optimizer = optim.Adam(model.parameters(), lr=CONFIG_LEARNING_RATE)

# COUNT AND PRINT PARAMETERS
total_params, trainable_params = count_parameters(model, "HFCN")

# Checkpoint paths
checkpoint_path = 'checkpoint_hfcn.pth'
best_checkpoint_path = 'checkpoint_best_hfcn.pth'

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
        
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        overall_least_loss = checkpoint.get('overall_least_loss', checkpoint.get('loss', float('inf')))
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
print(f"Current learning rate: {CONFIG_LEARNING_RATE}")
print()

for epoch in range(start_epoch, CONFIG_EPOCHS + 1):
    
    # =================================================================
    # 1. FIXED: Patience check -> Reload best model + Shuffle training
    # =================================================================
    if patience_counter >= patience:
        num_patience_counter_changed += 1
        
        # Reload best model weights (like reference code)
        if os.path.exists(best_checkpoint_path):
            print(f"\n>>> Patience reached ({patience_counter}). Reloading best model from {best_checkpoint_path}...")
            best_ckpt = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(best_ckpt['model_state_dict'])
            optimizer.load_state_dict(best_ckpt['optimizer_state_dict'])
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
    # =================================================================
    if num_patience_counter_changed >= 2:
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

        # Loss Calculation
        _, _, _, total_loss = custom_repo_loss(
            labels[0], outputs[0],
            labels[1], outputs[1], valid_masks[0],
            labels[2], outputs[2], valid_masks[1]
        )
        
        total_loss.backward()
        optimizer.step()

        # Metrics
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
    # 4. Validation Phase (Every 2 epochs)
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
        
        # =============================================================
        # FIXED: Best model check (mirrors reference code exactly)
        # - best_loss is NEVER set to float('inf')
        # - validation_shuffle_count resets ONLY on real improvement
        # - On patience, best model is reloaded (not continued with 
        #   degraded weights)
        # =============================================================
        if curr_val_loss <= best_loss:
            best_loss = curr_val_loss
            overall_least_loss = curr_val_loss
            patience_counter = 0
            validation_shuffle_count = 0
            
            print(f"  ✓ New Best Model saved! (Loss: {best_loss:.4f})")
            
            # Save best model weights
            torch.save(model.state_dict(), 'best_model_HFCN_pyt.pth')
            
            # Save best complete checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': curr_val_loss,
                'best_loss': best_loss,
                'overall_least_loss': overall_least_loss,
                'history': history,
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
            'loss': curr_val_loss,
            'best_loss': best_loss,
            'overall_least_loss': overall_least_loss,
            'history': history,
            'patience_counter': patience_counter,
            'num_patience_counter_changed': num_patience_counter_changed,
            'validation_shuffle_count': validation_shuffle_count
        }, checkpoint_path)
        
        # Visual confirmation of checkpoint saving (every 50 epochs)
        if epoch % 50 == 0:
            print(f"━━━ Checkpoint auto-saved at epoch {epoch} ━━━")

print("\n" + "=" * 70)
print("Training Complete.")
print(f"Best validation loss achieved: {overall_least_loss:.4f}")
print(f"Best model saved as: best_model_HFCN_pyt.pth")
print("=" * 70)

# Final plot generation
if epoch <= CONFIG_EPOCHS:
    save_plots(epoch, history)
else:
    save_plots(CONFIG_EPOCHS, history)