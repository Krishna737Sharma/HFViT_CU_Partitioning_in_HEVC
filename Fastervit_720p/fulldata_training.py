import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from timm.layers import DropPath, trunc_normal_
import time
import warnings

# Global debug flag
DEBUG = True 

# ==================== Constants ====================
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_LABEL_BYTES = 16
NUM_SAMPLE_LENGTH = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + 64 + (51 + 1) * NUM_LABEL_BYTES
SELECT_QP_LIST = [22, 27, 32, 37]
NUM_CLASSES = 21

TRAINSET_MAXSIZE = 40800
VALIDSET_MAXSIZE = 2400

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4  
EPOCHS = 10000

# File Paths
TRAIN_FILE_PATH = "/raid/somdyutiai/Krishna_24AI60R38/PycharmProjects/HEVC_Intra_Models-ETH-CNN_Pt/Data/720p_dataset/AI_Train_40800.dat_shuffled" 
VALID_FILE_PATH = "/raid/somdyutiai/Krishna_24AI60R38/PycharmProjects/HEVC_Intra_Models-ETH-CNN_Pt/Data/720p_dataset/AI_Valid_2400.dat_shuffled"

# Checkpoint Filenames
CHECKPOINT_PATH = 'best_fastervit_hevc_balanced.pth'
HISTORY_CHECKPOINT_PATH = 'history_fastervit.pth' # New file to store just the lists

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if DEBUG:
    print("=" * 60)
    print("DEBUG MODE ENABLED")
    print("=" * 60)
    print(f"IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"NUM_CHANNELS: {NUM_CHANNELS}")
    print(f"NUM_SAMPLE_LENGTH: {NUM_SAMPLE_LENGTH}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print("=" * 60)

# ==================== StreamingDataset Class ====================
class StreamingDataset(Dataset):
    def __init__(self, file_path, max_samples):
        if DEBUG:
            print(f"Initializing StreamingDataset: {file_path}")
            print(f"Max samples: {max_samples}")
        self.file_path = file_path
        self.max_samples = max_samples
        self.error_count = 0

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        try:
            with open(self.file_path, 'rb') as f:
                offset = idx * NUM_SAMPLE_LENGTH
                f.seek(offset)
                data = np.frombuffer(f.read(NUM_SAMPLE_LENGTH), dtype=np.uint8)

                # Image processing
                image = data[:4096].astype(np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
                image = torch.from_numpy(image).permute(2, 0, 1) / 255.0

                # QP processing
                qp = np.random.choice(SELECT_QP_LIST)
                qp_tensor = torch.tensor([float(qp) / 51.0], dtype=torch.float32)

                # Label processing
                label_offset = 4160 + int(qp) * NUM_LABEL_BYTES
                raw_label = data[label_offset : label_offset + NUM_LABEL_BYTES]
                y_grid = torch.tensor(raw_label, dtype=torch.float32).view(1, 4, 4)

                # Hierarchical Splits
                y_16 = F.relu(y_grid - 2) 
                y_flat_16 = y_16.view(-1)

                y_32_grid = F.avg_pool2d(y_grid, kernel_size=2)
                y_32 = F.relu(y_32_grid - 1) - F.relu(y_32_grid - 2)
                y_flat_32 = y_32.view(-1)
                y_valid_32 = F.relu(y_32_grid - 0) - F.relu(y_32_grid - 1)

                y_64_grid = F.avg_pool2d(y_grid, kernel_size=4)
                y_64 = F.relu(y_64_grid - 0) - F.relu(y_64_grid - 1)
                y_flat_64 = y_64.view(-1)
                  
                y_valid_16 = F.relu(y_grid - 1) - F.relu(y_grid - 2)

                target = torch.cat((y_flat_64, y_flat_32, y_flat_16), dim=0)

                return qp_tensor, image, y_flat_64, y_flat_32, y_flat_16, y_valid_32.view(-1), y_valid_16.view(-1), target
                  
        except Exception as e:
            self.error_count += 1
            if self.error_count <= 10 or DEBUG:
                print(f"ERROR loading sample {idx}: {e}")
            return (
                torch.zeros(1), 
                torch.zeros(NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), 
                torch.zeros(1), 
                torch.zeros(4), 
                torch.zeros(16), 
                torch.zeros(4), 
                torch.zeros(16), 
                torch.zeros(21)
            )

def create_subset_dataloader(file_path, total_samples, subset_size, batch_size, shuffle=True):
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % (2**32)
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
        pin_memory=True,
        worker_init_fn=worker_init_fn
    ), subset_indices

# ==================== DATA LOADERS ====================
train_loader, train_indices = create_subset_dataloader(TRAIN_FILE_PATH, TRAINSET_MAXSIZE, TRAINSET_MAXSIZE, BATCH_SIZE, shuffle=True)
validation_loader, validation_indices = create_subset_dataloader(VALID_FILE_PATH, VALIDSET_MAXSIZE, VALIDSET_MAXSIZE, BATCH_SIZE, shuffle=False)

# ==================== LOSS & ACCURACY ====================
def custom_hevc_loss(y64_true, y64_pred, y32_true, y32_pred, y32_valid, y16_true, y16_pred, y16_valid):
    epsilon = 1e-7
    y64_true, y64_pred = y64_true.reshape(-1), y64_pred.reshape(-1)
    y32_true, y32_pred, y32_valid = y32_true.reshape(-1), y32_pred.reshape(-1), y32_valid.reshape(-1)
    y16_true, y16_pred, y16_valid = y16_true.reshape(-1), y16_pred.reshape(-1), y16_valid.reshape(-1)

    loss_64 = -torch.mean(y64_true * torch.log(y64_pred + epsilon) + (1 - y64_true) * torch.log(1 - y64_pred + epsilon))
      
    loss_32 = -torch.sum(y32_valid * (y32_true * torch.log(y32_pred + epsilon) + (1 - y32_true) * torch.log(1 - y32_pred + epsilon)))
    loss_32 = loss_32 / (torch.sum(y32_valid) + epsilon)
      
    loss_16 = -torch.sum(y16_valid * (y16_true * torch.log(y16_pred + epsilon) + (1 - y16_true) * torch.log(1 - y16_pred + epsilon)))
    loss_16 = loss_16 / (torch.sum(y16_valid) + epsilon)

    if torch.isnan(loss_64) or torch.isnan(loss_32) or torch.isnan(loss_16):
        print("WARNING: NaN detected in loss!")
        print(f"loss_64: {loss_64}, loss_32: {loss_32}, loss_16: {loss_16}")

    return loss_64, loss_32, loss_16, (loss_64 + loss_32 + loss_16)

def calculate_accuracy_repo(y_flat_64, y_conv_flat_64, y_flat_32, y_conv_flat_32, y_flat_valid_32, y_flat_16, y_conv_flat_16, y_flat_valid_16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_flat_64, y_conv_flat_64 = y_flat_64.to(device), y_conv_flat_64.to(device)
    y_flat_32, y_conv_flat_32, y_flat_valid_32 = y_flat_32.to(device), y_conv_flat_32.to(device), y_flat_valid_32.to(device)
    y_flat_16, y_conv_flat_16, y_flat_valid_16 = y_flat_16.to(device), y_conv_flat_16.to(device), y_flat_valid_16.to(device)
    epsilon = 1e-12

    correct_prediction_64 = torch.round(y_conv_flat_64) == torch.round(y_flat_64)
    accuracy_64 = torch.mean(correct_prediction_64.float()) * 100

    correct_prediction_valid_32 = y_flat_valid_32 * (torch.round(y_conv_flat_32) == torch.round(y_flat_32)).float()
    accuracy_32 = torch.sum(y_flat_valid_32 * correct_prediction_valid_32) / (torch.sum(y_flat_valid_32) + epsilon) * 100

    correct_prediction_valid_16 = y_flat_valid_16 * (torch.round(y_conv_flat_16) == torch.round(y_flat_16)).float()
    accuracy_16 = torch.sum(y_flat_valid_16 * correct_prediction_valid_16) / (torch.sum(y_flat_valid_16) + epsilon) * 100

    avg_acc = (accuracy_64 + accuracy_32 + accuracy_16) / 3
    return avg_acc, accuracy_64, accuracy_32, accuracy_16

# ==================== OPTIMIZED ARCHITECTURE (From Inference Code) ====================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        # Split into Depthwise (groups=in_ch) + Pointwise (1x1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class EfficientResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Use Depthwise Separable here too
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        )
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.bn(self.conv(x)))

class StreamlinedHAT(nn.Module):
    def __init__(self, dim, num_heads=2, window_size=2, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        # self.norm1 = nn.LayerNorm(dim) # REMOVED for optimization
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

        self.pos_scale = nn.Parameter(torch.ones(num_heads) * 0.5)

    def forward(self, x, ct):
        B, C, H, W = x.shape

        x_win = x.permute(0, 2, 3, 1).reshape(-1, self.window_size ** 2, C)
        ct = ct.reshape(-1, 1, C)
        tokens = torch.cat([x_win, ct], dim=1)

        shortcut = tokens
        # tokens = self.norm1(tokens) # REMOVED for optimization

        qkv = self.qkv(tokens).reshape(
            tokens.size(0), tokens.size(1), 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn * self.pos_scale.view(1, -1, 1, 1)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(tokens.shape)
        tokens = shortcut + self.proj(out)
        tokens = tokens + self.mlp(self.norm2(tokens))

        x = tokens[:, :-1, :].reshape(B, H, W, C).permute(0, 3, 1, 2)
        ct = tokens[:, -1:, :]

        return x.contiguous(), ct

class BalancedFasterViT_HEVC(nn.Module):
    def __init__(self):
        super().__init__()

        dims = [8, 16, 24, 32]

        self.stem = ConvBNAct(NUM_CHANNELS, dims[0], 3, 2, 1)

        self.stage1 = nn.Sequential(
            EfficientResBlock(dims[0]),
            ConvBNAct(dims[0], dims[1], 3, 2, 1)
        )

        self.stage2 = nn.Sequential(
            EfficientResBlock(dims[1]),
            ConvBNAct(dims[1], dims[2], 3, 2, 1)
        )

        self.stage3 = nn.Sequential(
            EfficientResBlock(dims[2]),
            ConvBNAct(dims[2], dims[3], 3, 2, 1)
        )

        self.hat = StreamlinedHAT(dims[3], window_size=2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Original Large Head used in Inference code
        self.head = nn.Sequential(
            nn.Linear(dims[3] + 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(768, NUM_CLASSES),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, qp):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        ct = F.adaptive_avg_pool2d(x, (2, 2)).flatten(2).transpose(1, 2)
        x, _ = self.hat(x, ct)

        feat = self.gap(x).flatten(1)
        if qp.dim() == 1:
            qp = qp.unsqueeze(1)

        return self.head(torch.cat([feat, qp], dim=1))

# ==================== HELPER FUNCTION TO SAVE PLOTS ====================
def save_plots(history, epoch):
    if not os.path.exists('training_plots'):
        os.makedirs('training_plots')
       
    train_epochs_range = range(1, len(history['train_loss']) + 1)
    val_epochs_range = history['val_epochs']

    # 1. Training vs Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(val_epochs_range, history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title(f'Training and Validation Loss (Epoch {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_plots/loss_plot_epoch_{epoch}.png')
    plt.close()

    # 2. Training vs Validation Accuracy (Total)
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(val_epochs_range, history['val_acc'], label='Validation Accuracy', linestyle='--')
    plt.title(f'Training and Validation Accuracy (Epoch {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_plots/accuracy_plot_epoch_{epoch}.png')
    plt.close()

    # 3. L1 Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs_range, history['train_acc_l1'], label='Train L1 Acc')
    plt.plot(val_epochs_range, history['val_acc_l1'], label='Val L1 Acc', linestyle='--')
    plt.title(f'L1 Accuracy (64x64 Split) (Epoch {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_plots/l1_accuracy_plot_epoch_{epoch}.png')
    plt.close()

    # 4. L2 Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs_range, history['train_acc_l2'], label='Train L2 Acc')
    plt.plot(val_epochs_range, history['val_acc_l2'], label='Val L2 Acc', linestyle='--')
    plt.title(f'L2 Accuracy (32x32 Split) (Epoch {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_plots/l2_accuracy_plot_epoch_{epoch}.png')
    plt.close()

    # 5. L3 Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs_range, history['train_acc_l3'], label='Train L3 Acc')
    plt.plot(val_epochs_range, history['val_acc_l3'], label='Val L3 Acc', linestyle='--')
    plt.title(f'L3 Accuracy (16x16 Split) (Epoch {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_plots/l3_accuracy_plot_epoch_{epoch}.png')
    plt.close()
        
    print(f"    >>> Plots saved to 'training_plots/' for epoch {epoch}")

# ==================== TRAINING SETUP ====================
model = BalancedFasterViT_HEVC().to(device)
model = model.to(memory_format=torch.channels_last)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

count_parameters(model)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==================== TRAINING VARIABLES INITIALIZATION ====================
num_epochs = EPOCHS
start_epoch = 0 
best_loss = float('inf') 
overall_least_loss = float('inf')
patience = 10
patience_counter = 0 
num_patience_counter_changed = 0
validation_shuffle_count = 0 

# Storage for Plotting (Initialized Empty)
history = {
    'train_loss': [], 'val_loss': [], 'val_epochs': [],
    'train_acc': [], 'val_acc': [],
    'train_acc_l1': [], 'val_acc_l1': [],
    'train_acc_l2': [], 'val_acc_l2': [],
    'train_acc_l3': [], 'val_acc_l3': []
}

# ==================== CHECKPOINT LOADING LOGIC (CORRECTED) ====================

# 1. Load Main Checkpoint (Weights, Optimizer, Epoch)
if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    overall_least_loss = checkpoint['overall_least_loss']
    
    print(f"Resumed training from epoch {start_epoch} with best loss {best_loss:.4f}.")
else:
    print("No checkpoint found, starting fresh training.")

# 2. Load History Checkpoint (For continuous plotting)
if os.path.exists(HISTORY_CHECKPOINT_PATH):
    print(f"Loading history from {HISTORY_CHECKPOINT_PATH}...")
    history_checkpoint = torch.load(HISTORY_CHECKPOINT_PATH, weights_only=False)
    
    # Restore history dictionary
    history = history_checkpoint['history']
    
    # --- CRITICAL FIX: TRUNCATE HISTORY TO MATCH START_EPOCH ---
    if start_epoch > 0:
        print(f"Syncing history lengths to match start epoch: {start_epoch}")
        
        # A. Truncate Training Metrics (recorded every epoch)
        # We simply slice the list to keep data up to the start_epoch
        train_keys = ['train_loss', 'train_acc', 'train_acc_l1', 'train_acc_l2', 'train_acc_l3']
        for k in train_keys:
            if k in history:
                current_len = len(history[k])
                if current_len > start_epoch:
                    history[k] = history[k][:start_epoch]
        
        # B. Truncate Validation Metrics (recorded sparsely, e.g., every 2 epochs)
        # We must filter based on the recorded 'val_epochs'
        if 'val_epochs' in history and len(history['val_epochs']) > 0:
            # Find indices where the recorded epoch is less than or equal to start_epoch
            valid_indices = [i for i, ep in enumerate(history['val_epochs']) if ep <= start_epoch]
            
            if valid_indices:
                # The cut-off index is the last valid index + 1
                cut_idx = valid_indices[-1] + 1
                
                # Truncate val_epochs list
                history['val_epochs'] = history['val_epochs'][:cut_idx]
                
                # Truncate all validation metric lists to the same length
                val_keys = ['val_loss', 'val_acc', 'val_acc_l1', 'val_acc_l2', 'val_acc_l3']
                for k in val_keys:
                    if k in history:
                        history[k] = history[k][:cut_idx]
            else:
                # If start_epoch is very early and no validation happened yet
                history['val_epochs'] = []
                for k in ['val_loss', 'val_acc', 'val_acc_l1', 'val_acc_l2', 'val_acc_l3']:
                    if k in history:
                        history[k] = []

    print(f"History reloaded and synced. Train len: {len(history['train_loss'])}, Val len: {len(history['val_epochs'])}")

else:
    # Fallback: check if history is in the main checkpoint (legacy support)
    if 'history' in locals() and 'checkpoint' in locals() and 'history' in checkpoint:
        history = checkpoint['history']
        print("History loaded from main checkpoint.")
    else:
        print("No history checkpoint found, plots will start fresh.")

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

# ==================== TRAINING LOOP ====================

for epoch in range(start_epoch, num_epochs):
    
    # --- Checkpoint Reloading Logic on Patience Failure ---
    if patience_counter >= patience:
        num_patience_counter_changed += 1
        
        # Reload Best Model to reset weights
        if os.path.exists(CHECKPOINT_PATH):
            print(f"Patience reached ({patience_counter}). Reloading best model from {CHECKPOINT_PATH}...")
            checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # Note: We do NOT reset start_epoch, we continue from current epoch
            best_loss = checkpoint['best_loss']
        
        patience_counter = 0 # Reset counter

        print("Refreshing Training Data...")
        train_loader, train_indices = create_subset_dataloader(
            TRAIN_FILE_PATH, TRAINSET_MAXSIZE, TRAINSET_MAXSIZE, BATCH_SIZE, shuffle=True
        )

    # --- Dataset Refresh Logic (Termination Check) ---
    if num_patience_counter_changed >= 3:
        num_patience_counter_changed = 0
        validation_shuffle_count += 1 

        if validation_shuffle_count > 2:
            print("\n" + "=" * 60)
            print(f"TERMINATION: Validation dataset shuffled {validation_shuffle_count} times without loss improvement.")
            print("=" * 60)
            break
            
        print("Refreshing validation datasets...")
        new_validation_loader, new_validation_indices = create_subset_dataloader(
            VALID_FILE_PATH, VALIDSET_MAXSIZE, VALIDSET_MAXSIZE, BATCH_SIZE, shuffle=False
        )
        validation_loader, validation_indices = new_validation_loader, new_validation_indices


    # --- Standard Training Loop ---
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_acc_l1 = 0.0
    running_acc_l2 = 0.0
    running_acc_l3 = 0.0

    start_time = time.time()
    for batch_idx, batch in enumerate(train_loader):
        qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target = batch
        inputs = ctu_batch.to(device)
        qp_tensor = qp_batch.to(device)
            
        optimizer.zero_grad()
        outputs = model(inputs, qp_tensor)
            
        pred_64 = outputs[:, 0]
        pred_32 = outputs[:, 1:5]
        pred_16 = outputs[:, 5:21]
            
        l64, l32, l16, loss = custom_hevc_loss(
            y_flat_64.to(device), pred_64, 
            y_flat_32.to(device), pred_32, y_flat_valid_32.to(device), 
            y_flat_16.to(device), pred_16, y_flat_valid_16.to(device)
        )
            
        if torch.isnan(loss):
            print(f"NaN loss at epoch {epoch+1}, batch {batch_idx}")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()

        pred_64_for_acc = pred_64.unsqueeze(1) if pred_64.dim() == 1 else pred_64
        avg_acc, acc64, acc32, acc16 = calculate_accuracy_repo(
            y_flat_64, pred_64_for_acc,
            y_flat_32, pred_32, y_flat_valid_32,
            y_flat_16, pred_16, y_flat_valid_16
        )
        running_acc += avg_acc.item()
        running_acc_l1 += acc64.item()
        running_acc_l2 += acc32.item()
        running_acc_l3 += acc16.item()

    avg_loss = running_loss / len(train_loader)
    avg_acc = running_acc / len(train_loader)
    avg_acc_l1 = running_acc_l1 / len(train_loader)
    avg_acc_l2 = running_acc_l2 / len(train_loader)
    avg_acc_l3 = running_acc_l3 / len(train_loader)
        
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs} [{epoch_time:.1f}s] | Training Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | "
          f"Accuracy: {avg_acc:.2f}% | L1: {avg_acc_l1:.2f}% | L2: {avg_acc_l2:.2f}% | L3: {avg_acc_l3:.2f}%")

    # Store Training Metrics
    history['train_loss'].append(avg_loss)
    history['train_acc'].append(avg_acc)
    history['train_acc_l1'].append(avg_acc_l1)
    history['train_acc_l2'].append(avg_acc_l2)
    history['train_acc_l3'].append(avg_acc_l3)

    # Save checkpoint every 50 epochs (Similar to CNN code logic)
    if (epoch + 1) % 50 == 0:
        # Save History Separately
        torch.save({'history': history}, HISTORY_CHECKPOINT_PATH)
        
        # Save Model Checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch+1,
            'best_loss': best_loss,
            'overall_least_loss': overall_least_loss,
            'history': history 
        }, 'checkpoint_saved_50_epoch.pth')
        
        print(f"    >>> Checkpoint saved at epoch {epoch+1}")
        save_plots(history, epoch + 1)

    # Validation
    if (epoch+1) % 2 == 0:
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_acc_l1 = 0.0
        val_acc_l2 = 0.0
        val_acc_l3 = 0.0
            
        with torch.no_grad():
            for val_batch in validation_loader:
                qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target = val_batch
                inputs = ctu_batch.to(device)
                qp_tensor = qp_batch.to(device)
                    
                outputs = model(inputs, qp_tensor)
                    
                pred_64 = outputs[:, 0]
                pred_32 = outputs[:, 1:5]
                pred_16 = outputs[:, 5:21]
                    
                _, _, _, loss = custom_hevc_loss(
                    y_flat_64.to(device), pred_64, 
                    y_flat_32.to(device), pred_32, y_flat_valid_32.to(device), 
                    y_flat_16.to(device), pred_16, y_flat_valid_16.to(device)
                )
                val_loss += loss.item()
                    
                pred_64_for_acc = pred_64.unsqueeze(1) if pred_64.dim() == 1 else pred_64
                avg_acc, acc64, acc32, acc16 = calculate_accuracy_repo(
                    y_flat_64, pred_64_for_acc,
                    y_flat_32, pred_32, y_flat_valid_32,
                    y_flat_16, pred_16, y_flat_valid_16
                )
                val_acc += avg_acc.item()
                val_acc_l1 += acc64.item()
                val_acc_l2 += acc32.item()
                val_acc_l3 += acc16.item()

        avg_val_loss = val_loss / len(validation_loader)
        avg_val_acc = val_acc / len(validation_loader)
        avg_val_acc_l1 = val_acc_l1 / len(validation_loader)
        avg_val_acc_l2 = val_acc_l2 / len(validation_loader)
        avg_val_acc_l3 = val_acc_l3 / len(validation_loader)

        print(f"  Validation Loss: {avg_val_loss:.4f} | Accuracy: {avg_val_acc:.2f}% | L1: {avg_val_acc_l1:.2f}% | L2: {avg_val_acc_l2:.2f}% | L3: {avg_val_acc_l3:.2f}%")

        # Store Validation Metrics
        history['val_epochs'].append(epoch + 1)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        history['val_acc_l1'].append(avg_val_acc_l1)
        history['val_acc_l2'].append(avg_val_acc_l2)
        history['val_acc_l3'].append(avg_val_acc_l3)

        if avg_val_loss <= best_loss:
            best_loss = avg_val_loss
            overall_least_loss = avg_val_loss
            patience_counter = 0
            validation_shuffle_count = 0 
            
            # Save Best Model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch+1,
                'best_loss': best_loss,
                'overall_least_loss': overall_least_loss,
                'history': history
            }, CHECKPOINT_PATH)
            
            # Save History Separately (to ensure we always have lists even if best model isn't updated)
            torch.save({'history': history}, HISTORY_CHECKPOINT_PATH)

            print(f"   >>> Saved New Best Model (Loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            # Still save history even if model didn't improve
            torch.save({'history': history}, HISTORY_CHECKPOINT_PATH)
       
    scheduler.step()

# ==================== PLOTTING ====================
print("\n" + "=" * 60)
print("GENERATING FINAL PLOTS")
print("=" * 60)
save_plots(history, num_epochs) 
print("Plots saved as .png files.")

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print(f"Best Loss: {best_loss:.4f}")
print("=" * 60)