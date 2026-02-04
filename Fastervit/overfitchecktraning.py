import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
import os
import time
import wandb
from timm.layers import trunc_normal_

# Global debug flag
DEBUG = True 

# ==================== Constants ====================
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_LABEL_BYTES = 16
NUM_SAMPLE_LENGTH = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + 64 + (51 + 1) * NUM_LABEL_BYTES
SELECT_QP_LIST = [22, 27, 32, 37]
NUM_CLASSES = 21

TRAINSET_MAXSIZE = 1668975
VALIDSET_MAXSIZE = 98175

# ==================== OVERFIT CHECK SETTINGS ====================
BATCH_SIZE = 64
TRAIN_SUBSET_SIZE = 1000  # Small dataset for overfitting
VAL_SUBSET_SIZE = 500     # Small validation set
LEARNING_RATE = 1e-3
EPOCHS = 200 

# File Paths
TRAIN_FILE_PATH = "/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/Extract_Data/AI_Train_1668975.dat_shuffled" 
VALID_FILE_PATH = "/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/Extract_Data/AI_Valid_98175.dat_shuffled"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== WANDB INIT ====================
wandb.init(
    project="hevc-overfit-check",
    name="Small_Dataset_Optimized_Arch_Check",
    config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "subset_size": TRAIN_SUBSET_SIZE,
        "mode": "Overfitting_Check_Optimized"
    }
)

if DEBUG:
    print("=" * 60)
    print("DEBUG MODE: OVERFITTING CHECK (OPTIMIZED ARCHITECTURE)")
    print(f"Training on {TRAIN_SUBSET_SIZE} samples only.")
    print("=" * 60)

# ==================== StreamingDataset Class ====================
class StreamingDataset(Dataset):
    def __init__(self, file_path, max_samples):
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
    
    # Fixed seed for indices ensuring we get the SAME subset every time for overfit check
    random.seed(42) 
    subset_indices = random.sample(range(total_samples), real_subset_size)
      
    return DataLoader(
        Subset(full_dataset, subset_indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2, 
        pin_memory=True,
        worker_init_fn=worker_init_fn
    ), subset_indices

# ==================== DATA LOADERS ====================
print("Creating DataLoaders...")
train_loader, train_indices = create_subset_dataloader(TRAIN_FILE_PATH, TRAINSET_MAXSIZE, TRAIN_SUBSET_SIZE, BATCH_SIZE, shuffle=True)
validation_loader, validation_indices = create_subset_dataloader(VALID_FILE_PATH, VALIDSET_MAXSIZE, VAL_SUBSET_SIZE, BATCH_SIZE, shuffle=False)
print(f"Train Loader Size: {len(train_loader)} batches")

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

# ==================== ARCHITECTURE (OPTIMIZED) ====================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        # [MODIFIED] Using Depthwise Separable Convolution
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
        # [MODIFIED] Using Depthwise Separable Convolution
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

        # [MODIFIED] Removed self.norm1 for speed optimization
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
        # [MODIFIED] Removed norm1 execution
        
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
        self.stage1 = nn.Sequential(EfficientResBlock(dims[0]), ConvBNAct(dims[0], dims[1], 3, 2, 1))
        self.stage2 = nn.Sequential(EfficientResBlock(dims[1]), ConvBNAct(dims[1], dims[2], 3, 2, 1))
        self.stage3 = nn.Sequential(EfficientResBlock(dims[2]), ConvBNAct(dims[2], dims[3], 3, 2, 1))

        self.hat = StreamlinedHAT(dims[3], window_size=2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # [MAINTAINED] Using the original Large Head to preserve accuracy
        self.head = nn.Sequential(
            nn.Linear(dims[3] + 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(1024, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(1536, NUM_CLASSES),
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

# ==================== TRAINING SETUP ====================
model = BalancedFasterViT_HEVC().to(device)
model = model.to(memory_format=torch.channels_last)

wandb.watch(model, log="all", log_freq=50)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
# No annealing for overfit check - we want to force fit
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1.0) 

# ==================== TRAINING LOOP ====================
print("\n" + "=" * 60)
print("STARTING OVERFITTING CHECK (OPTIMIZED ARCHITECTURE)")
print("Expect Training Loss to approach 0.0 and Accuracy 100%")
print("=" * 60)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    start_time = time.time()
    
    # === TRAINING ===
    for batch_idx, batch in enumerate(train_loader):
        qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target = batch
        
        inputs = ctu_batch.to(device)
        qp_tensor = qp_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, qp_tensor)
        
        pred_64, pred_32, pred_16 = outputs[:, 0], outputs[:, 1:5], outputs[:, 5:21]
        
        _, _, _, loss = custom_hevc_loss(
            y_flat_64.to(device), pred_64, 
            y_flat_32.to(device), pred_32, y_flat_valid_32.to(device), 
            y_flat_16.to(device), pred_16, y_flat_valid_16.to(device)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        
        # Accuracy calc
        pred_64_for_acc = pred_64.unsqueeze(1) if pred_64.dim() == 1 else pred_64
        avg_acc, _, _, _ = calculate_accuracy_repo(
            y_flat_64, pred_64_for_acc, y_flat_32, pred_32, y_flat_valid_32, y_flat_16, pred_16, y_flat_valid_16
        )
        running_acc += avg_acc.item()

    avg_loss = running_loss / len(train_loader)
    avg_acc = running_acc / len(train_loader)
    
    # === VALIDATION (Run Every Epoch for Check) ===
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    with torch.no_grad():
        for val_batch in validation_loader:
            qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target = val_batch
            inputs = ctu_batch.to(device)
            qp_tensor = qp_batch.to(device)
            
            outputs = model(inputs, qp_tensor)
            pred_64, pred_32, pred_16 = outputs[:, 0], outputs[:, 1:5], outputs[:, 5:21]
            
            _, _, _, loss = custom_hevc_loss(
                y_flat_64.to(device), pred_64, 
                y_flat_32.to(device), pred_32, y_flat_valid_32.to(device), 
                y_flat_16.to(device), pred_16, y_flat_valid_16.to(device)
            )
            val_loss += loss.item()
            
            pred_64_for_acc = pred_64.unsqueeze(1) if pred_64.dim() == 1 else pred_64
            v_avg_acc, _, _, _ = calculate_accuracy_repo(
                y_flat_64, pred_64_for_acc, y_flat_32, pred_32, y_flat_valid_32, y_flat_16, pred_16, y_flat_valid_16
            )
            val_acc += v_avg_acc.item()

    avg_val_loss = val_loss / len(validation_loader)
    avg_val_acc = val_acc / len(validation_loader)

    epoch_time = time.time() - start_time
    
    print(f"Epoch {epoch+1}/{EPOCHS} [{epoch_time:.1f}s] | "
          f"Train Loss: {avg_loss:.4f} Acc: {avg_acc:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.2f}%")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "train_acc": avg_acc,
        "val_loss": avg_val_loss,
        "val_acc": avg_val_acc
    })
    
    # Save checkpoint periodically
    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f'overfit_check_epoch_{epoch+1}.pth')

wandb.finish()