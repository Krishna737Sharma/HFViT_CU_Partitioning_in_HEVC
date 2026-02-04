import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
import itertools
import random  # <--- FIXED: Missing import
import wandb   # <--- FIXED: Missing import
from torch.utils.data import Dataset, DataLoader, Subset
from timm.layers import trunc_normal_, SqueezeExcite

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

# [FIX]: Define device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ==================== StreamingDataset ====================
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
                # 64*64*1 = 4096 bytes
                image = data[:4096].astype(np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
                image = torch.from_numpy(image).permute(2, 0, 1) / 255.0

                # QP processing
                qp = np.random.choice(SELECT_QP_LIST)
                qp_tensor = torch.tensor([float(qp) / 51.0], dtype=torch.float32)

                # Label processing
                label_offset = 4160 + int(qp) * NUM_LABEL_BYTES
                raw_label = data[label_offset : label_offset + NUM_LABEL_BYTES]
                
                # Check if data read was sufficient
                if len(raw_label) < NUM_LABEL_BYTES:
                    raise ValueError("Insufficient data read for label")

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
            print(f"Error loading index {idx}: {e}")
            # Return zero tensors in case of error to prevent crash
            return (torch.zeros(1), torch.zeros(1, 64, 64), 
                    torch.zeros(1), torch.zeros(4), torch.zeros(16), 
                    torch.zeros(4), torch.zeros(16), torch.zeros(21))

# ==================== Data Loader Helper ====================
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

# ==================== Metrics & Losses ====================
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
    # Device is now global, but keeping .to(device) ensures tensors move if needed
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

# ==================== EfficientViT Modules ====================
class Conv2d_BN(nn.Module):
    def __init__(self, in_ch, out_ch, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        if ks > 1:
            self.c = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, ks, stride, pad, dilation, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
            )
        else:
            self.c = nn.Conv2d(in_ch, out_ch, ks, stride, pad, dilation, groups, bias=False)
            
        self.bn = nn.BatchNorm2d(out_ch)
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        return self.bn(self.c(x))

class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        return x + self.m(x)

class FFN(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.act = nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x

class CascadedGroupAttention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=4, attn_ratio=4, resolution=14, kernels=[5, 5, 5, 5]):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            # Input to QKV is dim // num_heads
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim))
        self.qkvs = nn.ModuleList(qkvs)
        self.dws = nn.ModuleList(dws)
        self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(self.d * num_heads, dim, bn_weight_init=0))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0: 
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            attn = ((q.transpose(-2, -1) @ k) * self.scale + 
                    (trainingab[i] if self.training else self.ab[i]))
            attn = attn.softmax(dim=-1)
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x

class LocalWindowAttention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=4, attn_ratio=4, resolution=14, window_resolution=7, kernels=[5, 5, 5, 5]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        self.window_resolution = window_resolution
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads, attn_ratio=attn_ratio, 
                                           resolution=window_resolution, kernels=kernels)

    def forward(self, x):
        H, W = self.resolution, self.resolution
        B, C, H_, W_ = x.shape
        
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H % self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W % self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            
            x = self.attn(x)
            
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution, C).transpose(2, 3).reshape(B, pH, pW, C)
            if padding:
                x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x

class EfficientViTBlock(nn.Module):    
    def __init__(self, type, ed, kd, nh=4, ar=4, resolution=14, window_resolution=7, kernels=[5, 5, 5, 5]):
        super().__init__()
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn0 = Residual(FFN(ed, int(ed * 2)))

        if type == 's':
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar, 
                                                       resolution=resolution, window_resolution=window_resolution, kernels=kernels))
                
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn1 = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))

class EfficientViT_HEVC(nn.Module):
    def __init__(self):
        super().__init__()
        
        embed_dim = [16, 24, 32]   
        num_heads = [2, 2, 2]      
        window_size = 4
        key_dim = [4, 6, 8]        
        attn_ratio = [max(1, int(embed_dim[i] / (key_dim[i] * num_heads[i]))) for i in range(len(embed_dim))]
        
        self.stem = nn.Sequential(
            Conv2d_BN(NUM_CHANNELS, 8, ks=3, stride=2, pad=1),
            nn.ReLU(),
            Conv2d_BN(8, embed_dim[0], ks=3, stride=2, pad=1)
        )
        
        self.stage1 = nn.Sequential(
            EfficientViTBlock('s', embed_dim[0], key_dim[0], num_heads[0], ar=attn_ratio[0],
                              resolution=16, window_resolution=window_size, kernels=[3, 3, 3, 3])
        )
        self.ds1 = PatchMerging(embed_dim[0], embed_dim[1])
        
        self.stage2 = nn.Sequential(
            EfficientViTBlock('s', embed_dim[1], key_dim[1], num_heads[1], ar=attn_ratio[1],
                              resolution=8, window_resolution=window_size, kernels=[3, 3, 3, 3])
        )
        self.ds2 = PatchMerging(embed_dim[1], embed_dim[2])
        
        self.stage3 = nn.Sequential(
            EfficientViTBlock('s', embed_dim[2], key_dim[2], num_heads[2], ar=attn_ratio[2],
                              resolution=4, window_resolution=window_size, kernels=[3, 3, 3, 3])
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim[2] + 1, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(),
            nn.Linear(1280, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, NUM_CLASSES),
            nn.Sigmoid()
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, qp):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.ds1(x)
        x = self.stage2(x)
        x = self.ds2(x)
        x = self.stage3(x)
        
        feat = self.gap(x).flatten(1)
        
        if qp.dim() == 1:
            qp = qp.unsqueeze(1)
            
        return self.head(torch.cat([feat, qp], dim=1))

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    
    # [FIX] Added basic check for data file existence
    if not os.path.exists(TRAIN_FILE_PATH):
        print(f"WARNING: Train file not found at {TRAIN_FILE_PATH}. Creating mock data to allow code to run.")
        # Create dummy file for testing purposes if you run this immediately
        with open(TRAIN_FILE_PATH, 'wb') as f:
            f.write(os.urandom(NUM_SAMPLE_LENGTH * TRAINSET_MAXSIZE))
        with open(VALID_FILE_PATH, 'wb') as f:
            f.write(os.urandom(NUM_SAMPLE_LENGTH * VALIDSET_MAXSIZE))

    print("Creating DataLoaders...")
    train_loader, train_indices = create_subset_dataloader(TRAIN_FILE_PATH, TRAINSET_MAXSIZE, TRAIN_SUBSET_SIZE, BATCH_SIZE, shuffle=True)
    validation_loader, validation_indices = create_subset_dataloader(VALID_FILE_PATH, VALIDSET_MAXSIZE, VAL_SUBSET_SIZE, BATCH_SIZE, shuffle=False)
    print(f"Train Loader Size: {len(train_loader)} batches")

    # [FIX] Unindented the main loop
    model = EfficientViT_HEVC().to(device)
    model = model.to(memory_format=torch.channels_last)

    # [FIX] Initialize wandb
    wandb.init(project="hevc_partition_efficientvit", config={"lr": LEARNING_RATE, "epochs": EPOCHS})
    wandb.watch(model, log="all", log_freq=50)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
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