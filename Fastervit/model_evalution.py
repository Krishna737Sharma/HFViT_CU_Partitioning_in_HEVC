import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
import os
import time
from timm.layers import trunc_normal_

# ==================== Constants ====================
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_LABEL_BYTES = 16
NUM_SAMPLE_LENGTH = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + 64 + (51 + 1) * NUM_LABEL_BYTES
SELECT_QP_LIST = [22, 27, 32, 37]
NUM_CLASSES = 21

VALIDSET_MAXSIZE = 196350
BATCH_SIZE = 64

# File Paths
VALID_FILE_PATH = "/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/Extract_Data/AI_Test_196350.dat_shuffled"
CHECKPOINT_PATH = 'best_fastervit_hevc_balanced.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
            if self.error_count <= 10:
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

def create_dataloader(file_path, total_samples, batch_size, shuffle=False):
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % (2**32)
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
       
    dataset = StreamingDataset(file_path, total_samples)
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

# ==================== MODEL ARCHITECTURE ====================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
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

# ==================== LOSS & ACCURACY FUNCTIONS ====================
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

# ==================== EVALUATION FUNCTION ====================
def evaluate_model(model, dataloader, device):
    """
    Comprehensive evaluation of the model on validation set
    """
    model.eval()
    
    total_loss = 0.0
    total_loss_64 = 0.0
    total_loss_32 = 0.0
    total_loss_16 = 0.0
    
    total_acc = 0.0
    total_acc_l1 = 0.0
    total_acc_l2 = 0.0
    total_acc_l3 = 0.0
    
    num_batches = 0
    num_samples = 0
    
    # For per-QP statistics
    qp_stats = {qp: {'count': 0, 'acc': 0.0, 'loss': 0.0} for qp in SELECT_QP_LIST}
    
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80)
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target = batch
            
            inputs = ctu_batch.to(device)
            qp_tensor = qp_batch.to(device)
            
            # Forward pass
            outputs = model(inputs, qp_tensor)
            
            # Split outputs
            pred_64 = outputs[:, 0]
            pred_32 = outputs[:, 1:5]
            pred_16 = outputs[:, 5:21]
            
            # Calculate loss
            l64, l32, l16, loss = custom_hevc_loss(
                y_flat_64.to(device), pred_64, 
                y_flat_32.to(device), pred_32, y_flat_valid_32.to(device), 
                y_flat_16.to(device), pred_16, y_flat_valid_16.to(device)
            )
            
            total_loss += loss.item()
            total_loss_64 += l64.item()
            total_loss_32 += l32.item()
            total_loss_16 += l16.item()
            
            # Calculate accuracy
            pred_64_for_acc = pred_64.unsqueeze(1) if pred_64.dim() == 1 else pred_64
            avg_acc, acc64, acc32, acc16 = calculate_accuracy_repo(
                y_flat_64, pred_64_for_acc,
                y_flat_32, pred_32, y_flat_valid_32,
                y_flat_16, pred_16, y_flat_valid_16
            )
            
            total_acc += avg_acc.item()
            total_acc_l1 += acc64.item()
            total_acc_l2 += acc32.item()
            total_acc_l3 += acc16.item()
            
            num_batches += 1
            num_samples += inputs.size(0)
            
            # Per-QP statistics (approximate, since QP is randomly chosen per sample)
            # This is an approximation for logging purposes
            
            # Progress indicator
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = num_samples / elapsed
                print(f"  Progress: {batch_idx + 1}/{len(dataloader)} batches | "
                      f"Samples: {num_samples} | Speed: {samples_per_sec:.1f} samples/s")
    
    eval_time = time.time() - start_time
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_loss_64 = total_loss_64 / num_batches
    avg_loss_32 = total_loss_32 / num_batches
    avg_loss_16 = total_loss_16 / num_batches
    
    avg_acc = total_acc / num_batches
    avg_acc_l1 = total_acc_l1 / num_batches
    avg_acc_l2 = total_acc_l2 / num_batches
    avg_acc_l3 = total_acc_l3 / num_batches
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nTotal Samples Evaluated: {num_samples}")
    print(f"Evaluation Time: {eval_time:.2f} seconds ({num_samples/eval_time:.1f} samples/sec)")
    print("\n" + "-"*80)
    print("LOSS METRICS:")
    print("-"*80)
    print(f"  Total Loss:        {avg_loss:.6f}")
    print(f"  L1 Loss (64x64):   {avg_loss_64:.6f}")
    print(f"  L2 Loss (32x32):   {avg_loss_32:.6f}")
    print(f"  L3 Loss (16x16):   {avg_loss_16:.6f}")
    
    print("\n" + "-"*80)
    print("ACCURACY METRICS:")
    print("-"*80)
    print(f"  Average Accuracy:  {avg_acc:.2f}%")
    print(f"  L1 Accuracy (64x64 split): {avg_acc_l1:.2f}%")
    print(f"  L2 Accuracy (32x32 split): {avg_acc_l2:.2f}%")
    print(f"  L3 Accuracy (16x16 split): {avg_acc_l3:.2f}%")
    
    print("\n" + "="*80)
    
    return {
        'loss': avg_loss,
        'loss_64': avg_loss_64,
        'loss_32': avg_loss_32,
        'loss_16': avg_loss_16,
        'accuracy': avg_acc,
        'accuracy_l1': avg_acc_l1,
        'accuracy_l2': avg_acc_l2,
        'accuracy_l3': avg_acc_l3,
        'num_samples': num_samples,
        'eval_time': eval_time
    }

# ==================== MAIN EVALUATION ====================
def main():
    print("\n" + "="*80)
    print("HEVC MODEL EVALUATION SCRIPT")
    print("="*80)
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nERROR: Checkpoint file not found at: {CHECKPOINT_PATH}")
        print("Please ensure the checkpoint file exists and the path is correct.")
        return
    
    # Load model
    print(f"\nLoading model from: {CHECKPOINT_PATH}")
    model = BalancedFasterViT_HEVC().to(device)
    model = model.to(memory_format=torch.channels_last)
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded successfully!")
    print(f"  Checkpoint Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best Loss: {checkpoint.get('best_loss', 'N/A'):.6f}")
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Create validation dataloader
    print(f"\nLoading validation dataset from: {VALID_FILE_PATH}")
    print(f"  Max samples: {VALIDSET_MAXSIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    validation_loader = create_dataloader(
        VALID_FILE_PATH, 
        VALIDSET_MAXSIZE, 
        BATCH_SIZE, 
        shuffle=False
    )
    
    print(f"✓ Validation dataloader created with {len(validation_loader)} batches")
    
    # Run evaluation
    results = evaluate_model(model, validation_loader, device)
    
    # Save results to file
    results_file = 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HEVC MODEL EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write(f"Validation Dataset: {VALID_FILE_PATH}\n")
        f.write(f"Total Samples: {results['num_samples']}\n")
        f.write(f"Evaluation Time: {results['eval_time']:.2f} seconds\n\n")
        f.write("-"*80 + "\n")
        f.write("LOSS METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Loss:        {results['loss']:.6f}\n")
        f.write(f"L1 Loss (64x64):   {results['loss_64']:.6f}\n")
        f.write(f"L2 Loss (32x32):   {results['loss_32']:.6f}\n")
        f.write(f"L3 Loss (16x16):   {results['loss_16']:.6f}\n\n")
        f.write("-"*80 + "\n")
        f.write("ACCURACY METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Average Accuracy:  {results['accuracy']:.2f}%\n")
        f.write(f"L1 Accuracy (64x64 split): {results['accuracy_l1']:.2f}%\n")
        f.write(f"L2 Accuracy (32x32 split): {results['accuracy_l2']:.2f}%\n")
        f.write(f"L3 Accuracy (16x16 split): {results['accuracy_l3']:.2f}%\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Results saved to: {results_file}")
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()