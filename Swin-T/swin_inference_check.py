# ----------------------------------------------------------------------------------
# Swin Transformer V2 for HEVC CU Partition Prediction
# ----------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import random
import os
import wandb
import time # <-- Added for timing
from fvcore.nn import FlopCountAnalysis
from thop import profile
from ptflops import get_model_complexity_info

torch.set_num_threads(1)
# ==================================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ==================================================================================
DEBUG = False
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_LABEL_BYTES = 16
NUM_SAMPLE_LENGTH = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + 64 + (51 + 1) * NUM_LABEL_BYTES
SELECT_QP_LIST = [22, 27, 32, 37]

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

            ctu_tensor = torch.from_numpy(image).float().permute(2, 0, 1) # [C, H, W]
            qp_tensor = torch.tensor(float(qp), dtype=torch.float32)

            ctu_tensor /= 255.0
            qp_tensor /= 51.0

            # Process labels for multi-level loss calculation
            y_image = torch.tensor(label, dtype=torch.float32).view(1, 4, 4)
            y_image_16 = F.relu(y_image - 2)
            avg_pool_result = F.avg_pool2d(y_image, kernel_size=2)
            y_image_32 = F.relu(avg_pool_result - 1) - F.relu(avg_pool_result - 2)
            y_image_64 = F.relu(F.avg_pool2d(y_image, kernel_size=4) - 0) - F.relu(F.avg_pool2d(y_image, kernel_size=4) - 1)
            y_image_valid_32 = F.relu(avg_pool_result - 0) - F.relu(avg_pool_result - 1)
            y_image_valid_16 = F.relu(y_image - 1) - F.relu(y_image - 2)

            y_flat_16 = y_image_16.view(-1)
            y_flat_32 = y_image_32.view(-1)
            y_flat_64 = y_image_64.view(-1)
            y_flat_valid_32 = y_image_valid_32.view(-1)
            y_flat_valid_16 = y_image_valid_16.view(-1)

            # Final target tensor for the model
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
# SECTION 2: SWIN TRANSFORMER V2 MODEL IMPLEMENTATION
# ==================================================================================

class MlpWithQP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features + 1, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, qp):
        B, N, C = x.shape
        qp_expanded = qp.view(B, 1, 1).expand(B, N, 1)
        x_with_qp = torch.cat([x, qp_expanded], dim=-1)
        x = self.fc1(x_with_qp)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij")).permute(1, 2, 0).contiguous().unsqueeze(0)
        
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=x.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpWithQP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, qp):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        
        x = x + self.drop_path(self.norm2(self.mlp(x, qp)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        
        x = self.reduction(x)
        x = self.norm(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, qp):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, qp)
            else:
                x = blk(x, qp)
        
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class SwinTransformerV2(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=1, num_classes=21,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=4, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes)

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, qp):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, qp)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, qp):
        x = self.forward_features(x, qp)
        logits = self.head(x)
        out = torch.sigmoid(logits)
        return out

# ==================================================================================
# SECTION 3: ACCURACY CALCULATION
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
    
    correct_prediction_valid_32 = y_flat_valid_32 * (torch.round(y_conv_flat_32) == torch.round(y_flat_32)).float()
    accuracy_32 = torch.sum(correct_prediction_valid_32) / (torch.sum(y_flat_valid_32) + epsilon) * 100
    
    correct_prediction_valid_16 = y_flat_valid_16 * (torch.round(y_conv_flat_16) == torch.round(y_flat_16)).float()
    accuracy_16 = torch.sum(correct_prediction_valid_16) / (torch.sum(y_flat_valid_16) + epsilon) * 100
    
    avg_acc = (accuracy_64 + accuracy_32 + accuracy_16) / 3
    return avg_acc, accuracy_64, accuracy_32, accuracy_16

# ==================================================================================
# SECTION 4: TRAINING SETUP & LOOP
# ==================================================================================

def main():
    # --- Configuration ---
    train_file_path = "/root/myproject/HEVC_Intra_Models-ViT/Data/AI_Train_1668975.dat_shuffled"
    validation_file_path = "/root/myproject/HEVC_Intra_Models-ViT/Data/AI_Valid_98175.dat_shuffled"
    TRAINSET_MAXSIZE = 1668975
    VALIDSET_MAXSIZE = 98175
    BATCH_SIZE = 512

    # --- PROPOSED LIGHTWEIGHT CONFIGURATION ---
    # By reducing embed_dim and depths, we drastically cut the parameter count.
    config = {
        "learning_rate": 0.001,
        "optimizer": "AdamW",
        "epochs": 10000,
        "architecture": "Swin-V2-HEVC-Light", # Renamed for clarity
        "batch_size": BATCH_SIZE,
        "embed_dim": 24,
        "depths":  [1, 2, 4, 1],
        "num_heads": [1, 2, 4, 8],
        "window_size": 4,
    }
    
    # --- Initialize wandb ---
    try:
        wandb.login(key="5c560f0045b5a49dcf8caa862e58469329427192") 
        wandb.init(
            project="Swin-V2-HEVC-Partition", 
            config=config
        )
    except Exception as e:
        print(f"Could not initialize wandb: {e}")
        class DummyWandb:
            def init(self, *args, **kwargs): pass
            def login(self, *args, **kwargs): pass
            def watch(self, *args, **kwargs): pass
            def log(self, *args, **kwargs): pass
        wandb = DummyWandb()
        wandb.init()


    device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- DataLoaders (commented out for pure inference test) ---
    # train_loader, train_indices = create_subset_dataloader(train_file_path, TRAINSET_MAXSIZE, 80000, BATCH_SIZE, shuffle=True)
    # validation_loader, validation_indices = create_subset_dataloader(validation_file_path, VALIDSET_MAXSIZE, 60000, BATCH_SIZE, shuffle=False)
    
    # --- Model Definition ---
    model = SwinTransformerV2(
        img_size=IMAGE_SIZE,
        patch_size=4,
        in_chans=NUM_CHANNELS,
        num_classes=21,
        window_size=config['window_size'],
        embed_dim=config['embed_dim'],
        depths=config['depths'],
        num_heads=config['num_heads']
    ).to(device)
    
    # wandb.watch(model) # Not needed for inference-only script
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ==================== INFERENCE TIMING SECTION ====================
    print("\n--- Preparing for Inference Time Measurement on LIGHTWEIGHT Swin ---")
    
    # No need to load a checkpoint for a speed test
    # timing_checkpoint_path = 'best_swinv2_model.pth'
    # if os.path.exists(timing_checkpoint_path):
    #     print(f"Loading checkpoint from {timing_checkpoint_path} for timing...")
    #     checkpoint_timing = torch.load(timing_checkpoint_path, map_location=device)
    #     model.load_state_dict(checkpoint_timing['model_state_dict'])
    # else:
    #     print("INFO: Timing will be on an UNTRAINED model, which is fine for a speed test.")

    model.eval()

    dummy_image = torch.randn(1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_qp = torch.tensor([27.0 / 51.0]).to(device)

    print("Performing a warm-up run...")
    with torch.no_grad():
        _ = model(dummy_image, dummy_qp)

    print("\n--- Starting Inference Time Measurement ---")

    # Synchronize before starting the timer
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_cpu = os.times()
    start_real = time.time()

    with torch.no_grad():
        output = model(dummy_image, dummy_qp)

    # Synchronize after the operation to get accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_real = time.time()
    end_cpu = os.times()

    real_time = (end_real - start_real) * 1000
    user_time = (end_cpu.user - start_cpu.user) * 1000
    system_time = (end_cpu.system - start_cpu.system) * 1000

    print(f"Inference Time (Forward Pass):")
    print(f"  - Real Time:   {real_time:.4f} ms")
    print(f"  - User Time:   {user_time:.4f} ms")
    print(f"  - System Time: {system_time:.4f} ms")
    print("-----------------------------------------\n")

   # ========================================================================

    print("\n--- Calculating FLOPs using thop ---")
    macs, params = profile(model, inputs=(dummy_image, dummy_qp))

    # Convert MACs to GFLOPs (FLOPs ≈ 2 * MACs)
    gflops = (macs * 2) / 1e9

    print(f"Model Parameters: {params:,}")
    print(f"MACs: {macs:,}")
    print(f"GFLOPs (estimated): {gflops:.4f} G")

    print("\n--- Calculating FLOPs using ptflops ---")

    # Define the input constructor required by ptflops for models with non-standard inputs
    # It must return a dictionary of keyword arguments for the model's forward method.
    def input_constructor(input_res):
        B = 1 # Batch size of 1 for calculation
        C, H, W = input_res
        # Create dummy inputs on the correct device, matching the forward(self, x, qp) signature
        dummy_image = torch.randn(B, C, H, W).to(device)
        dummy_qp = torch.tensor([32.0 / 51.0] * B).to(device) # Example: QP 32, normalized
        return {'x': dummy_image, 'qp': dummy_qp}

    # Define the input resolution for the 'x' tensor
    input_res = (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE) # (1, 64, 64)

    # Use ptflops to get MACs and params
    # We *must* use the 'aten' backend for Transformers, as per ptflops documentation.
    # We set as_strings=False to get raw numbers for our custom printing.
    macs, params = get_model_complexity_info(
        model,
        input_res,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
        input_constructor=input_constructor,
        backend='aten'
    )

    # Convert MACs to GFLOPs (FLOPs ≈ 2 * MACs)
    # This calculation is the same as the one used by 'thop'
    gflops = (macs * 2) / 1e9

    # Print in the *exact same format* as the original script
    print(f"Model Parameters: {params:,}")
    print(f"MACs: {macs:,}")
    print(f"GFLOPs (estimated): {gflops:.4f} G")

    # ==================== BATCH INFERENCE TIMING (AVERAGE PER SAMPLE) ====================
    print("\n--- Preparing for Batch Inference Time Measurement ---")

    # Use the BATCH_SIZE defined earlier (n=64)
    n_batch_size = BATCH_SIZE 
    print(f"Using batch size (n): {n_batch_size}")

    # 1. Create dummy batch inputs
    dummy_batch_image = torch.randn(n_batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_batch_qp = torch.tensor([32.0 / 51.0] * n_batch_size).to(device) # Example: QP 32, normalized

    # 2. Perform a "warm-up" run for the batch
    print("Performing a batch warm-up run...")
    with torch.no_grad():
        _ = model(dummy_batch_image, dummy_batch_qp)

    print("\n--- Starting Batch Inference Time Measurement ---")

    n_iterations = 10
    start_cpu_batch = os.times()
    start_real_batch = time.time()
    
    with torch.no_grad():
        for _ in range(n_iterations):
            output_batch = model(dummy_batch_image, dummy_batch_qp)

    end_real_batch = time.time()
    end_cpu_batch = os.times()

    batch_real_time_ms = (end_real_batch - start_real_batch) * 1000 / n_iterations
    batch_user_time_ms = (end_cpu_batch.user - start_cpu_batch.user) * 1000 / n_iterations
    batch_system_time_ms = (end_cpu_batch.system - start_cpu_batch.system) * 1000 / n_iterations
    batch_cpu_total_ms = batch_user_time_ms + batch_system_time_ms

    print(f"Total Batch Inference Time (for {n_batch_size} samples):")
    print(f"- Real Time:{batch_real_time_ms:.4f} ms")
    print(f"- CPU (User+System) Time: {batch_cpu_total_ms:.4f} ms")

    avg_real_time_ms = batch_real_time_ms / n_batch_size
    avg_cpu_time_ms = batch_cpu_total_ms / n_batch_size

    print(f"\nAverage Per-Sample Inference Time (in batch of {n_batch_size}):")
    print(f"- Avg Real Time:{avg_real_time_ms:.4f} ms/sample")
    print(f"- Avg CPU (User+System) Time: {avg_cpu_time_ms:.4f} ms/sample")
    print("-----------------------------------------\n")
    
    exit()

if __name__ == '__main__':
    main()