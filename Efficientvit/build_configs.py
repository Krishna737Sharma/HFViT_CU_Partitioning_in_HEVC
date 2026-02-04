# ----------------------------------------------------------------------------------
# Model Configuration Dictionaries
#
# This file stores hyperparameter configurations for different variants
# of the EfficientViT model. These dictionaries are imported by other
# scripts (like training and inference) to build a model with a specific
# size and set of parameters.
# ----------------------------------------------------------------------------------

# Configuration for the "TINY" M0 variant, modified for this project
# to have a reduced parameter count for faster testing and inference.
EfficientViT_m0_TINY = {
        'img_size': 64,             # Input image resolution (64x64 pixels).
        'patch_size': 16,           # Base patch size. The model's patch embedding uses stacked convs to reach a 16x stride (64/16 = 4x4 feature map).
        'embed_dim': [48, 96, 144], # Embedding dimension (number of channels) for each of the three stages.
        'depth': [1, 2, 3],         # Number of EfficientViT blocks in each stage.
        'num_heads': [2, 4, 2],     # Number of attention heads in each stage.
        'window_size': [7, 7, 7],   # Size of the local attention window (e.g., 7x7) for each stage.
        'kernels': [3, 3, 3, 3],    # Kernel sizes for the convolutional layers within the attention blocks.
        'ffn_exp_ratio': 1.5,       # Expansion ratio for the FeedForward Network (FFN). Hidden dim = embed_dim * 1.5.
    }

# Configuration for the standard EfficientViT M0 model.
EfficientViT_m0 = {
        'img_size': 64,
        'patch_size': 16,
        'embed_dim': [64, 128, 192], # Standard dimensions for M0.
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [5, 5, 5, 5],
        'ffn_exp_ratio': 2.0,       # Standard FFN expansion ratio.
    }

# Configuration for the standard EfficientViT M1 model.
EfficientViT_m1 = {
        'img_size': 224,            # Standard ImageNet resolution.
        'patch_size': 16,
        'embed_dim': [128, 144, 192],
        'depth': [1, 2, 3],
        'num_heads': [2, 3, 3],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
        'ffn_exp_ratio': 2.0,
    }

# Configuration for the standard EfficientViT M2 model.
EfficientViT_m2 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 192, 224],
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 2],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
        'ffn_exp_ratio': 2.0,
    }

# Configuration for the standard EfficientViT M3 model.
EfficientViT_m3 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 240, 320],
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 4],
        'window_size': [7, 7, 7],
        'kernels': [5, 5, 5, 5],
        'ffn_exp_ratio': 2.0,
    }

# Configuration for the standard EfficientViT M4 model.
EfficientViT_m4 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 256, 384],
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
        'ffn_exp_ratio': 2.0,
    }

# Configuration for the standard EfficientViT M5 model.
EfficientViT_m5 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [192, 288, 384],
        'depth': [1, 3, 4],
        'num_heads': [3, 3, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
        'ffn_exp_ratio': 2.0,
    }