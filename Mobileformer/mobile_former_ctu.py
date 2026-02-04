"""Mobile-Former V1 (MODIFIED FOR CTU TASK)

A PyTorch impl of MobileFromer-V1.
 
Paper: Mobile-Former: Bridging MobileNet and Transformer (CVPR 2022)
       https://arxiv.org/abs/2108.05895

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Use the original timm helpers and registry
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model

# *** MODIFIED: Import from our new CTU-specific blocks file ***
from dna_blocks_ctu import DnaBlock, DnaBlock3, _make_divisible, MergeClassifier, Local2Global
from timm.models.layers import DropPath

__all__ = ['MobileFormer']
  
def _cfg(url='', **kwargs):
    # Default CFG is fine, we will override in __init__
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (1, 1),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': (0.5,), 'std': (0.5,), # Adjusted for 1-channel
        'first_conv': 'stem.0', 'classifier': 'classifier',
        **kwargs
    }

default_cfgs = {
    'default': _cfg(url=''),
}

class MobileFormer(nn.Module):
    def __init__(
        self,
        block_args,
        num_classes=21,      # MODIFIED: Default to 21 classes
        img_size=64,         # MODIFIED: Default to 64x64
        width_mult=1.,
        in_chans=1,          # MODIFIED: Default to 1 channel
        stem_chs=16,
        num_features=1280,
        dw_conv='dw',
        kernel_size=(3,3),
        cnn_exp=(6,4),
        group_num=1,
        se_flag=[2,0,2,0],
        hyper_token_id=0,
        hyper_reduction_ratio=4,
        token_dim=128,
        token_num=6,
        cls_token_num=1,
        last_act='relu',
        last_exp=6,
        gbr_type='mlp',
        gbr_dynamic=[False, False, False],
        gbr_norm='post',
        gbr_ffn=False,
        gbr_before_skip=False,
        gbr_drop=[0.0, 0.0],
        mlp_token_exp=4,
        drop_rate=0.,
        drop_path_rate=0.,
        cnn_drop_path_rate=0.,
        attn_num_heads = 2,
        remove_proj_local=True,
        **ignored_kwargs  
        ):

        super(MobileFormer, self).__init__()

        cnn_drop_path_rate = drop_path_rate
        mdiv = 8 if width_mult > 1.01 else 4
        self.num_classes = num_classes

        #global tokens
        self.tokens = nn.Embedding(token_num, token_dim) 

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_chs, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_chs),
            nn.ReLU6(inplace=True)
        )
        input_channel = stem_chs

        # blocks
        layer_num = len(block_args)
        
        # MODIFIED: Calculate input resolution for 64x64 image with stride 2 stem
        # 64 / 2 = 32. 32*32 = 1024
        inp_res = (img_size // 2) * (img_size // 2) 
        
        layers = []
        for idx, val in enumerate(block_args):
            b, t, c, n, s, t2 = val # t2 for block2 the second expand
            block = eval(b)

            t = (t, t2)
            output_channel = _make_divisible(c * width_mult, mdiv) if idx > 0 else _make_divisible(c * width_mult, 4) 

            drop_path_prob = drop_path_rate * (idx+1) / layer_num
            cnn_drop_path_prob = cnn_drop_path_rate * (idx+1) / layer_num

            layers.append(block(
                input_channel, 
                output_channel, 
                s, 
                t, 
                dw_conv=dw_conv,
                kernel_size=kernel_size,
                group_num=group_num,
                se_flag=se_flag,
                hyper_token_id=hyper_token_id,
                hyper_reduction_ratio=hyper_reduction_ratio,
                token_dim=token_dim, 
                token_num=token_num,
                inp_res=inp_res,
                gbr_type=gbr_type,
                gbr_dynamic=gbr_dynamic,
                gbr_ffn=gbr_ffn,
                gbr_before_skip=gbr_before_skip,
                mlp_token_exp=mlp_token_exp,
                norm_pos=gbr_norm,
                drop_path_rate=drop_path_prob,
                cnn_drop_path_rate=cnn_drop_path_prob,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,        
            ))
            input_channel = output_channel

            if s == 2:
                inp_res = inp_res // 4

            for i in range(1, n):
                layers.append(block(
                    input_channel, 
                    output_channel, 
                    1, 
                    t, 
                    dw_conv=dw_conv,
                    kernel_size=kernel_size,
                    group_num=group_num,
                    se_flag=se_flag,
                    hyper_token_id=hyper_token_id,
                    hyper_reduction_ratio=hyper_reduction_ratio,
                    token_dim=token_dim, 
                    token_num=token_num,
                    inp_res=inp_res,
                    gbr_type=gbr_type,
                    gbr_dynamic=gbr_dynamic,
                    gbr_ffn=gbr_ffn,
                    gbr_before_skip=gbr_before_skip,
                    mlp_token_exp=mlp_token_exp,
                    norm_pos=gbr_norm,
                    drop_path_rate=drop_path_prob,
                    cnn_drop_path_rate=cnn_drop_path_prob,
                    attn_num_heads=attn_num_heads,
                    remove_proj_local=remove_proj_local,
                ))
                input_channel = output_channel

        self.features = nn.Sequential(*layers)

        # last layer of local to global
        self.local_global = Local2Global(
            input_channel,
            block_type = gbr_type,
            token_dim=token_dim,
            token_num=token_num,
            inp_res=inp_res,
            use_dynamic = gbr_dynamic[0],
            norm_pos=gbr_norm,
            drop_path_rate=drop_path_rate,
            attn_num_heads=attn_num_heads
        )

        # classifer
        self.classifier = MergeClassifier(
            input_channel, 
            oup=num_features, 
            ch_exp=last_exp,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_branch=gbr_drop,
            group_num=group_num,
            token_dim=token_dim,
            cls_token_num=cls_token_num,
            last_act = last_act,
            hyper_token_id=hyper_token_id,
            hyper_reduction_ratio=hyper_reduction_ratio
        )

        #initialize
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x, qp): # *** MODIFIED: Added 'qp' ***
        # setup tokens
        bs, _, _, _ = x.shape
        z = self.tokens.weight
        tokens = z[None].repeat(bs, 1, 1).clone()
        tokens = tokens.permute(1, 0, 2)
 
        # stem
        x = self.stem(x)

        # *** MODIFIED: Pass (x, tokens, qp) as a tuple to self.features ***
        x, tokens, _ = self.features((x, tokens, qp))
        
        # These final blocks do not need qp in this implementation
        tokens, attn = self.local_global((x, tokens))
        y = self.classifier((x, tokens))

        # *** MODIFIED: Add sigmoid for BCELoss ***
        return torch.sigmoid(y)

def _create_mobile_former(variant, pretrained=False, **kwargs):
    # This helper is from the original file, we can keep it
    # Note: We must pass pretrained=False as we don't have weights
    model = build_model_with_cfg(
        MobileFormer, 
        variant, 
        pretrained=False, # Always False for our custom task
        default_cfg=default_cfgs['default'],
        **kwargs)
    # print(model) # Optional: remove print
    return model

common_model_kwargs = dict(
    cnn_drop_path_rate = 0.1,
    dw_conv = 'dw',
    kernel_size=(3, 3),
    cnn_exp = (6, 4),
    cls_token_num = 1,
    hyper_token_id = 0,
    hyper_reduction_ratio = 4,
    attn_num_heads = 2,
    gbr_norm = 'post',
    mlp_token_exp = 4,
    gbr_before_skip = False,
    gbr_drop = [0., 0.],
    last_act = 'relu',
    remove_proj_local = True,
)

# --- Registering the model variants ---
# We keep these so you can easily call them by name

@register_model
def mobile_former_508m(pretrained=False, **kwargs):
    #stem = 24
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 2,  24, 1, 1, 0], #1 112x112 (1)
        ['DnaBlock3', 6,  40, 1, 2, 4], #2 56x56 (2)
        ['DnaBlock',  3,  40, 1, 1, 3], #3
        ['DnaBlock3', 6,  72, 1, 2, 4], #4 28x28 (2)
        ['DnaBlock',  3,  72, 1, 1, 3], #5
        ['DnaBlock3', 6, 128, 1, 2, 4], #6 14x14 (4)
        ['DnaBlock',  4, 128, 1, 1, 4], #7
        ['DnaBlock',  6, 176, 1, 1, 4], #8
        ['DnaBlock',  6, 176, 1, 1, 4], #9
        ['DnaBlock3', 6, 240, 1, 2, 4], #10 7x7 (3)
        ['DnaBlock',  6, 240, 1, 1, 4], #11
        ['DnaBlock',  6, 240, 1, 1, 4], #12
    ]
   
    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True, # This is True, so our QP injection will work
        num_features = 1920,
        stem_chs = 24,
        token_num = 6,
        token_dim = 192,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_mobile_former("mobile_former_508m", pretrained, **model_kwargs)
    return model

@register_model
def mobile_former_294m(pretrained=False, **kwargs):
    #stem = 16
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 2,  16, 1, 1, 0], #1 112x112 (1)
        ['DnaBlock3', 6,  24, 1, 2, 4], #2 56x56 (2)
        ['DnaBlock',  4,  24, 1, 1, 4], #3
        ['DnaBlock3', 6,  48, 1, 2, 4], #4 28x28 (2)
        ['DnaBlock',  4,  48, 1, 1, 4], #5
        ['DnaBlock3', 6,  96, 1, 2, 4], #6 14x14 (4)
        ['DnaBlock',  4,  96, 1, 1, 4], #7
        ['DnaBlock',  6, 128, 1, 1, 4], #8
        ['DnaBlock',  6, 128, 1, 1, 4], #9
        ['DnaBlock3', 6, 192, 1, 2, 4], #10 7x7 (3)
        ['DnaBlock',  6, 192, 1, 1, 4], #11
        ['DnaBlock',  6, 192, 1, 1, 4], #12
    ]
  
    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True, # This is True, so our QP injection will work
        num_features = 1920,
        stem_chs = 16,
        token_num = 6,
        token_dim = 192,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_mobile_former("mobile_former_294m", pretrained, **model_kwargs)
    return model

@register_model
def mobile_former_tiny_ctu(pretrained=False, **kwargs):

    stem_chs = 12 # Slightly larger stem
    token_num = 4 # Keep small number of tokens
    token_dim = 128 # Increased token dimension
    num_features = 768 # Increased classifier head features

    # Increased channel counts ('c' values) in later stages
    dna_blocks = [
        #b, e1, c, n, s, e2 (c = output channels)
        ['DnaBlock3', 3, 12, 1, 2, 0], # Stage 1 -> 32x32 out
        ['DnaBlock',  3, 12, 1, 1, 3], #
        ['DnaBlock3', 6, 24, 1, 2, 4], # Stage 2 -> 16x16 out
        ['DnaBlock',  3, 24, 1, 1, 3], #
        ['DnaBlock3', 6, 48, 1, 2, 4], # Stage 3 -> 8x8 out
        ['DnaBlock',  4, 48, 1, 1, 4], #
        ['DnaBlock',  6, 72, 1, 1, 4], # <--- Increased c
        ['DnaBlock3', 6, 96, 1, 2, 4], # Stage 4 -> 4x4 out <--- Increased c
        ['DnaBlock',  6, 96, 1, 1, 4], # <--- Increased c
    ]

    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True, # Enable FFN in GlobalBlock for QP injection
        num_features = num_features,
        stem_chs = stem_chs,
        token_num = token_num,
        token_dim = token_dim,
        # Keep common kwargs from the original file
        cnn_drop_path_rate = 0.1, # Keep defaults for now
        dw_conv = 'dw',
        kernel_size=(3, 3),
        cnn_exp = (6, 4),
        cls_token_num = 1,
        hyper_token_id = 0,
        hyper_reduction_ratio = 4,
        attn_num_heads = 2,
        gbr_norm = 'post',
        mlp_token_exp = 4,
        gbr_before_skip = False,
        gbr_drop = [0., 0.],
        last_act = 'relu',
        remove_proj_local = True,
        **kwargs,
    )
    # Note: pretrained must be False
    model = _create_mobile_former("mobile_former_tiny_ctu", pretrained=False, **model_kwargs)
    return model

@register_model
def mobile_former_1M_ctu(pretrained=False, **kwargs):
    # Aiming for ~1M params.
    # Start with 0.8M config and increase widths slightly.
    
    stem_chs = 8        # Keep stem small
    token_num = 3       # Keep token count low
    token_dim = 80      # Slightly increase token dim (was 64 in 0.8M)
    num_features = 512  # Keep classifier head small (was 768 in 2.3M)

    # 0.8M `c`: [ 8,  8, 16, 16, 32, 32, 48, 64, 64]
    # 2.3M `c`: [12, 12, 24, 24, 48, 48, 72, 96, 96]
    # New `c`:
    dna_blocks = [
        #b, e1, c, n, s, e2 (c = output channels)
        ['DnaBlock3', 3, 12, 1, 2, 0], # Stage 1 -> 32x32 out (c=12)
        ['DnaBlock',  3, 12, 1, 1, 3], # 
        ['DnaBlock3', 6, 20, 1, 2, 4], # Stage 2 -> 16x16 out (c=20)
        ['DnaBlock',  3, 20, 1, 1, 3], #
        ['DnaBlock3', 6, 40, 1, 2, 4], # Stage 3 -> 8x8 out (c=40)
        ['DnaBlock',  4, 40, 1, 1, 4], #
        ['DnaBlock',  6, 64, 1, 1, 4], # (c=64)
        ['DnaBlock3', 6, 80, 1, 2, 4], # Stage 4 -> 4x4 out (c=80)
        ['DnaBlock',  6, 80, 1, 1, 4], # (c=80)
    ]

    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True, # Enable FFN in GlobalBlock for QP injection
        num_features = num_features,
        stem_chs = stem_chs,
        token_num = token_num,
        token_dim = token_dim,
        # Keep common kwargs from the original file
        cnn_drop_path_rate = 0.1, 
        dw_conv = 'dw',
        kernel_size=(3, 3),
        cnn_exp = (6, 4),
        cls_token_num = 1,
        hyper_token_id = 0,
        hyper_reduction_ratio = 4,
        attn_num_heads = 2,
        gbr_norm = 'post',
        mlp_token_exp = 4,
        gbr_before_skip = False,
        gbr_drop = [0., 0.],
        last_act = 'relu',
        remove_proj_local = True,
        **kwargs,
    )
    model = _create_mobile_former("mobile_former_1M_ctu", pretrained=False, **model_kwargs)
    return model

# ... (You can keep the other model variants like 214m, 151m, etc. They will all work)