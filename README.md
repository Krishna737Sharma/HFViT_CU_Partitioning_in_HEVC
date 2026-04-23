# HEVC Intra Coding Acceleration via Deep Learning-Based CU Depth Prediction

A comprehensive research repository for accelerating HEVC (H.265) intra coding using deep learning models to predict Coding Unit (CU) depth decisions. This work benchmarks multiple architectures — ETH-CNN, HFCN, FasterViT, EfficientFormer, EfficientViT, LeViT, MobileFormer, and others — across four resolutions (720p, 1080p, 2K, 4K).

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Models](#models)
  - [ETH-CNN](#1-eth-cnn)
  - [HFCN](#2-hfcn)
  - [FasterViT](#3-fastervit)
  - [Lightweight Baselines](#4-lightweight-baselines)
- [HEVC Integration](#hevc-integration)
- [Evaluation & Analysis Tools](#evaluation--analysis-tools)
- [Visualization](#visualization)
- [Getting Started](#getting-started)

---

## Overview

Standard HEVC encoders perform exhaustive rate-distortion optimization to determine CU partition depths, which is computationally expensive. This project replaces that process with fast neural network inference, predicting optimal CU depth maps from raw video frames before encoding. Models are trained and evaluated at 720p, 1080p, 2K, and 4K resolutions.

---

## Repository Structure

```
├── ETH_CNN_720p/                    # ETH-CNN for 720p resolution
├── Eth_CNN_1080p/                   # ETH-CNN for 1080p resolution
├── Eth_CNN_2k/                      # ETH-CNN for 2K resolution
├── Pytorch EthCNN/                  # ETH-CNN for 4K resolution (PyTorch)
│
├── Fastervit_720p/                  # FasterViT for 720p resolution
├── Fastervit_1080p/                 # FasterViT for 1080p resolution
├── Fastervit_2k/                    # FasterViT for 2K resolution
├── Fastervit_4k/                    # FasterViT for 4K resolution
│
├── HFCN_720p/                       # HFCN for 720p resolution
├── HFCN_1080p/                      # HFCN for 1080p resolution
├── HFCN_2k/                         # HFCN for 2K resolution
├── HFCN_4k/                         # HFCN for 4K resolution
│
├── Efficientformer/                 # EfficientFormer inference benchmark
├── Efficientvit/                    # EfficientViT inference benchmark
├── LeViT/                           # LeViT inference benchmark
├── Mobileformer/                    # MobileFormer inference benchmark
├── Opencv Visualization/            # CU partition visualization (OpenCV)
│
├── BD_report1.py                    # BD-Rate calculation
├── RD_report1.py                    # RD-curve visualization
├── MS-SSIM_cal.py                   # MS-SSIM metric computation
├── cal_canfidence_interval.py       # Confidence interval calculation
└── SI&TI_test_sequences/            # SI/TI analysis with results
```

---

## Datasets

| Dataset | Resolutions | Link |
|---|---|---|
| CPH Dataset (720p, 1080p, 2K) | 720p · 1080p · 2K | [Kaggle – CPH Multi-Resolution](https://www.kaggle.com/datasets/krishnasharma737/cph-2k-720p-1080p-resolution-datasets) |
| CPH Dataset (4K) | 4K | [Kaggle – CPH 4K](https://www.kaggle.com/datasets/krishnasharma737/cph-dataset-4k-resolution) |
| Test Sequences | All resolutions | [Kaggle – Test Sequences](https://www.kaggle.com/datasets/krishnasharma737/test-sequences) |

The test sequences cover all 10 standard sequences used for SI/TI analysis, spanning **Class A** (2560×1600), **Class B** (1920×1080), and **Class E** (1280×720) as defined in the HEVC common test conditions.

---

## Models

### 1. ETH-CNN

A convolutional neural network adapted from the ETH-CNN architecture for CU depth prediction.

**Folders:** `ETH_CNN_720p`, `Eth_CNN_1080p`, `Eth_CNN_2k`, `Pytorch EthCNN` (4K)

#### Training

Each resolution folder contains `Eth_CNN.py` — the standalone training script.

```bash
# Example for 1080p
cd Eth_CNN_1080p
python Eth_CNN.py
```

**Output weight:** `best_model_ETH_CNN.pth` (720p / 1080p / 2K)

For **4K**, training is handled inside `Pytorch EthCNN/` and produces:

```
best_model_4qp_parallel_data_processing_loss_mod.pth
```

#### Inference Speed Test (4K)

```bash
cd "Pytorch EthCNN"
python cnn_inference_test.py
```

#### Model Evaluation (4K)

```bash
cd "Pytorch EthCNN"
python Model_Evaluation_ETH_CNN.py
```

#### HEVC Integration

Each resolution folder includes `video_to_cu_depth.py`. Copy this script into the `bin/` folder of your HEVC encoder build:

```bash
cp ETH_CNN_720p/video_to_cu_depth.py <HEVC_encoder_root>/bin/
```

Then invoke the script before encoding to generate CU depth maps that the encoder reads in place of exhaustive RDO search.

---

### 2. HFCN

A Hierarchical Fully Convolutional Network designed for multi-scale CU depth estimation.

**Folders:** `HFCN_720p`, `HFCN_1080p`, `HFCN_2k`, `HFCN_4k`

#### Training

```bash
cd HFCN_1080p          # or HFCN_720p / HFCN_2k / HFCN_4k
python HFCN_train.py
```

**Output weight:** `best_model_HFCN_pyt.pth`

#### HEVC Integration

```bash
cp HFCN_1080p/video_to_cu_depth.py <HEVC_encoder_root>/bin/
```

---

### 3. FasterViT

A hybrid CNN-ViT model optimised for high-resolution throughput, with a post-training layer fusion step to minimise inference latency.

**Folders:** `Fastervit_720p`, `Fastervit_1080p`, `Fastervit_2k`, `Fastervit_4k`

#### Training

```bash
cd Fastervit_1080p     # or any resolution folder
python train.py
```

**Output weight:** `best_fastervit_hevc_balanced.pth`

#### Layer Fusion (required before HEVC integration)

```bash
python apply_fuse_model.py
```

**Fused weight:** `best_fastervit_fused_<resolution>.pth`

#### HEVC Integration

Use the **fused** model weight together with the integration script:

```bash
cp Fastervit_1080p/video_to_cu_depth_fastervit.py <HEVC_encoder_root>/bin/
```

Ensure `best_fastervit_fused_1080p.pth` (or the appropriate resolution variant) is accessible from that path.

---

### 4. Lightweight Baselines

The following folders contain **inference speed benchmarks and training code** for lightweight vision transformer variants evaluated as alternative backbones:

| Folder | Architecture |
|---|---|
| `Efficientformer/` | EfficientFormer |
| `Efficientvit/` | EfficientViT |
| `LeViT/` | LeViT |
| `Mobileformer/` | MobileFormer |

Each folder exposes a common interface — a training script and a dedicated inference speed test script — so all architectures can be compared under identical conditions.

---

## HEVC Integration — General Workflow

1. **Train** the model for the target resolution using the provided training script.
2. **Fuse layers** (FasterViT only) using `apply_fuse_model.py`.
3. **Copy** the `video_to_cu_depth*.py` integration script and the best-model weight into `<HEVC_encoder_root>/bin/`.
4. **Run** the integration script ahead of the encoder. It reads raw video frames and writes predicted CU depth maps to a location the encoder is configured to read from.

> The integration scripts are designed to be a drop-in replacement for the RDO-based CTU partitioning stage, requiring no modifications to the encoder's core C/C++ source.

---

## Evaluation & Analysis Tools

### BD-Rate

```bash
python BD_report1.py
```

Computes the Bjøntegaard Delta Rate between two RD curves, reporting the average bitrate saving (or overhead) at equivalent quality.

### RD Curve Visualization

```bash
python RD_report1.py
```

Generates Rate-Distortion plots (PSNR / MS-SSIM vs bitrate) for all compared configurations.

### MS-SSIM

```bash
python MS-SSIM_cal.py
```

Calculates Multi-Scale Structural Similarity (MS-SSIM) between the original and reconstructed sequences.

### Confidence Interval

```bash
python cal_canfidence_interval.py
```

Computes 95 % confidence intervals over per-sequence metric results.

### SI / TI Analysis

The `SI&TI_test_sequences/` folder contains scripts and pre-computed results for Spatial Information (SI) and Temporal Information (TI) characterisation of all 10 test sequences.

---

## Visualization

```bash
cd "Opencv Visualization"
python <visualization_script>.py
```

Renders CU partition boundaries on top of decoded frames using OpenCV, enabling visual inspection of predicted vs. reference depth maps.

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch ≥ 1.12
- OpenCV (`pip install opencv-python`)
- Standard scientific stack: NumPy, Matplotlib, SciPy

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Krishna737Sharma/Deep-Learning-for-Fast-CTU-Partitioning-in-HEVC.git
cd Deep-Learning-for-Fast-CTU-Partitioning-in-HEVC

# 2. Download a dataset (example: 1080p training data)
#    https://www.kaggle.com/datasets/krishnasharma737/cph-2k-720p-1080p-resolution-datasets

# 3. Train ETH-CNN at 1080p
cd Eth_CNN_1080p
python Eth_CNN.py

# 4. Integrate with HEVC encoder
cp video_to_cu_depth.py <HEVC_encoder_root>/bin/

# 5. Evaluate
cd ..
python MS-SSIM_cal.py
python BD_report1.py
python RD_report1.py
```

---

## Citation

If you use this repository or the associated datasets in your research, please cite appropriately and link back to the Kaggle dataset pages listed in the [Datasets](#datasets) section.
