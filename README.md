# CSE463 Project

This repository contains my personal experiments on the "Lung & Colon Cancer Dataset-unstable dataset" and the "Brain Tumor MRI Dataset" using different hybrid architectures, including CNN, KAN, MLP, ConvNeXt-V2, SE blocks, etc., for comparative analysis.
> **Note:** These are personal experiments, so I cannot guarantee their correctness.  
> If you find any issues, have suggestions or want to contribute, feel free to open an issue or submit a pull request.
**Short Architecture Description:**  
You can find the architecture and implementation details in this file(MRI): [20-10 ConvNeXt-V2 KAN Notebook](https://github.com/lililiyabbayx/Computer-Vision-Project-based-on-KAN-and-Assignments/blob/main/20-10convnext-v2-kan-100ep.ipynb)  
*Description: Here, I will provide a summary of the architecture and key points from the notebook.*
## Architecture Overview

### 1️. ConvNeXt-V2 Baseline Block
- **Input:** (N, C, H, W) — keep a copy as shortcut for residual.
- **DWConv 7×7 (Depthwise Conv):** mixes spatial information per channel, keeps channel count C.
- **Permute → channels-last (N, H, W, C):** needed for position-wise MLPs and LayerNorm.
- **LayerNorm:** normalizes per channel to stabilize training.
- **PW1: expand (C → 4C):** position-wise MLP, acts like 1×1 conv per position to increase feature dimension.
- **GELU:** activation function, adds non-linearity for learning complex mappings.
- **GRN (Global Response Norm):** normalizes features globally, improves feature calibration.
- **PW2: project (4C → C):** reduces dimension back to original channel size.
- **Permute → channels-first (N, C, H, W)**
- **DropPath (optional):** randomly drops residual paths during training for regularization.
- **Residual Add:** adds shortcut input to output for stable training.

---

### 2️. ConvNeXt-V2 + SE Block (Squeeze-and-Excitation)
- Steps 1–8: same as baseline.
- **Global Average Pool over H, W:** squeezes spatial info into (N, C).
- **SE MLP:** Linear(C → C/reduction) → GELU → Linear(C/reduction → C) → Sigmoid — computes channel attention weights.
- **Scale features:** x = x * se — applies channel-wise gating.
- Steps 12–13: permute back → DropPath → Residual Add.  
> SE is applied after channel projection, before residual addition.

---

### 3️. ConvNeXt-V2 + KAN Block
- **Input:** (N, C, H, W) — keep shortcut.
- **DWConv 7×7:** spatial mixing.
- **Permute → channels-last & flatten positions (Npos, C)** for KAN.
- **KAN1: expand (C → 4C):** replaces PW1 MLP, uses spline-based KANLinear for richer channel mixing.
- **GELU**
- **GRN on 4C:** global normalization.
- **KAN2: project (4C → C):** replaces PW2 MLP, brings back original channels.
- **Permute → channels-first (N, C, H, W)**
- **DropPath → Residual Add**  
> KAN replaces MLPs for more expressive non-linear channel transformations.

---

### 4️. ConvNeXt-V2 + KAN + SE Block (SEKAN)
- Steps 1–7: same as KAN block.
- **Global Average Pool (H, W):** for SE gating.
- **SE MLP → Sigmoid:** compute attention per channel.
- **Scale features:** x = x * se.
- Steps 11–12: permute back → DropPath → Residual Add.  
> Combines KAN non-linear mixing with SE attention gating before residual addition.




