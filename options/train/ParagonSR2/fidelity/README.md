# ParagonSR2 Fidelity Training Configs

This folder contains the **Fidelity (PSNR)** training configurations. 

## What is Fidelity Training?
Fidelity training aims to maximize **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity). The goal is to reconstruct the ground truth image as accurately as possible, without hallucinating new details.

These models are excellent for:
1.  **General Upscaling**: Clean, artifact-free upscaling.
2.  **Pre-training**: These serve as the *perfect* starting point for GAN training. Train a fidelity model first before fine-tuning it with a GAN/Discriminator or use one of my pre-trained models.

## Key Features of these Configs
-   **Auto-VRAM Management**: Dynamically adjusts batch/patch size to maximize GPU utilization (targeted at 12GB cards like the RTX 3060).
-   **Training Automations**:
    -   `IntelligentLearningRateScheduler`: Adapts LR based on loss plateauing.
    -   `IntelligentEarlyStopping`: Prevents over-training when convergence is reached.
    -   `AdaptiveGradientClipping`: Maintains stability during high-learning rate phases.
-   **Modern Formats**: Uses `safetensors` by default for secure and fast checkpoint saving/loading.

## File naming convention
- `2x`: Scale factor 2.
- `ParagonSR2_{Variant}`: The model variant (Realtime, Stream, Photo).
- `fidelity`: Indicates purely pixel-based loss (L1 + MSSIM) without Adversarial Loss.

## Usage
Train using `traiNNer-redux`:
```bash
python train.py -opt options/train/ParagonSR2/fidelity/2xParagonSR2_Photo_fidelity.yml
```

> [!TIP]
> These configs are optimized for an **RTX 3060 12GB**. If you have more or less VRAM, the `DynamicBatchAndPatchSizeOptimizer` will automatically scale the workload for you.
