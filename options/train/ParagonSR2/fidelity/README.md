# ParagonSR2 Fidelity Training Configs

This folder contains the **Fidelity (PSNR)** training configurations. 

## What is Fidelity Training?
Fidelity training aims to maximize **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity). The goal is to reconstruct the ground truth image as accurately as possible, without hallucinating new details.

These models are excellent for:
1.  **General Upscaling**: Clean, artifact-free upscaling.
2.  **Pre-training**: These serve as the *perfect* starting point for GAN training. Train a fidelity model first before fine-tuning it with a GAN/Discriminator or use one of my pre-trained models.

## file naming convention
- `2x`: Scale factor 2.
- `ParagonSR2_{Variant}`: The model variant (Realtime, Stream, Photo, Pro).
- `fidelity`: Indicates purely pixel-based loss (L1/L2/Charbonnier) without Adversarial Loss.

## Usage
Train using `traiNNer-redux`:
```bash
python train.py -opt options/train/ParagonSR2/fidelity/2xParagonSR2_Photo_fidelity.yml
```
