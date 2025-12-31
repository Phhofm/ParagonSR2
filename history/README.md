# ParagonSR2: Architectural Evolution 🧬

The road to the final **ParagonSR2** was paved with many experiments. This document serves as a "Dev Log" summarizing the key architectural iterations found in the `history/` folder.

> **Why is this here?** Research is non-linear. My future self (or others) might find value in ideas that were discarded for this specific constraints but might work elsewhere.

---

## The Journey

### 1. `paragonsr_arch.py` (The Predecessor)
*   **Repo**: [ParagonSR (Legacy)](https://github.com/Phhofm/ParagonSR)
*   **Concept**: A Hybrid CNN with Reparameterization and Gated FFNs.
*   **Tech Stack**: `ReparamConvV2` + `InceptionDWConv2d` + `GatedFFN` (Simple Channel Multiplication). 
*   **Why Change?**: Training was rather slow. I wanted to build a successor to my network.
*   **Outcome**: Served as the baseline.

### 2. `paragonsr2_arch_version0.py` (Dynamic Genesis)
*   **Innovation**: **Dynamic Kernels**. Replaced `GatedFFN` with a new `DynamicTransformer`.
*   **Tech**: Introduced `DynamicKernelGenerator`—a tiny sub-network that looks at the input features and generates a *unique 3x3 convolution kernel* for every single sample in the batch.
*   **Goal**: "Smarter, Not Bigger". Give the network a "brain" to adapt its weights on the fly.
*   **Result**: ⚠️ **Mixed**. The quality seemed good, but training was unstable. Dynamic kernels are hard to optimize.

### 3. `paragonsr2_arch_version1.py` (The Deployment Update)
*   **Focus**: Making dynamic networks practical for release.
*   **Innovation**: Added two modes:
    1.  **"Cheap" Mode**: A lightweight SE-Block style channel modulation (Linear->ReLU->Sigmoid) for fast training.
    2.  **"Full" Mode**: The heavy per-sample kernel generation.
*   **Key Tech**: **EMA Kernel Tracking**. I implemented a system to track the "average" dynamic kernel during training. This allowed me to "freeze" the dynamic behavior into a static convolution for export, enabling ONNX/TensorRT compatibility!

### 4. `paragonsr2_arch_version2.py` (Normalization-Free)
*   **Focus**: Speed and INT8 Quantization.
*   **Change**: Removed `GroupNorm` layers entirely. Switched activation from `Mish` to `LeakyReLU`.
*   **Rationale**: Normalization layers slow down inference and complicate INT8 quantization. LeakyReLU is hardware-accelerated almost everywhere.
*   **Result**: A "Professional-Grade" architecture that was significantly faster and easier to deploy.

### 5. `paragonsr2_arch_version3.py` (Robustness Fix)
*   **Problem**: "ChatGPT's Concerns". The dynamic kernels in v0/v1 could sometimes diverge or oscillate.
*   **Fix**: Added **Convergence Monitoring**.
    *   `adaptive_update`: Automatically reduces the kernel update frequency if optimization stalls.
    *   `fallback_to_identity`: Safety mechanism to prevent NaN explosions.
*   **Result**: Much more stable training runs, allowing for deeper networks.

### 6. `paragonsr2_arch_version4.py` (Static Paragon)
*   **Pivot**: Sometimes, simple is better.
*   **Change**: Created a "Static" branch (`StaticDepthwiseTransformer`) that completely removed the dynamic kernel generator.
*   **Tech**: Replaced dynamic kernels with a fixed `DepthwiseConv3x3` + `CheapChannelModulation`.
*   **Use Case**: For ultra-low-power devices where even fused dynamic kernels were too heavy.

### 7. `paragonsr2_arch_version5.py` (The Dual-Path Hybrid)
*   **Major Shift**: Introduced the **Dual-Path Architecture**.
    *   **Path A (Detail)**: The deep network predicts *only* the fine textures/residuals.
    *   **Path B (Base)**: `MagicKernelSharp` provides a solid, artifact-free structural base.
*   **Innovation**: `ContentAwareDetailProcessor`. A module that analyzes deep features to decide *where* to add details (e.g., texture) and where to suppress them (e.g., sky).
*   **Result**: The birth of the modern ParagonSR2 design.

### 8. `paragonsr2_arch_version6.py` (The Synthesis / Release Candidate)
*   **Focus**: Final Polish & Attention.
*   **Tech**:
    *   **LocalWindowAttention**: Replaced `EfficientSelfAttention` with a Swin-style shifted window attention using PyTorch 2.0's `scaled_dot_product_attention` (FlashAttention).
    *   **Gradient Checkpointing**: Enabled training "Pro" models on 12GB cards.
    *   **Multi-Block Support**: Unified `Nano` (MBConv), `Stream` (Gate), and `Photo` (Paragon) into one file.
*   **Result**: The final architecture released in `traiNNer/archs/paragonsr2_arch.py`.

### 9. `paragonsr2_arch_version7.py` (The Performance & Compatibility Update)
*   **Focus**: Maximum training speed and seamless deployment.
*   **Innovations**:
    -   **FlexAttention Fusion**: Implemented fused RPB `score_mod` for fastest possible training on NVIDIA GPUs.
    -   **Intelligent Fallback**: Automatic switch to standard Attention for ONNX export.
    -   **RAttention Proxy**: Replaced recurrent units with convolution-based "Region-Aware Context" (3x3 DW Conv on K/V).
    -   **MSCF**: Introduced Multi-Scale Cross-Fusion (1x1, 3x3, 5x5 kernels).

### 10. `paragonsr2_arch_version8.py` (The Unified Refactor)
*   **Focus**: Lineup cleanup and structural unity.
*   **Change**: Refactored `ParagonBlock` to handle all variants more elegantly. 
*   **Goal**: Prepare the codebase for final release by ensuring all variants share high-level logic while maintaining specialized low-level kernels.
*   **Result**: A very stable but slightly heavy architecture file.

### 11. `paragonsr2_arch_version10.py` (The Professional Polish - RELEASE)
*   **Focus**: Extreme simplification and "Product-First" design.
*   **Major Change**: Stripped away modules that didn't provide a massive quality-to-latency gain (Removed MSCF, RAttention Proxy, and Content-Aware Gating from the main path).
*   **Tech**:
    - Refined **WindowAttention** for perfect ONNX export behavior.
    - Introduced **NanoBlock**, **StreamBlock**, and **PhotoBlock** as distinct, optimized engines.
    - Simplified **PixelShuffle** head for universal compatibility across all scales.
*   **Outcome**: This version represents the best balance of speed, quality, and deployment ease.

### 12. `paragonsr2_arch_version11.py` (The Universal Quality & Stability Update) 🚀
*   **Focus**: State-of-the-Art PSNR/SSIM across ALL image types and extreme training stability.
*   **Major Change**: Introduced the **Pro Tier** and high-depth configurations (up to 36 blocks).
*   **Innovations**:
    - **ProBlock**: A "universal engine" combining Conv, SE Channel Attention, Window Attention (local structure), and **Token Dictionary CA** (global textures).
    - **Numerical Stability**: Switched to `RMSNorm` with FP32 variance calculation and added `LayerScale` to every block to enable stable training of very deep networks.
    - **TokenDictionaryCA**: A cross-attention module that attends to a learned global dictionary of "visual concepts," helping the network reconstruct complex repeating patterns.
    - **Pro Window Size**: Optimized window sizes for the Pro variant to maximize receptive field without sacrificing throughput.
*   **Outcome**: The current flagship branch. It provides a significant quality jump over previous versions, especially for challenging archival or scientific content, while remaining rock-solid during deep training.

---

## Technical Summary of Current Lineup

| Variant | Logic | Primary Use | Deployment |
| :--- | :--- | :--- | :--- |
| **Realtime** | NanoBlock (MBConv) | High-Speed Video | Mobile / Edge |
| **Stream** | StreamBlock (Gated Conv) | Quality Streaming | Desktop Playback |
| **Photo** | PhotoBlock (Attn + Conv) | General Photography | Pro Photo Editing |
| **Pro** | ProBlock (Global + Local) | Scientific / Archival | Cloud / HPC |
