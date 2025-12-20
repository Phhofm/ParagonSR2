# ParagonSR2 scripts

This folder contains utility scripts for benchmarking and exporting ParagonSR2 models.

## 1. `convert_onnx_release.py`
Exports a trained PyTorch model (`.safetensors` or `.pth`) to ONNX format, optimized for inference.

### Key Features
-   **Auto-Patching**: Automatically replaces layers like `AdaptiveAvgPool2d(1)` with TensorRT-optimized versions.
-   **Validation**: Built-in PSNR validation to ensure the exported ONNX model matches PyTorch output exactly.
-   **Architecture Overrides**: Support for overriding training-time parameters (like `attention_mode` or `export_safe`) during export.
-   **Legacy Support**: Includes mapping logic to export models trained on older development versions of the architecture.

### Usage
```bash
python scripts/convert_onnx_release.py \
    --checkpoint /path/to/model.safetensors \
    --arch paragonsr2_photo \
    --scale 2 \
    --output output_folder \
    --val_dir /path/to/validation_dataset
```

## 2. `benchmark_release.py`
A comprehensive benchmarking tool to measure performance across different backends.

### Key Features
-   **Multi-Backend**: Benchmarks PyTorch (FP32), PyTorch (FP16/AMP), PyTorch (Compiled), and TensorRT (FP16).
-   **Markdown Export**: Automatically generates a formatted markdown table of results for direct use in READMEs.
-   **System Info**: Collects detailed hardware info (GPU, VRAM, Driver Version) to provide context for the results.
-   **Attention Suite**: Special flag `--benchmark_attention_modes` to compare No-Attn vs SDPA vs Flex performance.

### Usage
```bash
python scripts/benchmark_release.py \
    --input /path/to/test_dataset \
    --scale 2 \
    --pt_model /path/to/model.safetensors \
    --arch paragonsr2_photo \
    --trt_engine /path/to/engine.trt
```
