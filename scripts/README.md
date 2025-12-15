# ParagonSR2 scripts

This folder contains utility scripts for benchmarking and exporting ParagonSR2 models.

## 1. `convert_onnx_release.py`
Exports a trained PyTorch model (`.safetensors` or `.pth`) to ONNX format, optimized for inference.

### Usage
```bash
python scripts/convert_onnx_release.py \
    --checkpoint /path/to/model.safetensors \
    --arch paragonsr2_photo \
    --scale 2 \
    --output output_folder
```

## 2. `benchmark_release.py`
A comprehensive benchmarking tool to test:
- Inference Latency (ms)
- FPS
- Peak VRAM usage
- FLOPs / Params count

It supports benchmarking both **PyTorch** (FP32/FP16) and **TensorRT** (if `trtexec` converted engines are present).

### Usage
```bash
python scripts/benchmark_release.py
# Note: You may need to edit the script to point to your specific test dataset if defaults don't match.
```
