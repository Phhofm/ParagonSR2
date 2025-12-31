# ParagonSR2 Release Toolkit 🛠️

This directory contains the essential scripts for converting, benchmarking, and running inference with ParagonSR2 models. These tools are designed to bridge the gap between training and real-world deployment.

---

## 1. `convert_onnx_release.py`
Exports trained PyTorch models to high-performance ONNX format, specifically optimized for NVIDIA TensorRT.

### Key Features
- **TRT Pathing**: Automatically replaces `AdaptiveAvgPool` with `ReduceMean` ops for maximum TensorRT compatibility.
- **Dynamic Shapes**: Supports dynamic input resolutions, allowing a single engine to handle varied image sizes.
- **Video Mode (Feature Tap)**: The `--video` flag exports a specialized multi-input/multi-output model that enables temporal stabilization.
- **Accuracy Guard**: Built-in PSNR validation ensures the exported ONNX model matches PyTorch output within 0.01dB.

### Usage
```bash
python scripts/convert_onnx_release.py \
    --checkpoint "models/2x_photo.safetensors" \
    --arch paragonsr2_photo \
    --scale 2 \
    --output "export" \
    --video  # Enable if you plan to use it for video upscaling
```

---

## 2. `benchmark_release.py`
A scientific benchmarking tool to measure latency, throughput, and memory consumption across different backends.

### Backends Supported
- **PyTorch**: Standard FP32 and FP16/AMP.
- **Torch Compile**: Highly optimized PyTorch 2.0+ graph compilation.
- **TensorRT**: The gold standard for NVIDIA GPU inference.

### Usage
```bash
python scripts/benchmark_release.py \
    --input "datasets/val_set" \
    --scale 2 \
    --pt_model "models/2x_stream.safetensors" \
    --arch paragonsr2_stream \
    --trt_engine "export/2x_stream_fp16.trt"
```

---

## 3. `run_inference.py`
The "Universal Runner" for end-users. It automatically handles image and video processing with advanced features.

### Advanced Features
- **Auto-Backend**: Tries to use TensorRT if a `.trt` file exists next to your model, otherwise falls back to Compiled PyTorch.
- **Temporal Stabilization (Video Mode)**:
    - Automatically detects video inputs and enables **Feature-Tap Temporal Smoothing**.
    - Injects features from the previous frame into the current one to eliminate GAN-flicker and noise swimming.
    - **Scene Detection**: Monitors luma shifts and resets temporal state on scene cuts to prevent "ghosting" artifacts.
- **NaN / Inf Guard**: Automatically detects numerical instability in FP16 and retries the specific frame in FP32 to ensure perfect output.

### Usage
```bash
# Basic Image Upscaling
python scripts/run_inference.py \
    --input "photo.png" \
    --model "models/2x_photo.safetensors" \
    --arch paragonsr2_photo \
    --scale 2

# Advanced Video Upscaling with TensorRT
python scripts/run_inference.py \
    --input "movie.mp4" \
    --model "models/2x_stream.trt" \
    --arch paragonsr2_stream \
    --scale 2
```

---

## 🚀 Building a TensorRT Engine

For maximum performance, we recommend building TensorRT engines with `trtexec`.

### Standard Engine (Images)
```bash
trtexec --onnx=export/model_fp32.onnx \
        --saveEngine=export/model_fp16.trt \
        --fp16 \
        --minShapes=input:1x3x64x64 \
        --optShapes=input:1x3x720x1280 \
        --maxShapes=input:1x3x1080x1920
```

### Video Engine (Temporal)
If the ONNX was exported with the `--video` flag, you must provide the `prev_feat` dimension:
```bash
trtexec --onnx=export/model_video_fp32.onnx \
        --saveEngine=export/model_video_fp16.trt \
        --fp16 \
        --minShapes=input:1x3x64x64,prev_feat:1x64x64x64 \
        --optShapes=input:1x3x720x1280,prev_feat:1x64x720x1280 \
        --maxShapes=input:1x3x1080x1920,prev_feat:1x64x1080x1920
```
*(Note: `prev_feat` channel count depends on the variant: Realtime=16, Stream=32, Photo/Pro=64)*
