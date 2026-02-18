# HDR Real-Time Video Processing Framework

![Version](https://img.shields.io/badge/version-v0.5-blue)
![Status](https://img.shields.io/badge/status-active%20development-yellow)
![Thesis](https://img.shields.io/badge/type-academic%20research-green)

---

## 🚀 Overview

This repository contains the implementation framework for an undergraduate thesis:

**Mixed-Precision Quantization for HDR Reconstruction Networks (HDRTVNet++)**

The project studies performance-accuracy tradeoffs between FP32, FP16, and upcoming INT8 / mixed-precision methods for real-time SDR-to-HDR video reconstruction.

Version `v0.5` focuses on pipeline-level performance engineering before INT8 integration.

---

## ✨ Current Status (v0.5)

### Implemented

- FP32 and FP16 HDRTVNet++ ONNX inference
- Cross-backend ONNX Runtime provider selection
- GPU-first auto provider with CPU fallback
- Async video prefetch queue (`--prefetch`)
- Reused preprocess buffers for lower CPU overhead
- Stage timing breakdown (`pre`, `run`, `post`)
- Frame pacing stats (`fps`, `fps_1p_low`, `late`, `drop_est`)
- Benchmark matrix runner (`benchmark_matrix.py`)
- Static-shape ONNX export support (`--static`)
- Static-model runtime input guard (clear error on mismatch)

### Pipeline

`Video Source -> Preprocess -> ONNX Runtime -> Postprocess -> Renderer`

---

## 🛠 Installation

### Requirements

- Python 3.10+
- OpenCV
- NumPy
- ONNX Runtime backend package for your hardware

### Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### ONNX Runtime backend packages (install one)

- AMD / Windows (DirectML): `pip install onnxruntime-directml`
- NVIDIA (CUDA): `pip install onnxruntime-gpu`
- CPU only: `pip install onnxruntime`

---

## ▶️ Running

Default:

```bash
python src/main.py
```

Common:

```bash
python src/main.py --model hdrtvnet_fp16.onnx --provider dml --prefetch 8
python src/main.py --model hdrtvnet_fp32.onnx --provider dml --prefetch 8
python src/main.py --no-display --timing-interval 120
python src/main.py --model-stage-timing --timing-interval 120
python src/main.py --target-fps 8 --timing-interval 120
```

Provider options:

- `auto`
- `dml`
- `cuda`
- `rocm`
- `tensorrt`
- `coreml`
- `openvino`
- `cpu`

---

## 📊 Benchmarking

### Single run benchmark style

```bash
python src/main.py --model hdrtvnet_fp16.onnx --provider dml --prefetch 8 --no-display --warmup 30 --timing-interval 120 --model-stage-timing
```

### Matrix benchmark (auto CSV)

```bash
python benchmark_matrix.py --provider dml --prefetch-values 0,8 --max-frames 360 --warmup 30 --timing-interval 120 --target-fps 8
```

This writes `benchmark_results_*.csv` in the project root.

---

## 📦 Export ONNX

Dynamic shape:

```bash
python export_onnx_fp32.py --output hdrtvnet_fp32.onnx
python export_onnx_fp16.py --output hdrtvnet_fp16.onnx
```

Static shape (example 1440x1080):

```bash
python export_onnx_fp32.py --static --height 1080 --width 1440 --output hdrtvnet_fp32_1440x1080_static.onnx
python export_onnx_fp16.py --static --height 1080 --width 1440 --output hdrtvnet_fp16_1440x1080_static.onnx
```

---

## 📌 Notes

- Dynamic models are recommended when source resolution/aspect ratio varies.
- Static models are useful for fixed-resolution benchmarking.
- For 4:3 content (e.g., 1440x1080), keep matching width/height to avoid stretch.

---

## 🎓 Academic Context

This repository is the implementation component of an undergraduate thesis focused on precision-aware optimization for real-time HDR reconstruction.
