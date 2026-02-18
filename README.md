# HDR Real-Time Video Processing Framework

![Version](https://img.shields.io/badge/version-v0.3-blue)
![Status](https://img.shields.io/badge/status-active%20development-yellow)
![Thesis](https://img.shields.io/badge/type-academic%20research-green)

---

## 📌 Overview

This repository contains the implementation framework for an undergraduate thesis:

**Mixed-Precision Quantization for HDR Reconstruction Networks (HDRTVNet++)**

The project studies performance-accuracy tradeoffs between FP32, FP16, and future INT8/mixed-precision methods for real-time SDR-to-HDR video reconstruction.

---

## 🚀 Current Status (v0.3)

Version `v0.3` introduces ONNX Runtime GPU inference in the real-time pipeline with cross-backend provider selection.

### Implemented

- FP32 and FP16 HDRTVNet++ ONNX models
- ONNX Runtime inference integration
- Execution provider auto-selection (GPU-first, CPU fallback)
- Dual-input model handling (`input` + `condition`)
- Runtime timing output (`decode`, `resize`, `infer`, `render`)
- Optional ONNX Runtime profiling output

---

## 🧠 Architecture

Pipeline flow:

`Video Source -> Preprocess -> HDRTVNet (ONNX) -> Postprocess -> Renderer`

Inference backends are kept modular so you can benchmark variants without changing pipeline structure.

---

## 🛠 Installation

### Requirements

- Python 3.10+
- OpenCV
- NumPy
- ONNX Runtime package for your hardware/backend

### Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### ONNX Runtime backend packages

Install **one** backend package that matches your machine:

- AMD / Windows (DirectML): `pip install onnxruntime-directml`
- NVIDIA (CUDA): `pip install onnxruntime-gpu`
- CPU only: `pip install onnxruntime`

---

## ▶️ Running

Default:

```bash
python src/main.py
```

Examples:

```bash
python src/main.py --video input.mp4 --model hdrtvnet_fp16.onnx --provider auto
python src/main.py --provider dml
python src/main.py --provider cuda
python src/main.py --provider cpu
python src/main.py --no-display --timing-interval 120
python src/main.py --ort-profile
```

Available provider options:

- `auto`
- `dml`
- `cuda`
- `rocm`
- `tensorrt`
- `coreml`
- `openvino`
- `cpu`

---

## 📊 Benchmark Notes

- Use `--no-display` for throughput benchmarking.
- Keep video, resolution, drivers, and model constant when comparing FP32 vs FP16.
- `infer` includes preprocess + ONNX session + postprocess.

---

## 📦 Export ONNX

```bash
python export_onnx_fp32.py
python export_onnx_fp16.py
```

---

## 📂 Project Structure

```text
src/
  main.py
  video_source.py
  timer.py
  models/
    base_processor.py
    hdrtvnet_onnx.py
    hdrtvnet_fp32.py
    hdrtvnet_modules/
    weights/
```

---

## 🎓 Academic Context

This repository is the implementation component of an undergraduate thesis focused on precision-aware optimization for real-time HDR reconstruction.
