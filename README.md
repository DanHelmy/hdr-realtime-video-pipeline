# HDR Real-Time Video Processing Framework

![Version](https://img.shields.io/badge/version-v0.3-blue)
![Status](https://img.shields.io/badge/status-active%20development-yellow)
![Thesis](https://img.shields.io/badge/type-academic%20research-green)

---

## 📌 Overview

This repository contains the implementation framework for my undergraduate thesis:

Mixed-Precision Quantization for HDR Reconstruction Networks (HDRTVNet++)

The project investigates the performance–accuracy trade-off of FP32, FP16, and mixed-precision INT8 quantization techniques for real-time SDR-to-HDR video reconstruction.

The system is designed as a modular real-time video pipeline that progressively integrates optimized inference backends while maintaining clean architectural separation between decoding, inference, and rendering.

---

## 🚀 Current Status (v0.3)

Version v0.3 introduces ONNX Runtime + DirectML GPU acceleration into the real-time pipeline.

### Newly Implemented

- FP32 HDRTVNet++ exported to ONNX
- ONNX Runtime inference integration
- DirectML GPU acceleration (Windows 11 / DirectX 12)
- Dual-input model handling (image + condition tensor)
- Optimized session configuration (full graph optimization)
- Real-time FPS overlay
- Modular inference abstraction preserved

### Current Performance

- 1080p recommended for testing
- DirectML GPU execution
- ~6–7 FPS on mid-range GPU (baseline FP32 ONNX)
- CPU usage significantly reduced compared to PyTorch-only inference

This release establishes the ONNX + GPU accelerated FP32 baseline that will serve as the reference point for:

- FP16 acceleration
- INT8 quantization
- Mixed-Precision QAT optimization

---

## 🧠 Architecture Overview

System flow:

Video Source → Preprocess → HDRTVNet (ONNX FP32) → Postprocess → Renderer

The processor abstraction allows seamless integration of future inference backends:

- FP32 PyTorch (legacy baseline)
- FP32 ONNX (current GPU baseline)
- FP16 ONNX
- INT8 (PTQ)
- Mixed-Precision QAT
- DirectML backend acceleration

The architecture ensures inference engines evolve independently from the core video pipeline.

---

## 🛠 Installation

### Requirements

- Windows 11 (DirectX 12 required for DirectML)
- Python 3.10+
- ONNX Runtime (DirectML)
- OpenCV
- NumPy

### Create Virtual Environment

python -m venv venv
venv\Scripts\activate

### Install Dependencies

pip install onnxruntime-directml
pip install opencv-python numpy

If exporting models:
pip install torch torchvision onnx

---

## ▶️ Running the Pipeline

1. Place an SDR video file in the project directory.
2. Update VIDEO_PATH inside src/main.py.
3. Run:

python src/main.py

Press ESC to exit playback.

---

## 📂 Project Structure

src/
 ├── main.py
 ├── video_source.py
 ├── timer.py
 └── models/
      ├── hdrtvnet_onnx.py
      ├── base_processor.py
      └── hdrtvnet_modules/

The models directory contains:
- ONNX inference wrapper
- Extracted HDRTVNet++ architecture modules
- Modular processor abstraction layer

---

## 🎯 Research Objective

This project focuses on:

- Precision-aware optimization
- Real-time inference feasibility
- Latency benchmarking
- Performance–accuracy trade-off analysis
- Mixed-Precision Quantization with QAT

Network architecture modification is outside the scope of this study.

---

## ⚠️ Important Notes

- Current output is SDR-display compatible (PQ/BT.2020 display pipeline not yet implemented).
- True HDR10 output rendering will be implemented in a future release.
- GUI interface is under development.
- Quantization experiments (FP16 / INT8 / Mixed-QAT) are upcoming.

---

## 🎓 Academic Context

This repository represents the implementation component of an undergraduate thesis project.

The official HDRTVNet++ pretrained model is used as the FP32 baseline prior to quantization.

Experimental benchmarks, quantitative evaluation, and comparative analysis will be documented in subsequent releases.

---

Version v0.3 marks the first successful GPU-accelerated HDRTVNet++ inference inside a live real-time video processing pipeline.
