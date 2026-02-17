# HDR Real-Time Video Processing Framework

![Version](https://img.shields.io/badge/version-v0.2-blue)
![Status](https://img.shields.io/badge/status-active%20development-yellow)
![Thesis](https://img.shields.io/badge/type-academic%20research-green)

---

## 📌 Overview

This repository contains the implementation framework for my undergraduate thesis:

Mixed-Precision Quantization for HDR Reconstruction Networks (HDRTVNet++)

The project investigates the performance–accuracy trade-off of FP32, FP16, and mixed-precision INT8 quantization techniques for real-time SDR-to-HDR video reconstruction.

The system is designed to progressively integrate optimized inference backends while maintaining a clean and modular real-time video pipeline.

---

## 🚀 Current Status (v0.2)

Version v0.2 integrates the full HDRTVNet++ (Ensemble_AGCM_LE) model into the real-time video pipeline.

### Newly Implemented

- FP32 HDRTVNet++ model integration
- Clean inference-only wrapper (no training dependencies)
- Modular BaseProcessor interface
- SDR → HDR real-time video processing
- CPU-based PyTorch inference
- FPS performance measurement overlay

### Current Performance

- 4K input resolution supported
- CPU-only inference
- ~0.5 FPS expected (no optimization yet)

This release establishes the FP32 baseline inside the live pipeline, serving as the foundation for upcoming optimization and quantization experiments.

---

## 🧠 Architecture Overview

System flow:

Video Source → Processor Interface → HDRTVNet FP32 → Renderer

The processor abstraction allows seamless integration of future inference backends:

- FP32 baseline (current)
- FP16 mixed precision
- INT8 quantization (QAT / PTQ)
- ONNX Runtime backend
- DirectML acceleration

The architecture ensures that inference engines can evolve independently from the core video pipeline.

---

## 🛠 Installation

### Requirements

- Python 3.9+
- PyTorch (CPU)
- OpenCV
- NumPy

### Create Virtual Environment

python -m venv venv
venv\Scripts\activate

### Install Dependencies

pip install torch torchvision
pip install opencv-python numpy

---

## ▶️ Running the Pipeline

1. Place an SDR video file in the project directory.
2. Update VIDEO_PATH in main.py.
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
      ├── base_processor.py
      ├── hdrtvnet_fp32.py
      └── hdrtvnet_modules/

The models directory contains the inference abstraction layer and extracted HDRTVNet architecture modules.

---

## 🎓 Academic Context

This repository represents the implementation component of an undergraduate thesis project.

The official HDRTVNet++ pretrained model is used for baseline FP32 evaluation before quantization.

Future releases will introduce:

- ONNX export
- Mixed-precision acceleration
- INT8 quantization experiments
- Real-time optimization

---

This version marks the first successful integration of HDRTVNet++ into a live real-time video processing pipeline.
