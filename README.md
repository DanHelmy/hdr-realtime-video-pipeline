# HDR Real-Time Video Processing Framework

![Version](https://img.shields.io/badge/version-v0.1-blue)
![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Thesis](https://img.shields.io/badge/type-academic%20research-green)

---

## 📌 Overview

This repository contains the implementation framework for my undergraduate thesis:

**Mixed-Precision Quantization for HDR Reconstruction Networks (HDRTVNet++)**

The project investigates the performance–accuracy trade-off of FP32, FP16, and mixed-precision INT8 quantization techniques for real-time SDR-to-HDR video reconstruction.

The system is designed to progressively integrate optimized inference backends while maintaining a clean and modular real-time pipeline.

---

## 🚀 Current Status (v0.1)

Version **v0.1** establishes the foundational real-time video processing pipeline.

Currently implemented:

- Real-time video decoding
- Modular processor interface (dummy inference stage)
- FPS performance measurement
- Clean separation between source, processing, and rendering layers

HDRTVNet++ model integration and quantization experiments will be introduced in subsequent releases.

---

## 🧠 Architecture Overview

System flow:

Video Source → Processor Interface → Renderer

The processor abstraction allows seamless integration of:

- FP32 baseline model
- FP16 optimized inference
- Mixed-Precision INT8 (QAT)
- ONNX Runtime backend
- DirectML acceleration

This design ensures inference engines can evolve independently from the core video pipeline.

---

## 🛠 Installation

### Requirements
- Python 3.9+
- OpenCV
- NumPy

### Install Dependencies
pip install -r requirements.txt

### Run the Application
python src/main.py

---

## 🎯 Research Objective

This project focuses on:

- Precision-aware optimization
- Real-time inference feasibility
- Latency measurement and benchmarking
- Performance–accuracy evaluation under quantization

Network architecture modification is outside the scope of this study.

---

## 📚 Academic Context

This repository represents the implementation component of an academic thesis project.

Experimental benchmarks, quantitative evaluation, and comparative analysis will be documented in future releases.


