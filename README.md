# HDR Real-Time Video Processing Framework

![Version](https://img.shields.io/badge/version-v0.6-blue)
![Status](https://img.shields.io/badge/status-active%20development-yellow)
![Thesis](https://img.shields.io/badge/type-academic%20research-green)

---

## Overview

This repository contains the implementation framework for an undergraduate thesis:

**Real-Time SDR-to-HDR Video Reconstruction with HDRTVNet++ and PyTorch**

The project achieves real-time HDR reconstruction using a fully GPU-accelerated PyTorch pipeline with `torch.compile` optimizations, including support for AMD ROCm on Windows.

---

## Current Status (v0.6)

### Implemented

- Full PyTorch inference pipeline (FP16, FP32)
- `torch.compile` with Triton (`max-autotune`) — 2.4× model inference speedup
- ROCm-Windows support with automatic platform detection
- GPU-side preprocessing (BGR→RGB, normalize, permute on GPU)
- GPU-side postprocessing (clamp, scale, quantize, RGB→BGR on GPU)
- Pre-allocated GPU tensor buffers (zero per-frame allocation)
- `torch.inference_mode()` throughout
- Async video prefetch queue (`--prefetch`)
- Stage timing breakdown (`pre`, `run`, `post`)
- Frame pacing stats (`fps`, `fps_1p_low`, `late`, `drop_est`)
- Benchmark matrix runner (`benchmark_matrix.py`)
- Optional CUDA graph replay (`--cuda-graphs`)
- Optional channels_last memory format (`--channels-last`)

### Pipeline

```
Video Source → GPU Upload → GPU Preprocess → torch.compile Model → GPU Postprocess → CPU Download → Renderer
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- OpenCV, NumPy

### Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### PyTorch GPU backends

**AMD (ROCm/HIP):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

**NVIDIA (CUDA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Intel (CPU / oneAPI):**
```bash
pip install torch torchvision
```

### ROCm-Windows + torch.compile

To enable `torch.compile` on ROCm-Windows, install Triton and copy HIP SDK headers:

```bash
pip install triton-windows
xcopy /E /I "C:\Program Files\AMD\ROCm\7.1\include\hip" "venv\Lib\site-packages\_rocm_sdk_devel\include\hip"
```

Then use `--force-compile` when running.

---

## Running

Default (FP16, auto device):

```bash
python src/main.py
```

With torch.compile (recommended):

```bash
python src/main.py --force-compile --model-stage-timing
```

FP32 mode:

```bash
python src/main.py --force-compile --precision fp32 --model-stage-timing
```

Headless benchmark:

```bash
python src/main.py --force-compile --no-display --warmup 30 --timing-interval 120 --max-frames 360 --model-stage-timing
```

### CLI Flags

| Flag | Description |
|---|---|
| `--model PATH` | Model weights path (default: `src/models/weights/Ensemble_AGCM_LE.pth`) |
| `--device auto\|cuda\|cpu` | Device selection (default: auto) |
| `--precision auto\|fp16\|fp32` | Inference precision (default: auto → fp16 on GPU) |
| `--force-compile` | Enable `torch.compile` on ROCm (auto on NVIDIA) |
| `--no-compile` | Disable `torch.compile` entirely |
| `--channels-last` | Force channels_last memory format (auto on NVIDIA) |
| `--cuda-graphs` | Enable CUDA graph replay for static shapes |
| `--prefetch N` | Video reader prefetch queue size (default: 8) |
| `--model-stage-timing` | Report pre/run/post timing breakdown |
| `--no-display` | Headless mode for pure throughput testing |
| `--warmup N` | Frames to skip before collecting stats (default: 30) |
| `--timing-interval N` | Frames between timing reports (default: 120) |
| `--max-frames N` | Stop after N frames (0 = full video) |
| `--target-fps F` | Target FPS for late-frame and drop stats |
| `--max-width W` | Max processing width (default: 1920) |
| `--max-height H` | Max processing height (default: 1080) |
| `--static-input` | Force resize all frames to max-width × max-height |
| `--letterbox` | Preserve aspect ratio with black bars |

---

## Benchmarking

### Single run

```bash
python src/main.py --force-compile --no-display --warmup 30 --timing-interval 120 --model-stage-timing
```

### Matrix benchmark (auto CSV)

```bash
python benchmark_matrix.py
```

This runs the full matrix: precisions × compile modes × prefetch values, saving results to `benchmark_results_*.csv`.

Customize:

```bash
# Quick: just fp16, compile vs no-compile
python benchmark_matrix.py --precisions fp16 --prefetch-values 8 --max-frames 200

# Full matrix with target FPS stats
python benchmark_matrix.py --target-fps 24

# Custom output
python benchmark_matrix.py --output-csv thesis_benchmarks.csv
```

---

## Notes

- `torch.compile` requires a warmup compilation pass on the first run (cached by Triton afterwards).
- On ROCm-Windows, `--force-compile` is required because Triton ROCm codegen needs HIP SDK headers.
- `channels_last` and `cudnn.benchmark` are auto-enabled on NVIDIA but skipped on ROCm (MIOpen regression). Use `--channels-last` to test on ROCm.

---

## Citation

If this project is useful in your work, please cite the HDRTVNet/HDRTVNet++ papers:

```bibtex
@article{chen2023towards,
  title={Towards Efficient SDRTV-to-HDRTV by Learning from Image Formation},
  author={Chen, Xiangyu and Li, Zheyuan and Zhang, Zhengwen and Ren, Jimmy S and Liu, Yihao and He, Jingwen and Qiao, Yu and Zhou, Jiantao and Dong, Chao},
  journal={arXiv preprint arXiv:2309.04084},
  year={2023}
}
```

```bibtex
@InProceedings{chen2021hdrtvnet,
  author    = {Chen, Xiangyu and Zhang, Zhengwen and Ren, Jimmy S. and Tian, Lynhoo and Qiao, Yu and Dong, Chao},
  title     = {A New Journey From SDRTV to HDRTV},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2021},
  pages     = {4500-4509}
}
```

---

## License and Attribution

- This repository contains original real-time pipeline engineering and optimization work for thesis purposes.
- Model architecture and pretrained model lineage are based on HDRTVNet/HDRTVNet++ research code and publications.
- Please review upstream licenses/terms before redistributing pretrained weights or derived artifacts.
- Original HDRTVNet++ repository: `https://github.com/xiaom233/HDRTVNet-plus`

---

## Academic Context

This repository is the implementation component of an undergraduate thesis focused on real-time GPU-accelerated HDR reconstruction with precision-aware optimization.
