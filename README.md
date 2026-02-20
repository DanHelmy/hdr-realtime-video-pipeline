# HDR Real-Time Video Processing Framework

![Version](https://img.shields.io/badge/version-v0.8-blue)
![Status](https://img.shields.io/badge/status-active%20development-yellow)
![Thesis](https://img.shields.io/badge/type-academic%20research-green)

---

## Overview

This repository contains the implementation framework for an undergraduate thesis:

**Real-Time SDR-to-HDR Video Reconstruction with HDRTVNet++ and PyTorch**

The project achieves real-time HDR reconstruction using a fully GPU-accelerated PyTorch pipeline with `torch.compile` optimizations. Supports NVIDIA CUDA, AMD ROCm, and CPU backends.

---

## Current Status (v0.8)

### Implemented

- Full PyTorch inference pipeline (FP16, FP32, INT8)
- `torch.compile` with Triton — auto-enabled on all GPUs (auto-detects HIP SDK on ROCm-Windows)
- Cross-GPU support: NVIDIA (CUDA), AMD (ROCm), CPU
- INT8 full quantization (W8A8) — weights and activations
- INT8 mixed quantization — selective W8A8/W8A16 per layer
- Quantization-Aware Training (QAT) for mixed INT8 — fine-tunes against HDR ground truth
- GPU-side preprocessing (BGR→RGB, normalize, permute on GPU)
- GPU-side postprocessing (clamp, scale, quantize, RGB→BGR on GPU)
- Pre-allocated GPU tensor buffers (zero per-frame allocation)
- `torch.inference_mode()` throughout
- Async video prefetch queue (`--prefetch`)
- Stage timing breakdown (`pre`, `run`, `post`)
- Frame pacing stats (`fps`, `fps_1p_low`, `late`, `drop_est`)
- Benchmark matrix runner (`benchmark_matrix.py`)
- Optional CUDA graph replay (`--cuda-graphs`)
- Optional channels_last memory format (auto on NVIDIA)

### Pipeline

```
Video Source → GPU Upload → GPU Preprocess → torch.compile Model → GPU Postprocess → CPU Download → Renderer
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA or ROCm build)
- OpenCV, NumPy

### Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/macOS
pip install -r requirements.txt
```

### PyTorch GPU backends

**NVIDIA (CUDA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**AMD ROCm-Windows (Python 3.12):**
```cmd
pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz

pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl
```

**AMD ROCm-Linux:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

**CPU only:**
```bash
pip install torch torchvision
```

### torch.compile setup

**NVIDIA CUDA**: Works automatically when Triton is installed (included with modern PyTorch).

**AMD ROCm-Windows**: Auto-detects HIP SDK and enables compile automatically. Install `triton-windows` and ensure HIP SDK headers are available:
```bash
pip install triton-windows
xcopy /E /I "C:\Program Files\AMD\ROCm\7.1\include\hip" "venv\Lib\site-packages\_rocm_sdk_devel\include\hip"
```
If auto-detection fails, use `--force-compile` to enable manually.

**AMD ROCm-Linux**: Works automatically like NVIDIA.

---

## Running

Activate the virtual environment first (or use `.\venv\Scripts\python.exe` directly):

```bash
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/macOS
```

Default (FP16, max-autotune compile, auto device):

```bash
python src/main.py
```

Skip compile entirely:

```bash
python src/main.py --no-compile
```

INT8 quantized model:

```bash
# W8A8 (full INT8):
python src/main.py --precision int8-full --model src/models/weights/Ensemble_AGCM_LE_int8_full.pt

# Mixed INT8:
python src/main.py --precision int8-mixed --model src/models/weights/Ensemble_AGCM_LE_int8_mixed.pt

# Mixed INT8 (QAT fine-tuned):
python src/main.py --precision int8-mixed --model src/models/weights/Ensemble_AGCM_LE_int8_mixed_qat.pt
```

Headless benchmark:

```bash
python src/main.py --no-display --warmup 30 --timing-interval 120 --max-frames 360 --model-stage-timing
```

### CLI Flags

| Flag | Description |
|---|---|
| `--model PATH` | Model weights path (default: `src/models/weights/Ensemble_AGCM_LE.pth`) |
| `--device auto\|cuda\|cpu` | Device selection (default: auto) |
| `--precision auto\|fp16\|fp32\|int8-full\|int8-mixed` | Inference precision (default: auto → fp16 on GPU) |
| `--compile-mode auto\|default\|reduce-overhead\|max-autotune` | torch.compile mode (auto = max-autotune on all GPUs) |
| `--force-compile` | Force `torch.compile` on ROCm-Windows when HIP SDK auto-detection fails |
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

## INT8 Quantization

Two INT8 quantization modes are provided for model compression and (on supported hardware) inference acceleration:

### W8A8 (Full INT8)  
```bash
python quantize_int8_full.py
```
- Both weights and activations quantized to INT8
- Requires calibration data (uses `dataset/test_sdr/`)
- **3.11× compression**

### Mixed W8A8/W8A16
```bash
python quantize_int8_mixed.py
```
- Memory-bound 1×1 convs → W8A8 (INT8 activations help bandwidth)
- Compute-bound 3×3 convs → W8A16 (weight-only saves storage)
- **3.17× compression**

### Mixed W8A8/W8A16 + QAT (Quantization-Aware Training)
```bash
python quantize_int8_mixed_qat.py
```
- Starts from the PTQ mixed checkpoint and fine-tunes with fake quantization + STE
- Learnable weight/activation scales adapt to minimize reconstruction loss against HDR ground truth
- Trains on SDR/HDR pairs from `dataset/` (256×256 random crops, L1 loss)
- Output is fully compatible with `--precision int8-mixed` (same checkpoint format)
- **3.17× compression**
- Customizable: `--epochs 10 --lr 1e-5` or `--from-scratch` (no PTQ checkpoint needed)

---

## Benchmarking

### Single run

```bash
python src/main.py --no-display --warmup 30 --timing-interval 120 --model-stage-timing
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

## Platform Notes

| Feature | NVIDIA (CUDA) | AMD (ROCm) | CPU |
|---|---|---|---|
| torch.compile | Auto | `--force-compile` (Windows) | Not supported |
| channels_last | Auto | `--channels-last` to test | N/A |
| cudnn.benchmark | Auto | N/A (MIOpen) | N/A |
| FP16 inference | ✅ | ✅ | Fallback to FP32 |
| INT8 quantization | ✅ (tensor core acceleration possible) | ✅ (compression only, no INT8 compute path) | ✅ (compression only) |
| CUDA graphs | ✅ | ✅ | N/A |

### Compile time

`torch.compile` adds a one-time startup cost (cached by Triton on subsequent runs with the same resolution):

| Mode | Typical compile time | Best for |
|---|---|---|
| `default` | 30-60 seconds | Short clips, development |
| `max-autotune` | 2-5 minutes | Long videos, production benchmarks |

For a 2-hour movie (~172k frames), compile overhead is <1% of total processing time.

---

## Notes

- On ROCm-Windows, `--force-compile` is required because Triton codegen needs HIP SDK headers.
- `channels_last` and `cudnn.benchmark` are auto-enabled on NVIDIA but skipped on ROCm (MIOpen regression).
- INT8 inference speedup depends on hardware: NVIDIA tensor cores support native INT8 compute; AMD consumer GPUs (RDNA3) currently lack INT8 conv kernel support in MIOpen, so INT8 provides compression benefits only.

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
