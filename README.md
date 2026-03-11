# HDR Real-Time Video Processing Framework

![Version](https://img.shields.io/badge/version-v1.2-blue)
![Status](https://img.shields.io/badge/status-stable-brightgreen)
![Thesis](https://img.shields.io/badge/type-academic%20research-green)

---

## Overview

**Real-Time SDR-to-HDR Video Reconstruction with HDRTVNet++ and PyTorch**

This project converts standard dynamic range (SDR) video to high dynamic range (HDR) in real time using a deep learning model (HDRTVNet++). It runs entirely on the GPU with `torch.compile` optimizations and includes a full-featured desktop GUI.

Supports **NVIDIA CUDA**, **AMD ROCm**, and **CPU** backends.

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/DanHelmy/hdr-realtime-video-pipeline.git
cd hdr-realtime-video-pipeline
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Install PyTorch for your GPU (see "PyTorch GPU Backends" below)

# 3. Launch the GUI
python src/gui.py
```

Open a video and it plays in tabbed SDR/HDR views (with optional side-by-side tab).

---

## GUI (v1.2)

```bash
python src/gui.py
```

The GUI is the primary way to use the pipeline. It handles everything — kernel compilation, model loading, HDR display — automatically.

### New in v1.2

- **Film grain toggle** (mpv shader, auto-download; restores grain lost to upscalers)
- **Upscale options simplified**: default **EWA LanczosSharp** (EWA Lanczos removed), plus **FSR**
- **Film grain + upscale settings persist** in `.gui_prefs.json`
- **A/V relock polish** for UI transitions and pop/dock workflows

**Note:** QAT variants have been temporarily removed until the next update. All weights have been retrained (including the **+HG** option).

### Features

| Feature | Description |
|---|---|
| **Open any video** | Browse or drag-and-drop — playback starts automatically |
| **Tabbed views** | `SDR`, `HDR`, and `Side by Side` tabs |
| **Pop/Dock panes** | Detach SDR/HDR into separate windows and dock back |
| **Live precision switching** | FP16, FP32, INT8 variants — switch mid-playback |
| **HG toggle** | Enable/disable HG refinement (loads HG or no-HG INT8 weights) |
| **Playback controls** | Play / Pause / Resume / Stop |
| **Seek bar** | Drag to seek; when paused, seek is queued and applied on Resume for frame-accurate preview |
| **Live metrics** | FPS, latency, frame count, app VRAM/CPU memory, model size |
| **HDR metadata panel** | Color primaries, transfer function, peak luminance (nits), VO/GPU API |
| **Color handling** | SDR pane uses Rec.709 tagging; HDR pane uses BT.2020/PQ tagging; mpv auto-selects output mapping per display |
| **Automatic compilation** | Triton kernels compile in a clean subprocess; cached kernels load instantly |
| **Resolution + scaling** | Process at 1080p/720p/540p (or Source fallback) and scale to 1080p output using **EWA LanczosSharp** or **FSR** |
| **Film grain** | Optional film grain restoration (mpv shader) |
| **Audio support** | Auto-detect, attach external audio, and choose audio track |
| **Volume + stability policy** | Volume slider plus automatic mute below low FPS threshold, with fade-in restore on recovery |
| **Keyboard shortcuts** | `F11` borderless full-window, `Esc` exit borderless mode, `Space` pause/resume |
| **Cursor idle hide** | Optional auto-hide cursor during playback |
| **Pre-compile kernels** | Compile for any resolution/precision ahead of time |
| **Clear kernel cache** | Force recompilation (e.g. after PyTorch/driver update) |
| **Dark theme** | Modern dark UI, auto-applied |
| **Persistent GUI settings** | Saved in `.gui_prefs.json` (precision, resolution, upscale, film grain, metrics, volume, audio, cursor) |

### GUI Launch Flags

`src/gui.py` also accepts startup flags (used by restart/apply flows):

```bash
python src/gui.py --video input.mp4 --resolution 720p --precision FP16 --view Tabbed --autoplay 1 --start-frame 1200 --use-hg 1 --film-grain 1
```

### Tools Menu

- **Pre-compile Kernels** — compile for any resolution(s) ahead of time
- **Clear Kernel Cache** — force recompilation (e.g. after a PyTorch / driver update)

### mpv Display / Color Path

Both SDR and HDR panes are rendered through embedded **mpv** (D3D11):

- **SDR pane**: tagged as **Rec.709** (`bt.709` / `bt.1886`, full range)
- **HDR pane**: tagged as **BT.2020/PQ** (`bt.2020` / `pq`, full range)
- Output target is **auto-detected by mpv/display path** (no hard-forced target primaries/TRC)

> **Requires** `libmpv-2.dll` in the `src/` folder.
> Download from [mpv-winbuild](https://sourceforge.net/projects/mpv-player-windows/files/libmpv/)
> (the `mpv-dev-x86_64-*-git-*.7z` archive). If the DLL is missing, the GUI
> falls back to a standard QLabel preview.

### First Run

The first time you play a video at a given resolution, `torch.compile` with `max-autotune` needs to compile Triton kernels. This takes **2–5 minutes** and runs in a clean subprocess with a progress dialog. The compiled kernels are **cached to disk**, so subsequent runs at the same resolution load in **~5–10 seconds**.

All 1080p videos reuse the same cached kernels regardless of content, codec, or duration. A different resolution (different aspect ratio) triggers a one-time recompile.

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

### PyTorch GPU Backends

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

### torch.compile Setup

**NVIDIA CUDA**: Works automatically (Triton included with modern PyTorch).

**AMD ROCm-Windows**: Auto-detects HIP SDK. Install `triton-windows` and copy HIP headers:
```bash
pip install triton-windows
xcopy /E /I "C:\Program Files\AMD\ROCm\7.1\include\hip" "venv\Lib\site-packages\_rocm_sdk_devel\include\hip"
```
If auto-detection fails, use `--force-compile` in CLI mode.

**AMD ROCm-Linux**: Works automatically like NVIDIA.

---

## Architecture

### Pipeline

```
Video Source → GPU Upload → GPU Preprocess → torch.compile Model → GPU Postprocess → CPU Download → Renderer
```

### Key Optimizations

- GPU-side preprocessing (BGR→RGB, normalize, permute) and postprocessing (clamp, scale, quantize, RGB→BGR)
- Pre-allocated GPU tensor buffers (zero per-frame allocation)
- Pinned (page-locked) host memory for async H2D/D2H DMA transfers
- `torch.inference_mode()` throughout
- Async video prefetch queue
- mpv fast path: skips GPU→CPU postprocess when mpv handles HDR display

### Precision Modes

| Mode | Description | Compression |
|---|---|---|
| **FP16** | Half-precision (default on GPU) | — |
| **FP32** | Full precision | — |
| **INT8 Full** | W8A8 quantization (HG optional) | ~4.0× vs FP16+HG |
| **INT8 Mixed** | Mixed W8A8/W8A16 (HG optional) | ~4.0× vs FP16+HG |

**Note:** QAT variants are temporarily removed in v1.2 (weights retrained, including **+HG**). They will return in a future update.

INT8 modes include **pre-dequantization** for GPUs without INT8 tensor cores (AMD RDNA3, NVIDIA pre-Turing): INT8 weights are converted to FP16 once at load time, giving native FP16 speed with compressed checkpoint storage.

---

## CLI Mode

For headless benchmarking or scripted workflows:

```bash
# Default (FP16, max-autotune compile, auto device)
python src/main.py

# Skip compile
python src/main.py --no-compile

# FP32 precision
python src/main.py --precision fp32

# INT8 quantized models (HG on/off)
python src/main.py --precision int8-full --model src/models/weights/Ensemble_AGCM_LE_int8_full.pt
python src/main.py --precision int8-mixed --model src/models/weights/Ensemble_AGCM_LE_int8_mixed.pt
python src/main.py --precision int8-full --model src/models/weights/Ensemble_AGCM_LE_int8_full_nohg.pt --use-hg 0
python src/main.py --precision int8-mixed --model src/models/weights/Ensemble_AGCM_LE_int8_mixed_nohg.pt --use-hg 0

# Headless benchmark
python src/main.py --no-display --warmup 30 --timing-interval 120 --max-frames 360 --model-stage-timing
```

<details>
<summary><strong>All CLI Flags</strong></summary>

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
| `--predequantize auto\|on\|off` | Pre-dequantize INT8 weights to FP16 at load time (auto = enabled on GPUs without INT8 tensor cores) |
| `--use-hg 1\|0` | Enable HG refinement (1 = on, 0 = off) |
| `--cache-resolution WxH` | Pre-compile Triton kernels for this resolution at startup (default: auto = video resolution) |
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

</details>

---

## INT8 Quantization

<details>
<summary><strong>Quantization details</strong></summary>

**QAT note:** QAT variants are temporarily removed in v1.2 (weights retrained, including **+HG**). The QAT subsections below are kept for reference and will return in a future update.

### HG Weights (Optional)
HG refinement is optional. If you enable HG (`--use-hg 1`), place the pretrained HG
weights at:

`Source Pipeline/pretrained_models/HG_weights.pth`

You can override with `--hg-weights` in CLI mode.

### W8A8 (Full INT8, HG optional)
```bash
python quantize_int8_full.py
```
- Both weights and activations quantized to INT8
- Requires calibration data (uses `dataset/test_sdr/`)
  - **~4.0× compression vs FP16+HG** (HG adds significant weight size)
  - Outputs:
    - `src/models/weights/Ensemble_AGCM_LE_int8_full.pt` (HG)
    - `src/models/weights/Ensemble_AGCM_LE_int8_full_nohg.pt` (no-HG)

### W8A8 (Full INT8) + QAT (HG optional) — *temporarily removed in v1.2*
```bash
python quantize_int8_full_qat.py
```
- Starts from the PTQ full checkpoint and fine-tunes with fake quantization + STE
- Learnable weight/activation scales adapt to minimize reconstruction loss against HDR ground truth
- Trains on SDR/HDR pairs from `dataset/` (256×256 random crops, L1 loss)
  - **~4.0× compression vs FP16+HG**
  - Outputs:
    - `src/models/weights/Ensemble_AGCM_LE_int8_full_qat.pt` (HG)
    - `src/models/weights/Ensemble_AGCM_LE_int8_full_qat_nohg.pt` (no-HG)
- Customizable: `--epochs 10 --lr 1e-5` or `--from-scratch`

### Mixed W8A8/W8A16 (HG optional)
```bash
python quantize_int8_mixed.py
```
- Memory-bound 1×1 convs → W8A8 (INT8 activations help bandwidth)
- Compute-bound 3×3 convs → W8A16 (weight-only saves storage)
  - **~4.0× compression vs FP16+HG**
  - Outputs:
    - `src/models/weights/Ensemble_AGCM_LE_int8_mixed.pt` (HG)
    - `src/models/weights/Ensemble_AGCM_LE_int8_mixed_nohg.pt` (no-HG)

### Mixed W8A8/W8A16 + QAT (Quantization-Aware Training, HG optional) — *temporarily removed in v1.2*
```bash
python quantize_int8_mixed_qat.py
```
- Starts from the PTQ mixed checkpoint and fine-tunes with fake quantization + STE
- Learnable weight/activation scales adapt to minimize reconstruction loss against HDR ground truth
- Trains on SDR/HDR pairs from `dataset/` (256×256 random crops, L1 loss)
  - **~4.0× compression vs FP16+HG**
  - Outputs:
    - `src/models/weights/Ensemble_AGCM_LE_int8_mixed_qat.pt` (HG)
    - `src/models/weights/Ensemble_AGCM_LE_int8_mixed_qat_nohg.pt` (no-HG)
- Customizable: `--epochs 10 --lr 1e-5` or `--from-scratch`

### Pre-Dequantization

On GPUs without native INT8 compute, per-inference dequantization adds ~55% overhead. Pre-dequantization converts INT8 weights to FP16 **once at load time**:

```bash
python src/main.py --precision int8-mixed --model src/models/weights/Ensemble_AGCM_LE_int8_mixed.pt --predequantize on
```

**Result:** Same FP16 speed + 2.94× compressed storage.

</details>

---

## Platform Notes

| Feature | NVIDIA (CUDA) | AMD (ROCm) | CPU |
|---|---|---|---|
| torch.compile | Auto | Auto (Windows: needs HIP SDK) | Not supported |
| FP16 inference | ✅ | ✅ | Fallback to FP32 |
| INT8 quantization | ✅ (tensor cores) | ✅ (compression only) | ✅ (compression only) |
| CUDA graphs | ✅ | ✅ | N/A |
| channels_last | Auto | `--channels-last` | N/A |

### Compile Cache

| Scenario | Time |
|---|---|
| First run at a resolution | 2–5 minutes |
| Cached resolution | ~5–10 seconds |
| Different resolution | 2–5 minutes (one-time) |

You can also pre-compile manually:
```bash
python src/compile_kernels.py 1920x1080
python src/compile_kernels.py --clear-cache 1920x1080
```

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

- Original real-time pipeline engineering and optimization work for thesis purposes.
- Model architecture and pretrained model lineage based on HDRTVNet/HDRTVNet++ research code.
- Please review upstream licenses before redistributing pretrained weights.
- Original HDRTVNet++ repository: [github.com/xiaom233/HDRTVNet-plus](https://github.com/xiaom233/HDRTVNet-plus)

---

## Academic Context

This repository is the implementation component of an undergraduate thesis focused on real-time GPU-accelerated HDR reconstruction with precision-aware optimization.
