# HDR Real-Time Video Processing Framework

![Version](https://img.shields.io/badge/version-v2.0-blue)
![Status](https://img.shields.io/badge/status-stable-brightgreen)
![Thesis](https://img.shields.io/badge/type-academic%20research-green)

---

## Overview

**Real-Time SDR-to-HDR Video Reconstruction with HDRTVNet++ and PyTorch**

This project converts standard dynamic range (SDR) video to high dynamic range (HDR) in real time using a deep learning model (HDRTVNet++). It runs entirely on the GPU with `torch.compile` optimizations and includes a full-featured desktop GUI.

Windows-only project with **NVIDIA CUDA**, **AMD ROCm-Windows**, and **CPU** backends.

---

## Manual Downloads (Do This First)

Download required runtime files from the shared Google Drive assets folder:

`https://drive.google.com/drive/folders/1jh8gXBVzqRse-7w_2Dztca1_KVh5eRu1?usp=drive_link`

Place these files before first run:

1. `HG_weights.pth` -> `src/models/weights/HG_weights.pth`
2. `libmpv-2.dll` -> `src/libmpv-2.dll`

Startup behavior for clone users:
- The app now checks these files on launch.
- If either file is missing, a blocking dialog appears with:
  - Google Drive download button
  - exact placement instructions
  - Restart App button

Optional download:
- Install **GNU Octave** (and add to `PATH`) if you want HDR-VDP3 metrics.

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/DanHelmy/hdr-realtime-video-pipeline.git
cd hdr-realtime-video-pipeline

# 2. Double-click setup.bat
# (or run it in terminal)
.\setup.bat

# 3. Launch the GUI
.\run_gui.bat
```

Open a video and it plays in tabbed SDR/HDR views (with optional side-by-side tab).

---

## GUI (v2.0)

```bash
python src/gui.py
```

The GUI is the primary way to use the pipeline. It handles everything — kernel compilation, model loading, HDR display — automatically.

### New in v2.0

- **Modular GUI refactor**: `gui.py` now composes focused mixins/modules (`gui_ui_builder.py`, `gui_signal_wiring.py`, `gui_playback_runtime.py`, `gui_pipeline_worker_*.py`, etc.)
- **QAT precision modes restored** in GUI:
  - `INT8 Mixed (QAT)`
  - `INT8 Full (QAT)`
- **Compare precision refresh**: re-run frame-compare at the same anchor frame with a different precision directly from the compare dialog
- **Upscaler set expanded**: **EWA LanczosSharp**, **FSR**, and **SSimSuperRes**
- **A/V relock and mute policy polish** for seek/pause/pop-dock transitions and low-FPS recovery
- **Unified runtime cache root** for Triton/TorchInductor/HDR-VDP3 (`HDRTVNET_CACHE_DIR` override supported)

### Features

| Feature | Description |
|---|---|
| **Open any video** | Browse or drag-and-drop — playback starts automatically |
| **Modular codebase** | GUI and worker logic split into maintainable mixins/modules for easier iteration |
| **Tabbed views** | `SDR`, `HDR`, and `Side by Side` tabs |
| **Pop/Dock panes** | Detach SDR/HDR into separate windows and dock back |
| **Live precision switching** | FP16, FP32, INT8 PTQ/QAT variants — switch mid-playback |
| **HG toggle** | Enable/disable HG refinement (loads HG or no-HG INT8 weights) |
| **Playback controls** | Play / Pause / Resume / Stop |
| **Seek bar** | Drag to seek; when paused, seek is queued and applied on Resume for frame-accurate preview |
| **Performance metrics panel** | FPS, latency, frame count, app VRAM/CPU memory, model size, precision, processing resolution |
| **Compare metrics dialog** | Pauses playback and opens 3-way frame compare (SDR, HDR GT, HDR Convert) with PSNR, SSSIM, DeltaEITP, normalized variants, and optional HDR-VDP3 |
| **HDR metadata panel** | Color primaries, transfer function, peak luminance (nits), VO/GPU API |
| **Color handling** | SDR pane uses Rec.709 tagging; HDR pane uses BT.2020/PQ tagging; mpv auto-selects output mapping per display |
| **Automatic compilation** | Triton kernels compile in a clean subprocess; cached kernels load instantly |
| **Resolution + scaling** | Process at 1080p/720p/540p (or Source fallback) and scale to 1080p output using **EWA LanczosSharp**, **FSR**, or **SSimSuperRes** |
| **Film grain** | Optional film grain restoration (mpv shader) |
| **Audio support** | Auto-detect, attach external audio, and choose audio track |
| **Volume + stability policy** | Volume slider plus automatic mute below low FPS threshold, with fade-in restore on recovery |
| **Keyboard shortcuts** | `F11` borderless full-window, `Esc` exit borderless mode, `Space` pause/resume |
| **Cursor idle hide** | Optional auto-hide cursor during playback |
| **Pre-compile kernels** | Compile for any resolution/precision ahead of time |
| **Clear kernel cache** | Force recompilation (e.g. after PyTorch/driver update) |
| **Dark theme** | Modern dark UI, auto-applied |
| **Persistent GUI settings** | Saved in `.gui_prefs.json` (precision, resolution, view/tab, upscale, film grain, metrics visibility, HG toggle, volume, audio track, cursor hide, last-open directory) |

### GUI Launch Flags

`src/gui.py` also accepts startup flags (used by restart/apply flows):

```bash
python src/gui.py --video input.mp4 --resolution 720p --precision FP16 --view Tabbed --autoplay 1 --start-frame 1200 --use-hg 1 --film-grain 1 --hdr-gt hdr_reference.mkv
```

### Objective Metrics (PSNR / SSSIM / DeltaEITP / HDR-VDP3)

- Use **HDR GT ...** in the GUI, then click **Compare** to compute per-frame accuracy metrics.
- In `v2.0`, objective scoring is compare-driven by default (instead of a continuously updating runtime panel) for cleaner playback behavior.
- Ground-truth should be the same content/timing as the input clip for valid measurements.
- `HDR-VDP3` now has a built-in local bridge at `scripts/hdrvdp3_bridge.py`.
  - The GUI will use it automatically when `HDRTVNET_HDRVDP3_CMD` is not set.
  - First HDR-VDP3 run auto-downloads toolbox files to a user cache folder:
    - `%LOCALAPPDATA%\HDRTVNetCache\hdrvdp\`
  - Requires **GNU Octave** installed and available in `PATH`.
- You can still override with your own command using env var `HDRTVNET_HDRVDP3_CMD`.
  - Template placeholders: `{test}` / `{pred}` and `{reference}` / `{ref}`.

### Tools Menu

- **Pre-compile Kernels** — compile for any resolution(s) ahead of time
- **Clear Kernel Cache** — force recompilation (e.g. after a PyTorch / driver update)

### mpv Display / Color Path

Both SDR and HDR panes are rendered through embedded **mpv** (D3D11):

- **SDR pane**: tagged as **Rec.709** (`bt.709` / `bt.1886`, full range)
- **HDR pane**: tagged as **BT.2020/PQ** (`bt.2020` / `pq`, full range)
- Output target is **auto-detected by mpv/display path** (no hard-forced target primaries/TRC)

> **Requires** `libmpv-2.dll` in the `src/` folder.
> Download it from the shared Google Drive assets folder above (same folder as
> `HG_weights.pth`).  
> Fallback source: [mpv-winbuild](https://sourceforge.net/projects/mpv-player-windows/files/libmpv/)
> (`mpv-dev-x86_64-*-git-*.7z`).

### First Run

The first time you play a video at a given resolution, `torch.compile` with `max-autotune` needs to compile Triton kernels. This takes **2–5 minutes** and runs in a clean subprocess with a progress dialog. The compiled kernels are **cached to disk**, so subsequent runs at the same resolution load in **~5–10 seconds**.

---

## Installation

### Requirements

- Python 3.10+ (3.12 for AMD Windows)
- PyTorch 2.0+ (CUDA or ROCm build)
- OpenCV, NumPy

### Setup

```bash
# Auto-detect backend and install (double-clickable):
.\setup.bat

# Optional manual override:
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Backend nvidia
# or: -Backend amd
# or: -Backend cpu
```

Optional flags:
- `-RecreateVenv` to rebuild `venv` from scratch
- `-RunGui` to auto-launch after setup

### PyTorch GPU Backends

This repo now provides backend-specific requirement files under `requirements/`:

- `requirements/requirements-nvidia.txt` -> common deps + CUDA PyTorch (`cu124`)
- `requirements/requirements-amd.txt` -> common deps + ROCm-Windows SDK/PyTorch wheels (+ `triton-windows`)
- `requirements/requirements-common.txt` -> shared app deps only (use with manual CPU PyTorch install)

Equivalent setup scripts:
- `setup.bat` (double-click entry point)
- `scripts/setup.ps1` (auto-detect + override support)
- `run_gui.bat` (double-click GUI launcher)
- `scripts/setup_nvidia.ps1`
- `scripts/setup_amd.ps1`
- `scripts/setup_cpu.ps1`
- `scripts/run_gui.ps1`

**NVIDIA (CUDA):**
```bash
pip install -r requirements/requirements-nvidia.txt
```

**AMD ROCm-Windows (Python 3.12):**
```bash
pip install -r requirements/requirements-amd.txt
```

**CPU only:**
```bash
pip install -r requirements/requirements-common.txt
pip install torch torchvision
```

### torch.compile Setup

**NVIDIA CUDA**: Works automatically (Triton included with modern PyTorch).

**AMD ROCm-Windows**: Auto-detects HIP SDK. `requirements/requirements-amd.txt` already includes `triton-windows`.

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
| **INT8 Full (PTQ)** | W8A8 quantization (HG optional) | ~4.0× vs FP16+HG |
| **INT8 Full (QAT)** | W8A8 + quantization-aware fine-tuning (HG optional) | ~4.0× vs FP16+HG |
| **INT8 Mixed (PTQ)** | Mixed W8A8/W8A16 (HG optional) | ~4.0× vs FP16+HG |
| **INT8 Mixed (QAT)** | Mixed W8A8/W8A16 + quantization-aware fine-tuning (HG optional) | ~4.0× vs FP16+HG |

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
python src/main.py --precision int8-full --model src/models/weights/Ensemble_AGCM_LE_int8_full_qat.pt
python src/main.py --precision int8-mixed --model src/models/weights/Ensemble_AGCM_LE_int8_mixed_qat.pt
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

### HG Weights (Required Download)
`HG_weights.pth` is not included in this GitHub repo because it is too large for
normal GitHub tracking. For clone users, startup checks require this file.

Download it from the shared Google Drive assets folder:

`https://drive.google.com/drive/folders/1jh8gXBVzqRse-7w_2Dztca1_KVh5eRu1?usp=drive_link`

This folder also contains `libmpv-2.dll`.

Steps:
1. Open the Google Drive folder above.
2. Download `HG_weights.pth`.
3. Place it at:

`src/models/weights/HG_weights.pth`

You can also override the location in CLI mode:

```bash
python src/main.py --hg-weights "FULL/PATH/TO/HG_weights.pth"
```

### W8A8 (Full INT8, HG optional)
```bash
python scripts/quantize/quantize_int8_full.py
```
- Both weights and activations quantized to INT8
- Requires calibration data (default: `dataset/train_sdr/`)
  - **~4.0× compression vs FP16+HG** (HG adds significant weight size)
  - Outputs:
    - `src/models/weights/Ensemble_AGCM_LE_int8_full.pt` (HG)
    - `src/models/weights/Ensemble_AGCM_LE_int8_full_nohg.pt` (no-HG)

### W8A8 (Full INT8) + QAT (HG optional)
```bash
python scripts/quantize/quantize_int8_full_qat.py
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
python scripts/quantize/quantize_int8_mixed.py
```
- Memory-bound 1×1 convs → W8A8 (INT8 activations help bandwidth)
- Compute-bound 3×3 convs → W8A16 (weight-only saves storage)
  - **~4.0× compression vs FP16+HG**
  - Outputs:
    - `src/models/weights/Ensemble_AGCM_LE_int8_mixed.pt` (HG)
    - `src/models/weights/Ensemble_AGCM_LE_int8_mixed_nohg.pt` (no-HG)

### Mixed W8A8/W8A16 + QAT (Quantization-Aware Training, HG optional)
```bash
python scripts/quantize/quantize_int8_mixed_qat.py
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
