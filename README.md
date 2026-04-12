# HDR Real-Time Video Processing Framework

![Version](https://img.shields.io/badge/version-v4.0-blue)
![Status](https://img.shields.io/badge/status-active%20development-brightgreen)
![Focus](https://img.shields.io/badge/focus-GUI%20Grand%20Update-orange)
![Type](https://img.shields.io/badge/type-academic%20research-green)

---

## Overview

**Real-Time SDR-to-HDR Video Reconstruction with HDRTVNet++ and PyTorch**

This project converts SDR video to HDR in real time with a low-latency desktop app designed for thesis-grade experimentation, live playback, and export workflows.

`v4.0` is the **Grand UI Update** release: a clearer interface narrative, stronger user flows, and documentation that is now organized around how users actually operate the app.

Windows-only project with support for:
- **NVIDIA CUDA**
- **AMD ROCm-Windows**
- **CPU fallback**

---

## What Is New In v4.0 (Grand UI Update)

- Reframed UI documentation around real usage: **Open -> Play -> Compare -> Export**
- Added a visual-first README structure with dedicated screenshot slots
- Consolidated core feature descriptions to reduce duplicate information
- Kept Browser Window Capture and Chrome Audio Sync guidance explicit and safer to follow
- Preserved advanced controls (precision switching, HG toggle, runtime mode, pre-dequantization, kernel tools) with clearer context

---

## Quick Start

```bash
# 1. Clone and enter repo
git clone https://github.com/DanHelmy/hdr-realtime-video-pipeline.git
cd hdr-realtime-video-pipeline

# 2. Setup environment + dependencies
.\setup.bat

# 3. Launch app
.\run_gui.bat
```

If assets are missing, setup and first launch attempt to auto-download required files (`libmpv-2.dll`, `HG_weights.pth`).

Manual fallback assets:

`https://drive.google.com/drive/folders/1jh8gXBVzqRse-7w_2Dztca1_KVh5eRu1?usp=drive_link`

---

## UI Tour (v4.0)

### 1. Main Workspace

- Tabbed views: `SDR`, `HDR`, `Side by Side`
- Playback controls: play/pause/resume/stop, seek, frame pipeline metrics
- Live precision/runtime controls with minimal interruption

![TODO-main-ui-overview](docs/images/v4-main-ui-overview.png)

> Replace with a full-window screenshot showing tabs, transport controls, and metrics panel.

### 2. Browser Window Capture (Experimental)

- Direct visible-window Chrome capture for video path
- Extension provides delayed local Chrome audio sync
- Dynamic frame-following and adaptive paced presentation

![TODO-browser-capture-flow](docs/images/v4-browser-capture-flow.png)

> Replace with screenshot showing source selection on Browser Window Capture and active playback.

### 3. Compare / Objective Metrics Dialog

- 3-way comparison: SDR input, HDR reference, HDR output
- Metrics: PSNR, SSSIM, DeltaEITP, normalized variants, optional HDR-VDP3

![TODO-compare-dialog](docs/images/v4-compare-dialog.png)

> Replace with screenshot of compare window including metric values and frame preview panes.

### 4. Export Dialog

- Separate export pipeline settings
- Native source resolution/FPS defaults
- ProRes 422 HQ + PCM output path
- Advanced controls for compile reuse and INT8 pre-dequantization override

![TODO-export-dialog](docs/images/v4-export-dialog.png)

> Replace with screenshot of export dialog (basic + advanced tabs).

---

## Screenshot Plan (What To Capture)

Use this checklist to add polished visuals quickly:

1. **Hero screenshot**
- Full app window while processing a visually rich SDR scene
- Include tabs + metrics panel + active controls
- Suggested filename: `docs/images/v4-main-ui-overview.png`

2. **Browser capture setup screenshot**
- Source mode set to Browser Window Capture (Experimental)
- Chrome window visible in frame, extension active
- Suggested filename: `docs/images/v4-browser-capture-flow.png`

3. **Compare dialog screenshot**
- Show all three frames and metrics table
- Use a frame with obvious highlights/colors to show HDR reconstruction differences
- Suggested filename: `docs/images/v4-compare-dialog.png`

4. **Export workflow screenshot**
- Export dialog with resolution/FPS + model/precision settings visible
- Optional second image for Advanced tab if you want deeper docs
- Suggested filename: `docs/images/v4-export-dialog.png`

5. **Tools menu screenshot**
- Capture open Tools menu showing pre-dequantization and kernel actions
- Suggested filename: `docs/images/v4-tools-menu.png`

Optional bonus visuals:
- Before/after SDR vs HDR still frame comparison strip
- Performance panel close-up during steady playback
- Kernel compile progress dialog screenshot (first-run compile)

---

## Browser Audio Sync Extension (Chrome)

The extension is bundled in this repo and is now audio-only.

Extension folder:

`browser_tab_capture_extension/`

Recommended flow:

1. Run app (`run_gui.bat` or `python src\gui.py`)
2. In Chrome: `Settings > System`
3. Disable `Use graphics acceleration when available`
4. Restart Chrome
5. Open `chrome://extensions`
6. Enable `Developer mode`
7. Click `Load unpacked`
8. Select `browser_tab_capture_extension`
9. In app, choose `Browser Window Capture (Experimental)`
10. In Chrome, open target tab and click extension `Start Chrome Audio Sync`
11. In app, pick the matching visible Chrome window
12. Tune delay slider for lip sync
13. Stop sync later from extension popup

Important notes:
- Experimental feature
- Chrome only (for synced browser-window playback)
- Hardware acceleration in Chrome must be off for this path
- App stays silent during this mode while Chrome replays delayed tab audio

---

## Core GUI Features

| Feature | Description |
|---|---|
| Open any video | Browse or drag-and-drop; playback starts automatically |
| Browser Window Capture (Experimental) | Direct visible-Chrome video capture + bundled Chrome audio sync extension |
| Tabbed views | `SDR`, `HDR`, `Side by Side` |
| Pop/Dock panes | Undock SDR/HDR panes and dock them back |
| Live precision switching | FP16, FP32, INT8 PTQ/QAT variants |
| HG toggle | Enable/disable HG refinement |
| Performance metrics panel | FPS, model-stage latency, frame count, app VRAM/CPU memory, model size, precision, processing resolution |
| Compare dialog | Frame-wise objective metrics and visual inspection |
| Resolution + scaling | Source/1080p/720p/540p with EWA LanczosSharp, FSR, or SSimSuperRes |
| Film grain | Optional restoration shader |
| Export workflow | Separate dialog with independent model/precision settings |
| Persistent settings | Stored in `.gui_prefs.json` |

---

## GUI Launch

```bash
python src/gui.py
```

Startup flags example:

```bash
python src/gui.py --video input.mp4 --resolution 720p --precision FP16 --view Tabbed --autoplay 1 --start-frame 1200 --use-hg 1 --film-grain 1 --hdr-gt hdr_reference.mkv
```

---

## Objective Metrics (PSNR / SSSIM / DeltaEITP / HDR-VDP3)

- Use **HDR GT ...** in the GUI, then click **Compare** for per-frame objective scoring
- Ground-truth clip should match source timing/content
- `scripts/hdrvdp3_bridge.py` is used automatically when `HDRTVNET_HDRVDP3_CMD` is not set
- GNU Octave is required for HDR-VDP3 runs

Toolbox download/cache behavior:
- Preferred location: `third_party/hdrvdp/`
- Fallback: `%LOCALAPPDATA%\HDRTVNetCache\hdrvdp\`

---

## Export (Production-Oriented)

- Open **File -> Export Video...**
- Export defaults to source-native resolution/FPS
- Aspect ratio can be locked; mismatches are fit with padding
- Supports all available presets (`FP16`, `FP32`, `INT8 PTQ/QAT`, HG on/off)
- Output is intentionally limited to **ProRes 422 HQ (`.mov`) + PCM audio**
- Tagged through HDR path as BT.2020 / PQ with `1001 nit` expectation
- HDR input sources are rejected in export selection

---

## Installation

### Requirements

- Python 3.12
- PyTorch 2.0+
- OpenCV, NumPy

### Setup

```bash
# Auto-detect backend and install
.\setup.bat

# Optional override
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Backend nvidia
# or: -Backend amd
# or: -Backend cpu
```

Optional setup flags:
- `-RecreateVenv`
- `-RunGui`

### Backend Requirement Files

- `requirements/requirements-nvidia.txt` (CUDA path)
- `requirements/requirements-amd.txt` (ROCm-Windows path)
- `requirements/requirements-common.txt` (shared dependencies)

---

## CLI Mode

```bash
# Default (FP16, compile enabled)
python src/main.py

# No compile
python src/main.py --no-compile

# FP32
python src/main.py --precision fp32

# INT8 examples
python src/main.py --precision int8-full --model src/models/weights/Ensemble_AGCM_LE_int8_full.pt
python src/main.py --precision int8-mixed --model src/models/weights/Ensemble_AGCM_LE_int8_mixed.pt

# Headless benchmark
python src/main.py --no-display --warmup 30 --timing-interval 120 --max-frames 360 --model-stage-timing
```

---

## First Run And Compile Cache

First run at a new resolution/precision/HG/pre-dequantize combination may trigger `torch.compile` + Triton kernel build.

Typical timing:
- First compile: `2-5 minutes`
- Cache hit startup: `~5-10 seconds`

Caches are project-scoped to avoid collisions across separate local clones.

Manual compile tools:

```bash
python src/compile_kernels.py 1920x1080
python src/compile_kernels.py --clear-cache 1920x1080
```

---

## Platform Notes

| Feature | NVIDIA (CUDA) | AMD (ROCm) | CPU |
|---|---|---|---|
| torch.compile | Auto | Auto (HIP SDK needed on Windows) | Not supported |
| FP16 inference | Yes | Yes | Falls back to FP32 |
| INT8 quantization | Yes (tensor cores) | Yes (compression-focused) | Yes (compression-focused) |
| channels_last | Auto | Optional (`--channels-last`) | N/A |

---

## INT8 Quantization Scripts

```bash
python scripts/quantize/quantize_int8_full.py
python scripts/quantize/quantize_int8_full_qat.py
python scripts/quantize/quantize_int8_mixed.py
python scripts/quantize/quantize_int8_mixed_qat.py
```

Notes:
- `HG_weights.pth` is downloaded automatically when missing (or place manually under `src/models/weights/HG_weights.pth`)
- Pre-dequantization is recommended on GPUs without native INT8 tensor cores

---

## Architecture

```text
Video Source -> GPU Upload -> GPU Preprocess -> torch.compile Model -> GPU Postprocess -> CPU Download -> Renderer
```

Key optimizations:
- GPU-side pre/post processing
- Pre-allocated GPU buffers
- Pinned host memory for DMA transfers
- `torch.inference_mode()` runtime
- Async prefetch queue

---

## Citation

If this project is useful in your work, please cite HDRTVNet/HDRTVNet++:

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

## License And Attribution

- Real-time pipeline engineering and optimization for thesis work
- Model lineage based on HDRTVNet/HDRTVNet++ research code
- Review upstream licenses before redistributing pretrained weights
- Upstream repository: [github.com/xiaom233/HDRTVNet-plus](https://github.com/xiaom233/HDRTVNet-plus)

---

## Academic Context

This repository is the implementation component of an undergraduate thesis focused on real-time GPU-accelerated HDR reconstruction with precision-aware optimization.
