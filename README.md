# HDR Real-Time Video Processing Framework

![Version](https://img.shields.io/badge/version-v6.0-blue)
![Stable](https://img.shields.io/badge/stable-v6.0-brightgreen)
![Status](https://img.shields.io/badge/status-stable%20release-brightgreen)
![Thesis](https://img.shields.io/badge/type-academic%20research-green)

---

## Overview

**Real-Time SDR-to-HDR Video Reconstruction with HDRTVNet++ using TensorRT on NVIDIA and PyTorch on AMD**

This project converts standard dynamic range (SDR) video to high dynamic range (HDR) in real time using HDRTVNet++ and a desktop GUI built around low-latency playback, backend-specific inference, export, and live browser-window viewing.

`main` currently tracks the **v6.0 stable line**, including a hybrid inference runtime where NVIDIA devices use TensorRT engines and AMD devices keep the PyTorch pipeline with `torch.compile`/Triton optimization.

Latest stable release: **v6.0**.

For normal use, download the latest stable build from GitHub **Releases** instead of cloning `main`.

Core updates include:

- NVIDIA TensorRT backend for playback, export, benchmark, preview, and CLI inference
- on-demand TensorRT engine creation when a model/resolution/mode is activated
- cached TensorRT engines under `src/models/engines/` using `{model}_{resolution}_{mode}.engine`
- no NVIDIA PyTorch fallback: TensorRT build/load errors are logged and shown to the user without crashing the app
- PyTorch-specific tuning controls are hidden on NVIDIA
- AMD keeps the PyTorch runtime, with contiguous tensors by default and opt-in channels-last testing
- AMD benchmark runs use cached `max-autotune` PyTorch kernels when the exact compile cache is already present, otherwise they stay eager to avoid surprise compile stalls
- AMD INT8 `Auto` pre-dequantization behavior is preserved
- new `Model Quality Benchmark` tool in `Tools` for both single-video and dataset benchmarking, including queued multi-run benchmark batches
- deterministic frame/pair selection options with eager-mode quality runs for repeatable objective comparisons
- objective metric domains are now explicit and shared across compare and benchmark:
  - `PSNR` / `SSIM` run on linear HDR frames
  - `DeltaEITP` / `HDR-VDP3` run on a BT.2020/PQ color-managed path
- display-side mpv tone mapping now uses `tone-mapping=spline` with `tone-mapping-param=0.45`, shared across playback, compare, and benchmark previews; it does not affect objective metrics or exported HDR content
- pane-aware playback upscaling: mpv upscale quality follows the actual HDR pane size, so normal windowed, side-by-side, fullscreen-with-UI, and hide-UI fullscreen layouts only upscale when the pane is larger than the processed frame
- FSR playback now uses an RGB `MAIN` mpv shader for the app's RGB48 feed, with a high-quality residual mpv scaler instead of bilinear fallback when FSR's EASU pass does not cover the full target
- `SSimDownscaler.glsl` is loaded automatically as the mpv downscale shader companion when present (`HDRTVNET_MPV_SSIM_DOWNSCALER=0` disables it)
- SDR panes now use SDR-specific downscaling (`mitchell`, non-linear) instead of the HDR linear-light downscale path, reducing glow around tiny SDR text in side-by-side/windowed layouts
- SDR preview panes feed mpv as BGR24 instead of RGB48 when HDR metadata is not forced, cutting SDR-only/side-by-side pipe bandwidth and conversion cost
- normal video playback now uses a fixed small mpv/worker frame buffer before startup, seek, and resume release, so audio and both panes restart together instead of relying on a mute-until-stable gate
- timeline scrubbing uses a paused mpv file preview for exact seek thumbnails while the main HDR/SDR raw-video pipes stay isolated
- the status bar now keeps the live pane upscale state visible during playback instead of only reporting it on monitor moves; audio recovery notices yield to that persistent scaling state
- live HDR playback and export use raw model output; the old high-highlight debanding and sky/temporal cleanup bandaids are no longer part of the app/export path
- built-in `HDR-VDP3` bridge now converts BT.2100 PQ inputs back to absolute display luminance before scoring
- benchmark result viewer with SDR/HDR GT/HDR Convert previews, run metadata, and summary reloading
- benchmark session hierarchy (`source_name/timestamp__precision__resolution__n<count>/...`) plus exportable metrics and sample images
- benchmark queue can add the current setup once or add every visible precision preset in one click, making FP32/FP16/INT8 sweeps less tedious
- deterministic video frame detection now uses FFmpeg keyframe timestamps plus bounded low-resolution preview scoring when available, returns large requested pools from the keyframe timeline without decoding every candidate, caches the scored pool, and avoids repeated OpenCV random-seek QC passes
- compare, objective metrics, and benchmark frame previews share a guarded FFmpeg SDR frame fast-seek path with OpenCV fallback for requested noncurrent frames
- mpv-preview thesis figure renderer for benchmark PNG/TIFF frames, using the same embedded libmpv display path instead of FFmpeg tone-map approximations
- benchmark interaction lock so playback controls (and compare) are frozen while benchmarking is open
- first-run GUI defaults are tuned for the balanced NVIDIA path: `INT8 Mixed (QAT)`, `1080p`, `SSimSuperRes`, and HG off
- NVIDIA runtime now defaults to quant-friendly TensorRT ModelOpt Torch builds from self-describing HR/ACGM/LE/HG source checkpoints; mixed presets use the proven runtime include mask, while Full INT8 forces all ModelOpt Torch quantizers
- normal `FP16` / `FP32` now use the quant-friendly distilled HR/HG architecture; untouched original HR/HG checkpoints remain reference assets for manual experiments
- current RTX 5060 Ti 4K TensorRT checks include distilled FP16 HG ~20.06 ms/frame, INT8 Mixed QAT Film no-HG ~13.57 ms/frame, INT8 Mixed QAT Film HG ~15.74 ms/frame; Full INT8 QAT Film verifies `74/74` no-HG and `104/104` HG quantizers
- QAT INT8 checkpoints now ship in two included families: FP32-anchored tone-protected `QAT` and movie-accuracy `QAT (Film)`, with Full/Mixed and HG/no-HG variants trained for the same TensorRT composition used at deployment
- unsupported extra low-precision experiment presets and builder paths have been removed; the supported NVIDIA deployment surface is FP16/FP32 plus INT8 PTQ/QAT/QAT Film
- HDR GT fast path processing for significantly faster ground-truth video alignment and frame mapping
- optimized GT sync algorithm with improved frame scanning and alignment detection
- HDR GT status indicator in the GUI status line showing when fast path is active
- cached GT alignment results to avoid redundant processing for the same video pairs
- HDR GT pairing now tolerates small duration/frame-count differences, encoded black bars vs cropped active-picture sources, and conservative cached constant frame offsets
- compare, objective metrics, and benchmark now map SDR frames to the matching HDR GT frame instead of assuming raw frame numbers always line up
- benchmark video runs now post-verify GT frames by exact decode and local frame alignment before final metrics are reported
- queued video benchmarks reuse a bounded in-memory post-verify GT cache for repeated SDR/HDR/frame/resolution pairs, skipping redundant exact GT decode/alignment while still recomputing HDR Convert outputs and metrics per checkpoint
- compare preparation now uses a cancelable application-modal progress dialog so pending compare seeks can be canceled cleanly
- native `Browser Window Capture (Experimental)` video path
- modern Windows direct window capture backend for browser-window video
- Chrome Audio Sync extension kept audio-only, with manual start/stop in Chrome
- Browser Window Capture now observes Chrome compositor frames separately from the low-cost process-FPS budget
- Browser Window Capture now feeds mpv a steady low-FPS stream and lets mpv repeat frames on display vsync instead of forcing 60 fps pipe writes
- playback timing now uses Windows high-resolution waitable timers, MMCSS `Playback` scheduling, precise live-feeder deadline polling, and lower-copy mpv named-pipe writes to reduce scheduler-induced frame cadence misses
- Browser Window Capture waits for the HDR mpv display handoff after cold compile, avoiding the occasional black HDR pane / inactive-HDR startup race
- cleaner startup logging by filtering harmless Qt DPI-awareness warning spam
- repo-local PyTorch kernel caches on AMD under `src/models/compile_cache/` so users can see and delete generated kernels
- bounded cached-kernel verification with clear-and-recompile recovery when an incompatible cache would otherwise hang warmup
- continued playback/export cleanup from the `v2.x` and `v3.0` work

Windows-only project with **NVIDIA CUDA/TensorRT**, **AMD ROCm-Windows/PyTorch**, and **CPU** backends.

---

## How To Run (Recommended Release Workflow)

No manual asset download is required before setup.

1. Open the [GitHub Releases page](https://github.com/DanHelmy/hdr-realtime-video-pipeline/releases).
2. Download the latest stable release ZIP.
3. Extract the ZIP.
4. Run `setup.bat` (double-click or terminal).
5. Run `run_gui.bat` to start the app.

Do not clone `main` for normal use. `main` may contain unreleased changes, experimental features, or in-progress thesis work.

What now happens automatically:
- `setup.bat` tries to download `libmpv-2.dll` and `HG.pt` into the repo.
- `run_gui.bat` checks for missing Python dependencies and offers to rerun setup if an update added a new library.
- If `libmpv-2.dll` or `HG.pt` are still missing, the GUI tries to download them on first launch before showing a recovery dialog.

Manual fallback assets link:

`https://drive.google.com/drive/folders/1jh8gXBVzqRse-7w_2Dztca1_KVh5eRu1?usp=drive_link`

The automatic downloader uses fixed Google Drive file IDs and saves HG locally as `src/models/weights/original/HG.pt`, so the Drive file does not need to be renamed. For manual placement, only the local filename/path matters.

---

## Releases

Use the latest stable release unless you are testing local development work:

- Latest stable release: **v6.0**
- Releases page: [github.com/DanHelmy/hdr-realtime-video-pipeline/releases](https://github.com/DanHelmy/hdr-realtime-video-pipeline/releases)
- `main`: current v6.0 stable/development head

For tagged release ZIPs, use GitHub Releases. The green `Code -> Download ZIP` button downloads the current `main` tree.

---

## Quick Start

```bash
# 1. Download the latest stable ZIP from GitHub Releases
# 2. Extract it and open a terminal in the extracted folder

# 3. Double-click setup.bat
# (or run it in terminal)
.\setup.bat

# 4. Launch the GUI
.\run_gui.bat
```

Open a video and it plays in tabbed SDR/HDR views (with optional side-by-side tab).

Developer-only clone workflow:

```bash
git clone https://github.com/DanHelmy/hdr-realtime-video-pipeline.git
cd hdr-realtime-video-pipeline
.\setup.bat
.\run_gui.bat
```

---

## UI Tour (v6.0)

### 1. Main Workspace

- Tabbed views: `SDR`, `HDR`, `Side by Side`
- Playback controls, timeline, and live runtime metrics, with hide-UI playback surfacing the playhead, `Show UI`, and metrics overlay only while the cursor is moving

![Main UI Overview](docs/images/v4-main-ui-overview.png)

### 2. Browser Window Capture (Experimental)

- Direct visible-window Chrome capture for video
- Chrome extension handles delayed local audio sync
- Video processing stays on a low-FPS budget while mpv handles display-vsync frame repeat

![Browser Capture Flow](docs/images/v4-browser-capture-flow.png)

### 3. Compare / Objective Metrics Dialog

- Side-by-side objective frame comparison workflow
- PSNR / SSIM on linear HDR frames, plus DeltaEITP / HDR-VDP3 on the color-managed HDR path
- Compare pauses playback while preparing the exact requested frame and shows a cancelable modal progress dialog
- Cancel clears the pending compare seek/request; playback resumes automatically when Compare was the thing that paused it

![Compare Dialog](docs/images/v4-compare-dialog.png)

### 4. Export Dialog

- Independent export settings (separate from playback controls)
- Resolution/FPS/model/precision export controls

![Export Dialog](docs/images/v4-export-dialog.png)

### 5. Model Quality Benchmark Tool

- Open from `Tools -> Model Quality Benchmark ...`
- Benchmark a single `SDR video + HDR GT` pair or paired `SDR/HDR GT` dataset folders
- Queue multiple benchmark configurations, preview/remove individual queued runs, and run the remaining queue back-to-back without manually restarting each run
- Review objective results with SDR/HDR GT/HDR Convert previews and run metadata (`source`, `precision`, `resolution`)
- On AMD, benchmark uses cached `torch.compile` / `max-autotune` kernels on an exact cache hit and falls back to eager mode on a miss
- Video pairs can differ slightly in length, start offset, or encoded black bars as long as the active content matches
- Video frame selection can detect a larger deterministic candidate pool, then use the same average-mode workflow as datasets (`selected`, `all`, or deterministic subset) for final scoring
- Video benchmark metrics are finalized by an exact GT decode/post-verify pass; JSON/CSV results include the GT frame, alignment offset, and alignment score used for each row
- Queued video runs reuse a bounded in-memory post-verify GT cache for repeated source/GT/frame/resolution pairs. This reuses only the verified GT frame/alignment side; HDR Convert outputs and metrics are still recomputed for every checkpoint.
- Dataset image pairs stay on the image decoder path; video sync/crop probes are only used for actual video files

![Benchmark Tab](docs/images/v5-benchmark-tab.png)

---

## Browser Audio Sync Extension

The browser extension is already included in this repo. You do not download it separately, and it is now audio-only.

Extension folder:

`browser_tab_capture_extension/`

Use this flow for `Browser Window Capture`:

1. Run the app with `run_gui.bat` or `py src\gui.py`
2. In Chrome, open `Settings > System`
3. Turn off `Use graphics acceleration when available`
4. Restart Chrome
5. Open `chrome://extensions`
6. Turn on `Developer mode`
7. Click `Load unpacked`
8. Select the repo's `browser_tab_capture_extension` folder
9. In HDRTVNet++, choose `Browser Window Capture (Experimental)`
10. In Chrome, open the tab you want and click the extension's `Start Chrome Audio Sync`
11. Back in HDRTVNet++, pick the matching visible Chrome window
12. Adjust the extension delay slider while playback is running until lip-sync looks right
13. Stop Chrome Audio Sync later from the extension popup when you are done

Audio delay slider range: 0-2000 ms.

Browser capture pacing:

- The GUI exposes a Browser Window Capture FPS cap: `24`, `30`, or `60` fps. `24 fps` is the default smooth low-latency cap.
- Browser-window processing/output presets are capped by both the captured window size and the largest attached monitor, so a 4K monitor can expose 2160p capture while smaller monitors stay capped to their own useful maximum.
- Chrome compositor observation runs faster than processing, defaulting to 2x the selected process cap (`48`, `60`, or `120` fps) unless `HDRTVNET_LIVE_CAPTURE_OBSERVE_FPS` overrides it.
- Video processing samples the latest visible Chrome window frame at the selected GUI cap, or `HDRTVNET_LIVE_CAPTURE_PROCESS_FPS` before the GUI setting is loaded. Default: `24`.
- The mpv live feed repeats the latest processed frame at the selected cap for steadier display cadence. `HDRTVNET_LIVE_CAPTURE_DISPLAY_FPS` can lower that feed, but values above process FPS are capped to avoid uneven duplicate cadence.
- mpv owns the final display timing with vsync-aware frame repeat, so Python does not need to write 60 frames per second.
- mpv keeps a tiny live jitter buffer so short wake-up or pipe-write stalls repeat a frame instead of creating a visible cadence hole. Default: `HDRTVNET_LIVE_CAPTURE_MPV_BUFFER_FRAMES=8`.
- The live mpv feeder allows a tiny bounded refill after a late write so mpv's raw-video pipe does not underflow into visible pauses.
- Playback, capture, and feeder timing threads opt into Windows high-resolution timers plus MMCSS `Playback` scheduling so frame-deadline wakeups are less likely to slip under CPU load.
- Once a live feeder has a frame to repeat, it uses precise short-slice polling instead of relying on coarse `Queue.get(timeout=...)` wakeups for presentation deadlines.
- The mpv named-pipe writer avoids per-chunk `bytes` copies when writing frame buffers, reducing avoidable jitter in the final delivery step.
- mpv display debanding and built-in `fruit` output dithering are enabled by default for both Browser Window Capture and normal video playback to soften codec/compositor banding after SDR-to-HDR expansion.
- Captured browser SDR frames are tagged as full-range sRGB for the SDR preview path; the converted HDR pane is still tagged as BT.2020/PQ after HDRTVNet++ inference.
- Static browser windows keep feeding by repeating the latest visible frame at process FPS. If WinRT has no compositor frame yet, a visible-window fallback captures one startup seed frame so playback can start even before the tab video is moving.
- Browser source delivery now resets after a meaningfully late frame instead of immediately catching up with a short interval. This favors steady motion cadence over shaving a few milliseconds of live latency.
- To reduce load further, set the variables before launching:

```powershell
$env:HDRTVNET_LIVE_CAPTURE_OBSERVE_FPS="48"
$env:HDRTVNET_LIVE_CAPTURE_PROCESS_FPS="24"
$env:HDRTVNET_LIVE_CAPTURE_DISPLAY_FPS="24"
$env:HDRTVNET_LIVE_CAPTURE_MPV_BUFFER_FRAMES="3"
.\run_gui.bat
```

Advanced testing: `HDRTVNET_FEEDER_GPU_RGB48=1` enables the experimental GPU-side HDR tensor-to-RGB48 feeder conversion and is currently the default. If it hangs or misbehaves on a GPU/driver stack, launch with `HDRTVNET_FEEDER_GPU_RGB48=0`.

Display deband/dither tuning:

- `HDRTVNET_MPV_DEBAND=1|0` enables/disables normal video display debanding; default is `1`.
- HDRTVNET_MPV_DEBAND_ITERATIONS=3
- HDRTVNET_MPV_DEBAND_THRESHOLD=100
- HDRTVNET_MPV_DEBAND_RANGE=32
- HDRTVNET_MPV_DEBAND_GRAIN=8
- `HDRTVNET_MPV_DITHER=1|0` enables/disables mpv output dithering; default is `1`.
- `HDRTVNET_MPV_DITHER_ALGO=fruit` (`ordered` and `error-diffusion` are also accepted by mpv builds that support them)
- `HDRTVNET_MPV_DITHER_DEPTH=auto`
- `HDRTVNET_MPV_DITHER_SIZE_FRUIT=6`
- `HDRTVNET_MPV_TEMPORAL_DITHER=1|0` changes the dither pattern over time; default is `1`.
- `HDRTVNET_MPV_TEMPORAL_DITHER_PERIOD=1`

Browser Window Capture uses the same display cleanup defaults as normal video playback:

- `HDRTVNET_BROWSER_MPV_DEBAND=1|0`; default is `1`.
- `HDRTVNET_BROWSER_MPV_DEBAND_ITERATIONS=3`
- `HDRTVNET_BROWSER_MPV_DEBAND_THRESHOLD=100`
- `HDRTVNET_BROWSER_MPV_DEBAND_RANGE=32`
- `HDRTVNET_BROWSER_MPV_DEBAND_GRAIN=8`
- `HDRTVNET_BROWSER_MPV_TEMPORAL_DITHER=1|0` enables/disables temporal dither for Browser Window Capture; default is `1`.
- `HDRTVNET_BROWSER_MPV_TEMPORAL_DITHER_PERIOD=1`

If you see shimmer or panel-dither interference on a 6-bit/FRC display, first try `HDRTVNET_MPV_TEMPORAL_DITHER=0`.

Export defaults now follow the same direction:

- HDR export adds `deband` before colorspace conversion.
- HDR export uses `zscale` dithering (`error_diffusion`) instead of disabling dithering.

WinRT pacing tuning:

- `HDRTVNET_WINRT_FRAME_POOL_BUFFERS=4` controls the Windows Graphics Capture frame-pool depth. Default: `4`.
- `HDRTVNET_WINRT_DRAIN_TO_LATEST=0` keeps queued WinRT frames in order for steadier motion cadence. Set to `1` for the older lowest-latency behavior that always jumps to the newest queued frame.
- Live capture interpolation is currently enabled by default.

Important:

- `Browser Window Capture` is experimental.
- Google Chrome is the only supported browser for synced browser-window playback in this mode.
- Chrome hardware acceleration must be off for this path.
- HDRTVNet++ captures video directly from the visible Chrome window.
- Browser-window video uses a native Windows output-capture backend; performance still depends on window size and monitor configuration.
- The extension is audio-only and delays tab audio locally inside Chrome.
- Chrome Audio Sync now stays active until you stop it manually in the extension.
- HDRTVNet++ stays silent during browser-window playback.
- Browser-window capture observes Chrome separately (default 2x the selected GUI cap), feeds mpv a steady `24`/`30`/`60` fps stream, and lets mpv repeat frames on display vsync.
- After a cold compile, playback waits for mpv to attach before the worker starts producing frames; this prevents the HDR pane from staying black while the worker silently falls back to CPU output.
- Without Chrome Audio Sync, Chrome keeps playing audio locally and it can lead the video.
- If the Chrome source window disappears unexpectedly, HDRTVNet++ now treats that as source loss and restarts cleanly instead of holding onto a dead browser feed.

---

## GUI (v6.0)

```bash
python src/gui.py
```

The GUI is the primary way to use the pipeline. It handles backend selection, model/engine loading, HDR display, export, dedicated objective benchmarking, and live browser-window viewing.

### v6.0 Highlights

- **Hybrid inference runtime**
  - NVIDIA devices use TensorRT engines for all inference
  - AMD devices keep the PyTorch inference path
  - preprocessing and postprocessing are shared so backend output handling stays consistent

- **TensorRT on-demand engine cache for NVIDIA**
  - engines are built only when a mode/model/resolution is activated
  - cache files are saved under `src/models/engines/`
- engine names follow `{model}_{resolution}_{mode}.engine`
- INT8 TensorRT engine names include a Q/DQ version suffix so older incompatible cache files are not reused
  - existing engines load directly on later runs
  - if engine build/load fails, the app logs the error, informs the user, and does not crash
  - NVIDIA does not fall back to PyTorch inference

- **AMD PyTorch path keeps existing controls**
  - PyTorch compile/eager controls remain available on AMD
  - INT8 pre-dequantization controls remain available on AMD
  - channels-last is opt-in for AMD PyTorch tensors; contiguous is the default compile/runtime format
  - `Auto` pre-dequantization for AMD INT8 still resolves to pre-dequantize-on

- **NVIDIA UI is TensorRT-focused**
  - PyTorch-specific options are hidden on NVIDIA:
    - INT8 pre-dequantization menu
    - runtime execution mode menu
    - pre-compile kernels
    - clear kernel cache
    - export max-autotune/pre-dequantize advanced controls
  - NVIDIA exposes a dedicated `Clear TensorRT Engine Cache` tool for cached `.engine` files
  - visible NVIDIA controls focus on model, precision/mode, resolution, and HG selection

- **TensorRT replaces PyTorch max-autotune for NVIDIA**
  - TensorRT optimizes during engine build
  - optimized engines are cached and reused
  - no max-autotune background-load warning is shown on NVIDIA

- **HDR GT pairing is more practical for real movie files**
  - GT validation allows short duration/frame-count mismatches by default
  - encoded black bars are handled with one stable pair-level crop rectangle, so cropped and letterboxed versions of the same movie can still be paired without per-frame aspect shifts
  - **fast path processing** significantly accelerates HDR GT alignment with optimized frame scanning
  - **improved sync algorithm** with better constant offset detection and frame mapping
  - **cached alignment results** prevent redundant processing for repeated video pairs
  - **GUI status indicator** shows "HDR GT: fast path active" when using optimized processing
  - a cached sync scan estimates constant SDR/HDR lead-lag offsets and reports the detected frame offset, but keeps offset `0` unless a nonzero offset is clearly better
  - benchmark video runs add a per-sample local GT-frame search during post-verify, so final metrics are based on the best nearby exact-decoded GT frame instead of a stale fast seek
  - queued benchmark runs reuse verified exact GT frames from a bounded memory cache when the SDR file, HDR GT file, selected frame, mapped GT frame, resolution, and alignment settings match
  - compare, objective metrics, and benchmark use the same mapped GT-frame lookup
  - same-canvas GT files are no longer cropped per frame, avoiding accidental zoomed GT previews/metrics

- **Compare preparation is cancelable**
  - clicking `Compare` opens a modal progress dialog while the exact frame is prepared
  - canceling clears the pending compare request and temporary seek instead of leaving playback pinned to the compare frame
  - stale compare results are ignored after cancel
  - recompare requests for noncurrent SDR video frames use a guarded FFmpeg fast seek first and fall back to OpenCV random access if the fast seek cannot verify the requested timestamp closely enough

- **Safer default quality preset**
  - clean first launches default to `INT8 Mixed (QAT)`, `1080p`, `SSimSuperRes`, and HG off
  - existing `.gui_prefs.json` settings still win, so local user choices are preserved

- **Pane-aware playback upscale**
  - the `Resolution` control selects model processing size (`2160p`, `1440p`, `1080p`, `720p`, `540p`, or source-limited fallback); every preset above the source size is hidden, so a 720p source only offers 720p/540p
  - the HDR pane's actual drawable size drives the upscale target, so fullscreen-with-UI and side-by-side do not request full-monitor upscale unless the pane itself fills that space
  - hide-UI fullscreen naturally becomes monitor-sized because the HDR pane fills the display
  - while UI is hidden, the playhead, `Show UI`, and performance metrics appear together as a temporary overlay when the cursor moves, then fade back out
  - the `Upscale` control is always editable; the selected method is saved as a preference and becomes active whenever the current pane target is larger than the processed frame
  - Apply feedback now reports whether the selected upscale is active or only saved as an inactive preference for the current pane size
  - FSR now runs through an RGB `MAIN` shader path for RGB48 playback, and any residual scale after EASU uses the best mpv scaler instead of bilinear
  - the status bar keeps the current pane upscale state visible during playback, and audio recovery notices yield to that live scaling status
  - moving or resizing the main window or popped-out HDR view updates the mpv scale/CAS settings without restarting inference

- **Smoother normal video playback**
  - startup, seek, and resume fill a fixed `HDRTVNET_VIDEO_PLAYBACK_BUFFER_FRAMES` buffer before releasing audio and HDR/SDR panes together; default is `12`
  - side-by-side SDR/HDR mpv feeds preserve ordered frames during normal video playback, while Browser Window Capture keeps its low-delay latest-frame stabilizer
  - dragging the playhead shows an exact mpv scrub preview from the source file instead of relying on OpenCV seeking
  - side-by-side relock is less aggressive, so normal timestamp noise does not repeatedly flush both mpv panes

### Previous v5.1 Highlights

- **Objective metric domains are now consistent across compare and benchmark**
  - `PSNR` and `SSIM` are computed from true linear HDR image pairs
  - `DeltaEITP` and `HDR-VDP3` are computed from BT.2020/PQ display-managed image pairs
  - compare and benchmark now call the same shared metric path, so they no longer drift
  - the built-in `HDR-VDP3` bridge now decodes PQ back to absolute photometric values before invoking `hdrvdp3`

- **Model Quality Benchmark tool is now built in (Tools menu)**
  - supports two workflows: `Video (SDR + HDR GT)` and `Dataset Folders (SDR + HDR GT)`
  - queue controls can add the current run or add all visible precision presets for the current source/resolution/HG setup
  - runs quality benchmarking through TensorRT on NVIDIA, cached max-autotune on AMD when available, or eager fallback when no safe compile cache exists
  - video workflow includes deterministic distinct-frame candidate pools, average modes (`selected`, `all`, deterministic subset), and manual frame checkboxes
  - video frame detection now prefers FFmpeg packet-level keyframe timestamps plus bounded tiny preview-frame scoring, then reuses cached scores for deterministic subset/all-frame modes instead of re-reading every detected frame through OpenCV
  - dataset workflow includes paired-file scanning with the same averaging modes (`selected`, `all`, deterministic subset)
  - result page shows run info (`source name`, `precision`, `resolution`) and supports loading existing JSON/CSV summaries
  - result previews now use the same compare-style color-managed display path for `SDR`, `HDR GT`, and `HDR Convert`
  - session outputs are structured under `logs/benchmark_sessions/<source>/<timestamp>__<precision>__<resolution>__n<count>/...`
  - benchmark locks playback interactions while open and compare is blocked until benchmark closes

- **Native Browser Window Capture replaces the old browser-video bridge**
  - HDRTVNet++ now captures browser video directly from the visible Chrome window
  - the extension is kept only for delayed local Chrome audio
  - the app no longer depends on browser JPEG frame uploads in this mode
- **Chrome Audio Sync is simpler and more explicit**
  - Chrome-only
  - experimental
  - manual delay slider in the extension popup
  - manual stop behavior stays in the extension popup
  - HDRTVNet++ stays silent while Chrome replays delayed tab audio locally
- **Browser Window Capture is more usable as a live viewer**
  - Browser Window Capture now observes Chrome compositor frames separately, then runs HDRTVNet++ only under the low-cost process-FPS budget
  - the direct-window capture path remains the only active browser-video path
  - live browser presentation now uses mpv timed playback (`display-resample`) so frame repeat is handled by the display clock instead of a Python pacing loop
  - playback timing now uses Windows high-resolution waitable timers, MMCSS `Playback` scheduling, precise live-feeder deadline polling, and lower-copy mpv pipe writes to reduce scheduler-induced frame cadence misses
  - HDR output waits for an explicit mpv display handoff after compile, so cold compile startup no longer races into a black HDR pane with inactive HDR metadata
  - if the Chrome source window disappears unexpectedly, the app now restarts cleanly instead of leaving a dead live source attached
- **Startup logging is cleaner**
  - harmless Qt DPI-awareness warnings are filtered so launch logs stay focused on real problems
- **Kernel cache behavior is safer and more local**
  - AMD PyTorch compile caches are stored under `src/models/compile_cache/`, so generated kernels stay visible beside the repo instead of hiding in AppData
  - multiple local copies no longer reuse or wipe each other's kernels
  - FP16 and predequantized mixed INT8 cache markers now line up consistently in both directions when they share the same effective compiled graph
  - cached-kernel verification now detects incompatible "compiled" caches before playback/export enters a stuck warmup path
  - when verification stops, the dialog prioritizes retrying verification or clean recompilation instead of immediately suggesting cache deletion
- **Export workflow remains production-oriented**
  - separate `File > Export Video...` flow with independent precision/model/HG selection
  - source-native resolution/FPS defaults
  - aspect-ratio-safe fit/pad resizing
  - ProRes 422 HQ (`.mov`) + PCM audio only
- **Export and playback polish continues**
  - HDR sources are rejected immediately when chosen
  - keep-aspect ratio editing no longer fights typed values
  - pressing `Enter` in resolution/FPS/path fields no longer starts export accidentally
  - cancel tears down the export runtime and releases GPU resources more cleanly
  - progress/finalization reporting is more truthful during long exports
- **HDR metadata/tagging improved**
  - ProRes exports now use a more reliable BT.2020 / PQ tagging path
  - export conversion path now targets a `1001 nit` HDR peak expectation
  - export now applies `deband` + `error_diffusion` dithering by default to better match interactive playback cleanup
- **HDR GT fast path utilities**
  - dedicated `gui_hdr_gt_fast_path.py` module for optimized ground-truth video processing
  - `true_hdr_video_fast` mode indicator for improved HDR GT handling
  - faster frame alignment with reduced computational overhead
  - better memory management for large HDR GT video files
- **Compare pane display transitions are smoother**
  - compare no longer force-recreates mpv surfaces on show/screen-change, reducing visible color-space flashing
- **HDR cleanup bandaids are removed from the app path**
  - live mpv playback and export use raw model output for clean FPS/artifact A/B testing
  - playback, export, compare, benchmarks, and objective metrics all use the same raw model output path
- **Live metrics are more thesis-friendly**
  - the `Latency` field now reflects model-stage latency instead of mixing in more source-path timing differences
  - app `VRAM` / `CPU` memory remain whole-app runtime metrics, which is more honest than pretending they are model-only allocations
- **Playback session logging is built in**
  - the playback toolbar now includes `Log Session`
  - a logged session saves full runtime metrics such as `FPS`, `latency`, `VRAM`, `CPU`, model precision, and objective metric fields when present
  - compare clicks also save per-frame compare metrics such as `PSNR`, `SSIM`, `DeltaEITP`, normalized variants, and optional `HDR-VDP3`
  - logs are written to `logs/playback_sessions/<timestamp>_<source>/`
  - each session folder includes `summary.txt`, `session.json`, `runtime_metrics.csv`, and `compare_events.csv` when compare was used
- **Experimental AMD max-autotune export reuses the playback compile cache**
  - export uses the same compile/cache keying and compile dialogs as playback
  - max-autotune export uses full Stop behavior first to avoid stale GPU/MIOpen state
- **Modular GUI refactor remains in place**
  - `gui.py` composes focused mixins/modules (`gui_ui_builder.py`, `gui_signal_wiring.py`, `gui_playback_runtime.py`, `gui_pipeline_worker_*.py`, etc.)

### Features

| Feature | Description |
|---|---|
| **Open any video** | Browse or drag-and-drop — playback starts automatically |
| **Browser Window Capture (Experimental)** | Native visible-Chrome window capture with bundled Chrome Audio Sync; samples latest frames under a process-FPS budget and lets mpv repeat them on display vsync |
| **Modular codebase** | GUI and worker logic split into maintainable mixins/modules for easier iteration |
| **Tabbed views** | `SDR`, `HDR`, and `Side by Side` tabs |
| **Pop/Dock panes** | Detach SDR/HDR into separate windows and dock back |
| **Live precision switching** | FP16, FP32, INT8 PTQ/QAT variants — switch mid-playback |
| **HG toggle** | Enable/disable HG refinement (loads HG or no-HG INT8 weights) |
| **Playback controls** | Play / Pause / Resume / Stop |
| **Seek bar** | Drag to seek; when paused, seek is queued and applied on Resume for frame-accurate preview |
| **Paused hot-swap preview** | Precision changes can redraw the current paused frame without resuming playback; AMD pre-dequantize changes can do the same on PyTorch runtimes |
| **Performance metrics panel** | FPS, model-stage latency, frame count, app VRAM/CPU memory, checkpoint size, precision, processing resolution; in hide-UI playback it joins the temporary cursor-triggered overlay with the playhead and `Show UI` button |
| **Compare metrics dialog** | Pauses playback and opens 3-way frame compare (SDR, HDR GT, HDR Convert) with PSNR/SSIM on linear HDR frames, DeltaEITP on the color-managed HDR path, normalized variants, and optional HDR-VDP3 |
| **Model Quality Benchmark tool** | Tools-menu benchmark dialog for video or dataset objective evaluation, queued multi-run batches, one-click all-precision queueing, deterministic selection, GT sync/crop handling, run metadata display, preview images, and summary export/load |
| **Deterministic compare snapshots** | Compare recomputes the selected frame in an isolated path so the first snapshot matches refresh behavior more reliably |
| **Playback session logs** | `Log Session` saves full runtime metric samples plus compare metrics to `logs/playback_sessions/` as text/JSON/CSV |
| **HDR metadata panel** | Color primaries, transfer function, peak luminance (nits), VO/GPU API |
| **Color handling** | SDR pane uses Rec.709 tagging and SDR-specific downscaling; HDR pane uses BT.2020/PQ tagging and HDR-oriented linear downscaling; mpv auto-selects output mapping per display |
| **Hybrid backend selection** | NVIDIA uses TensorRT engines exclusively; AMD uses the existing PyTorch runtime |
| **TensorRT engine cache** | NVIDIA engines are built on demand per model/resolution/mode and reused from `src/models/engines/` |
| **Clear TensorRT engine cache** | NVIDIA-only tool for deleting selected or all cached `.engine` files |
| **PyTorch kernel compilation** | AMD can use Triton/torch.compile caches in a clean subprocess; caches are repo-local under `src/models/compile_cache/` and verified before reuse |
| **Resolution + scaling** | Process at 2160p/1440p/1080p/720p/540p (source-capped when the video is smaller); playback uses **EWA LanczosSharp**, **FSR**, or **SSimSuperRes** through a pane-aware target, so the upscale stage follows the HDR pane's actual drawable size |
| **Single-frame processing path** | Temporal stabilization is disabled globally for more predictable playback/export cost and latency |
| **Film grain** | Optional film grain restoration (mpv shader) |
| **Video export** | Separate export dialog with native resolution/FPS defaults, independent model preset selection, and ProRes 422 HQ output |
| **Audio support** | Auto-detect, attach external audio, and choose audio track |
| **Volume + stability policy** | Volume slider plus automatic mute below low FPS threshold, with fade-in restore on recovery |
| **Timeline recovery** | Backward seeks and relocks reset stale frame-drop state more reliably to avoid frozen HDR video after the audio already moved |
| **Keyboard shortcuts** | `F11` borderless full-window, `Esc` exit borderless mode, `Space` pause/resume |
| **Cursor idle hide** | Optional auto-hide cursor during playback |
| **INT8 pre-dequantization control** | AMD Tools-menu toggle for `Auto` / `On` / `Off` on INT8 PyTorch runtimes |
| **Runtime execution mode** | AMD Tools-menu toggle for `Compile (recommended)` / `Eager (not recommended)` |
| **Pre-compile kernels** | AMD-only PyTorch kernel precompile for any resolution/precision ahead of time |
| **Clear kernel cache** | AMD-only PyTorch kernel cache clearing for the current project checkout |
| **Dark theme** | Modern dark UI, auto-applied |
| **Persistent GUI settings** | Saved in `.gui_prefs.json` (precision, resolution, browser capture FPS, view/tab, upscale, film grain, metrics visibility, HG toggle, AMD pre-dequantization/runtime mode, volume, audio track, cursor hide, last-open directory). On a clean first launch the default preset is `INT8 Mixed (QAT)` / `1080p` / `SSimSuperRes` / HG off |

### GUI Launch Flags

`src/gui.py` also accepts startup flags (used by restart/apply flows):

```bash
python src/gui.py --video input.mp4 --resolution 720p --precision FP16 --view Tabbed --autoplay 1 --start-frame 1200 --use-hg 1 --film-grain 1 --hdr-gt hdr_reference.mkv
python src/gui.py --source-mode window_capture --live-fps 30 --resolution 1440p --precision "INT8 Mixed (QAT)" --use-hg 0
```

### Objective Metrics (PSNR / SSIM / DeltaEITP / HDR-VDP3)

- Use **HDR GT ...** in the GUI, then click **Compare** to compute per-frame accuracy metrics.
- In `v6.0`, compare and the `Model Quality Benchmark` tool use the same shared full-reference metric pipeline.
- `PSNR` and `SSIM` are computed from the linear HDR image pair.
- `DeltaEITP` and `HDR-VDP3` are computed from a BT.2020/PQ display-managed path derived from that linear HDR pair.
- `DeltaEITP-N` is grade-normalized in absolute linear RGB before BT.2020/PQ conversion; it is not normalized on PQ code values and it is not re-normalized after PQ encoding.
- Shared padded black or near-black borders are cropped from both images before objective metrics when a substantial common letterbox/pillarbox region is detected.
- Ground-truth should be the same content/timing as the input clip for valid measurements.
- GT pairing accepts small practical differences between real encodes:
  - `HDRTVNET_GT_SYNC_TOLERANCE_S` controls when GT pairing can stay on simple overlap-sync before it falls back to content-sync notes. Default: `2.0`.
  - active-picture detection allows one file to be letterboxed while the other is cropped, then applies a stable pair-level geometric crop before preview/metrics only when the encoded pair genuinely needs it.
  - `HDRTVNET_GT_SYNC_OFFSET_SEARCH_S` controls the constant-offset search window. Default: `2.0`.
  - `HDRTVNET_GT_SYNC_OFFSET_MIN_GAIN` controls how much better a nonzero global offset must be before it replaces frame offset `0`. Default: `0.06`; tiny offsets require a stronger gain to avoid false `+/-1` to `+/-5` frame shifts.
  - the detected offset is cached per file signature and used by Compare, objective logging, and benchmark frame reads.
  - if a movie starts more than two seconds apart between SDR and HDR GT files, raise `HDRTVNET_GT_SYNC_OFFSET_SEARCH_S` before launch.
- Compare recompare, benchmark previews, and deterministic video frame display use guarded FFmpeg fast seeks for SDR frame reads when available, with OpenCV fallback:
  - `HDRTVNET_SDR_FRAME_FAST_SEEK=0` disables the SDR fast-seek reader.
  - `HDRTVNET_SDR_FRAME_FAST_SEEK_PTS_GUARD=0` disables timestamp verification for the fast reader.
  - `HDRTVNET_SDR_FRAME_CACHE_MAX` controls the small repeated-frame cache. Default: `8`.
- Model Quality Benchmark video candidate detection uses FFmpeg packet-level keyframe timestamps and tiny preview-frame scoring when available, then stores the deterministic scored pool:
  - `HDRTVNET_FRAME_DETECT_FFMPEG=0` disables this keyframe detector and falls back to the OpenCV scanner.
- Benchmark video post-verification keeps the first pass fast, then exact-decodes GT frames before final metrics:
  - `HDRTVNET_BENCHMARK_AUTO_POST_VERIFY` enables/disables this pass. Default: `1`.
  - `HDRTVNET_BENCHMARK_AUTO_POST_VERIFY_MAX_ITEMS` limits verified rows, or `all` verifies every video row. Default: `all`.
  - `HDRTVNET_BENCHMARK_GT_LOCAL_SEARCH_FRAMES` controls the per-sample local GT search radius around the mapped frame. Default: `0`.
  - `HDRTVNET_BENCHMARK_GT_LOCAL_SEARCH_MIN_GAIN` controls how much better a nearby GT frame must be before it replaces the mapped frame. Default: `0.035`.
  - benchmark summaries and CSV exports include `gt_frame`, `gt_alignment_offset_frames`, and `gt_alignment_score` for auditability.
- `HDR-VDP3` now has a built-in local bridge at `scripts/hdrvdp3_bridge.py`.
  - The built-in bridge accepts BT.2100 PQ input frames and converts them to absolute display luminance / photometric values before calling `hdrvdp3`.
  - The GUI will use it automatically when `HDRTVNET_HDRVDP3_CMD` is not set.
  - If an HDR-VDP3 toolbox is already present under `third_party/hdrvdp/`, it is reused.
  - New toolbox downloads only happen when GNU Octave is available.
  - First HDR-VDP3 run auto-downloads toolbox files into:
    - `third_party/hdrvdp/`
  - If the repo location is not writable, set `HDRTVNET_HDRVDP_CACHE_DIR` intentionally to choose a custom cache path.
  - Requires **GNU Octave** installed and available in `PATH`.
- You can still override with your own command using env var `HDRTVNET_HDRVDP3_CMD`.
  - Template placeholders: `{test}` / `{pred}`, `{reference}` / `{ref}`, and `{encoding}`.
- Objective HDR peak for the managed metric path can be adjusted with `HDRTVNET_OBJECTIVE_HDR_PEAK_NITS` (default: `1000`).

### Playback Session Logs

- Click `Log Session` in the playback toolbar to arm logging for the next active playback session.
- Stop playback, close the app, let playback finish, or click the button again to flush the session log to disk.
- Logs are saved under:
  - `logs/playback_sessions/<timestamp>_<source>/`
- Saved files:
  - `summary.txt` for a quick human-readable report
  - `session.json` for the full structured session payload
  - `runtime_metrics.csv` for the sampled runtime stream (`FPS`, latency, `VRAM`, `CPU`, frame index, precision, objective fields when present)
  - `compare_events.csv` for per-click compare metrics (`PSNR`, `SSIM`, `DeltaEITP`, normalized variants, `HDR-VDP3`, notes)
- The worker summary also stores the exact average inference latency across logged inference frames.

### Tools Menu

- **Model Quality Benchmark** — quality benchmarking dialog for video or dataset workflows, with previews, averages, all-precision queueing, and result export/load
- **INT8 Pre-dequantization** — AMD PyTorch only; choose `Auto`, `On`, or `Off` for INT8 runtime loading behavior
- **Runtime Execution Mode** — AMD PyTorch only; choose compiled or eager execution
- **Pre-compile Kernels** — AMD PyTorch only; compile for any resolution(s) ahead of time
- **Clear Kernel Cache** — AMD PyTorch only; force recompilation (e.g. after a PyTorch / driver update)
- **Clear TensorRT Engine Cache** — NVIDIA only; delete selected or all cached `.engine` files

On NVIDIA, PyTorch-specific tuning and kernel-cache tools are hidden because inference always uses TensorRT engines.

### Video Export

- Open **File -> Export Video...** to export with settings that are independent from the playback controls
- Export defaults to the source clip's **native resolution** and **native FPS**
- You can override resolution/FPS and keep aspect ratio locked; mismatched aspect ratios are fit into the target frame with padding
- Export supports any available model preset (`FP16`, `FP32`, `INT8 PTQ/QAT`, HG on/off) directly from the export dialog
- Output is intentionally limited to **ProRes 422 HQ (`.mov`) + PCM audio**
  - This is a high-quality mezzanine format that avoids recompressing already-compressed sources back into delivery codecs like H.265 during intermediate work
- ProRes export is tagged through the HDR path as **BT.2020 / PQ**, with a `1001 nit` target peak expectation in the export conversion path
- HDR input sources are rejected immediately when selected in the export dialog
- On AMD, export has an **Advanced** tab for:
  - experimental max-autotune compile reuse
  - INT8 pre-dequantize override (`Auto` / `Force On` / `Force Off`)
- On NVIDIA, export uses TensorRT engines and hides PyTorch-specific export tuning controls.
- Export uses raw model output.
- Starting a normal export keeps playback paused/locked for the export run
- Starting an AMD export with **experimental max-autotune** may trigger the same full **Stop** behavior first so compile/warmup can run cleanly
- Canceling an export tears down the export model/runtime and releases GPU resources without requiring an app close

### mpv Display / Color Path

Both SDR and HDR panes are rendered through embedded **mpv** (d3d11):

- **SDR pane**: tagged as **Rec.709** (`bt.709` / `bt.1886`, full range)
- **HDR pane**: tagged as **BT.2020/PQ** (`bt.2020` / `pq`, full range)
- Output target is **auto-detected by mpv/display path** (no hard-forced target primaries/TRC)
- SDR and HDR panes intentionally use different downscale defaults: SDR uses non-linear `mitchell` downscaling to avoid small-pane bloom around bright text, while HDR keeps linear-light `catmull_rom` downscaling for the HDR presentation path. Override with `HDRTVNET_MPV_SDR_DSCALE`, `HDRTVNET_MPV_SDR_DSCALE_ANTIRING`, `HDRTVNET_MPV_DSCALE`, and `HDRTVNET_MPV_DSCALE_ANTIRING`.

Playback scaling is pane-aware:

- `Resolution` controls model processing size and cache/engine selection, not the final display target.
- `Upscale` is an always-editable preference for the mpv presentation scaler; it is applied whenever the current HDR pane target is larger than the processed frame.
- Fullscreen-with-UI, side-by-side, and windowed playback follow the HDR pane's drawable size instead of assuming the whole monitor is available.
- Hide-UI fullscreen becomes monitor-sized naturally because the HDR pane fills the display.
- In Hide-UI playback, the playhead, `Show UI`, and performance metrics reappear only while the cursor is moving.
- Compare and benchmark preview panes use the same high-quality mpv scaler family without extra CAS sharpening.
- FSR is adapted for the RGB48 playback path by running EASU/RCAS on `MAIN` RGB; if FSR only scales partway to the target, mpv finishes the residual scale with the best configured scaler instead of bilinear.
- Moving or resizing the app or a popped-out HDR view hot-swaps the mpv scale/CAS settings without restarting the model pipeline.

Thesis figure screenshots can be rendered from saved benchmark frames through the same embedded mpv preview path:

```powershell
.\venv\Scripts\python.exe scripts\render_mpv_preview_figures.py --input logs\benchmark_sessions\Thesis --limit 5
```

The script opens a temporary Qt/libmpv render window, feeds `sdr.png`, `hdr_gt.tiff`, and `hdr_convert.tiff` as raw RGB48 frames, and saves mpv `screenshot-to-file window` PNGs plus an optional side-by-side contact sheet under `docs/images/thesis_mpv_figures/`. It does not use FFmpeg for rendering or tone mapping. Use `--input <frame_dir>` for one exact benchmark frame, `--render-size 1920x1080` to capture a specific pane size, `--scale fsr` or `--scale ssim_superres` to test a presentation scaler, and `--png-depth 16` if you want to keep mpv's high-bit-depth screenshot output instead of thesis/PDF-friendly 8-bit PNGs.

### HDR Model Output

HDR playback and export use raw model output with no tensor-side highlight stabilizer, sky deblocker, temporal smoother, or export cleanup layer. This is the correct baseline now that HR/HG are the actual deployment models. Compare snapshots, HDR GT, benchmark images, objective metrics, and exported videos all measure or encode the model output before display-side tone mapping.

### Tone Mapping Behavior

HDR playback in the GUI uses mpv's display pipeline. When `force_hdr_metadata` is enabled, frames are interpreted as **BT.2020 / PQ HDR** within the playback pipeline, as raw frame data does not retain original container metadata

Tone mapping behavior is **display-side only**:

- The app configures mpv with `tone-mapping=spline` and `tone-mapping-param=0.45` for its display pipeline.
- That tone-mapping choice is shared across the mpv-backed playback, compare, and benchmark preview panes, and is **for visualization purposes only**.

Key distinctions:

- **Metrics (PSNR, SSIM, DeltaEITP, HDR-VDP3)** are computed **before display** and operate on the signal domain; they are **not affected by tone mapping**.
- **Exported videos** are encoded as **BT.2020/PQ HDR signals without display-side tone mapping**, preserving the intended HDR luminance range
- **Spline tone mapping is only used to improve visual appearance** (e.g., smoother highlight rolloff, reduced perceived banding) during playback/preview.

For fair comparison:

- When enabled, tone mapping should be applied **equally to both HDR GT and HDR Convert**.
- For benchmarking and thesis results, it is recommended to **disable tone mapping overrides** to avoid altering perceived contrast.

In summary:

- **Metrics → no tone mapping**
- **Export → no tone mapping; raw model output by default**
- **Playback → tone mapping (display-only, for visualization)**

> **Requires** `libmpv-2.dll` in the `src/` folder.
> `setup.bat` and first GUI launch now try to download it automatically.
> Manual fallback: shared Google Drive assets folder above (same folder as
> `HG.pt`).
> Fallback source: [mpv-winbuild](https://sourceforge.net/projects/mpv-player-windows/files/libmpv/)
> (`mpv-dev-x86_64-*-git-*.7z`).

### First Run

NVIDIA and AMD now have different first-run behavior.

**NVIDIA TensorRT**

The first time you play, export, or benchmark a given model/resolution/mode combination, the app checks for a cached TensorRT engine:

`src/models/engines/{model}_{resolution}_{mode}.engine`

INT8 TensorRT modes include a versioned ModelOpt Torch/QDQ suffix in the mode portion of the filename. Mixed INT8 uses the runtime include mask and disables output quantizers by default; Full INT8 turns every ModelOpt Torch quantizer on and disables FP16 tactics when FP16 islands are off.

If the engine is missing, the app loads the selected `.pt` model, exports a temporary ONNX file with the same model/resolution/mode stem, builds a TensorRT engine, saves it, removes the ONNX file, and then runs inference through that engine. Later runs load the `.engine` directly when the cached engine metadata still matches the model, resolution, precision, mode, TensorRT export settings, and CUDA device fingerprint.

Checkpoint source selection is preset-driven. FP32/FP16 modes use `src/models/weights/distilled/hr/HR_qfriendly_spatialmixglobal_fp32.pt` plus `src/models/weights/distilled/hg/HG_qfriendly_directh16_fp32.pt` when HG is enabled. INT8 no-HG modes use the matching TensorRT source under `distilled/hr`; INT8 HG modes use the HR source split from the exact HR+HG composite under `distilled/hr_hg` plus the matching HG source under `distilled/hg`. Untouched original HR/HG checkpoints are kept under `src/models/weights/original/` as reference/manual-experiment assets; they stay hidden from normal precision, benchmark, and compare lists, and are reached only through `Tools -> Use Original HR/HR+HG Model ...`, `--checkpoint-family original`, or explicit CLI `--model` / `--hg-weights` paths.

TensorRT engines are not universal binaries. The same checkpoint and command can be run on other TensorRT-capable NVIDIA GPUs, but each machine should build its own `.engine` because TensorRT chooses tactics for the local GPU architecture, driver, TensorRT version, workspace, and timing cache. Tensor Core GPUs normally benefit most from FP16/INT8. Older or non-Tensor-Core NVIDIA GPUs may still build through TensorRT, but INT8 speedups are not guaranteed and TensorRT may choose DP4A, FP16, or FP32 tactics depending on what the hardware supports. Non-NVIDIA GPUs do not use TensorRT; AMD/CPU use the PyTorch path.

The GUI creates the TensorRT engine only when the user activates a mode whose engine is missing. You can also pre-build NVIDIA engines manually:

```bash
python src/build_tensorrt_engines.py 3840x2160 --precision fp16 --use-hg 1 --benchmark-runs 20
python src/build_tensorrt_engines.py 3840x2160 --precision int8-mixed-qat --use-hg 0 --benchmark-runs 20
python src/build_tensorrt_engines.py 3840x2160 --precision int8-mixed-qat --use-hg 1 --benchmark-runs 20
python src/build_tensorrt_engines.py 3840x2160 --precision int8-full-qat --use-hg 1 --benchmark-runs 10
```

TensorRT performs optimization during engine build time and caches the result. No PyTorch max-autotune warning is shown on NVIDIA.

New TensorRT builds use the highest TensorRT builder optimization level by default and persist a shared timing cache at `src/models/engines/tensorrt_timing.cache`. Existing `.engine` files keep their current tactics until rebuilt. To compare or refresh them, clear the TensorRT cache from the GUI or run the prebuild script with `--force`.

Advanced TensorRT build environment overrides:

- `HDRTVNET_TRT_BUILDER_OPT_LEVEL=0..5` controls TensorRT builder search depth; default is `5`.
- `HDRTVNET_TRT_WORKSPACE_GB=4` controls builder workspace size in GiB.
- `HDRTVNET_TRT_TIMING_CACHE=path|none` changes or disables the shared timing cache.
- `HDRTVNET_TRT_DEDICATED_STREAM=1|0` runs TensorRT enqueue on a non-default CUDA stream; default is `1`.
- `HDRTVNET_TRT_AUX_STREAMS=N` optionally sets TensorRT build-time auxiliary stream count.

The manual prebuild script also exposes `--opt-level`, `--workspace-gb`, `--timing-cache`, `--aux-streams`, `--force-onnx`, and `--benchmark-runs` so NVIDIA test machines can rebuild and report comparable numbers from one command. `--force-onnx` is mainly for clearing stale ONNX files left by older builds.

If the TensorRT engine build or load fails, the error is logged and shown to the user. The app does not silently fall back to PyTorch on NVIDIA.

### CLI mpv Playback Benchmark Recipes

Run these from the repo root after `setup.bat`.

Set your video path once:

```powershell
$VIDEO="D:\path\to\your_sdr_video.mkv"
```

One-line 1080p mpv playback commands:

```powershell
.\venv\Scripts\python.exe src\cli_playback_benchmark.py --checkpoint-family original --video "$VIDEO" --resolutions 1920x1080 --runs fp32 fp16 int8-mixed-ptq int8-full-ptq int8-mixed-qat int8-full-qat --use-hg 1 --duration-s 180 --warmup-frames 120 --sample-interval 120 --display --display-backend mpv --out-root logs\playback_sessions\cli_original_mpv
.\venv\Scripts\python.exe src\cli_playback_benchmark.py --checkpoint-family distilled --video "$VIDEO" --resolutions 1920x1080 --runs fp32 fp16 int8-mixed-ptq int8-full-ptq int8-mixed-qat int8-full-qat --use-hg 1 --duration-s 180 --warmup-frames 120 --sample-interval 120 --display --display-backend mpv --out-root logs\playback_sessions\cli_distilled_mpv
```

Use `--use-hg 0` to benchmark HR/no-HG. Add `2560x1440` or `3840x2160` after `--resolutions` when the source video is at least that large. The script writes GUI-style playback logs under the selected `--out-root` and includes mpv display/feed cost in `render_ms` and total latency.

On NVIDIA, missing TensorRT engines are built on first use before the timed playback window. With the default ModelOpt path, `--trt-qdq-fusion native` still exports explicit Q/DQ and lets TensorRT fuse it natively. If ModelOpt is disabled for legacy implicit INT8 experiments, calibration defaults to the benchmark video unless you pass `--trt-calibration-dataset` or `--trt-calibration-cache`.

**AMD PyTorch**

The first time you play a video at a given resolution/precision/HG combination on AMD, `torch.compile` with `max-autotune` may need to compile Triton kernels. AMD INT8 pre-dequantize mode is part of that PyTorch cache key. This takes **2–5 minutes** and runs in a clean subprocess with a progress dialog.

Compiled PyTorch kernels are **cached to disk**, so:

- subsequent playback on an exact cache hit skips the clean precompile subprocess
- export max-autotune reuses the same compile cache and only compiles cleanly on a real cache miss
- benchmark reuses the same cache when available but does not launch a new compile on a miss; it falls back to eager for that run
- caches are stored in `src/models/compile_cache/` by default, next to the project instead of in AppData
- caches are scoped to the current local project checkout instead of being shared implicitly across different local copies of the repo
- older builds may have left `%LOCALAPPDATA%\HDRTVNetCache\`; current AMD compile caches do not use that location
- if an old or incompatible cache looks "compiled" but would hang warmup, the app can stop early and ask to clear/recompile the current project's cache

PyTorch compile defaults are tuned for fixed-resolution video: `dynamic=False`, `mode=max-autotune`, and two warmup passes. Advanced overrides:

- `HDRTVNET_CACHE_DIR=path` intentionally moves the AMD PyTorch/Triton compile cache to a custom location; unset it to keep caches repo-local.
- `HDRTVNET_COMPILE_DYNAMIC=0|1|auto` controls shape specialization; default is `0`.
- `HDRTVNET_COMPILE_FULLGRAPH=1` can be used for experiments, but default is `0` for compatibility.
- `HDRTVNET_COMPILE_WARMUP_RUNS=N` controls compile warmup passes; default is `2`.
- `HDRTVNET_BENCHMARK_USE_COMPILE_CACHE=1|0` controls whether the benchmark may use an already-present max-autotune cache; default is `1`. Cache misses still run eager.

---

## Installation

### Requirements

- Python 3.12 (setup scripts target 3.12 for all backends)
- Backend-specific PyTorch wheels from the requirement files
- NVIDIA: CUDA 12.6 PyTorch wheels plus TensorRT CUDA 12 bindings/libs and ONNX export dependencies
- AMD: ROCm-Windows 7.2.1 PyTorch stack plus optional HIP SDK for compiled PyTorch kernels
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

- `requirements/requirements-nvidia.txt` -> common deps + CUDA PyTorch (`torch==2.9.1+cu126`, `torchvision==0.24.1`, `torchaudio==2.9.1`) + ONNX/TensorRT engine build/runtime deps
- `requirements/requirements-amd.txt` -> common deps + ROCm-Windows 7.2.1 SDK/PyTorch wheels (`torch==2.9.1+rocm7.2.1`, `torchvision==0.24.1+rocm7.2.1`, `torchaudio==2.9.1+rocm7.2.1`) + `triton-windows`
- `requirements/requirements-common.txt` -> shared app deps only (use with manual CPU PyTorch install)

Equivalent setup scripts:
- `setup.bat` (double-click entry point)
- `scripts/setup.ps1` (auto-detect + override support)
- `run_gui.bat` (double-click GUI launcher)
- `scripts/setup_nvidia.ps1`
- `scripts/setup_amd.ps1`
- `scripts/setup_cpu.ps1`
- `scripts/run_gui.ps1`

**NVIDIA (CUDA, Python 3.12):**
```bash
.\venv\Scripts\python.exe -m pip install --prefer-binary -r requirements/requirements-nvidia.txt
```
NVIDIA uses TensorRT for inference. PyTorch is still required to load `.pt` / `.pth` checkpoints and export temporary ONNX artifacts during first-time model/resolution/mode builds, but Triton is not required for NVIDIA inference.

The NVIDIA requirement file installs PyTorch from the CUDA 12.6 wheel index, then pulls `onnx>=1.16`, `onnxscript>=0.1.0`, `tensorrt_cu12_bindings>=10.0`, and `tensorrt_cu12_libs>=10.0`. The Python import name remains `tensorrt`; the split package names are the NVIDIA CUDA 12 wheel names.

`setup.bat` / `scripts/setup_nvidia.ps1` performs a post-install NVIDIA runtime check:

- NVIDIA CUDA driver DLL
- `torch.cuda`
- `onnx`
- `onnxscript`
- `tensorrt_libs`
- `tensorrt` import from the CUDA 12 bindings wheel
- TensorRT builder creation

A separate CUDA Toolkit/SDK is not required in the AMD HIP SDK sense when the pip wheels provide the needed runtime libraries. If the TensorRT check fails, update the NVIDIA driver first, then rerun setup with a fresh venv. If the wheel-provided TensorRT runtime still cannot import/build, install the matching NVIDIA CUDA Toolkit / TensorRT runtime from NVIDIA and rerun setup.

**AMD ROCm-Windows (Python 3.12):**
```bash
pip install -r requirements/requirements-amd.txt
```
Recommended for best compatibility/performance:
- Install AMD HIP SDK: `https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html`
- Default install path expected by warning checks: `C:\Program Files\AMD\ROCm`

**CPU only:**
```bash
pip install -r requirements/requirements-common.txt
pip install torch torchvision
```

### Backend Setup

**NVIDIA CUDA/TensorRT**: Uses cached TensorRT `.engine` files. Engines are created on demand from the selected PyTorch checkpoint when the matching cache file is missing.

**AMD ROCm-Windows/PyTorch**: Auto-detects HIP SDK for `torch.compile`. `requirements/requirements-amd.txt` already includes `triton-windows`.

---

## Architecture

### Pipeline

NVIDIA:

```
Video Source → GPU Upload → GPU Preprocess → TensorRT Engine → GPU Postprocess → CPU Download / mpv Renderer
```

AMD:

```
Video Source → GPU Upload → GPU Preprocess → torch.compile Model → GPU Postprocess → CPU Download → Renderer
```

### Key Optimizations

- GPU-side preprocessing (BGR→RGB, normalize, permute) and postprocessing (clamp, scale, quantize, RGB→BGR)
- Pre-allocated GPU tensor buffers (zero per-frame allocation)
- Pinned (page-locked) host memory for async H2D/D2H DMA transfers
- `torch.inference_mode()` throughout
- NVIDIA TensorRT engines built on demand and cached per model/resolution/mode; default INT8 uses quant-friendly ModelOpt Torch Q/DQ builds, with legacy native-implicit INT8 calibration-cache support kept for experiments
- AMD PyTorch tensors use contiguous memory format by default, with channels-last available as an opt-in test
- Async video prefetch queue
- mpv fast path: skips GPU→CPU postprocess when mpv handles HDR display

### Precision Modes

| Mode | Description | Compression |
|---|---|---|
| **FP16** | Half-precision quant-friendly distilled HR/HG path | — |
| **FP32** | Full-precision quant-friendly distilled HR/HG path | — |
| **INT8 Full (PTQ)** | W8A8 quantization (HG optional) | ~4.0× vs FP16+HG |
| **INT8 Full (QAT)** | W8A8 + quantization-aware fine-tuning (HG optional) | ~4.0× vs FP16+HG |
| **INT8 Full (QAT) (Film)** | W8A8 + movie-accuracy QAT with stronger FP32/HDR anchoring (HG optional) | ~4.0× vs FP16+HG |
| **INT8 Mixed (PTQ)** | Mixed W8A8/W8A16/FP16-sensitive layers (HG optional) | ~4.0× vs FP16+HG |
| **INT8 Mixed (QAT)** | Mixed W8A8/W8A16/FP16-sensitive layers + quantization-aware fine-tuning (HG optional) | ~4.0× vs FP16+HG |
| **INT8 Mixed (QAT) (Film)** | Mixed movie-accuracy QAT with protected tone-control paths (HG optional) | ~4.0× vs FP16+HG |

The app includes ready-to-use PTQ, QAT, and QAT (Film) INT8 checkpoints. On NVIDIA, these presets route through self-describing quant-friendly TensorRT source checkpoints (`HR_qfriendly_*` / `HG_qfriendly_*`) so the engine builder can recreate the HR/ACGM/LE/HG architecture without relying on environment overrides. The Film variants are separate deployable checkpoints rather than runtime color presets.

### Checkpoint Selection Guide

There is no single checkpoint that is best for every device, movie, and metric. Pick the preset that matches the target hardware, required FPS, and preferred visual response:

| Preset | Recommended use |
|---|---|
| **FP32** | Quant-friendly full-precision reference / teacher / debugging path. It is useful for comparisons and QAT anchoring, but is usually not the practical deployment choice once FP16 or INT8 variants match its quality. |
| **FP16** | Quant-friendly half-precision fallback when TensorRT INT8 is unavailable or when maximum compatibility is more important than checkpoint size. |
| **INT8 Mixed (QAT)** | Current default NVIDIA preset. Mixed precision protects sensitive tone/control layers while QAT recovers accuracy, making it the preferred non-Film INT8 deployment candidate. |
| **INT8 Full (PTQ)** | Fully quantized W8A8 baseline. It can sometimes score surprisingly well because PTQ clipping/precision loss acts like an implicit regularizer, but that behavior is accidental and not directly controllable. |
| **INT8 Full (QAT) / QAT (Film)** | Most moldable INT8 path. Full W8A8 gives QAT more low-bit behavior to reshape, so these variants can be useful when a specific dataset or movie-domain look is preferred. |
| **INT8 Mixed (QAT) (Film)** | Optional movie-domain preset. It keeps the same mixed-precision structure, but nudges the output toward the Film objective. |

For thesis interpretation, treat quantization as more than compression. PTQ can accidentally improve agreement with HDR targets by suppressing FP32 biases, while QAT makes that behavior controllable by optimizing the quantized checkpoint toward a chosen objective. In practice, use whichever checkpoint gives the best FPS and the visual style you prefer on the target runtime.

On AMD, INT8 modes include **pre-dequantization** for GPUs without native INT8 convolution support: INT8 weights are converted to FP16 once at load time, giving native FP16 speed with compressed checkpoint storage. The default `Auto` mode resolves to pre-dequantize-on for AMD.

On NVIDIA, the selected INT8 preset maps to a quant-friendly TensorRT source checkpoint. No-HG uses `src/models/weights/distilled/hr`; HG uses the composite-split HR source in `src/models/weights/distilled/hr_hg` plus the matching HG source in `src/models/weights/distilled/hg`. **ModelOpt Torch** inserts Q/DQ from the same runtime composition used for PTQ, QAT, and QAT (Film). Mixed presets use the current proven include mask (`AGCM.spatial`, `AGCM.global`, `LE.low_in`, `LE.recon_trunk3`, plus the matching HG low/trunk regions when HG is enabled) and keep output quantizers disabled unless explicitly requested. Full INT8 is the strict baseline: selecting an `INT8 Full` preset forces every ModelOpt Torch quantizer on. Eager PyTorch/AMD INT8 deploy checkpoints live separately under `src/models/weights/pytorch_int8/hr` and `src/models/weights/pytorch_int8/hg`.

The default ModelOpt Torch/QDQ TensorRT path does not use shipped TensorRT `.calib` files. ModelOpt performs its export-time calibration from the app's deterministic calibration loop, and TensorRT then consumes the Q/DQ graph directly. Quant-friendly HR/ACGM/LE and HR+HG engines export as single-input graphs when the condition path is structurally unused, reducing runtime binding overhead and avoiding stale condition-tensor metadata.

For the legacy native-implicit INT8 path (`HDRTVNET_TRT_INT8_MODELOPT=0`), calibration still uses SDR/source inputs that match deployment content, not HDR targets. CLI and matrix tools accept `--trt-calibration-dataset` / `--calibration-dataset` for image directories, single images, or text manifests, plus `--trt-calibration-cache` / `--calibration-cache` for a prebuilt TensorRT cache. If no cache or dataset is provided, CLI playback can calibrate from the current video for first-time local engine builds.

Legacy prebuilt calibration caches live under `src/models/tensorrt_calibration/` and are ignored by git. To generate a native-implicit GUI matrix on an NVIDIA machine:

```powershell
py scripts/build_tensorrt_calibration_caches.py `
  --calibration-dataset dataset/train_sdr `
  --calibration-frames 0 `
  --force
```

This builds the 36 INT8 GUI cache files: 6 INT8 presets x 3 GUI resolutions x HG on/off. `--calibration-frames 0` means all available calibration images/frames; positive values such as `256` use deterministic content-ranked image selection for a faster representative subset. Rebuild or delete cached `.engine` files whenever replacing `.calib` files; engine metadata fingerprints the calibration cache contents so changed caches invalidate stale engines automatically.

---

## CLI Mode

For CLI playback, headless benchmarking, or scripted workflows. `src/main.py` uses the shared CLI display path: mpv by default, OpenCV only when requested with `--display-backend opencv`. The displayed video is clean output without the old on-frame metrics overlay; timing is still printed to the console.

```bash
# Default (auto device: TensorRT on NVIDIA, PyTorch on AMD/CPU)
python src/main.py

# AMD/CPU PyTorch only: skip torch.compile
python src/main.py --no-compile

# FP32 precision
python src/main.py --precision fp32

# NVIDIA default fast path: TensorRT INT8 Mixed QAT no-HG
python src/main.py --precision int8-mixed --use-hg 0

# NVIDIA HR+HG distilled path
python src/main.py --precision int8-mixed --use-hg 1

# AMD/CPU eager INT8 checkpoint path
python src/main.py --precision int8-mixed --model src/models/weights/pytorch_int8/hr/HR_int8_mixed_qat.pt --use-hg 0

# Headless benchmark
python src/main.py --no-display --warmup 30 --timing-interval 120 --max-frames 360 --model-stage-timing
```

### Playback Log Batch Benchmark

For unattended runtime logging, use `src/cli_playback_benchmark.py`. This is intentionally a CLI workflow rather than a GUI feature: the GUI `Model Quality Benchmark` is still the main tool for objective quality metrics and preview-based comparisons, while this script is for collecting playback-style FPS/latency logs across several precision/resolution combinations without babysitting the app.

The script writes the same playback-session style files used by the GUI:

- batch summary: `logs/playback_sessions/<timestamp>_<source>_cli_batch/batch_summary.csv`
- per run: `summary.txt`, `session.json`, and `runtime_metrics.csv`
- per-run folders: `<resolution>_<preset>_<hg|nohg>/`

Example from the repo root, using the mpv display path and logging 3 minutes per run:

```cmd
.\venv\Scripts\python.exe src\cli_playback_benchmark.py --video "path\to\video.mp4" --resolutions 1280x720 1920x1080 --runs fp32 fp16 int8-mixed-ptq int8-full-ptq int8-mixed-qat int8-full-qat int8-mixed-qat-film int8-full-qat-film --duration-s 180 --warmup-frames 120 --sample-interval 120 --use-hg 0 --compile-mode max-autotune --display
```

`--display` uses the embedded mpv HDR path by default, including the raw RGB48LE feed/display cost in `render_ms` and total frame latency. The CLI prepends the repo `src/` folder to the Windows DLL search path so `src/libmpv-2.dll` is found the same way the GUI finds it. If mpv is unavailable or you explicitly want the older OpenCV display path, add:

```cmd
--display-backend opencv
```

Useful flags:

| Flag | Description |
|---|---|
| `--video PATH` | Video to benchmark; required |
| `--resolutions WxH ...` | Processing resolutions, e.g. `1280x720 1920x1080` |
| `--runs ...` | Presets: `fp32`, `fp16`, `int8-mixed-ptq`, `int8-full-ptq`, `int8-mixed-qat`, `int8-full-qat`, `int8-mixed-qat-film`, `int8-full-qat-film` |
| `--checkpoint-family distilled\|original` | Checkpoint family to benchmark. `distilled` is the quant-friendly default; `original` uses untouched HR/HG plus matching original INT8 files. |
| `--model PATH` | HR/model checkpoint override for all selected runs |
| `--hg-weights PATH` | HG checkpoint override; use with `--model` for original HR/HG playback benchmarks |
| `--duration-s N` | Timed duration per run after warmup |
| `--warmup-frames N` | Frames skipped before logging stats |
| `--sample-interval N` | Timed frames between console/log samples |
| `--use-hg 1\|0` | Enable/disable HG refinement |
| `--display` | Show processed frames and include display cost |
| `--display-backend mpv\|opencv` | Display backend used with `--display`; default is `mpv` |

<details>
<summary><strong>All CLI Flags</strong></summary>

| Flag | Description |
|---|---|
| `--model PATH` | Model weights path (default: `src/models/weights/distilled/hr/HR_qfriendly_spatialmixglobal_fp32.pt`) |
| `--device auto\|cuda\|cpu` | Device selection (default: auto) |
| `--precision auto\|fp16\|fp32\|int8-full\|int8-mixed` | Inference precision (default: auto → fp16 on GPU) |
| `--compile-mode auto\|default\|reduce-overhead\|max-autotune\|max-autotune-no-cudagraphs` | PyTorch backend only; torch.compile mode (auto = max-autotune) |
| `--force-compile` | AMD PyTorch only; force `torch.compile` on ROCm-Windows when HIP SDK auto-detection fails |
| `--no-compile` | PyTorch backend only; disable `torch.compile` entirely |
| `--channels-last` | PyTorch backend only; force channels_last memory format for testing (AMD defaults to contiguous) |
| `--cuda-graphs` | PyTorch backend only; enable CUDA graph replay for static shapes |
| `--predequantize auto\|on\|off` | AMD PyTorch INT8; pre-dequantize INT8 weights to FP16 at load time (`auto` resolves on for AMD) |
| `--use-hg 1\|0` | Enable HG refinement (1 = on, 0 = off) |
| `--cache-resolution WxH` | AMD PyTorch only; pre-compile Triton kernels for this resolution at startup (default: auto = video resolution) |
| `--prefetch N` | Video reader prefetch queue size (default: 8) |
| `--model-stage-timing` | Report pre/run/post timing breakdown |
| `--no-display` | Headless mode for pure throughput testing |
| `--display-backend mpv\|opencv` | Display backend for `src/main.py`; default is `mpv` |
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

### Weight Layout

The repo uses one naming scheme for all shipped `.pt` weights:

| Folder | Meaning |
|---|---|
| `src/models/weights/original/HR.pt` | untouched upstream HR/ACGM/LE checkpoint |
| `src/models/weights/original/HG.pt` | untouched upstream HG checkpoint, downloaded by setup because it is too large for normal git tracking |
| `src/models/weights/distilled/hr/*.pt` | quant-friendly standalone HR/no-HG TensorRT/FP source checkpoints |
| `src/models/weights/distilled/hr_hg/*.pt` | quant-friendly HR source split from the exact HR+HG composite INT8 checkpoint |
| `src/models/weights/distilled/hg/*.pt` | quant-friendly HG TensorRT/FP source checkpoints |
| `src/models/weights/pytorch_int8/hr/*.pt` | quant-friendly eager PyTorch/AMD HR-only deploy checkpoints |
| `src/models/weights/pytorch_int8/hg/*.pt` | quant-friendly eager PyTorch/AMD HR+HG deploy checkpoints |
| `src/models/weights/original/pytorch_int8/*/*.pt` | untouched original eager PyTorch/AMD INT8 checkpoints |
| `src/models/weights/original/tensorrt/hr/*.pt` | untouched original standalone HR/no-HG TensorRT source checkpoints |
| `src/models/weights/original/tensorrt/hr_hg/*.pt` | untouched original HR source split from the exact original HR+HG composite INT8 checkpoint |
| `src/models/weights/original/tensorrt/hg/*.pt` | untouched original HG TensorRT source checkpoints; these are over 100 MB and stay ignored/release-local |

`hr` means HR/ACGM/LE only. `hg` means the eager deploy checkpoint contains the combined HR+HG PyTorch path. TensorRT keeps HR and HG source checkpoints split so the builder can load HR and, only when `--use-hg 1`, load the matching HG source separately. `hr_hg` is the HR half of the composite checkpoint; it exists because the standalone HR checkpoint and the HR part of the HR+HG checkpoint can differ after PTQ/QAT.

### Engine Creation

The GUI and CLI precision preset picks three paths:

- `model`: eager PyTorch deploy checkpoint, used on AMD/CPU and as the fallback identity for the preset
- `trt_model`: TensorRT HR source checkpoint under `distilled/hr` for no-HG or `distilled/hr_hg` for HG
- `trt_hg_weights`: optional TensorRT HG source checkpoint under `distilled/hg`

On NVIDIA, the TensorRT builder uses `trt_model`, the current resolution, precision, and HG toggle to export a temporary ONNX and cache an `.engine` under `src/models/engines/`. PTQ, QAT, and QAT Film are separate source checkpoints. Mixed INT8 uses the selected mixed mask; Full INT8 is the strict contract path and forces all ModelOpt Torch quantizers on.

Regenerate TensorRT sources after replacing eager INT8 checkpoints:

```powershell
.\venv\Scripts\python.exe scripts\quantize\split_distilled_tensorrt_sources.py
```

That command also regenerates original-family TensorRT sources when `src\models\weights\original\pytorch_int8` is present. The generated `original\tensorrt\hg` files are local/release assets because each original HG source is over the normal GitHub file-size limit.

### Rebuild PTQ Checkpoints

Default `--use-hg 1` writes to `pytorch_int8/hg/HR_HG_*.pt`. Passing `--use-hg 0` writes to `pytorch_int8/hr/HR_*.pt` unless you override `--output`.

```powershell
.\venv\Scripts\python.exe scripts\quantize\quantize_int8_mixed.py --device cuda --sensitivity-device cuda --calibration-device cuda --calibration-dir dataset\train_sdr --num-calibrate 0 --num-validate 16
.\venv\Scripts\python.exe scripts\quantize\quantize_int8_mixed.py --use-hg 0 --device cuda --sensitivity-device cuda --calibration-device cuda --calibration-dir dataset\train_sdr --num-calibrate 0 --num-validate 16

.\venv\Scripts\python.exe scripts\quantize\quantize_int8_full.py --device cuda --calibration-device cuda --calibration-dir dataset\train_sdr --num-calibrate 0 --num-validate 32
.\venv\Scripts\python.exe scripts\quantize\quantize_int8_full.py --use-hg 0 --device cuda --calibration-device cuda --calibration-dir dataset\train_sdr --num-calibrate 0 --num-validate 32
```

### Rebuild QAT Checkpoints

QAT optimizes toward FP32/HDR accuracy. QAT Film uses the same quantized composition but is trained for movie-domain accuracy. PTQ, QAT, and QAT Film must keep the same precision composition inside a family; only the trained weights/objective change.

The QAT recipes expect paired SDR/HDR data staged locally as `dataset/train_sdr`, `dataset/train_hdr`, `dataset/test_sdr`, and `dataset/test_hdr`. These recipes use HDRTV1K by `chxy95`: [huggingface.co/datasets/chxy95/HDRTV1K](https://huggingface.co/datasets/chxy95/HDRTV1K). Follow the dataset page terms when downloading or redistributing the data.

Mixed INT8 is an explicit W8A8 layer recipe:

| Family | Mode | W8A8 file | Composition |
|---|---|---|---|
| Distilled/q-friendly | HR/no-HG | `configs/qat_layouts/spatialmixglobalh8w64_nohg_mixed_w8a8.txt` | 23 W8A8, 1 W8A16 |
| Distilled/q-friendly | HR+HG | `configs/qat_layouts/spatialmixglobalh8w64_hg_mixed_w8a8.txt` | 32 W8A8, 2 W8A16 |
| Original | HR/no-HG | `configs/qat_layouts/original_nohg_mixed_w8a8.txt` | 29 W8A8, 78 W8A16, 21 FP16 |
| Original | HR+HG | `configs/qat_layouts/original_hg_composite_mixed_w8a8.txt` | 51 W8A8, 74 W8A16, 24 FP16 |

The distilled mixed path was selected from the fastest stable TensorRT layout: quantize the dense HR/ACGM/LE spatial/global/recon trunk regions and the direct HG low/trunk region, while keeping tiny control/output-sensitive layers out of W8A8.

```text
HR/no-HG: AGCM.spatial;AGCM.global;LE.low_in;LE.recon_trunk3
HR+HG:    base.AGCM.spatial;base.AGCM.global;base.LE.low_in;base.LE.recon_trunk3;hg.low_in;hg.trunk
```

Original mixed comes from the legacy sensitivity/protection sweep recovered from the old `Ensemble_AGCM_LE_int8_*` checkpoint metadata. Full INT8 is not a mask: Full means every quantizable conv/linear in that checkpoint family is W8A8, even if it is slower than Mixed.

When inspecting TensorRT logs, W8A16 layers may not appear as enabled Q/DQ quantizers because they keep FP16 activations. The full mixed recipe is still the table above; TensorRT speed coverage is mostly the W8A8 subset.

QAT defaults:

| Variant | Main settings |
|---|---|
| Mixed QAT | `epochs=10`, `lr=2e-6`, `crop-size=384`, `batch-size=1`, `max-long-edge=1080`, `teacher-source=fp32`, `monitor-score=hybrid`, `teacher-loss-weight=0.68`, `highlight-teacher-weight=0.38`, `dark-teacher-weight=0.36`, `protect-agcm-controls=1`, `protect-sft-controls=1`, `fp16-sensitive-layers=1`, `early-stop-patience=4` |
| Full QAT | `epochs=6`, `lr=1.5e-6`, `teacher-loss-weight=0.65`, `highlight-teacher-weight=0.35`, `dark-teacher-weight=0.34`, `freeze-sensitive-layers=1`, `freeze-sft-controls=1`, `early-stop-patience=3` |
| Mixed QAT Film | same Mixed composition, but `teacher-loss-weight=0.72`, `teacher-luma-weight=0.12`, `teacher-chroma-weight=0.07`, `highlight-teacher-weight=0.40`, `dark-teacher-weight=0.38`, and source-chroma/source-shadow auxiliary weights set to `0` |
| Full QAT Film | same Full composition, but `lr=2e-6`, `teacher-loss-weight=0.70`, `teacher-luma-weight=0.11`, `teacher-chroma-weight=0.065`, `highlight-teacher-weight=0.38`, `dark-teacher-weight=0.36`, and source-chroma/source-shadow auxiliary weights set to `0` |

For original-family QAT, use the same recipe with `src\models\weights\original\pytorch_int8\...` input/output paths. After replacing eager INT8 checkpoints, regenerate TensorRT sources and validate parity:

```powershell
.\venv\Scripts\python.exe scripts\quantize\split_distilled_tensorrt_sources.py
.\venv\Scripts\python.exe scripts\validate_tensorrt_sources.py --skip-onnx --resolution 128x128
```

The quick-sync target is `metadata=True pt_exact=True` for every source checkpoint.

```powershell
.\venv\Scripts\python.exe scripts\quantize\quantize_int8_mixed_qat.py --device cuda --ptq-checkpoint src\models\weights\pytorch_int8\hg\HR_HG_int8_mixed.pt --output src\models\weights\pytorch_int8\hg\HR_HG_int8_mixed_qat.pt --sdr-dir dataset\train_sdr --hdr-dir dataset\train_hdr --val-sdr-dir dataset\test_sdr --val-hdr-dir dataset\test_hdr --teacher-source fp32 --epochs 10 --lr 2e-6
.\venv\Scripts\python.exe scripts\quantize\quantize_int8_mixed_qat.py --use-hg 0 --device cuda --ptq-checkpoint src\models\weights\pytorch_int8\hr\HR_int8_mixed.pt --output src\models\weights\pytorch_int8\hr\HR_int8_mixed_qat.pt --sdr-dir dataset\train_sdr --hdr-dir dataset\train_hdr --val-sdr-dir dataset\test_sdr --val-hdr-dir dataset\test_hdr --teacher-source fp32 --epochs 10 --lr 2e-6

.\venv\Scripts\python.exe scripts\quantize\quantize_int8_full_qat.py --device cuda --ptq-checkpoint src\models\weights\pytorch_int8\hg\HR_HG_int8_full.pt --output src\models\weights\pytorch_int8\hg\HR_HG_int8_full_qat.pt --sdr-dir dataset\train_sdr --hdr-dir dataset\train_hdr --val-sdr-dir dataset\test_sdr --val-hdr-dir dataset\test_hdr --teacher-source fp32 --epochs 6 --lr 1.5e-6
.\venv\Scripts\python.exe scripts\quantize\quantize_int8_full_qat.py --use-hg 0 --device cuda --ptq-checkpoint src\models\weights\pytorch_int8\hr\HR_int8_full.pt --output src\models\weights\pytorch_int8\hr\HR_int8_full_qat.pt --sdr-dir dataset\train_sdr --hdr-dir dataset\train_hdr --val-sdr-dir dataset\test_sdr --val-hdr-dir dataset\test_hdr --teacher-source fp32 --epochs 6 --lr 1.5e-6
```

### AMD/CPU Eager INT8

AMD and CPU do not use TensorRT source files. They load the eager checkpoints from `pytorch_int8/hr` or `pytorch_int8/hg`. On AMD, `--predequantize auto` converts INT8 weights to FP16 once at load time when native INT8 convolution is not beneficial.

```powershell
.\venv\Scripts\python.exe src\main.py --precision int8-mixed --model src\models\weights\pytorch_int8\hr\HR_int8_mixed_qat.pt --use-hg 0 --predequantize auto
.\venv\Scripts\python.exe src\main.py --precision int8-mixed --model src\models\weights\pytorch_int8\hg\HR_HG_int8_mixed_qat.pt --use-hg 1 --predequantize auto
```

</details>

---

## Platform Notes

| Feature | NVIDIA (CUDA) | AMD (ROCm) | CPU |
|---|---|---|---|
| Inference backend | TensorRT only | PyTorch | PyTorch |
| Engine/cache behavior | On-demand `.engine` build + cached load | `torch.compile` cache when enabled | N/A |
| torch.compile | Not used | Auto (Windows: needs HIP SDK) | Not supported |
| FP16 inference | ✅ | ✅ | Fallback to FP32 |
| INT8 quantization | Quant-friendly source checkpoints exported through TensorRT ModelOpt Torch Q/DQ by default; legacy native-implicit INT8 with `.calib` support remains available for experiments | ✅ (compression/pre-dequantized FP16 path) | ✅ (compression only) |
| CUDA graphs | Not used | ✅ | N/A |
| channels_last | Not used in TensorRT engine runtime | Opt-in on AMD PyTorch | N/A |

### TensorRT Engine Cache (NVIDIA)

| Scenario | Behavior |
|---|---|
| First run for model/resolution/mode | Load checkpoint, export temporary `.onnx`, build `.engine`, remove `.onnx` |
| Cached model/resolution/mode | Load `.engine` directly |
| Different model/resolution/mode | Build a new `.engine` once |
| Build/load failure | Log and inform user; no NVIDIA PyTorch fallback |
| Manual clear | `Tools -> Clear TensorRT Engine Cache ...` |
| Live size metric | GUI and playback logs keep the `Checkpoint: ... MB` metric label; NVIDIA reports the active cached `.engine` size, while AMD/CPU report the selected `.pt` / `.pth` checkpoint size |

Manual engine prebuild:

```bash
python src/build_tensorrt_engines.py 1920x1080 1280x720 --precision fp16
python src/build_tensorrt_engines.py 1920x1080 --precision int8-full --use-hg 0
```

### PyTorch Compile Cache (AMD)

| Scenario | Time |
|---|---|
| First run at a resolution | 2–5 minutes |
| Cached resolution | ~5–10 seconds |
| Different resolution | 2–5 minutes (one-time) |

You can also pre-compile AMD PyTorch kernels manually:
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
- QAT training/validation recipes use HDRTV1K by `chxy95`: [huggingface.co/datasets/chxy95/HDRTV1K](https://huggingface.co/datasets/chxy95/HDRTV1K). The dataset page lists the license as MIT; review that page before redistributing data.

---

## Academic Context

This repository is the implementation component of an undergraduate thesis focused on real-time GPU-accelerated HDR reconstruction with precision-aware optimization.
