# HDR Real-Time Video Pipeline

Real-time SDR-to-HDR reconstruction pipeline built for an undergraduate thesis on mixed-precision inference.

Current focus:
- ONNX Runtime inference
- DirectML acceleration on AMD/Windows
- FP32 vs FP16 performance analysis

## Highlights
- Modular pipeline (`decode -> preprocess -> infer -> postprocess -> render`)
- Dual-input HDRTVNet ONNX execution (`input`, `condition`)
- DirectML-first provider selection with CPU fallback
- Runtime timing output for benchmarking

## Requirements
- Windows 11
- Python 3.10+
- AMD GPU with DirectX 12 drivers (for DirectML path)

## Install
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run
Default run:
```bash
python src/main.py
```

Useful flags:
```bash
python src/main.py --video input.mp4 --model hdrtvnet_fp16.onnx --provider auto
python src/main.py --provider dml
python src/main.py --provider cpu
python src/main.py --no-display --timing-interval 120
python src/main.py --ort-profile
```

## Benchmark Notes
- Use `--no-display` to measure throughput without GUI overhead.
- Timing output fields:
  - `decode`: video read time
  - `resize`: pre-inference resize time
  - `infer`: preprocess + ONNX inference + postprocess
  - `render`: display overlay + window update
- Keep input video, resolution, and drivers fixed between FP32/FP16 runs.

## Export ONNX
```bash
python export_onnx_fp32.py
python export_onnx_fp16.py
```

## Project Layout
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

## Thesis Context
This repository is the implementation framework for evaluating precision-performance tradeoffs in real-time HDR reconstruction, including FP32, FP16, and later INT8/mixed-precision paths.
