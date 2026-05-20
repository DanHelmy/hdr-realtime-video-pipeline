"""
TensorRT-specific mixed INT8 PTQ for HDRTVNet++.

This keeps the exact W8A8 / W8A16 / FP16 layer assignment from an existing
mixed PT checkpoint, then recalibrates only the W8A8 activation scales in a
TensorRT CUDA-friendly format:

  - Same layer mask as the source PT checkpoint.
  - W8A8 activations are symmetric signed INT8 with zero_point=0.
  - W8A16 layers remain weight-only and export as FP TensorRT ops.
  - FP16 layers remain native FP16/FP32 modules.

The output checkpoint is intended for TensorRT Q/DQ export/build, not for the
AMD/ROCm runtime path.

Usage
-----
    python scripts/quantize/quantize_int8_mixed_tensorrt.py ^
      --source-checkpoint src/models/weights/Ensemble_AGCM_LE_int8_mixed_nohg.pt ^
      --output src/models/weights/Ensemble_AGCM_LE_int8_mixed_nohg_trt.pt ^
      --use-hg 0 --calibration-video path/to/video.mp4 ^
      --engine-size 1280x720 --export-onnx
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from models.hdrtvnet_modules.Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE
from models.hdrtvnet_modules.HG_Composite_arch import HG_Composite
from models.hdrtvnet_torch import (
    W8A8Conv2d,
    W8A8Linear,
    W8Conv2d,
    W8Linear,
    _export_tensorrt_onnx_from_model,
    _quantize_model_mixed_v2,
    calibrate_w8a8,
    tensorrt_mode_name,
    tensorrt_onnx_path,
)


def _weight(name: str) -> str:
    return os.path.join(_REPO_ROOT, "src", "models", "weights", name)


def _parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = str(text).lower().split("x", 1)
        return int(w), int(h)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid size '{text}', expected WxH such as 1280x720."
        ) from exc


def _clean_state_dict(state: dict) -> dict:
    return {(k[7:] if str(k).startswith("module.") else k): v for k, v in state.items()}


def load_fp32_model(model_path: str, hg_weights: str, use_hg: bool) -> nn.Module:
    if use_hg:
        model = HG_Composite(
            classifier="color_condition",
            cond_c=6,
            in_nc=3,
            out_nc=3,
            nf=32,
            act_type="relu",
            weighting_network=False,
            hg_nf=64,
            mask_r=0.75,
        )
    else:
        model = Ensemble_AGCM_LE(
            classifier="color_condition",
            cond_c=6,
            in_nc=3,
            out_nc=3,
            nf=32,
            act_type="relu",
            weighting_network=False,
        )

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    cleaned = _clean_state_dict(state)
    if use_hg:
        model.base.load_state_dict(cleaned, strict=True)
        hg_state = torch.load(hg_weights, map_location="cpu")
        if isinstance(hg_state, dict) and "state_dict" in hg_state:
            hg_state = hg_state["state_dict"]
        model.hg.load_state_dict(hg_state, strict=True)
    else:
        model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def load_source_mask(path: str) -> tuple[dict, list[str], list[str]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Not a checkpoint dictionary: {path}")
    if checkpoint.get("quantization") != "w8a8_mixed":
        raise ValueError(
            "Source checkpoint must be a mixed INT8 checkpoint "
            f"(quantization=w8a8_mixed), got {checkpoint.get('quantization')!r}."
        )
    w8a8_layers = list(checkpoint.get("w8a8_layers") or [])
    if not w8a8_layers:
        raise ValueError(
            "Source checkpoint does not contain an explicit w8a8_layers mask."
        )
    fp16_layers = list(checkpoint.get("fp16_layers") or [])
    return checkpoint, w8a8_layers, fp16_layers


def _load_image_paths(calib_dir: str, max_items: int) -> list[str]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(calib_dir, pattern)))
    paths = sorted(dict.fromkeys(paths))
    if max_items > 0:
        paths = paths[:max_items]
    return paths


def _frame_to_tensor_bgr(frame_bgr: np.ndarray, width: int | None, height: int | None) -> torch.Tensor:
    if width and height and (frame_bgr.shape[1] != width or frame_bgr.shape[0] != height):
        frame_bgr = cv2.resize(frame_bgr, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    else:
        h, w = frame_bgr.shape[:2]
        new_w = max(8, int(round(w / 8)) * 8)
        new_h = max(8, int(round(h / 8)) * 8)
        if new_w != w or new_h != h:
            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)


def load_calibration_tensors(
    *,
    calibration_dir: str | None,
    calibration_video: str | None,
    num_calibrate: int,
    width: int | None,
    height: int | None,
) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    limit = max(0, int(num_calibrate))

    if calibration_video:
        cap = cv2.VideoCapture(calibration_video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open calibration video: {calibration_video}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0 and limit > 0:
            indices = np.linspace(0, max(0, total - 1), limit, dtype=np.int64)
        else:
            indices = np.arange(limit if limit > 0 else total, dtype=np.int64)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                tensors.append(_frame_to_tensor_bgr(frame, width, height))
        cap.release()

    if calibration_dir and (limit <= 0 or len(tensors) < limit):
        remaining = 0 if limit <= 0 else max(0, limit - len(tensors))
        paths = _load_image_paths(calibration_dir, remaining)
        for path in paths:
            frame = cv2.imread(path)
            if frame is not None:
                tensors.append(_frame_to_tensor_bgr(frame, width, height))

    return tensors


def prepare_model_input(img_tensor: torch.Tensor, device: torch.device, dtype: torch.dtype):
    img_dev = img_tensor.to(device=device, dtype=dtype)
    try:
        cond = F.interpolate(
            img_dev,
            scale_factor=0.25,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        cond = F.interpolate(
            img_dev,
            scale_factor=0.25,
            mode="bicubic",
            align_corners=False,
        )
    return img_dev, cond


def _print_composition(model: nn.Module) -> None:
    w8a8 = []
    w8 = []
    fp = []
    for name, module in model.named_modules():
        if isinstance(module, (W8A8Conv2d, W8A8Linear)):
            w8a8.append(name)
        elif isinstance(module, (W8Conv2d, W8Linear)):
            w8.append(name)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            fp.append(name)
    print(f"  Exact W8A8 layers: {len(w8a8)}")
    print(f"  Exact W8A16 layers: {len(w8)}")
    print(f"  Exact FP layers: {len(fp)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recalibrate an existing mixed PTQ checkpoint for NVIDIA TensorRT "
            "while preserving its exact layer assignment."
        )
    )
    parser.add_argument(
        "--source-checkpoint",
        default=_weight("Ensemble_AGCM_LE_int8_mixed_nohg.pt"),
        help="Existing mixed PT checkpoint whose layer mask is reused exactly.",
    )
    parser.add_argument(
        "--model",
        default=_weight("Ensemble_AGCM_LE.pth"),
        help="FP32/FP16 base model weights used to regenerate quantized weights.",
    )
    parser.add_argument(
        "--output",
        default=_weight("Ensemble_AGCM_LE_int8_mixed_nohg_trt.pt"),
        help="Output TensorRT-calibrated mixed checkpoint.",
    )
    parser.add_argument(
        "--hg-weights",
        default=_weight("HG_weights.pth"),
        help="HG weights path when --use-hg 1.",
    )
    parser.add_argument("--use-hg", default="0", choices=["0", "1"])
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--calibration-video", default=None)
    parser.add_argument("--num-calibrate", type=int, default=32)
    parser.add_argument("--calibration-method", default="percentile", choices=["max", "percentile"])
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--max-calib-samples", type=int, default=200000)
    parser.add_argument(
        "--engine-size",
        type=_parse_size,
        default=None,
        help="Optional TensorRT export size WxH. Also resizes calibration frames.",
    )
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-output", default=None)
    parser.add_argument(
        "--mode-name",
        default=None,
        help="TensorRT mode/cache token. Default: int8-mixed_trt_nohg/hg.",
    )
    parser.add_argument(
        "--qdq-fusion",
        default="none",
        choices=["none", "add"],
        help="Experimental TensorRT Q/DQ placement for exported ONNX.",
    )
    args = parser.parse_args()

    use_hg = str(args.use_hg).strip() != "0"
    compute_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cpu":
        compute_dtype = torch.float32

    if args.export_onnx and device.type != "cuda":
        raise RuntimeError("--export-onnx requires CUDA/TensorRT export on this project path.")

    source_ckpt, w8a8_layers, fp16_layers = load_source_mask(args.source_checkpoint)
    print(f"Source checkpoint: {args.source_checkpoint}")
    print(f"  Keeping exact W8A8 mask: {len(w8a8_layers)} layers")
    print(f"  Keeping exact FP16 mask: {len(fp16_layers)} layers")
    if source_ckpt.get("activation_quant") != "symmetric":
        print("  Source activation quantization is asymmetric; TensorRT output will be symmetric.")

    print(f"Loading base model: {args.model}")
    model = load_fp32_model(args.model, args.hg_weights, use_hg)
    _quantize_model_mixed_v2(
        model,
        compute_dtype=compute_dtype,
        w8a8_layers=w8a8_layers,
        fp16_layers=fp16_layers,
        asymmetric=False,
    )
    _print_composition(model)
    model = model.to(dtype=compute_dtype, device=device).eval()

    width = height = None
    if args.engine_size is not None:
        width, height = args.engine_size

    tensors = load_calibration_tensors(
        calibration_dir=args.calibration_dir,
        calibration_video=args.calibration_video,
        num_calibrate=args.num_calibrate,
        width=width,
        height=height,
    )
    if not tensors:
        raise RuntimeError(
            "No calibration samples loaded. Pass --calibration-video or --calibration-dir."
        )
    print(f"Calibrating symmetric W8A8 activation scales with {len(tensors)} sample(s)")
    calibration_inputs = [
        prepare_model_input(tensor, device, compute_dtype) for tensor in tensors
    ]
    calibrate_w8a8(
        model,
        calibration_inputs,
        method=args.calibration_method,
        percentile=args.percentile,
        percentile_low=0.0,
        max_samples=args.max_calib_samples,
    )

    save_data = {
        "state_dict": model.state_dict(),
        "compute_dtype": str(compute_dtype),
        "quantization": "w8a8_mixed",
        "activation_quant": "symmetric",
        "w8a8_layers": w8a8_layers,
        "fp16_layers": fp16_layers,
        "selection_mode": "tensorrt_exact_pt_mask",
        "source_checkpoint": os.path.abspath(args.source_checkpoint),
        "calibration_method": args.calibration_method,
        "calibration_percentile": args.percentile,
        "calibration_samples": len(tensors),
        "tensorrt_calibrated": True,
        "architecture": {
            "classifier": "color_condition",
            "cond_c": 6,
            "in_nc": 3,
            "out_nc": 3,
            "nf": 32,
            "act_type": "relu",
            "weighting_network": False,
            "use_hg": use_hg,
            "hg_nf": 64,
            "mask_r": 0.75,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(save_data, args.output)
    print(f"Saved TensorRT-calibrated checkpoint: {args.output}")

    if args.export_onnx:
        if args.engine_size is None:
            raise RuntimeError("--export-onnx requires --engine-size WxH.")
        base_mode = args.mode_name or f"int8-mixed_trt_{'hg' if use_hg else 'nohg'}"
        trt_mode = tensorrt_mode_name(
            "int8-mixed",
            base_mode,
            predequantize=False,
            qdq_fusion=args.qdq_fusion,
        )
        onnx_path = args.onnx_output or tensorrt_onnx_path(
            args.output,
            int(width),
            int(height),
            trt_mode,
        )
        _export_tensorrt_onnx_from_model(
            model=model,
            onnx_path=onnx_path,
            width=int(width),
            height=int(height),
            dtype=compute_dtype,
            device=device,
            precision="int8-mixed",
            flat_model=False,
            qdq_fusion=args.qdq_fusion,
        )
        print(f"Exported TensorRT Q/DQ ONNX: {onnx_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
