"""Validate TensorRT source checkpoints before building engines.

This is intentionally runnable on non-NVIDIA machines. It checks that the
TensorRT source checkpoints preserve the original PyTorch checkpoint behavior,
exports TensorRT-style explicit-Q/DQ ONNX graphs, runs them with ONNX Runtime,
and writes visual comparison sheets for quick inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
_WEIGHTS = _SRC / "models" / "weights"
_TRT_SOURCES = _WEIGHTS / "tensorrt_sources"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.hdrtvnet_torch import (  # noqa: E402
    HDRTVNetTorch,
    _export_tensorrt_onnx_from_model,
    _resolve_tensorrt_qdq_fusion,
    tensorrt_source_checkpoint_validation_error,
)


def _configure_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _parse_resolution(text: str) -> tuple[int, int]:
    try:
        w, h = str(text).lower().split("x", 1)
        return int(w), int(h)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution '{text}', expected WxH"
        ) from exc


def _default_image() -> Path:
    for rel in (
        "dataset/test_sdr/001.png",
        "dataset/train_sdr/001.png",
        "dataset/test_hdr/001.png",
    ):
        path = _ROOT / rel
        if path.is_file():
            return path
    raise FileNotFoundError("No default dataset image found")


def _default_checkpoints() -> list[Path]:
    return sorted(
        p
        for p in _WEIGHTS.glob("Ensemble_AGCM_LE_int8_*.pt")
        if p.is_file() and p.parent.name != "tensorrt_sources"
    )


def _source_path(checkpoint: Path) -> Path:
    return _TRT_SOURCES / checkpoint.name


def _precision_from_name(path: Path) -> str:
    name = path.name.lower()
    if "_full" in name:
        return "int8-full"
    if "_mixed" in name:
        return "int8-mixed"
    raise ValueError(f"Cannot infer precision from {path.name}")


def _expected_counts(checkpoint: dict) -> dict[str, int]:
    arch = checkpoint.get("architecture", {})
    total = 149 if bool(arch.get("use_hg", True)) else 128
    quantization = str(checkpoint.get("quantization") or "")
    fp16 = len(checkpoint.get("fp16_layers") or [])
    if quantization == "w8a8_full":
        w8a8 = total
    elif quantization == "w8a8_mixed":
        w8a8 = len(checkpoint.get("w8a8_layers") or [])
    else:
        w8a8 = 0
    return {
        "total": int(total),
        "w8a8": int(w8a8),
        "w8a16": int(max(0, total - w8a8 - fp16)),
        "fp16": int(fp16),
    }


def _load_sample_frame(args) -> np.ndarray:
    if args.video:
        path = Path(args.video)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(args.frame_index)))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {args.frame_index} from {path}")
        return frame

    path = Path(args.image) if args.image else _default_image()
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.integer):
            max_value = float(np.iinfo(image.dtype).max)
            image = np.clip(image.astype(np.float32) * (255.0 / max_value), 0, 255)
        else:
            image = np.clip(image.astype(np.float32) * 255.0, 0, 255)
        image = image.astype(np.uint8)
    return image


def _resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame.copy()
    interp = cv2.INTER_AREA if frame.shape[1] > width or frame.shape[0] > height else cv2.INTER_CUBIC
    return cv2.resize(frame, (width, height), interpolation=interp)


def _as_tensor(output) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.detach().float().cpu()


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a - b).abs()
    mse = torch.mean((a - b) ** 2).item()
    mae = diff.mean().item()
    max_abs = diff.max().item()
    ref = a.abs().mean().clamp_min(1e-8).item()
    psnr = 99.0 if mse <= 1e-20 else -10.0 * math.log10(mse)
    return {
        "mae": float(mae),
        "max_abs": float(max_abs),
        "rmse": float(math.sqrt(max(mse, 0.0))),
        "psnr": float(psnr),
        "rel_mae": float(mae / ref),
    }


def _image_metrics(a_bgr: np.ndarray, b_bgr: np.ndarray) -> dict[str, float]:
    a = a_bgr.astype(np.float32)
    b = b_bgr.astype(np.float32)
    diff = np.abs(a - b)
    mse = float(np.mean((a - b) ** 2))
    psnr = 99.0 if mse <= 1e-12 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return {
        "u8_mae": float(np.mean(diff)),
        "u8_max_abs": float(np.max(diff)),
        "u8_psnr": float(psnr),
    }


def _postprocess(processor: HDRTVNetTorch, tensor: torch.Tensor) -> np.ndarray:
    return processor.postprocess(tensor.detach().clone())


def _label_panel(panel: np.ndarray, label: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        label,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _write_visual_sheet(
    path: Path,
    *,
    input_bgr: np.ndarray,
    original_bgr: np.ndarray,
    source_bgr: np.ndarray,
    onnx_bgr: np.ndarray,
    title: str,
) -> None:
    diff = np.abs(original_bgr.astype(np.int16) - onnx_bgr.astype(np.int16))
    diff_gray = np.clip(diff.mean(axis=2) * 8.0, 0, 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_gray, cv2.COLORMAP_INFERNO)
    panels = [
        _label_panel(input_bgr, "input"),
        _label_panel(original_bgr, "original pt"),
        _label_panel(source_bgr, "trt source pt"),
        _label_panel(onnx_bgr, "onnx qdq"),
        _label_panel(diff_color, "abs diff x8"),
    ]
    sheet = np.concatenate(panels, axis=1)
    cv2.rectangle(sheet, (0, sheet.shape[0] - 28), (sheet.shape[1], sheet.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        sheet,
        title[:180],
        (8, sheet.shape[0] - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def _run_onnx(onnx_path: Path, tensor: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError(
            "onnxruntime is required for ONNX output validation. "
            "Install it with: python -m pip install onnxruntime"
        ) from exc

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    outputs = session.run(
        None,
        {
            "input": tensor.detach().cpu().numpy(),
            "cond": cond.detach().cpu().numpy(),
        },
    )
    return torch.from_numpy(outputs[0]).float()


def _onnx_counts(path: Path) -> dict[str, int]:
    import onnx
    from collections import Counter

    model = onnx.load(path)
    counts = Counter(node.op_type for node in model.graph.node)
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers = {}
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            consumers.setdefault(input_name, []).append((node, index))
    qdq_casts = 0
    compute_ops = {"Conv", "Gemm", "MatMul"}
    for node in model.graph.node:
        if node.op_type != "Cast" or not node.input or not node.output:
            continue
        producer = producers.get(node.input[0])
        node_consumers = consumers.get(node.output[0], [])
        if producer is not None and producer.op_type == "DequantizeLinear":
            if any(consumer.op_type in compute_ops for consumer, _ in node_consumers):
                qdq_casts += 1
        if any(consumer.op_type == "QuantizeLinear" for consumer, _ in node_consumers):
            qdq_casts += 1
    return {
        "onnx_quantize_linear": int(counts.get("QuantizeLinear", 0)),
        "onnx_dequantize_linear": int(counts.get("DequantizeLinear", 0)),
        "onnx_conv": int(counts.get("Conv", 0)),
        "onnx_gemm": int(counts.get("Gemm", 0)),
        "onnx_add": int(counts.get("Add", 0)),
        "onnx_mul": int(counts.get("Mul", 0)),
        "onnx_cast": int(counts.get("Cast", 0)),
        "onnx_qdq_cast": int(qdq_casts),
    }


def _validate_one(
    checkpoint: Path,
    frame_bgr: np.ndarray,
    output_dir: Path,
    args,
) -> dict[str, object]:
    source = _source_path(checkpoint)
    if not source.is_file():
        raise FileNotFoundError(f"TensorRT source missing: {source}")

    original_ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_ckpt = torch.load(source, map_location="cpu", weights_only=False)
    counts = _expected_counts(original_ckpt)
    source_counts = _expected_counts(source_ckpt)
    use_hg = bool(original_ckpt.get("architecture", {}).get("use_hg", True))
    precision = _precision_from_name(checkpoint)

    source_validation_error = tensorrt_source_checkpoint_validation_error(
        str(source),
        str(checkpoint),
    )
    metadata_ok = (
        source_validation_error is None
        and source_ckpt.get("target_backend") == "tensorrt"
        and source_ckpt.get("tensorrt_source_checkpoint") is True
        and source_ckpt.get("checkpoint_format") == "portable_fake_quant_v1"
        and source_ckpt.get("state_format") == "native_fp32"
        and counts == source_counts
        and len(source_ckpt.get("activation_qparams") or {}) == counts["w8a8"]
    )

    original_processor = HDRTVNetTorch(
        str(checkpoint),
        device="cpu",
        precision=precision,
        compile_model=False,
        predequantize="off",
        use_hg=use_hg,
        warmup_passes=0,
    )
    source_processor = HDRTVNetTorch(
        str(source),
        device="cpu",
        precision=precision,
        compile_model=False,
        predequantize="off",
        use_hg=use_hg,
        warmup_passes=0,
    )

    tensor, cond = original_processor.preprocess(frame_bgr)
    with torch.inference_mode():
        original_out = _as_tensor(original_processor.model((tensor, cond)))
        source_out = _as_tensor(source_processor.model((tensor, cond)))

    pt_metrics = _metrics(original_out, source_out)
    original_bgr = _postprocess(original_processor, original_out)
    source_bgr = _postprocess(source_processor, source_out)
    pt_u8_metrics = _image_metrics(original_bgr, source_bgr)

    row: dict[str, object] = {
        "checkpoint": checkpoint.name,
        "source": source.name,
        "precision": precision,
        "use_hg": bool(use_hg),
        "metadata_ok": bool(metadata_ok),
        "metadata_error": source_validation_error or "",
        "activation_quant": source_ckpt.get("activation_quant"),
        "source_activation_quant": source_ckpt.get("source_activation_quant"),
        **counts,
        **{f"pt_{k}": v for k, v in pt_metrics.items()},
        **{f"pt_{k}": v for k, v in pt_u8_metrics.items()},
    }

    if not args.skip_onnx:
        effective_qdq_fusion = _resolve_tensorrt_qdq_fusion(precision, args.qdq_fusion)
        row["qdq_fusion"] = effective_qdq_fusion
        mode = f"{checkpoint.stem}_validation_{args.resolution}_{effective_qdq_fusion}"
        onnx_path = output_dir / "onnx" / f"{mode}.onnx"
        export_processor = HDRTVNetTorch(
            str(source),
            device="cpu",
            precision=precision,
            compile_model=False,
            predequantize="off",
            use_hg=use_hg,
            warmup_passes=0,
        )
        _export_tensorrt_onnx_from_model(
            model=export_processor.model,
            onnx_path=str(onnx_path),
            width=frame_bgr.shape[1],
            height=frame_bgr.shape[0],
            dtype=tensor.dtype,
            device=torch.device("cpu"),
            precision=precision,
            flat_model=getattr(export_processor, "_is_flat_model", False),
            qdq_fusion=args.qdq_fusion,
        )
        onnx_out = _run_onnx(onnx_path, tensor, cond)
        onnx_metrics = _metrics(original_out, onnx_out)
        onnx_bgr = _postprocess(original_processor, onnx_out)
        onnx_u8_metrics = _image_metrics(original_bgr, onnx_bgr)
        row.update(_onnx_counts(onnx_path))
        row.update({f"onnx_{k}": v for k, v in onnx_metrics.items()})
        row.update({f"onnx_{k}": v for k, v in onnx_u8_metrics.items()})
        row["onnx_path"] = str(onnx_path)

        visual_path = output_dir / "visuals" / f"{checkpoint.stem}.png"
        _write_visual_sheet(
            visual_path,
            input_bgr=frame_bgr,
            original_bgr=original_bgr,
            source_bgr=source_bgr,
            onnx_bgr=onnx_bgr,
            title=(
                f"{checkpoint.name} | ONNX MAE={row['onnx_mae']:.6g}, "
                f"max={row['onnx_max_abs']:.6g}, PSNR={row['onnx_psnr']:.2f} dB"
            ),
        )
        row["visual_path"] = str(visual_path)

        if not args.keep_onnx:
            data_path = Path(f"{onnx_path}.data")
            try:
                onnx_path.unlink(missing_ok=True)
                data_path.unlink(missing_ok=True)
            except Exception:
                pass

    return row


def _write_report(output_dir: Path, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = output_dir / "report.csv"
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    _configure_console()
    parser = argparse.ArgumentParser(
        description="Validate TensorRT source checkpoints and exported ONNX outputs."
    )
    parser.add_argument(
        "checkpoints",
        nargs="*",
        type=Path,
        help="Original INT8 checkpoints. Default: all Ensemble_AGCM_LE_int8_*.pt files.",
    )
    parser.add_argument("--resolution", default="256x256", help="Validation frame resolution.")
    parser.add_argument("--image", default=None, help="Input image path. Default: dataset/test_sdr/001.png.")
    parser.add_argument("--video", default=None, help="Optional input video path.")
    parser.add_argument("--frame-index", type=int, default=120, help="Video frame index. Default: 120.")
    parser.add_argument("--output-dir", default=None, help="Report output directory.")
    parser.add_argument(
        "--qdq-fusion",
        default="auto",
        help="TensorRT ONNX Q/DQ fusion mode. Default: auto.",
    )
    parser.add_argument("--skip-onnx", action="store_true", help="Only run checkpoint/PT parity checks.")
    parser.add_argument("--keep-onnx", action="store_true", help="Keep generated ONNX files.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of checkpoints for quick tests.")
    parser.add_argument(
        "--max-onnx-mae",
        type=float,
        default=0.02,
        help="Fail if any ONNX output MAE exceeds this value. Default: 0.02.",
    )
    parser.add_argument(
        "--max-onnx-u8-mae",
        type=float,
        default=5.0,
        help="Fail if any postprocessed ONNX uint8 MAE exceeds this value. Default: 5.0.",
    )
    parser.add_argument(
        "--max-qdq-casts",
        type=int,
        default=0,
        help="Fail if Q/DQ-boundary Cast count exceeds this value. Default: 0.",
    )
    args = parser.parse_args()

    width, height = _parse_resolution(args.resolution)
    args.resolution = f"{width}x{height}"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _ROOT / "logs" / "tensorrt_source_validation" / time.strftime("%Y%m%d_%H%M%S")
    )

    checkpoints = [p.resolve() for p in args.checkpoints] if args.checkpoints else _default_checkpoints()
    if args.limit > 0:
        checkpoints = checkpoints[: args.limit]
    if not checkpoints:
        raise FileNotFoundError("No checkpoints selected")

    frame = _resize_frame(_load_sample_frame(args), width, height)
    rows: list[dict[str, object]] = []
    print(f"[validate] output: {output_dir}")
    print(f"[validate] checkpoints: {len(checkpoints)}")
    print(f"[validate] resolution: {width}x{height}")
    for index, checkpoint in enumerate(checkpoints, 1):
        print(f"[validate] {index}/{len(checkpoints)} {checkpoint.name}", flush=True)
        row = _validate_one(checkpoint, frame, output_dir, args)
        rows.append(row)
        pt_ok = row.get("pt_max_abs", 1.0) <= 1e-7
        onnx_msg = ""
        if not args.skip_onnx:
            onnx_msg = (
                f", onnx_mae={row.get('onnx_mae', float('nan')):.6g}, "
                f"onnx_max={row.get('onnx_max_abs', float('nan')):.6g}, "
                f"u8_mae={row.get('onnx_u8_mae', float('nan')):.3f}"
            )
        print(
            f"  metadata={row['metadata_ok']} pt_exact={pt_ok} "
            f"W8A8={row['w8a8']} W8A16={row['w8a16']} FP16={row['fp16']}{onnx_msg}",
            flush=True,
        )

    _write_report(output_dir, rows)
    failures = []
    for row in rows:
        reasons = []
        if not row.get("metadata_ok"):
            reasons.append("metadata")
        if float(row.get("pt_max_abs", 1.0)) > 1e-7:
            reasons.append("pt-parity")
        if not args.skip_onnx:
            if float(row.get("onnx_mae", 0.0)) > float(args.max_onnx_mae):
                reasons.append("onnx-mae")
            if float(row.get("onnx_u8_mae", 0.0)) > float(args.max_onnx_u8_mae):
                reasons.append("onnx-u8-mae")
            if int(row.get("onnx_qdq_cast", 0)) > int(args.max_qdq_casts):
                reasons.append("qdq-cast")
        if reasons:
            failed_row = dict(row)
            failed_row["failure_reasons"] = ",".join(reasons)
            failures.append(failed_row)
    print(f"[validate] report: {output_dir / 'report.json'}")
    print(f"[validate] csv   : {output_dir / 'report.csv'}")
    if failures:
        print(f"[validate] FAILED: {len(failures)} checkpoint(s) failed validation")
        for row in failures:
            print(f"  {row.get('checkpoint')}: {row.get('failure_reasons')}")
        return 1
    print("[validate] checkpoint, ONNX, and Q/DQ validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
