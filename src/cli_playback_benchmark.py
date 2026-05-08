from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import re
import time
from collections import deque
from datetime import datetime

from windows_runtime import (
    configure_rocm_sdk_environment,
    ensure_windows_supported,
    project_cache_root,
)

ensure_windows_supported("HDRTVNet++ CLI playback benchmark")
configure_rocm_sdk_environment()

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
_CACHE_ROOT = project_cache_root(__file__)
_DLL_DIR_HANDLES = []


def _prepend_dll_search_path(path: pathlib.Path) -> None:
    path_text = str(path)
    if not path_text or not path.is_dir():
        return
    os.environ["PATH"] = path_text + os.pathsep + os.environ.get("PATH", "")
    try:
        _DLL_DIR_HANDLES.append(os.add_dll_directory(path_text))
    except Exception:
        pass


_prepend_dll_search_path(_HERE)
os.makedirs(_CACHE_ROOT, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_CACHE_ROOT, "torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_CACHE_ROOT, "triton"))
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

import cv2
import numpy as np
import psutil
import torch

from gui_compile_cache import _mark_compiled
from models.hdrtvnet_torch import HDRTVNetTensorRT, HDRTVNetTorch, _IS_NVIDIA
from video_source import VideoSource


def _weight(name: str) -> str:
    return str(_HERE / "models" / "weights" / name)


_RUN_PRESETS = {
    "fp16": {
        "precision": "fp16",
        "model": _weight("Ensemble_AGCM_LE.pth"),
        "label": "fp16",
    },
    "fp32": {
        "precision": "fp32",
        "model": _weight("Ensemble_AGCM_LE.pth"),
        "label": "fp32",
    },
    "int8-mixed-ptq": {
        "precision": "int8-mixed",
        "model": _weight("Ensemble_AGCM_LE_int8_mixed.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_mixed_nohg.pt"),
        "predequantize": "on",
        "label": "int8_mixed_ptq_predeq",
    },
    "int8-full-ptq": {
        "precision": "int8-full",
        "model": _weight("Ensemble_AGCM_LE_int8_full.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_full_nohg.pt"),
        "predequantize": "on",
        "label": "int8_full_ptq_predeq",
    },
    "int8-mixed-qat": {
        "precision": "int8-mixed",
        "model": _weight("Ensemble_AGCM_LE_int8_mixed_qat.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_mixed_qat_nohg.pt"),
        "predequantize": "on",
        "label": "int8_mixed_qat_predeq",
    },
    "int8-full-qat": {
        "precision": "int8-full",
        "model": _weight("Ensemble_AGCM_LE_int8_full_qat.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_full_qat_nohg.pt"),
        "predequantize": "on",
        "label": "int8_full_qat_predeq",
    },
    "int8-mixed-qat-film": {
        "precision": "int8-mixed",
        "model": _weight("Ensemble_AGCM_LE_int8_mixed_qat_film.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_mixed_qat_film_nohg.pt"),
        "predequantize": "on",
        "label": "int8_mixed_qat_film_predeq",
    },
    "int8-full-qat-film": {
        "precision": "int8-full",
        "model": _weight("Ensemble_AGCM_LE_int8_full_qat_film.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_full_qat_film_nohg.pt"),
        "predequantize": "on",
        "label": "int8_full_qat_film_predeq",
    },
}
_DEFAULT_RUNS = [
    "fp32",
    "fp16",
    "int8-mixed-ptq",
    "int8-full-ptq",
    "int8-mixed-qat",
    "int8-full-qat",
    "int8-mixed-qat-film",
    "int8-full-qat-film",
]

_CSV_FIELDS = [
    "elapsed_s",
    "logged_at_local",
    "fps",
    "latency_ms",
    "model_latency_ms",
    "live_video_latency_ms",
    "frame",
    "cpu_mb",
    "gpu_mb",
    "model_mb",
    "model_size_label",
    "precision",
    "proc_res",
    "psnr_db",
    "sssim",
    "delta_e_itp",
    "hdr_vdp3",
    "objective_enabled",
    "objective_note",
    "hdr_vdp3_note",
    "is_live_capture",
    "decode_ms",
    "resize_ms",
    "infer_ms",
    "pre_ms",
    "run_ms",
    "post_ms",
    "fps_1p_low",
]


def _parse_resolution(text: str) -> tuple[int, int]:
    try:
        w, h = str(text).lower().split("x", 1)
        return int(w), int(h)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution '{text}', expected WxH."
        ) from exc


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    value = value.strip("._-")
    return (value or "session")[:80]


def _stats(samples: list[dict], key: str) -> dict | None:
    vals = []
    for sample in samples:
        try:
            value = float(sample.get(key))
        except Exception:
            continue
        if math.isfinite(value):
            vals.append(value)
    if not vals:
        return None
    return {
        "count": len(vals),
        "avg": sum(vals) / float(len(vals)),
        "min": min(vals),
        "max": max(vals),
        "last": vals[-1],
    }


def _fmt_stats(label: str, stats: dict | None, suffix: str = "") -> str:
    if not stats:
        return f"{label}: n/a"
    return (
        f"{label}: avg {stats['avg']:.3f}{suffix}, "
        f"min {stats['min']:.3f}{suffix}, "
        f"max {stats['max']:.3f}{suffix}, "
        f"last {stats['last']:.3f}{suffix}"
    )


def _resize_frame(frame, width: int, height: int):
    if frame.shape[1] == int(width) and frame.shape[0] == int(height):
        return frame
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def _bgr_to_rgb48_bytes(frame: np.ndarray, host_state: dict) -> bytes:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected HxWx3 frame.")
    shape = (int(frame.shape[0]), int(frame.shape[1]), 3)
    arr = host_state.get("numpy")
    if host_state.get("shape") != shape or arr is None:
        arr = np.empty(shape, dtype=np.uint16)
        host_state["shape"] = shape
        host_state["numpy"] = arr
    if frame.dtype == np.uint8:
        np.multiply(frame[:, :, ::-1], np.uint16(257), out=arr, casting="unsafe")
    elif frame.dtype == np.uint16:
        np.copyto(arr, frame[:, :, ::-1], casting="unsafe")
    else:
        src = frame.astype(np.float32, copy=False)
        if src.max(initial=0.0) <= 1.0:
            src = src * 65535.0
        arr[:] = np.clip(src[:, :, ::-1], 0.0, 65535.0).astype(np.uint16)
    return arr.tobytes()


class _DisplaySink:
    def __init__(self, args, width: int, height: int, fps: float):
        self.enabled = bool(args.display)
        self.backend = str(getattr(args, "display_backend", "mpv") or "mpv").strip().lower()
        self.window_name = "HDRTVNet++ CLI Benchmark"
        self._app = None
        self._mpv_widget = None
        self._rgb48_state: dict = {}

        if not self.enabled:
            return
        if self.backend == "opencv":
            return
        if self.backend != "mpv":
            raise ValueError(f"Unknown display backend: {self.backend}")
        self._start_mpv(int(width), int(height), float(fps) if fps and fps > 0 else 30.0)

    def _start_mpv(self, width: int, height: int, fps: float) -> None:
        _prepend_dll_search_path(_HERE)
        try:
            import mpv as mpv_lib
        except (OSError, ImportError) as exc:
            raise RuntimeError(
                "mpv display requested, but python-mpv/libmpv is unavailable. "
                "Use --display-backend opencv for the old display path."
            ) from exc

        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError as exc:
            raise RuntimeError(
                "mpv display requested, but PyQt6 is unavailable. "
                "Use --display-backend opencv for the old display path."
            ) from exc

        from gui_mpv_widget import MpvHDRWidget
        from gui_scaling import (
            BEST_MPV_SCALE,
            FILMGRAIN_SHADER_PATH,
            FSR_SHADER_PATH,
            SSIM_SUPERRES_SHADER_PATH,
            _ensure_filmgrain_shader,
            _ensure_fsr_shader,
            _ensure_ssim_superres_shader,
            _normalize_shader_paths,
        )

        app = QApplication.instance()
        if app is None:
            app = QApplication(["HDRTVNet++ CLI Benchmark"])
        self._app = app

        mpv_diag = str(os.environ.get("HDRTVNET_MPV_DIAG", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        widget = MpvHDRWidget(
            mpv_lib=mpv_lib,
            mpv_diag=mpv_diag,
            normalize_shader_paths=_normalize_shader_paths,
            ensure_fsr_shader=_ensure_fsr_shader,
            ensure_ssim_superres_shader=_ensure_ssim_superres_shader,
            ensure_filmgrain_shader=_ensure_filmgrain_shader,
            best_mpv_scale=BEST_MPV_SCALE,
            fsr_shader_path=FSR_SHADER_PATH,
            ssim_superres_shader_path=SSIM_SUPERRES_SHADER_PATH,
            filmgrain_shader_path=FILMGRAIN_SHADER_PATH,
        )
        widget.setWindowTitle(self.window_name)
        widget.resize(int(width), int(height))
        widget.show()
        app.processEvents()

        started = widget.start_playback(
            int(width),
            int(height),
            fps=float(fps),
            scale_kernel="bicubic",
            scale_antiring=0.0,
            force_hdr_metadata=True,
            vsync_timed=False,
        )
        if not started:
            widget.close()
            app.processEvents()
            raise RuntimeError(
                getattr(widget, "_last_scale_error", None) or "mpv display startup failed."
            )
        self._mpv_widget = widget

    def show(self, frame) -> bool:
        if not self.enabled:
            return True
        if self.backend == "opencv":
            cv2.imshow(self.window_name, frame)
            return (cv2.waitKey(1) & 0xFF) != 27

        widget = self._mpv_widget
        app = self._app
        if widget is None or app is None:
            return False
        widget.feed_frame(_bgr_to_rgb48_bytes(frame, self._rgb48_state))
        app.processEvents()
        return bool(widget.isVisible())

    def close(self) -> None:
        if self.backend == "opencv":
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            return
        widget = self._mpv_widget
        app = self._app
        if widget is not None:
            try:
                widget.stop_playback()
            except Exception:
                pass
            try:
                widget.close()
            except Exception:
                pass
        if app is not None:
            try:
                app.processEvents()
            except Exception:
                pass


def _compiled_marker_predequantize_mode(precision: str, selected_mode: str, processor) -> str:
    if not str(precision or "").strip().lower().startswith("int8"):
        return "auto"
    mode = str(selected_mode or "auto").strip().lower()
    if mode not in {"auto", "on", "off"}:
        mode = "auto"
    if mode != "auto":
        return mode
    try:
        is_w8_model = bool(getattr(processor, "_is_w8_model"))
    except Exception:
        return "auto"
    return "off" if is_w8_model else "on"


def _mark_cache(processor, width: int, height: int, args, run: dict) -> None:
    if _IS_NVIDIA or not bool(getattr(processor, "_compiled", False)):
        return
    precision = str(run["precision"])
    try:
        _mark_compiled(
            int(width),
            int(height),
            precision,
            model_path=str(run["model"]),
            use_hg=bool(args.use_hg),
            predequantize_mode=_compiled_marker_predequantize_mode(
                precision,
                str(run.get("predequantize", "auto")),
                processor,
            ),
            compile_mode=str(
                getattr(processor, "_compile_mode", None)
                or ("max-autotune" if str(args.compile_mode) == "auto" else args.compile_mode)
            ),
            memory_format=str(getattr(processor, "_memory_format_name", None) or "contiguous"),
        )
    except Exception as exc:
        print(f"WARNING: could not write compile marker: {exc}", flush=True)


def _make_processor(args, run: dict, width: int, height: int):
    predeq_text = str(run.get("predequantize", "auto"))
    predeq = {"auto": "auto", "on": True, "off": False}[predeq_text]
    if _IS_NVIDIA and str(args.device).lower() != "cpu":
        return HDRTVNetTensorRT(
            run["model"],
            device=args.device,
            precision=run["precision"],
            engine_width=int(width),
            engine_height=int(height),
            mode_name=f"{run['precision']}_{'hg' if args.use_hg else 'nohg'}",
            use_hg=bool(args.use_hg),
        )
    return HDRTVNetTorch(
        run["model"],
        device=args.device,
        precision=run["precision"],
        compile_model=not args.no_compile,
        force_compile=args.force_compile,
        compile_mode=args.compile_mode,
        use_cuda_graphs=False,
        force_channels_last=False,
        predequantize=predeq,
        use_hg=bool(args.use_hg),
        warmup_passes=0,
    )


def _write_runtime_csv(path: pathlib.Path, samples: list[dict]) -> None:
    extras = []
    seen = set(_CSV_FIELDS)
    for sample in samples:
        for key in sample:
            if key not in seen:
                seen.add(key)
                extras.append(key)
    fields = _CSV_FIELDS + sorted(extras)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({key: sample.get(key) for key in fields})


def _write_session_files(
    *,
    session_dir: pathlib.Path,
    reason: str,
    source_label: str,
    settings: dict,
    runtime_samples: list[dict],
    worker_summary: dict,
    started_at: str,
    ended_at: str,
) -> dict:
    runtime_summary = {
        key: _stats(runtime_samples, key)
        for key in (
            "fps",
            "latency_ms",
            "model_latency_ms",
            "live_video_latency_ms",
            "gpu_mb",
            "cpu_mb",
            "model_mb",
            "psnr_db",
            "sssim",
            "delta_e_itp",
            "hdr_vdp3",
        )
    }
    for sample in reversed(runtime_samples):
        label = str(sample.get("model_size_label") or "").strip()
        if label:
            runtime_summary["model_size_label"] = label
            break

    lines = [
        "HDRTVNet++ Playback Log",
        f"Reason: {reason}",
        f"Saved To: {session_dir}",
        f"Started: {started_at or 'n/a'}",
        f"Ended: {ended_at or 'n/a'}",
        f"Source: {source_label or 'n/a'}",
        f"Source Mode: {settings.get('source_mode') or 'cli_headless'}",
        "",
        "Settings:",
        f"  Precision: {settings.get('precision') or 'n/a'}",
        f"  Resolution: {settings.get('resolution') or 'n/a'}",
        f"  Upscale: {settings.get('upscale_mode') or 'n/a'}",
        f"  Use HG: {settings.get('use_hg')}",
        f"  Film Grain: {settings.get('film_grain')}",
        f"  Runtime Mode: {settings.get('runtime_execution_mode') or 'n/a'}",
        f"  Predequantize: {settings.get('predequantize_mode') or 'n/a'}",
        f"  HDR GT: {settings.get('hdr_ground_truth_path') or 'none'}",
        "",
        "Runtime Metrics:",
        f"  Samples Saved: {len(runtime_samples)}",
        "  " + _fmt_stats("FPS", runtime_summary.get("fps"), ""),
        "  " + _fmt_stats("Latency", runtime_summary.get("latency_ms"), " ms"),
        "  "
        + _fmt_stats(
            "Inference Latency (sampled UI)",
            runtime_summary.get("model_latency_ms"),
            " ms",
        ),
        "  " + _fmt_stats("GPU Memory", runtime_summary.get("gpu_mb"), " MB"),
        "  " + _fmt_stats("CPU Memory", runtime_summary.get("cpu_mb"), " MB"),
    ]
    if runtime_summary.get("model_mb"):
        size_label = str(runtime_summary.get("model_size_label") or "Checkpoint").strip()
        lines.append(
            "  "
            + _fmt_stats(
                f"{size_label} Size",
                runtime_summary.get("model_mb"),
                " MB",
            )
        )
    exact_avg = worker_summary.get("avg_model_latency_ms")
    exact_count = int(worker_summary.get("model_latency_samples", 0) or 0)
    if exact_avg is None:
        lines.append("  Exact Inference Average: n/a")
    else:
        lines.append(
            "  Exact Inference Average: "
            f"{float(exact_avg):.3f} ms over {exact_count} frames"
        )
    lines.extend(["", "Compare Events: 0", "  None", ""])

    session_payload = {
        "reason": reason,
        "saved_at_local": ended_at,
        "started_at_local": started_at or None,
        "session_elapsed_s": worker_summary.get("session_elapsed_s", 0.0),
        "source_label": source_label or None,
        "settings": settings,
        "worker_summary": worker_summary,
        "runtime_metric_summary": runtime_summary,
        "runtime_metrics": runtime_samples,
        "compare_events": [],
        "files": {
            "summary_txt": "summary.txt",
            "session_json": "session.json",
            "runtime_metrics_csv": "runtime_metrics.csv",
            "compare_events_csv": None,
        },
    }

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    with (session_dir / "session.json").open("w", encoding="utf-8") as handle:
        json.dump(session_payload, handle, indent=2)
    _write_runtime_csv(session_dir / "runtime_metrics.csv", runtime_samples)
    return {
        "session_dir": str(session_dir),
        "summary_txt": str(session_dir / "summary.txt"),
        "session_json": str(session_dir / "session.json"),
        "runtime_metrics_csv": str(session_dir / "runtime_metrics.csv"),
        "avg_model_latency_ms": exact_avg,
        "avg_fps": (runtime_summary.get("fps") or {}).get("avg"),
    }


def _run_one(args, run: dict, resolution: tuple[int, int], batch_dir: pathlib.Path) -> dict:
    width, height = resolution
    source = VideoSource(args.video, prefetch=args.prefetch)
    fps = float(source.fps or 30.0)
    timed_frames_target = max(1, int(round(float(args.duration_s) * fps)))
    max_frames = int(args.warmup_frames) + timed_frames_target
    source_label = pathlib.Path(args.video).name
    run_label = f"{width}x{height}_{run['label']}_{'hg' if args.use_hg else 'nohg'}"
    session_dir = batch_dir / run_label

    processor = _make_processor(args, run, width, height)
    if bool(getattr(processor, "_compiled", False)) and not args.skip_cache_warmup:
        processor.warmup_compile(int(width), int(height))
        _mark_cache(processor, width, height, args, run)

    process = psutil.Process(os.getpid())
    use_cuda = torch.cuda.is_available() and str(args.device).lower() != "cpu"
    model_size_mb = os.path.getsize(run["model"]) / (1024 * 1024)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started_t = time.perf_counter()

    frame_idx = 0
    stats_frames = 0
    decode_ms = resize_ms = infer_ms = render_ms = pre_ms = run_ms = post_ms = 0.0
    frame_ms_sum = 0.0
    fps_samples = deque(maxlen=10000)
    model_latency_values: list[float] = []
    runtime_samples: list[dict] = []
    display = None

    try:
        if args.display:
            display = _DisplaySink(args, width, height, fps)
        while frame_idx < max_frames:
            t0 = time.perf_counter()
            ok, frame = source.read()
            t1 = time.perf_counter()
            if not ok:
                break
            frame = _resize_frame(frame, width, height)
            t2 = time.perf_counter()
            out, pre_t, run_t, post_t = processor.process_timed(frame)
            t3 = time.perf_counter()
            if display is not None:
                if not display.show(out):
                    break
            t4 = time.perf_counter()

            frame_idx += 1
            if frame_idx <= int(args.warmup_frames):
                continue

            stats_frames += 1
            decode_ms += (t1 - t0) * 1000.0
            resize_ms += (t2 - t1) * 1000.0
            infer_ms += (t3 - t2) * 1000.0
            render_ms += (t4 - t3) * 1000.0
            pre_ms += float(pre_t)
            run_ms += float(run_t)
            post_ms += float(post_t)
            frame_ms = (t4 - t0) * 1000.0
            frame_ms_sum += frame_ms
            if frame_ms > 0:
                fps_samples.append(1000.0 / frame_ms)
            model_latency = float(run_t) if float(run_t) > 0.0 else (t3 - t2) * 1000.0
            model_latency_values.append(model_latency)

            if stats_frames % int(args.sample_interval) == 0 or stats_frames == timed_frames_target:
                avg_frame_ms = frame_ms_sum / max(1, stats_frames)
                avg_fps = 1000.0 / avg_frame_ms if avg_frame_ms > 0 else 0.0
                sorted_fps = sorted(fps_samples)
                if sorted_fps:
                    k = max(1, int(len(sorted_fps) * 0.01))
                    one_percent_low = sum(sorted_fps[:k]) / k
                else:
                    one_percent_low = 0.0
                gpu_mb = 0.0
                if use_cuda:
                    try:
                        gpu_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                    except Exception:
                        gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                sample = {
                    "elapsed_s": round(max(0.0, time.perf_counter() - started_t), 3),
                    "logged_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "fps": float(avg_fps),
                    "latency_ms": float(avg_frame_ms),
                    "model_latency_ms": float(sum(model_latency_values) / len(model_latency_values)),
                    "live_video_latency_ms": 0.0,
                    "is_live_capture": False,
                    "frame": int(frame_idx),
                    "cpu_mb": process.memory_info().rss / (1024 * 1024),
                    "gpu_mb": float(gpu_mb),
                    "model_mb": float(model_size_mb),
                    "model_size_label": "Checkpoint",
                    "precision": str(run["label"]),
                    "proc_res": f"{width}x{height}",
                    "psnr_db": None,
                    "sssim": None,
                    "delta_e_itp": None,
                    "hdr_vdp3": None,
                    "objective_enabled": False,
                    "objective_note": "",
                    "hdr_vdp3_note": "",
                    "decode_ms": decode_ms / max(1, stats_frames),
                    "resize_ms": resize_ms / max(1, stats_frames),
                    "infer_ms": infer_ms / max(1, stats_frames),
                    "pre_ms": pre_ms / max(1, stats_frames),
                    "run_ms": run_ms / max(1, stats_frames),
                    "post_ms": post_ms / max(1, stats_frames),
                    "render_ms": render_ms / max(1, stats_frames),
                    "fps_1p_low": float(one_percent_low),
                }
                runtime_samples.append(sample)
                print(
                    f"[bench] {run_label}: frames={stats_frames} "
                    f"fps={avg_fps:.2f} model={sample['model_latency_ms']:.3f}ms",
                    flush=True,
                )
            if stats_frames >= timed_frames_target:
                break
    finally:
        try:
            source.release()
        except Exception:
            pass
        if display is not None:
            display.close()
        del processor
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    ended_at = datetime.now().astimezone().isoformat(timespec="seconds")
    elapsed_s = round(max(0.0, time.perf_counter() - started_t), 3)
    avg_model_latency = (
        sum(model_latency_values) / len(model_latency_values)
        if model_latency_values
        else None
    )
    source_mode = (
        f"cli_display_{args.display_backend}"
        if bool(args.display)
        else "cli_headless"
    )
    settings = {
        "source_mode": source_mode,
        "precision": str(run["label"]),
        "resolution": f"{width}x{height}",
        "upscale_mode": "none",
        "use_hg": bool(args.use_hg),
        "film_grain": False,
        "runtime_execution_mode": "compiled" if not args.no_compile else "eager",
        "predequantize_mode": str(run.get("predequantize", "auto")),
        "objective_metrics_enabled": False,
        "hdr_ground_truth_path": None,
        "model_path": str(run["model"]),
        "duration_s": float(args.duration_s),
        "warmup_frames": int(args.warmup_frames),
        "display_backend": str(args.display_backend) if bool(args.display) else None,
    }
    worker_summary = {
        "avg_model_latency_ms": avg_model_latency,
        "model_latency_samples": len(model_latency_values),
        "session_elapsed_s": elapsed_s,
        "processed_frames": int(frame_idx),
        "timed_frames": int(stats_frames),
    }
    return _write_session_files(
        session_dir=session_dir,
        reason="cli benchmark finished",
        source_label=source_label,
        settings=settings,
        runtime_samples=runtime_samples,
        worker_summary=worker_summary,
        started_at=started_at,
        ended_at=ended_at,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CLI playback benchmarks and write GUI-style playback logs."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Video path to benchmark.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=_parse_resolution,
        default=[(1280, 720), (1920, 1080)],
        help="Resolutions to benchmark, e.g. 1280x720 1920x1080.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(_DEFAULT_RUNS),
        choices=tuple(_RUN_PRESETS.keys()),
        help="Precision/model presets to run.",
    )
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--warmup-frames", type=int, default=120)
    parser.add_argument("--sample-interval", type=int, default=120)
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--use-hg", default="0", choices=["0", "1"])
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-cache-warmup", action="store_true")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show processed frames and include display cost in timing.",
    )
    parser.add_argument(
        "--display-backend",
        default="mpv",
        choices=["mpv", "opencv"],
        help="Display backend used with --display. Defaults to mpv, matching the GUI HDR path.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--out-root",
        default=str(_ROOT / "logs" / "playback_sessions"),
        help="Root folder for GUI-style playback session logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    args.use_hg = str(args.use_hg).strip() != "0"

    source_slug = _slug(pathlib.Path(args.video).stem)
    batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = pathlib.Path(args.out_root) / f"{batch_stamp}_{source_slug}_cli_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] Video: {args.video}", flush=True)
    print(f"[bench] Output: {batch_dir}", flush=True)
    if args.display:
        print(f"[bench] Display: {args.display_backend}", flush=True)

    results = []
    for resolution in args.resolutions:
        for run_key in args.runs:
            preset = dict(_RUN_PRESETS[run_key])
            if (not args.use_hg) and preset.get("model_nohg"):
                preset["model"] = preset["model_nohg"]
            if not os.path.isfile(preset["model"]):
                raise FileNotFoundError(f"Model not found: {preset['model']}")
            print(
                f"[bench] Starting {resolution[0]}x{resolution[1]} {preset['label']} "
                f"({'HG' if args.use_hg else 'no-HG'})",
                flush=True,
            )
            results.append(_run_one(args, preset, resolution, batch_dir))

    summary_csv = batch_dir / "batch_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "session_dir",
            "runtime_metrics_csv",
            "avg_model_latency_ms",
            "avg_fps",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in fields})
    with (batch_dir / "batch_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"video": args.video, "results": results}, handle, indent=2)
    print(f"[bench] Done: {batch_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
