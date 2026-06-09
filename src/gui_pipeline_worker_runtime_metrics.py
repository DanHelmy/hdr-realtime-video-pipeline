from __future__ import annotations

import os
import time
import warnings
import torch

from gui_config import _select_model_path
from models.hdrtvnet_torch import HDRTVNetTensorRT, _IS_NVIDIA


class PipelineWorkerRuntimeMetricsMixin:
    """Live metrics emit helper for PipelineWorker."""

    @staticmethod
    def _trimmed_latency_average(values) -> float:
        vals = [float(v) for v in values if float(v) > 0.0]
        if not vals:
            return 0.0
        if len(vals) < 8:
            return sum(vals) / len(vals)
        vals.sort()
        trim = max(1, len(vals) // 10)
        kept = vals[trim:-trim] if len(vals) > (trim * 2) else vals
        return sum(kept) / len(kept)

    @staticmethod
    def _tensorrt_device_memory_mb(processor) -> float | None:
        if not isinstance(processor, HDRTVNetTensorRT):
            return None

        def _read_size(obj) -> float | None:
            if obj is None:
                return None
            for attr in ("device_memory_size_v2", "device_memory_size"):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        value = getattr(obj, attr, None)
                        size = value() if callable(value) else value
                except Exception:
                    continue
                if isinstance(size, (int, float)) and size > 0:
                    return float(size)
            if hasattr(obj, "get_device_memory_size"):
                try:
                    size = obj.get_device_memory_size()
                except Exception:
                    return None
                if isinstance(size, (int, float)) and size > 0:
                    return float(size)
            return None

        size_bytes = _read_size(getattr(processor, "_trt_context", None))
        if size_bytes is None:
            size_bytes = _read_size(getattr(processor, "_trt_engine", None))
        if size_bytes is None:
            return None
        return size_bytes / (1024.0 * 1024.0)

    @classmethod
    def _runtime_gpu_memory_mb(cls, processor, use_cuda: bool) -> float:
        if not use_cuda:
            return 0.0
        trt_device_mb = cls._tensorrt_device_memory_mb(processor)
        if trt_device_mb is not None:
            return float(trt_device_mb)
        try:
            reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            if reserved_mb > 0.0:
                return float(reserved_mb)
        except Exception:
            pass
        try:
            return float(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            return 0.0

    def _emit_runtime_metrics_if_ready(
        self,
        *,
        frame_idx: int,
        frame_times,
        model_times,
        presented_times,
        metrics_warmup_frames: int,
        force: bool = False,
        process,
        use_cuda: bool,
        proc_w: int,
        proc_h: int,
        objective_note: str,
        hdr_vdp3_note: str,
        psnr_avg,
        sssim_avg,
        deitp_avg,
        hdr_vdp3_avg,
        hg_weights_path: str,
        live_video_latency_ms: float = 0.0,
        is_live_capture: bool = False,
    ) -> None:
        if not force and (not frame_times or metrics_warmup_frames != 0):
            return
        if not frame_times:
            return
        now_t = time.perf_counter()
        if not force:
            emit_interval_s = max(
                0.05,
                float(getattr(self, "_metrics_emit_interval_s", 0.20) or 0.20),
            )
            last_emit_t = float(getattr(self, "_last_metrics_emit_t", 0.0) or 0.0)
            if last_emit_t > 0.0 and (now_t - last_emit_t) < emit_interval_s:
                return

        avg = sum(frame_times) / len(frame_times)
        model_avg = (
            (sum(model_times) / len(model_times))
            if model_times
            else 0.0
        )
        model_display_avg = self._trimmed_latency_average(model_times)
        if len(presented_times) >= 2:
            dt = presented_times[-1] - presented_times[0]
            fps = ((len(presented_times) - 1) / dt) if dt > 0 else 0.0
        else:
            fps = 1000.0 / avg if avg > 0 else 0.0

        cpu_mb = process.memory_info().rss / (1024 * 1024)
        gpu_mb = self._runtime_gpu_memory_mb(getattr(self, "_processor", None), use_cuda)

        if self._input_is_hdr:
            model_mb = 0.0
            prec_label = "Bypass (HDR input)"
            model_size_label = "Checkpoint"
        else:
            engine_path = ""
            engine_size_token = 0
            if _IS_NVIDIA:
                engine_path = str(
                    getattr(getattr(self, "_processor", None), "engine_path", "") or ""
                )
                if engine_path and os.path.isfile(engine_path):
                    try:
                        engine_size_token = int(os.path.getsize(engine_path))
                    except OSError:
                        engine_size_token = 0
            cache_key = (
                str(self._precision_key),
                bool(self._use_hg),
                int(proc_w),
                int(proc_h),
                engine_path if _IS_NVIDIA else "",
                engine_size_token if _IS_NVIDIA else 0,
            )
            model_size_label = "Checkpoint"
            if getattr(self, "_metrics_model_size_key", None) != cache_key:
                model_mb = 0.0
                cached_label = "Checkpoint"
                if _IS_NVIDIA and engine_path and os.path.isfile(engine_path):
                    model_mb = os.path.getsize(engine_path) / (1024 * 1024)
                else:
                    model_path = _select_model_path(self._precision_key, self._use_hg)
                    model_mb = os.path.getsize(model_path) / (1024 * 1024)
                    if self._use_hg and self._precision_key in ("FP16", "FP32"):
                        if os.path.isfile(hg_weights_path):
                            model_mb += os.path.getsize(hg_weights_path) / (1024 * 1024)
                self._metrics_model_size_key = cache_key
                self._metrics_model_size_mb = model_mb
                self._metrics_model_size_label = cached_label
            model_mb = float(getattr(self, "_metrics_model_size_mb", 0.0))
            model_size_label = str(
                getattr(self, "_metrics_model_size_label", "Checkpoint") or "Checkpoint"
            )
            prec_label = self._precision_key

        self.metrics_updated.emit({
            "fps": fps,
            "latency_ms": avg,
            "model_latency_ms": float(model_avg),
            "model_latency_display_ms": float(model_display_avg),
            "live_video_latency_ms": float(live_video_latency_ms),
            "is_live_capture": bool(is_live_capture),
            "frame": frame_idx,
            "cpu_mb": cpu_mb,
            "gpu_mb": gpu_mb,
            "model_mb": model_mb,
            "model_size_label": model_size_label,
            "precision": prec_label,
            "proc_res": f"{proc_w}x{proc_h}",
            "psnr_db": psnr_avg.value,
            "sssim": sssim_avg.value,
            "delta_e_itp": deitp_avg.value,
            "hdr_vdp3": hdr_vdp3_avg.value,
            "objective_enabled": bool(self._objective_metrics_enabled),
            "objective_note": objective_note,
            "hdr_vdp3_note": hdr_vdp3_note,
        })
        self._last_metrics_emit_t = now_t
