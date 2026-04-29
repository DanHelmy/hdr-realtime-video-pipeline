from __future__ import annotations

import os
import time
import torch

from gui_config import _select_model_path

try:
    from models.hdrtvnet_torch import _IS_NVIDIA
except Exception:
    _IS_NVIDIA = False


class PipelineWorkerRuntimeMetricsMixin:
    """Live metrics emit helper for PipelineWorker."""

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
        if len(presented_times) >= 2:
            dt = presented_times[-1] - presented_times[0]
            fps = ((len(presented_times) - 1) / dt) if dt > 0 else 0.0
        else:
            fps = 1000.0 / avg if avg > 0 else 0.0

        cpu_mb = process.memory_info().rss / (1024 * 1024)
        gpu_mb = self._app_vram_mb
        if gpu_mb <= 0.0 and use_cuda:
            # Fallback when Windows counters are unavailable:
            # allocator reservation is closer to "reserved VRAM"
            # than memory_allocated() (live tensor bytes only).
            gpu_mb = torch.cuda.memory_reserved() / (1024 * 1024)

        if self._input_is_hdr:
            model_mb = 0.0
            prec_label = "Bypass (HDR input)"
            model_size_label = "Checkpoint"
        else:
            cache_key = (str(self._precision_key), bool(self._use_hg))
            model_size_label = "Engine" if _IS_NVIDIA else "Checkpoint"
            if getattr(self, "_metrics_model_size_key", None) != cache_key:
                model_mb = 0.0
                engine_path = getattr(self._processor, "engine_path", None)
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
            model_mb = float(getattr(self, "_metrics_model_size_mb", 0.0))
            prec_label = self._precision_key

        self.metrics_updated.emit({
            "fps": fps,
            "latency_ms": avg,
            "model_latency_ms": float(model_avg),
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
