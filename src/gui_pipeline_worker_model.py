from __future__ import annotations

import os
import sys

import torch

from models.hdrtvnet_torch import (
    HDRTVNetTensorRT,
    HDRTVNetTorch,
    _IS_NVIDIA,
    tensorrt_engine_path,
    tensorrt_engine_is_valid,
    tensorrt_mode_name,
    tensorrt_prebuilt_calibration_cache_path,
)
from gui_config import PRECISIONS, _select_model_path
from gui_output_capture import capture_output_to_gui


def _resolve_predequantize_arg(mode: str):
    m = str(mode).strip().lower()
    if m == "on":
        return True
    if m == "off":
        return False
    return "auto"


class PipelineWorkerModelMixin:
    """Model load/swap helpers for PipelineWorker."""

    @staticmethod
    def _silent_warmup(processor, w, h):
        """Prime either the compiled graph cache or the eager runtime once."""
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        try:
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
            if getattr(processor, "_compiled", False):
                processor.warmup_compile(w, h)
            else:
                import numpy as np

                dummy = np.zeros((max(1, int(h)), max(1, int(w)), 3), dtype=np.uint8)
                processor.process(dummy)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        finally:
            sys.stdout.close()
            sys.stderr.close()
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr
            os.close(devnull_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

    def _load_model(
        self,
        key,
        announce_ready: bool = True,
        *,
        compile_model: bool | None = None,
        force_compile: bool = False,
        compile_mode: str | None = None,
        warmup: bool = True,
    ):
        cfg = PRECISIONS.get(key, {})
        path = _select_model_path(key, self._use_hg)
        if not os.path.isfile(path):
            if key in PRECISIONS and PRECISIONS[key].get("precision", "").startswith("int8"):
                alt = PRECISIONS[key].get("model_nohg" if self._use_hg else "model")
                note = ""
                if alt and alt != path:
                    note = f" (alt: {alt})"
                self.status_message.emit(
                    f"ERROR: weights not found - {path}{note}"
                )
            else:
                self.status_message.emit(f"ERROR: weights not found - {path}")
            return False

        cw, ch = self._proc_w, self._proc_h

        if announce_ready:
            self.status_message.emit(f"Loading model: {key} ...")

        if self._processor is not None:
            del self._processor
            self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            torch._dynamo.reset()
        except (AssertionError, RuntimeError):
            pass

        if compile_model is None:
            compile_model = (
                str(getattr(self, "_runtime_execution_mode", "compile")).strip().lower()
                != "eager"
            )

        try:
            if _IS_NVIDIA:
                mode_name = f"{key}_{'hg' if self._use_hg else 'nohg'}"
                trt_predeq = _resolve_predequantize_arg(
                    getattr(self, "_predequantize_mode", "auto")
                )
                trt_mode_name = tensorrt_mode_name(
                    cfg["precision"],
                    mode_name,
                    predequantize=trt_predeq,
                    qdq_fusion="native",
                )
                trt_calibration_cache = tensorrt_prebuilt_calibration_cache_path(
                    path,
                    int(cw),
                    int(ch),
                    cfg["precision"],
                    mode_name,
                    use_hg=self._use_hg,
                    predequantize=trt_predeq,
                    qdq_fusion="native",
                    require_exists=True,
                )
                trt_engine = tensorrt_engine_path(
                    path,
                    int(cw),
                    int(ch),
                    trt_mode_name,
                )
                if announce_ready:
                    if tensorrt_engine_is_valid(
                        trt_engine,
                        model_path=path,
                        width=int(cw),
                        height=int(ch),
                        precision=cfg["precision"],
                        mode_name=mode_name,
                        use_hg=self._use_hg,
                        predequantize=trt_predeq,
                        qdq_fusion="native",
                        calibration_cache=trt_calibration_cache,
                    ):
                        self.status_message.emit(
                            f"Loading cached TensorRT engine for {cw}x{ch} ({key}) ..."
                        )
                    else:
                        self.status_message.emit(
                            f"Building TensorRT engine for {cw}x{ch} ({key}) ..."
                        )
                with capture_output_to_gui(self.status_message.emit):
                    self._processor = HDRTVNetTensorRT(
                        path,
                        device="auto",
                        precision=cfg["precision"],
                        engine_width=int(cw),
                        engine_height=int(ch),
                        mode_name=mode_name,
                        use_hg=self._use_hg,
                        predequantize=trt_predeq,
                        qdq_fusion="native",
                        calibration_cache=trt_calibration_cache,
                    )
            else:
                resolved_compile_mode = (
                    "auto"
                    if compile_mode is None and bool(compile_model)
                    else (compile_mode or "default")
                )
                self._processor = HDRTVNetTorch(
                    path,
                    device="auto",
                    precision=cfg["precision"],
                    compile_model=bool(compile_model),
                    force_compile=bool(force_compile),
                    compile_mode=resolved_compile_mode,
                    predequantize=_resolve_predequantize_arg(
                        getattr(self, "_predequantize_mode", "auto")
                    ),
                    use_hg=self._use_hg,
                    warmup_passes=0,
                )
        except Exception as exc:
            self._processor = None
            self.status_message.emit(f"ERROR: model backend failed - {exc}")
            print(f"ERROR: model backend failed: {exc}", file=sys.stderr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

        if announce_ready and warmup:
            if getattr(self._processor, "_trt_engine", None) is not None:
                self.status_message.emit(
                    f"Priming TensorRT runtime for {cw}x{ch} ({key}) ..."
                )
            elif getattr(self._processor, "_compiled", False):
                self.status_message.emit(
                    f"Warming up kernels for {cw}x{ch} ({key}) ..."
                )
            else:
                self.status_message.emit(
                    f"Priming model for {cw}x{ch} ({key}) ..."
                )
        if warmup:
            self._silent_warmup(self._processor, cw, ch)

        self._precision_key = key
        if announce_ready:
            suffix = " [TensorRT]" if _IS_NVIDIA else ""
            self.status_message.emit(f"Ready - {key}{suffix}")
        return True

    @staticmethod
    def _tensor_mb(t: torch.Tensor | None) -> float:
        if t is None:
            return 0.0
        try:
            return (t.numel() * t.element_size()) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def _model_memory_breakdown_mb(self) -> tuple[float, float, float]:
        """Return (total_mb, cpu_mb, gpu_mb) for loaded model tensors only."""
        p = self._processor
        if p is None:
            return 0.0, 0.0, 0.0

        model = getattr(p, "model", None)
        if model is None:
            return 0.0, 0.0, 0.0
        model = getattr(model, "_orig_mod", model)

        cpu_mb = 0.0
        gpu_mb = 0.0
        seen = set()

        def _accumulate_tensor(t: torch.Tensor | None):
            nonlocal cpu_mb, gpu_mb
            if t is None:
                return
            tid = id(t)
            if tid in seen:
                return
            seen.add(tid)
            mb = self._tensor_mb(t)
            if mb <= 0:
                return
            try:
                dev_type = t.device.type
            except Exception:
                dev_type = "cpu"
            if dev_type == "cuda":
                gpu_mb += mb
            else:
                cpu_mb += mb

        try:
            for t in model.parameters(recurse=True):
                _accumulate_tensor(t)
            for t in model.buffers(recurse=True):
                _accumulate_tensor(t)
        except Exception:
            pass

        total_mb = cpu_mb + gpu_mb
        return total_mb, cpu_mb, gpu_mb
