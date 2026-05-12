from __future__ import annotations

import os

import torch

from gui_compile_cache import _is_compiled, _precision_to_compile_arg
from gui_config import _select_model_path
from models.hdrtvnet_torch import (
    _HAS_COMPILE,
    _HAS_TRITON,
    _IS_NVIDIA,
    _IS_ROCM,
)

_BENCHMARK_USE_COMPILE_CACHE = str(
    os.environ.get("HDRTVNET_BENCHMARK_USE_COMPILE_CACHE", "1")
).strip().lower() not in {"0", "false", "no", "off"}


def _normalize_benchmark_predequantize_mode(mode: str | None) -> str:
    m = str(mode or "auto").strip().lower()
    if m in {"on", "off"}:
        return m
    return "auto"


def _effective_benchmark_predequantize_mode(
    precision_arg: str,
    selected_mode: str | None,
) -> str:
    mode = _normalize_benchmark_predequantize_mode(selected_mode)
    if not str(precision_arg or "").startswith("int8"):
        return mode
    if mode in {"on", "off"}:
        return mode
    if _IS_ROCM:
        return "on"
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            has_int8_tc = (
                props.major > 7
                or (props.major == 7 and props.minor >= 5)
            )
            return "off" if has_int8_tc else "on"
        except Exception:
            return "auto"
    return "auto"


def _benchmark_compile_cache_ready(
    *,
    w: int,
    h: int,
    precision_key: str,
    model_path: str,
    use_hg: bool,
    predequantize_mode: str,
) -> tuple[bool, str, str]:
    """Return whether benchmark-style cached max-autotune can be used safely."""
    if _IS_NVIDIA or not _BENCHMARK_USE_COMPILE_CACHE:
        return False, "disabled", "auto"
    if not (_HAS_COMPILE and _HAS_TRITON and torch.cuda.is_available()):
        return False, "unavailable", "auto"

    precision_arg = _precision_to_compile_arg(precision_key)
    selected_pdq = _normalize_benchmark_predequantize_mode(predequantize_mode)
    effective_pdq = _effective_benchmark_predequantize_mode(
        precision_arg,
        selected_pdq,
    )
    if _is_compiled(
        int(w),
        int(h),
        precision_arg,
        model_path=model_path,
        use_hg=bool(use_hg),
        predequantize_mode=effective_pdq,
    ):
        return True, precision_arg, effective_pdq

    if (
        str(precision_arg).startswith("int8")
        and selected_pdq == "auto"
        and effective_pdq != "auto"
        and _is_compiled(
            int(w),
            int(h),
            precision_arg,
            model_path=model_path,
            use_hg=bool(use_hg),
            predequantize_mode="auto",
        )
    ):
        return True, precision_arg, "auto"

    if str(precision_arg).startswith("int8") and effective_pdq == "on":
        fp16_model = _select_model_path("FP16", bool(use_hg))
        if _is_compiled(
            int(w),
            int(h),
            "fp16",
            model_path=fp16_model,
            use_hg=bool(use_hg),
            predequantize_mode="auto",
        ):
            return True, precision_arg, effective_pdq

    return False, precision_arg, effective_pdq
