"""Kernel compile-cache marker utilities shared by GUI compile flows."""

from __future__ import annotations

import hashlib as _hashlib
import os
import pathlib as _pathlib

from windows_runtime import default_cache_root

_TRITON_CACHE = (
    _pathlib.Path(
        os.environ.get("TRITON_CACHE_DIR", os.path.join(default_cache_root(), "triton"))
    )
    / "cache"
)


def _compiled_marker_path() -> _pathlib.Path:
    return _TRITON_CACHE / "hdrtvnet_compiled.txt"


_MODEL_HASH_CACHE: dict[str, str] = {}


def _model_hash(path: str) -> str:
    if not path:
        return "missing"
    cached = _MODEL_HASH_CACHE.get(path)
    if cached:
        return cached
    try:
        data = _pathlib.Path(path).read_bytes()
    except Exception:
        digest = "missing"
    else:
        digest = _hashlib.sha256(data).hexdigest()
    _MODEL_HASH_CACHE[path] = digest
    return digest


def _model_compile_id(precision: str, model_path: str) -> str:
    """Return model identity used in compile cache keys.

    PTQ/QAT INT8 variants share kernel shapes, so they intentionally share
    one compile-id bucket to avoid unnecessary restart/recompile boundaries.
    """
    if str(precision).startswith("int8"):
        return "int8shared"
    return _model_hash(model_path)


def _compiled_key(
    w: int,
    h: int,
    precision: str,
    model_path: str,
    use_hg: bool,
    predequantize_mode: str = "auto",
    compile_mode: str = "max-autotune",
) -> str:
    pdq = str(predequantize_mode or "auto").strip().lower()
    if pdq not in {"auto", "on", "off"}:
        pdq = "auto"
    mh = _model_compile_id(precision, model_path)
    return f"{w}x{h}_{precision}_hg{int(use_hg)}_{compile_mode}_{pdq}_{mh}"


def _legacy_compiled_keys(
    w: int,
    h: int,
    precision: str,
    model_path: str,
    use_hg: bool,
    compile_mode: str = "max-autotune",
) -> tuple[str]:
    mh = _model_hash(model_path)
    # Legacy GUI marker format (no predequantization mode)
    k1 = f"{w}x{h}_{precision}_hg{int(use_hg)}_{compile_mode}_{mh}"
    return (k1,)


def _is_compiled(
    w: int,
    h: int,
    precision: str,
    model_path: str,
    use_hg: bool,
    predequantize_mode: str = "auto",
    compile_mode: str = "max-autotune",
) -> bool:
    """Check if clean-compiled kernels exist for this config."""
    mp = _compiled_marker_path()
    if mp.is_file():
        pdq = str(predequantize_mode or "auto").strip().lower()
        if pdq not in {"auto", "on", "off"}:
            pdq = "auto"
        lines = set(mp.read_text(encoding="utf-8").splitlines())
        key = _compiled_key(
            w,
            h,
            precision,
            model_path,
            use_hg,
            predequantize_mode=pdq,
            compile_mode=compile_mode,
        )
        if key in lines:
            return True
        if str(precision).startswith("int8"):
            # Compatibility: older markers used model-hash suffixes, so PTQ/QAT
            # variants can differ only in the final token.
            prefix = f"{w}x{h}_{precision}_hg{int(use_hg)}_{compile_mode}_{pdq}_"
            for line in lines:
                if line.startswith(prefix):
                    return True
        # Legacy fallback is only safe in auto mode. For explicit on/off we
        # must avoid false positives across predequantization modes.
        if pdq == "auto":
            for old_key in _legacy_compiled_keys(
                w, h, precision, model_path, use_hg, compile_mode=compile_mode
            ):
                if old_key in lines:
                    return True
            if str(precision).startswith("int8"):
                legacy_prefix = f"{w}x{h}_{precision}_hg{int(use_hg)}_{compile_mode}_"
                for line in lines:
                    if not line.startswith(legacy_prefix):
                        continue
                    suffix = line[len(legacy_prefix):]
                    if not suffix:
                        continue
                    # Exclude keyed entries that already include pdq mode.
                    if suffix.startswith(("auto_", "on_", "off_")):
                        continue
                    return True
    return False


def _mark_compiled(
    w: int,
    h: int,
    precision: str,
    model_path: str,
    use_hg: bool,
    predequantize_mode: str = "auto",
    compile_mode: str = "max-autotune",
):
    """Record that kernels for this config were compiled cleanly."""
    mp = _compiled_marker_path()
    mp.parent.mkdir(parents=True, exist_ok=True)
    key = _compiled_key(
        w,
        h,
        precision,
        model_path,
        use_hg,
        predequantize_mode=predequantize_mode,
        compile_mode=compile_mode,
    )
    existing = set()
    if mp.is_file():
        existing = set(mp.read_text(encoding="utf-8").splitlines())
    existing.add(key)
    mp.write_text("\n".join(sorted(existing)) + "\n", encoding="utf-8")


def _precision_to_compile_arg(gui_precision: str) -> str:
    """Map GUI precision label to compile/precompile precision argument."""
    return {
        "FP16": "fp16",
        "FP32": "fp32",
        "INT8 Mixed (PTQ)": "int8-mixed",
        "INT8 Mixed (QAT)": "int8-mixed",
        "INT8 Full (PTQ)": "int8-full",
        "INT8 Full (QAT)": "int8-full",
    }.get(str(gui_precision), "fp16")
