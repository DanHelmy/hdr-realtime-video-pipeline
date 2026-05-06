"""Kernel compile-cache marker utilities shared by GUI compile flows."""

from __future__ import annotations

import hashlib as _hashlib
import os
import pathlib as _pathlib

from windows_runtime import compile_cache_namespace, project_cache_root

_TRITON_CACHE = (
    _pathlib.Path(
        os.environ.get("TRITON_CACHE_DIR", os.path.join(project_cache_root(__file__), "triton"))
    )
    / "cache"
)
_CACHE_NAMESPACE = compile_cache_namespace(__file__)
DEFAULT_COMPILE_MODE = "max-autotune"
DEFAULT_MEMORY_FORMAT = (
    "channels-last"
    if str(os.environ.get("HDRTVNET_CHANNELS_LAST", "")).strip().lower()
    in {"1", "true", "yes", "on", "y"}
    else "contiguous"
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


def _normalize_memory_format(memory_format: str | None = None) -> str:
    text = str(memory_format or DEFAULT_MEMORY_FORMAT).strip().lower()
    text = text.replace("_", "-")
    if text in {"channels-last", "nhwc"}:
        return "channels-last"
    return "contiguous"


def _compiled_key(
    w: int,
    h: int,
    precision: str,
    model_path: str,
    use_hg: bool,
    predequantize_mode: str = "auto",
    compile_mode: str = DEFAULT_COMPILE_MODE,
    memory_format: str = DEFAULT_MEMORY_FORMAT,
) -> str:
    pdq = str(predequantize_mode or "auto").strip().lower()
    if pdq not in {"auto", "on", "off"}:
        pdq = "auto"
    fmt = _normalize_memory_format(memory_format)
    mh = _model_compile_id(precision, model_path)
    return f"{_CACHE_NAMESPACE}:{w}x{h}_{precision}_hg{int(use_hg)}_{compile_mode}_{fmt}_{pdq}_{mh}"


def _payload_shares_fp16_predequantized_graph(
    payload: str,
    *,
    w: int,
    h: int,
    precision: str,
    use_hg: bool,
    compile_mode: str,
    predequantize_mode: str,
    memory_format: str,
) -> bool:
    text = str(payload or "").strip()
    if not text:
        return False
    prefix_base = f"{w}x{h}_"
    if not text.startswith(prefix_base):
        return False

    fmt = _normalize_memory_format(memory_format)
    if str(precision).startswith("int8") and str(predequantize_mode) == "on":
        fp16_prefix = f"{w}x{h}_fp16_hg{int(use_hg)}_{compile_mode}_{fmt}_"
        return text.startswith(fp16_prefix)

    if str(precision) == "fp16":
        int8_prefix = f"{w}x{h}_int8-"
        int8_suffix = f"_hg{int(use_hg)}_{compile_mode}_{fmt}_on_"
        return text.startswith(int8_prefix) and int8_suffix in text

    return False


def _legacy_compiled_keys(
    w: int,
    h: int,
    precision: str,
    model_path: str,
    use_hg: bool,
    compile_mode: str = DEFAULT_COMPILE_MODE,
    memory_format: str = DEFAULT_MEMORY_FORMAT,
) -> tuple[str]:
    mh = _model_hash(model_path)
    fmt = _normalize_memory_format(memory_format)
    # Legacy GUI marker format (no predequantization mode)
    k1 = f"{w}x{h}_{precision}_hg{int(use_hg)}_{compile_mode}_{fmt}_{mh}"
    return (k1,)


def _is_compiled(
    w: int,
    h: int,
    precision: str,
    model_path: str,
    use_hg: bool,
    predequantize_mode: str = "auto",
    compile_mode: str = DEFAULT_COMPILE_MODE,
    memory_format: str = DEFAULT_MEMORY_FORMAT,
) -> bool:
    """Check if clean-compiled kernels exist for this config."""
    mp = _compiled_marker_path()
    if mp.is_file():
        def _marker_payload(line: str) -> str:
            text = str(line or "").strip()
            if ":" in text:
                return text.split(":", 1)[1]
            return text

        pdq = str(predequantize_mode or "auto").strip().lower()
        if pdq not in {"auto", "on", "off"}:
            pdq = "auto"
        fmt = _normalize_memory_format(memory_format)
        lines = set(mp.read_text(encoding="utf-8").splitlines())
        key = _compiled_key(
            w,
            h,
            precision,
            model_path,
            use_hg,
            predequantize_mode=pdq,
            compile_mode=compile_mode,
            memory_format=fmt,
        )
        key_payload = _marker_payload(key)
        if key in lines:
            return True
        for line in lines:
            if _marker_payload(line) == key_payload:
                return True
        for line in lines:
            if _payload_shares_fp16_predequantized_graph(
                _marker_payload(line),
                w=w,
                h=h,
                precision=str(precision),
                use_hg=bool(use_hg),
                compile_mode=str(compile_mode),
                predequantize_mode=str(pdq),
                memory_format=fmt,
            ):
                return True
        if not str(precision).startswith("int8"):
            # FP16/FP32 do not have distinct runtime graphs for pre-dequantize
            # modes. Accept older markers that were mistakenly written with
            # "_on" / "_off" suffixes.
            prefix_payload = (
                f"{w}x{h}_{precision}_hg{int(use_hg)}_{compile_mode}_{fmt}_"
            )
            for line in lines:
                if _marker_payload(line).startswith(prefix_payload):
                    return True
        if str(precision).startswith("int8"):
            # Compatibility inside the current repo namespace: older markers
            # used model-hash suffixes, so PTQ/QAT variants could differ only
            # in the final token.
            prefix_payload = (
                f"{w}x{h}_{precision}_hg{int(use_hg)}_{compile_mode}_{fmt}_{pdq}_"
            )
            for line in lines:
                if _marker_payload(line).startswith(prefix_payload):
                    return True
    return False


def _mark_compiled(
    w: int,
    h: int,
    precision: str,
    model_path: str,
    use_hg: bool,
    predequantize_mode: str = "auto",
    compile_mode: str = DEFAULT_COMPILE_MODE,
    memory_format: str = DEFAULT_MEMORY_FORMAT,
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
        memory_format=memory_format,
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
        "INT8 Mixed (QAT) (Film)": "int8-mixed",
        "INT8 Full (PTQ)": "int8-full",
        "INT8 Full (QAT)": "int8-full",
        "INT8 Full (QAT) (Film)": "int8-full",
    }.get(str(gui_precision), "fp16")
