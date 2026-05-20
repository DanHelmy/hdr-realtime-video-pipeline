"""Kernel compile-cache marker utilities shared by GUI compile flows."""

from __future__ import annotations

import hashlib as _hashlib
import os
import pathlib as _pathlib
import time as _time

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


def _marker_payload(line: str) -> str:
    text = str(line or "").strip()
    if ":" in text:
        return text.split(":", 1)[1]
    return text


def _read_marker_lines(marker_path: _pathlib.Path) -> set[str]:
    try:
        return set(marker_path.read_text(encoding="utf-8").splitlines())
    except Exception:
        return set()


def _current_profile_root() -> _pathlib.Path | None:
    try:
        # marker path is <profile>/triton/cache/hdrtvnet_compiled.txt
        return _compiled_marker_path().parents[2]
    except Exception:
        return None


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


def _normalize_predequantize_mode(predequantize_mode: str | None) -> str:
    pdq = str(predequantize_mode or "auto").strip().lower()
    if pdq not in {"auto", "on", "off"}:
        pdq = "auto"
    return pdq


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


def _marker_lines_match_config(
    lines: set[str],
    *,
    key: str,
    w: int,
    h: int,
    precision: str,
    use_hg: bool,
    predequantize_mode: str,
    compile_mode: str,
    memory_format: str,
) -> bool:
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
            predequantize_mode=str(predequantize_mode),
            memory_format=memory_format,
        ):
            return True
    if not str(precision).startswith("int8"):
        # FP16/FP32 do not have distinct runtime graphs for pre-dequantize
        # modes. Accept older markers that were mistakenly written with
        # "_on" / "_off" suffixes.
        prefix_payload = (
            f"{w}x{h}_{precision}_hg{int(use_hg)}_{compile_mode}_{memory_format}_"
        )
        for line in lines:
            if _marker_payload(line).startswith(prefix_payload):
                return True
    if str(precision).startswith("int8"):
        # Compatibility inside the current repo namespace: older markers used
        # model-hash suffixes, so PTQ/QAT variants could differ only in the
        # final token.
        prefix_payload = (
            f"{w}x{h}_{precision}_hg{int(use_hg)}_{compile_mode}_{memory_format}_{predequantize_mode}_"
        )
        for line in lines:
            if _marker_payload(line).startswith(prefix_payload):
                return True
    return False


def _profile_has_kernel_cache(profile_root: _pathlib.Path) -> bool:
    return (
        (profile_root / "torchinductor").is_dir()
        and (profile_root / "triton").is_dir()
        and (profile_root / "triton" / "cache" / "hdrtvnet_compiled.txt").is_file()
    )


def _safe_archive_path(path: _pathlib.Path) -> _pathlib.Path:
    stamp = _time.strftime("%Y%m%d_%H%M%S")
    base = path.with_name(f"{path.name}.stale-{stamp}")
    candidate = base
    index = 1
    while candidate.exists():
        candidate = path.with_name(f"{base.name}-{index}")
        index += 1
    return candidate


def _promote_compatible_profile(
    *,
    key: str,
    w: int,
    h: int,
    precision: str,
    use_hg: bool,
    predequantize_mode: str,
    compile_mode: str,
    memory_format: str,
) -> bool:
    """Rename a compatible older namespace into the current cache namespace."""
    if str(os.environ.get("HDRTVNET_CACHE_DIR", "")).strip():
        return False

    current_profile = _current_profile_root()
    if current_profile is None:
        return False
    profiles_root = current_profile.parent
    if not profiles_root.is_dir():
        return False

    try:
        current_profile_resolved = current_profile.resolve()
    except Exception:
        current_profile_resolved = current_profile

    candidates: list[_pathlib.Path] = []
    try:
        for path in profiles_root.iterdir():
            if not path.is_dir():
                continue
            try:
                if path.resolve() == current_profile_resolved:
                    continue
            except Exception:
                if path == current_profile:
                    continue
            if not _profile_has_kernel_cache(path):
                continue
            marker = path / "triton" / "cache" / "hdrtvnet_compiled.txt"
            lines = _read_marker_lines(marker)
            if _marker_lines_match_config(
                lines,
                key=key,
                w=w,
                h=h,
                precision=precision,
                use_hg=use_hg,
                predequantize_mode=predequantize_mode,
                compile_mode=compile_mode,
                memory_format=memory_format,
            ):
                candidates.append(path)
    except Exception:
        return False

    if not candidates:
        return False
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    source_profile = candidates[0]

    try:
        current_profile.parent.mkdir(parents=True, exist_ok=True)
        if current_profile.exists():
            current_marker = _compiled_marker_path()
            current_lines = _read_marker_lines(current_marker) if current_marker.is_file() else set()
            if _marker_lines_match_config(
                current_lines,
                key=key,
                w=w,
                h=h,
                precision=precision,
                use_hg=use_hg,
                predequantize_mode=predequantize_mode,
                compile_mode=compile_mode,
                memory_format=memory_format,
            ):
                return True
            current_profile.rename(_safe_archive_path(current_profile))
        source_profile.rename(current_profile)
        print(
            "Kernel compile cache profile promoted: "
            f"{source_profile.name} -> {current_profile.name}"
        )
        return True
    except Exception as exc:
        try:
            print(f"Kernel compile cache profile promotion skipped: {exc}")
        except Exception:
            pass
        return False


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
    pdq = _normalize_predequantize_mode(predequantize_mode)
    fmt = _normalize_memory_format(memory_format)
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
    mp = _compiled_marker_path()
    if mp.is_file():
        lines = _read_marker_lines(mp)
        if _marker_lines_match_config(
            lines,
            key=key,
            w=w,
            h=h,
            precision=precision,
            use_hg=use_hg,
            predequantize_mode=pdq,
            compile_mode=compile_mode,
            memory_format=fmt,
        ):
            return True
    if _promote_compatible_profile(
        key=key,
        w=w,
        h=h,
        precision=precision,
        use_hg=use_hg,
        predequantize_mode=pdq,
        compile_mode=compile_mode,
        memory_format=fmt,
    ):
        lines = _read_marker_lines(mp)
        return _marker_lines_match_config(
            lines,
            key=key,
            w=w,
            h=h,
            precision=precision,
            use_hg=use_hg,
            predequantize_mode=pdq,
            compile_mode=compile_mode,
            memory_format=fmt,
        )
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
