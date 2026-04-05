from __future__ import annotations

import os
import sys
from pathlib import Path

from windows_runtime import (
    default_cache_root,
    enable_high_resolution_timer,
    ensure_windows_supported,
)


def _prepend_dll_search_path(path: str):
    if not path or not os.path.isdir(path):
        return
    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(path)
    except Exception:
        pass


def _install_rocm_preload_guard():
    """Make torch ROCm preload more resilient in frozen app layouts.

    Some packaged environments can fail on the first `amd_comgr` preload even
    when the rest of ROCm runtime is usable. In that case we retry without that
    entry and, as a last resort, skip preload instead of hard-crashing at import
    time.
    """
    try:
        import rocm_sdk
    except Exception:
        return

    if getattr(rocm_sdk, "_hdrtvnet_preload_guard_installed", False):
        return

    original_init = getattr(rocm_sdk, "initialize_process", None)
    if original_init is None:
        return

    def _safe_initialize_process(**kwargs):
        try:
            return original_init(**kwargs)
        except Exception as exc:
            preload = list(kwargs.get("preload_shortnames") or [])
            if preload and "amd_comgr" in preload:
                retry_kwargs = dict(kwargs)
                retry_kwargs["preload_shortnames"] = [
                    name for name in preload if name != "amd_comgr"
                ]
                try:
                    print(
                        "[HDRTVNet] ROCm preload fallback: retrying without "
                        "'amd_comgr'.",
                        file=sys.stderr,
                    )
                except Exception:
                    pass
                try:
                    return original_init(**retry_kwargs)
                except Exception:
                    pass
            try:
                print(
                    f"[HDRTVNet] ROCm preload skipped due to startup error: {exc}",
                    file=sys.stderr,
                )
            except Exception:
                pass
            return None

    rocm_sdk.initialize_process = _safe_initialize_process
    rocm_sdk._hdrtvnet_preload_guard_installed = True


def _try_preload_bundled_rocm_dlls(search_roots: list[str]):
    """Best-effort preload for bundled ROCm runtime DLLs.

    In some frozen environments, proactively loading ROCm DLLs stabilizes
    subsequent torch/rocm_sdk initialization.
    """
    try:
        import ctypes
    except Exception:
        return

    ordered_patterns = [
        "amd_comgr*.dll",
        "amdhip64*.dll",
        "hiprtc0*.dll",
        "rocm-openblas*.dll",
        "hipblas*.dll",
        "hipfft*.dll",
        "hiprand*.dll",
        "hipsparse*.dll",
        "hipsolver*.dll",
        "libhipblaslt*.dll",
        "MIOpen*.dll",
    ]

    for root in search_roots:
        p = Path(root)
        if not p.is_dir():
            continue
        for pattern in ordered_patterns:
            for dll_path in sorted(p.glob(pattern)):
                try:
                    ctypes.CDLL(str(dll_path))
                except Exception:
                    # Non-fatal: keep trying the rest.
                    continue


def prepare_runtime_environment(current_file: str) -> tuple[str, str]:
    """Configure cache and DLL lookup paths before torch/PyQt are imported."""
    ensure_windows_supported("HDRTVNet++ GUI")
    enable_high_resolution_timer(1)
    cache_root = os.environ.get("HDRTVNET_CACHE_DIR", default_cache_root())
    try:
        os.makedirs(cache_root, exist_ok=True)
    except Exception:
        pass

    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        os.path.join(cache_root, "torchinductor"),
    )
    os.environ.setdefault(
        "TRITON_CACHE_DIR",
        os.path.join(cache_root, "triton"),
    )
    # Persist autotune graph cache so subprocess-compiled kernels are reused.
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

    here = os.path.dirname(os.path.abspath(current_file))
    root = os.path.dirname(here)
    # Ensure bundled DLLs are discoverable before torch import.
    _prepend_dll_search_path(here)

    # Frozen onedir layout keeps bundled libs under _internal/.
    rocm_bins = [
        os.path.join(here, "_rocm_sdk_core", "bin"),
        os.path.join(here, "_rocm_sdk_devel", "bin"),
        os.path.join(here, "_rocm_sdk_libraries_custom", "bin"),
        os.path.join(root, "_internal", "_rocm_sdk_core", "bin"),
        os.path.join(root, "_internal", "_rocm_sdk_devel", "bin"),
        os.path.join(root, "_internal", "_rocm_sdk_libraries_custom", "bin"),
    ]
    for bin_dir in rocm_bins:
        _prepend_dll_search_path(bin_dir)

    _try_preload_bundled_rocm_dlls(rocm_bins)
    _install_rocm_preload_guard()

    return here, root
