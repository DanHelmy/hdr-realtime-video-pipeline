"""
Standalone Triton kernel compiler - runs in a clean process with zero
GPU overhead (no GUI, no mpv, no display) so that torch.compile
max-autotune benchmarks are accurate.

Usage:
    # Compile for one resolution:
    python src/compile_kernels.py 1920x1080

    # Compile for multiple resolutions:
    python src/compile_kernels.py 1920x1080 1440x1080

    # Different precision:
    python src/compile_kernels.py --precision int8-mixed --model path/to/model.pt 1920x1080

    # Clear cache first then compile:
    python src/compile_kernels.py --clear-cache 1920x1080

    # Force compile when HIP SDK auto-detection misses on ROCm-Windows:
    python src/compile_kernels.py --force-compile 1920x1080

The GUI's "Tools -> Pre-compile Kernels" runs this script as a subprocess
and monitors its stdout for progress.

Exit codes:
    0  - success
    1  - error (message on stderr)
"""

import argparse
import os
import sys
import time

from windows_runtime import ensure_windows_supported, project_cache_root

# Pin caches to a stable user path (avoid Temp churn/permissions).
ensure_windows_supported("HDRTVNet++ kernel compiler")

_cache_root = project_cache_root(__file__)
try:
    os.makedirs(_cache_root, exist_ok=True)
except Exception:
    pass

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_cache_root, "torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache_root, "triton"))
# Force UTF-8 stdout/stderr on Windows - model code prints Unicode
# characters (->, x, etc.) that crash under the default cp1252 encoding.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Persist autotune decisions to disk so the GUI process can reuse them
# without re-benchmarking.
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

# Ensure imports resolve when run from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _weight(name):
    return os.path.join(_HERE, "models", "weights", name)


# Precision -> (precision_str, default_model_path)
_PRECISION_MAP = {
    "fp16": ("fp16", _weight("Ensemble_AGCM_LE.pth")),
    "fp32": ("fp32", _weight("Ensemble_AGCM_LE.pth")),
    "int8-mixed": ("int8-mixed", _weight("Ensemble_AGCM_LE_int8_mixed_qat.pt")),
    "int8-full": ("int8-full", _weight("Ensemble_AGCM_LE_int8_full.pt")),
}


def _mark_compiled(
    w: int,
    h: int,
    precision: str,
    *,
    model_path: str,
    use_hg: bool,
    predequantize_mode: str,
):
    """Record that kernels for this exact compile config were compiled."""
    from gui_compile_cache import _mark_compiled as _mark_compiled_cache

    _mark_compiled_cache(
        w,
        h,
        precision,
        model_path=model_path,
        use_hg=bool(use_hg),
        predequantize_mode=str(predequantize_mode or "auto"),
    )


def _effective_marker_predequantize_mode(
    precision: str,
    args_predequantize: str,
    processor,
) -> str:
    if not str(precision or "").strip().lower().startswith("int8"):
        return "auto"
    mode = str(args_predequantize or "auto").strip().lower()
    if mode not in {"auto", "on", "off"}:
        mode = "auto"
    if mode != "auto":
        return mode
    # Match the actual runtime graph that got compiled. Older versions wrote
    # "auto" literally, which made GUI playback miss AMD/NVIDIA auto-resolved
    # INT8 cache entries and show false first-time warnings.
    try:
        is_w8_model = bool(getattr(processor, "_is_w8_model"))
    except Exception:
        return "auto"
    return "off" if is_w8_model else "on"


def _clear_caches():
    """Delete Triton and TorchInductor kernel caches."""
    import shutil
    import pathlib
    import getpass
    import tempfile

    cleared = []

    triton_root = pathlib.Path(
        os.environ.get("TRITON_CACHE_DIR", os.path.join(project_cache_root(__file__), "triton"))
    )
    triton_dir = triton_root / "cache"
    if triton_dir.exists():
        shutil.rmtree(triton_dir, ignore_errors=True)
        cleared.append(str(triton_dir))

    inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not inductor_dir:
        inductor_dir = os.path.join(
            tempfile.gettempdir(),
            f"torchinductor_{getpass.getuser()}",
        )
    inductor_path = pathlib.Path(inductor_dir)
    if inductor_path.exists():
        shutil.rmtree(inductor_path, ignore_errors=True)
        cleared.append(str(inductor_path))

    return cleared


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compile Triton kernels for specific resolutions."
    )
    parser.add_argument(
        "resolutions",
        nargs="+",
        metavar="WxH",
        help="One or more resolutions to compile (e.g. 1920x1080 1440x1080)",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=list(_PRECISION_MAP.keys()),
        help="Model precision (default: fp16)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model weights path (default: auto based on precision)",
    )
    parser.add_argument(
        "--use-hg",
        default="1",
        choices=["1", "0"],
        help="Enable HG refinement (1/0). Default: 1",
    )
    parser.add_argument(
        "--hg-weights",
        default=None,
        help="Path to HG_weights.pth (overrides default path)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete kernel caches before compiling",
    )
    parser.add_argument(
        "--force-compile",
        action="store_true",
        help="Force torch.compile on ROCm-Windows even if HIP SDK auto-detection fails",
    )
    parser.add_argument(
        "--predequantize",
        default="auto",
        choices=["auto", "on", "off"],
        help=(
            "INT8 pre-dequantization mode: "
            "'auto' (default), 'on' (force), 'off' (disable)."
        ),
    )
    args = parser.parse_args()

    # Parse resolutions first (fail fast on bad input)
    parsed_res = []
    for r in args.resolutions:
        try:
            w, h = r.lower().split("x")
            parsed_res.append((int(w), int(h)))
        except (ValueError, AttributeError):
            print(f"ERROR: Invalid resolution '{r}', expected WxH (e.g. 1920x1080)",
                  file=sys.stderr)
            sys.exit(1)

    # Clear caches if requested
    if args.clear_cache:
        print("[compile] Clearing kernel caches ...")
        cleared = _clear_caches()
        for d in cleared:
            print(f"[compile]   Deleted: {d}")
        if not cleared:
            print("[compile]   No caches found.")
        print("[compile] Cache cleared.")

    # Resolve model path
    prec_str, default_model = _PRECISION_MAP[args.precision]
    model_path = args.model or default_model
    if not os.path.isfile(model_path):
        print(f"ERROR: Model weights not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    if args.hg_weights and not os.path.isfile(args.hg_weights):
        print(f"ERROR: HG weights not found: {args.hg_weights}", file=sys.stderr)
        sys.exit(1)

    # Import heavy dependencies only after arg parsing
    import torch
    from models.hdrtvnet_torch import HDRTVNetTorch

    print(f"[compile] Loading model: {args.precision} - {os.path.basename(model_path)}")
    sys.stdout.flush()

    predeq = {"auto": "auto", "on": True, "off": False}[args.predequantize]

    processor = HDRTVNetTorch(
        model_path,
        device="auto",
        precision=prec_str,
        compile_model=True,
        force_compile=bool(args.force_compile),
        compile_mode="max-autotune",
        predequantize=predeq,
        hg_weights=args.hg_weights,
        use_hg=str(args.use_hg).strip() != "0",
    )

    if not (hasattr(processor, "_compiled") and processor._compiled):
        print("[compile] WARNING: torch.compile not active - nothing to compile.")
        sys.exit(0)

    total = len(parsed_res)
    for i, (w, h) in enumerate(parsed_res, 1):
        print(f"[compile] ({i}/{total}) Compiling kernels for {w}x{h} ...")
        sys.stdout.flush()
        t0 = time.perf_counter()
        processor.warmup_compile(w, h)
        elapsed = time.perf_counter() - t0
        print(f"[compile] ({i}/{total}) {w}x{h} done in {elapsed:.1f}s")
        sys.stdout.flush()

        # Write marker so the GUI knows this resolution is compiled
        _mark_compiled(
            w,
            h,
            args.precision,
            model_path=model_path,
            use_hg=str(args.use_hg).strip() != "0",
            predequantize_mode=_effective_marker_predequantize_mode(
                args.precision,
                args.predequantize,
                processor,
            ),
        )

    print("[compile] All resolutions compiled successfully.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

