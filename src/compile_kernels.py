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

The GUI's "Tools -> Pre-compile Kernels" runs this script as a subprocess
and monitors its stdout for progress.

Exit codes:
    0  - success
    1  - error (message on stderr)
"""

import argparse
import os
import pathlib
import sys
import time

# Pin caches to a stable user path (avoid Temp churn/permissions).
def _default_cache_root():
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        return os.path.join(local_app, "HDRTVNetCache")
    return os.path.join(os.path.expanduser("~"), ".cache", "hdrtvnet")

_cache_root = os.environ.get("HDRTVNET_CACHE_DIR", _default_cache_root())
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


# -- Marker file (shared with gui.py) -----------------------------------------
_TRITON_CACHE = (
    pathlib.Path(os.environ.get("TRITON_CACHE_DIR", pathlib.Path.home() / ".triton"))
    / "cache"
)


def _mark_compiled(w: int, h: int, precision: str):
    """Record that kernels for this resolution+precision were compiled."""
    mp = _TRITON_CACHE / "hdrtvnet_compiled.txt"
    mp.parent.mkdir(parents=True, exist_ok=True)
    key = f"{w}x{h}_{precision}"
    existing = set()
    if mp.is_file():
        existing = set(mp.read_text(encoding="utf-8").splitlines())
    existing.add(key)
    mp.write_text("\n".join(sorted(existing)) + "\n", encoding="utf-8")


def _clear_caches():
    """Delete Triton and TorchInductor kernel caches."""
    import shutil
    import pathlib
    import getpass
    import tempfile

    cleared = []

    triton_root = pathlib.Path(
        os.environ.get("TRITON_CACHE_DIR", pathlib.Path.home() / ".triton")
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
        "--clear-cache",
        action="store_true",
        help="Delete kernel caches before compiling",
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

    # Import heavy dependencies only after arg parsing
    import torch
    from models.hdrtvnet_torch import HDRTVNetTorch

    print(f"[compile] Loading model: {args.precision} - {os.path.basename(model_path)}")
    sys.stdout.flush()

    processor = HDRTVNetTorch(
        model_path,
        device="auto",
        precision=prec_str,
        compile_model=True,
        compile_mode="max-autotune",
        predequantize="auto",
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
        _mark_compiled(w, h, args.precision)

    print("[compile] All resolutions compiled successfully.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

