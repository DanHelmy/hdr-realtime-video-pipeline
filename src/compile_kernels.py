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

from windows_runtime import (
    configure_rocm_sdk_environment,
    ensure_windows_supported,
    install_torch_windows_warning_filter,
    project_cache_root,
)

# Pin PyTorch/Triton caches inside this checkout so generated kernels are visible.
ensure_windows_supported("HDRTVNet++ kernel compiler")
install_torch_windows_warning_filter()
configure_rocm_sdk_environment()

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


def _int_env(name: str, default: int, *, minimum: int = 0, maximum: int = 999) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = default
    return min(maximum, max(minimum, value))


def _float_env(name: str, default: float, *, minimum: float = 0.0, maximum: float = 999.0) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except Exception:
        value = default
    return min(maximum, max(minimum, value))


def _finish_success(message: str) -> None:
    """Print success and exit without waiting on ROCm/PyTorch teardown.

    On ROCm-Windows the helper can finish compiling, flush all logs, and then
    sit in process shutdown for a long time. The cache marker is already on
    disk by this point, so a hard successful exit is safer for the GUI than
    letting users cancel an already-finished compile.
    """
    print(message)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    if str(os.environ.get("HDRTVNET_COMPILE_HARD_EXIT", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        os._exit(0)


def _weight(name):
    return os.path.join(_HERE, "models", "weights", name)


# Precision -> (precision_str, default_hg_model_path, default_nohg_model_path)
_PRECISION_MAP = {
    "fp16": (
        "fp16",
        _weight("distilled/hr/HR_qfriendly_spatialmixglobal_fp32.pt"),
        _weight("distilled/hr/HR_qfriendly_spatialmixglobal_fp32.pt"),
    ),
    "fp32": (
        "fp32",
        _weight("distilled/hr/HR_qfriendly_spatialmixglobal_fp32.pt"),
        _weight("distilled/hr/HR_qfriendly_spatialmixglobal_fp32.pt"),
    ),
    "int8-mixed": (
        "int8-mixed",
        _weight("pytorch_int8/hg/HR_HG_int8_mixed_qat.pt"),
        _weight("pytorch_int8/hr/HR_int8_mixed_qat.pt"),
    ),
    "int8-full": (
        "int8-full",
        _weight("pytorch_int8/hg/HR_HG_int8_full.pt"),
        _weight("pytorch_int8/hr/HR_int8_full.pt"),
    ),
}


def _mark_compiled(
    w: int,
    h: int,
    precision: str,
    *,
    model_path: str,
    use_hg: bool,
    predequantize_mode: str,
    compile_mode: str,
    memory_format: str,
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
        compile_mode=str(compile_mode or "max-autotune"),
        memory_format=str(memory_format or "contiguous"),
    )


def _is_marked_compiled(
    w: int,
    h: int,
    precision: str,
    *,
    model_path: str,
    use_hg: bool,
    predequantize_mode: str,
    compile_mode: str,
    memory_format: str,
) -> bool:
    from gui_compile_cache import _is_compiled as _is_compiled_cache

    return bool(
        _is_compiled_cache(
            w,
            h,
            precision,
            model_path=model_path,
            use_hg=bool(use_hg),
            predequantize_mode=str(predequantize_mode or "auto"),
            compile_mode=str(compile_mode or "max-autotune"),
            memory_format=str(memory_format or "contiguous"),
        )
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
        inductor_dir = os.path.join(project_cache_root(__file__), "torchinductor")
    inductor_path = pathlib.Path(inductor_dir)
    if inductor_path.exists():
        shutil.rmtree(inductor_path, ignore_errors=True)
        cleared.append(str(inductor_path))

    return cleared


def _cache_targets():
    """Return the cache folders that define one autotune result."""
    import pathlib

    triton_root = pathlib.Path(
        os.environ.get("TRITON_CACHE_DIR", os.path.join(project_cache_root(__file__), "triton"))
    )
    inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not inductor_dir:
        inductor_dir = os.path.join(project_cache_root(__file__), "torchinductor")
    return (
        ("triton_cache", triton_root / "cache"),
        ("torchinductor", pathlib.Path(inductor_dir)),
    )


def _snapshot_caches(snapshot_dir: str) -> None:
    import pathlib
    import shutil

    dst_root = pathlib.Path(snapshot_dir)
    dst_root.mkdir(parents=True, exist_ok=True)
    for label, source in _cache_targets():
        if not source.exists():
            continue
        dst = dst_root / label
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(source, dst)


def _restore_caches(snapshot_dir: str) -> None:
    import pathlib
    import shutil

    _clear_caches()
    src_root = pathlib.Path(snapshot_dir)
    for label, target in _cache_targets():
        source = src_root / label
        if not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)


def _benchmark_quality_score(processor, parsed_res, *, warmup: int, runs: int) -> float:
    """Return median end-to-end processing time across compiled resolutions."""
    import statistics
    import numpy as np
    import torch

    warmup = max(0, int(warmup))
    runs = max(1, int(runs))
    scores: list[float] = []
    for w, h in parsed_res:
        dummy = np.zeros((int(h), int(w), 3), dtype=np.uint8)
        for _ in range(warmup):
            processor.process(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        samples: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            processor.process(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)

        median_ms = statistics.median(samples)
        scores.append(float(median_ms))
        print(
            f"[quality] benchmark {int(w)}x{int(h)}: "
            f"median {median_ms:.3f} ms over {runs} runs",
            flush=True,
        )

    if not scores:
        return float("inf")
    return sum(scores) / len(scores)


def _prewarm_gpu_for_autotune(processor, torch_mod, seconds: float) -> None:
    """Stabilize GPU clocks before Triton starts timing kernel candidates.

    This deliberately avoids the HDRTVNet model, so it cannot compile or cache
    the wrong input shape. It is only meant to wake the GPU power/clock state.
    """
    seconds = max(0.0, float(seconds or 0.0))
    if seconds <= 0.0:
        return
    if not bool(getattr(processor, "_use_cuda", False)):
        return
    if not torch_mod.cuda.is_available():
        return

    try:
        size = _int_env(
            "HDRTVNET_AUTOTUNE_GPU_PREWARM_SIZE",
            2048,
            minimum=256,
            maximum=4096,
        )
        device = getattr(processor, "device", torch_mod.device("cuda:0"))
        dtype = getattr(processor, "_dtype", torch_mod.float16)
        if dtype not in {torch_mod.float16, torch_mod.bfloat16, torch_mod.float32}:
            dtype = torch_mod.float16

        print(
            f"[compile] Pre-warming GPU clocks for {seconds:.1f}s "
            f"before autotune ({size}x{size} matmul) ...",
            flush=True,
        )
        a = torch_mod.randn((size, size), device=device, dtype=dtype)
        b = torch_mod.randn((size, size), device=device, dtype=dtype)
        out = torch_mod.empty_like(a)

        t0 = time.perf_counter()
        deadline = t0 + seconds
        iters = 0
        with torch_mod.inference_mode():
            while time.perf_counter() < deadline:
                torch_mod.mm(a, b, out=out)
                iters += 1
                if iters % 4 == 0:
                    torch_mod.cuda.synchronize(device)
        torch_mod.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        print(
            f"[compile] GPU pre-warm complete "
            f"({iters} matmuls in {elapsed:.1f}s).",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[compile] WARNING: GPU pre-warm skipped: {exc}",
            flush=True,
        )


def _run_quality_trials(args, parsed_res) -> int:
    """Run isolated clean compiles and keep the fastest measured cache."""
    import pathlib
    import shutil
    import subprocess
    import tempfile

    trials = max(1, int(args.quality_trials))
    trial_roots: list[str] = []
    results: list[tuple[float, str, int]] = []
    script = os.path.abspath(__file__)

    print(
        f"[quality] Running {trials} clean autotune trials; "
        "the fastest measured cache will be kept.",
        flush=True,
    )
    if args.clear_cache:
        print("[quality] --clear-cache is implied for each trial.", flush=True)

    try:
        for trial in range(1, trials + 1):
            trial_root = tempfile.mkdtemp(prefix=f"hdrtvnet_autotune_trial_{trial}_")
            trial_roots.append(trial_root)
            score_file = os.path.join(trial_root, "score.txt")
            snapshot_dir = os.path.join(trial_root, "snapshot")

            cmd = [sys.executable, "-u", script]
            cmd += list(args.resolutions)
            cmd += ["--precision", args.precision]
            if args.model:
                cmd += ["--model", args.model]
            cmd += ["--use-hg", str(args.use_hg)]
            if args.hg_weights:
                cmd += ["--hg-weights", args.hg_weights]
            cmd += ["--predequantize", str(args.predequantize)]
            if args.force_compile:
                cmd += ["--force-compile"]
            cmd += ["--compile-mode", str(args.compile_mode)]
            cmd += ["--gpu-prewarm-seconds", str(args.gpu_prewarm_seconds)]
            cmd += [
                "--clear-cache",
                "--quality-child",
                "--quality-score-file",
                score_file,
                "--quality-benchmark-warmup",
                str(args.quality_benchmark_warmup),
                "--quality-benchmark-runs",
                str(args.quality_benchmark_runs),
            ]

            print(f"[quality] Trial {trial}/{trials} starting ...", flush=True)
            completed = subprocess.run(cmd, cwd=os.path.dirname(_HERE))
            if completed.returncode != 0:
                print(
                    f"[quality] Trial {trial}/{trials} failed "
                    f"(exit {completed.returncode}).",
                    file=sys.stderr,
                    flush=True,
                )
                return completed.returncode or 1

            try:
                score = float(pathlib.Path(score_file).read_text(encoding="utf-8").strip())
            except Exception:
                print(
                    f"[quality] Trial {trial}/{trials} did not write a valid score.",
                    file=sys.stderr,
                    flush=True,
                )
                return 1

            _snapshot_caches(snapshot_dir)
            results.append((score, snapshot_dir, trial))
            print(
                f"[quality] Trial {trial}/{trials} score: {score:.3f} ms",
                flush=True,
            )

        best_score, best_snapshot, best_trial = min(results, key=lambda item: item[0])
        _restore_caches(best_snapshot)
        print(
            f"[quality] Keeping trial {best_trial}/{trials} "
            f"({best_score:.3f} ms median aggregate).",
            flush=True,
        )
        print("[compile] All resolutions compiled successfully.", flush=True)
        return 0
    finally:
        for root in trial_roots:
            shutil.rmtree(root, ignore_errors=True)


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
        help="Path to original/HG.pt or distilled HG weights (overrides default path)",
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
        "--compile-mode",
        default="auto",
        choices=[
            "auto",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help=(
            "torch.compile mode for PyTorch backends "
            "(auto = max-autotune)."
        ),
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
    parser.add_argument(
        "--verify-cache-only",
        action="store_true",
        help=(
            "Verify an existing compile cache by running warmup in a clean "
            "process. Fails instead of compiling when markers are missing."
        ),
    )
    parser.add_argument(
        "--gpu-prewarm-seconds",
        type=float,
        default=_float_env(
            "HDRTVNET_AUTOTUNE_GPU_PREWARM_SECONDS",
            8.0,
            minimum=0.0,
            maximum=120.0,
        ),
        help=(
            "Run simple GPU work before first autotune compile so clocks are "
            "stable before kernel candidates are timed. Default: 8.0 seconds; "
            "set 0 to disable."
        ),
    )
    parser.add_argument(
        "--quality-trials",
        type=int,
        default=_int_env("HDRTVNET_AUTOTUNE_QUALITY_TRIALS", 1, minimum=1, maximum=10),
        help=(
            "Run N independent clean max-autotune compiles and keep the fastest "
            "measured cache. Default: 1."
        ),
    )
    parser.add_argument(
        "--quality-benchmark-warmup",
        type=int,
        default=_int_env("HDRTVNET_AUTOTUNE_QUALITY_BENCHMARK_WARMUP", 2, minimum=0, maximum=20),
        help="Warmup frames per resolution before scoring a quality trial. Default: 2.",
    )
    parser.add_argument(
        "--quality-benchmark-runs",
        type=int,
        default=_int_env("HDRTVNET_AUTOTUNE_QUALITY_BENCHMARK_RUNS", 8, minimum=1, maximum=100),
        help="Timed frames per resolution used to score a quality trial. Default: 8.",
    )
    parser.add_argument(
        "--quality-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quality-score-file",
        default=None,
        help=argparse.SUPPRESS,
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

    if (
        not args.verify_cache_only
        and not args.quality_child
        and int(args.quality_trials) > 1
    ):
        sys.exit(_run_quality_trials(args, parsed_res))

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
    use_hg = str(args.use_hg).strip() != "0"
    prec_str, default_hg_model, default_nohg_model = _PRECISION_MAP[args.precision]
    default_model = default_hg_model if use_hg else default_nohg_model
    model_path = args.model or default_model
    if not os.path.isfile(model_path):
        print(f"ERROR: Model weights not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    if use_hg and not args.hg_weights:
        default_hg_weights = ""
        if args.precision in {"fp16", "fp32"}:
            default_hg_weights = _weight("distilled/hg/HG_qfriendly_directh16_fp32.pt")
        if default_hg_weights and os.path.isfile(default_hg_weights):
            args.hg_weights = default_hg_weights
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
        compile_mode=str(args.compile_mode),
        predequantize=predeq,
        hg_weights=args.hg_weights,
        use_hg=use_hg,
        warmup_passes=0,
    )

    if not (hasattr(processor, "_compiled") and processor._compiled):
        if args.verify_cache_only:
            print(
                "[verify] torch.compile is not active in the verification subprocess.",
                file=sys.stderr,
            )
            sys.exit(2)
        print("[compile] WARNING: torch.compile not active - nothing to compile.")
        sys.exit(0)

    if not args.verify_cache_only:
        _prewarm_gpu_for_autotune(
            processor,
            torch,
            seconds=float(args.gpu_prewarm_seconds),
        )

    total = len(parsed_res)
    effective_marker_mode = _effective_marker_predequantize_mode(
        args.precision,
        args.predequantize,
        processor,
    )
    effective_compile_mode = str(
        getattr(processor, "_compile_mode", None)
        or (
            "max-autotune"
            if str(args.compile_mode) == "auto"
            else args.compile_mode
        )
    )
    effective_memory_format = str(
        getattr(processor, "_memory_format_name", None)
        or "contiguous"
    )
    for i, (w, h) in enumerate(parsed_res, 1):
        if args.verify_cache_only and not _is_marked_compiled(
            w,
            h,
            args.precision,
            model_path=model_path,
            use_hg=use_hg,
            predequantize_mode=effective_marker_mode,
            compile_mode=effective_compile_mode,
            memory_format=effective_memory_format,
        ):
            print(
                f"[verify] Missing compile marker for {w}x{h}; cache is not ready.",
                file=sys.stderr,
            )
            sys.exit(2)

        prefix = "[verify]" if args.verify_cache_only else "[compile]"
        action = "Verifying cached kernels" if args.verify_cache_only else "Compiling kernels"
        print(f"{prefix} ({i}/{total}) {action} for {w}x{h} ...")
        sys.stdout.flush()
        t0 = time.perf_counter()
        try:
            processor.warmup_compile(w, h)
        except Exception as exc:
            print(
                f"{prefix} ERROR: torch.compile warmup failed for {w}x{h}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(3)
        if not bool(getattr(processor, "_compiled", False)):
            print(
                f"{prefix} ERROR: torch.compile fell back to eager for {w}x{h}; "
                "compiled cache was not generated.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(3)
        elapsed = time.perf_counter() - t0
        print(f"{prefix} ({i}/{total}) {w}x{h} done in {elapsed:.1f}s")
        sys.stdout.flush()

        if not args.verify_cache_only:
            # Write marker so the GUI knows this resolution is compiled
            _mark_compiled(
                w,
                h,
                args.precision,
                model_path=model_path,
                use_hg=use_hg,
                predequantize_mode=effective_marker_mode,
                compile_mode=effective_compile_mode,
                memory_format=effective_memory_format,
            )

    if args.quality_score_file:
        score = _benchmark_quality_score(
            processor,
            parsed_res,
            warmup=int(args.quality_benchmark_warmup),
            runs=int(args.quality_benchmark_runs),
        )
        with open(args.quality_score_file, "w", encoding="utf-8") as fh:
            fh.write(f"{score:.9f}\n")
        print(f"[quality] aggregate score: {score:.3f} ms", flush=True)

    if args.verify_cache_only:
        _finish_success("[verify] Cached kernels verified successfully.")
    else:
        _finish_success("[compile] All resolutions compiled successfully.")


if __name__ == "__main__":
    main()

