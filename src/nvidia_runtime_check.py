from __future__ import annotations

import argparse
import ctypes
import importlib.metadata as metadata
import json
import os
import sys
from typing import Any


EXPECTED_TORCH_CUDA_PREFIX = "13."
EXPECTED_TENSORRT_VERSION = "11.0.0.114"
EXPECTED_TENSORRT_PACKAGES = (
    "tensorrt_cu13",
    "tensorrt_cu13_bindings",
    "tensorrt_cu13_libs",
)
OBSOLETE_TENSORRT_PACKAGES = (
    "tensorrt_cu12",
    "tensorrt_cu12_bindings",
    "tensorrt_cu12_libs",
)


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _nvidia_driver_present() -> bool:
    if os.name == "nt":
        try:
            ctypes.WinDLL("nvcuda.dll")
            return True
        except Exception:
            return False
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_nvidia_python_runtime() -> dict[str, Any]:
    """Validate the NVIDIA Python runtime without building a TensorRT engine."""
    issues: list[str] = []
    details: dict[str, Any] = {}
    nvidia_detected = _nvidia_driver_present()

    if not nvidia_detected:
        return {
            "ok": True,
            "nvidia_detected": False,
            "issues": issues,
            "details": details,
        }

    try:
        import torch

        torch_version = str(getattr(torch, "__version__", "unknown"))
        torch_cuda = str(getattr(torch.version, "cuda", "") or "")
        details["torch"] = torch_version
        details["torch_cuda"] = torch_cuda
        if not torch_cuda.startswith(EXPECTED_TORCH_CUDA_PREFIX):
            issues.append(
                "PyTorch CUDA runtime is "
                f"{torch_cuda or 'none'}, expected {EXPECTED_TORCH_CUDA_PREFIX}x."
            )
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:
            cuda_available = False
            details["torch_cuda_error"] = f"{type(exc).__name__}: {exc}"
        details["torch_cuda_available"] = cuda_available
        if not cuda_available:
            issues.append("PyTorch cannot access the NVIDIA CUDA device.")
        else:
            try:
                details["cuda_device"] = str(torch.cuda.get_device_name(0) or "")
            except Exception:
                pass
    except Exception as exc:
        issues.append(f"PyTorch import failed: {type(exc).__name__}: {exc}")

    for package in EXPECTED_TENSORRT_PACKAGES:
        version = _package_version(package)
        details[package] = version
        if version != EXPECTED_TENSORRT_VERSION:
            issues.append(
                f"{package} is {version or 'not installed'}, "
                f"expected {EXPECTED_TENSORRT_VERSION}."
            )

    for package in OBSOLETE_TENSORRT_PACKAGES:
        version = _package_version(package)
        details[package] = version
        if version is not None:
            issues.append(
                f"Obsolete {package} {version} is still installed; setup.bat must remove it."
            )

    try:
        import tensorrt_libs  # noqa: F401

        details["tensorrt_libs_import"] = True
    except Exception as exc:
        details["tensorrt_libs_import"] = False
        issues.append(f"TensorRT CUDA libraries import failed: {type(exc).__name__}: {exc}")

    try:
        import tensorrt as trt

        trt_version = str(getattr(trt, "__version__", "unknown"))
        details["tensorrt"] = trt_version
        if trt_version != EXPECTED_TENSORRT_VERSION:
            issues.append(
                f"TensorRT import resolves to {trt_version}, "
                f"expected {EXPECTED_TENSORRT_VERSION}."
            )
    except Exception as exc:
        details["tensorrt"] = None
        issues.append(f"TensorRT import failed: {type(exc).__name__}: {exc}")

    return {
        "ok": len(issues) == 0,
        "nvidia_detected": True,
        "issues": issues,
        "details": details,
    }


def format_nvidia_runtime_issues(status: dict[str, Any]) -> str:
    issues = list(status.get("issues") or [])
    if not issues:
        return ""
    lines = "\n".join(f"- {issue}" for issue in issues)
    return (
        "The NVIDIA CUDA/TensorRT Python runtime is outdated or inconsistent.\n\n"
        f"{lines}\n\n"
        "Run setup.bat again to refresh the environment, then launch the app again."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the HDRTVNet++ NVIDIA Python runtime."
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)
    status = check_nvidia_python_runtime()
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        if status["ok"]:
            if status.get("nvidia_detected"):
                print("NVIDIA runtime OK.")
            else:
                print("No NVIDIA runtime detected; NVIDIA check skipped.")
        else:
            print(format_nvidia_runtime_issues(status))
    return 0 if status["ok"] else 10


if __name__ == "__main__":
    sys.exit(main())
