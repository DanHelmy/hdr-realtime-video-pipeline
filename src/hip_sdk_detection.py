from __future__ import annotations

import os
import shutil
import sys


def has_hip_headers(root: str) -> bool:
    if not root:
        return False
    root = str(root).strip().strip('"')
    if not root:
        return False
    candidates = (
        os.path.join(root, "include", "hip"),
        os.path.join(root, "hip", "include", "hip"),
    )
    return any(os.path.isdir(p) for p in candidates)


def candidate_hip_roots_windows() -> list[str]:
    roots: list[str] = []

    # Explicit env vars first.
    for env_key in ("HIP_PATH", "ROCM_PATH", "ROCM_HOME"):
        value = os.environ.get(env_key, "")
        if value:
            roots.append(value)

    # Standard ROCm install path.
    rocm_root = r"C:\Program Files\AMD\ROCm"
    roots.append(rocm_root)
    if os.path.isdir(rocm_root):
        try:
            for entry in os.listdir(rocm_root):
                roots.append(os.path.join(rocm_root, entry))
        except Exception:
            pass

    # Python install roots (works when ROCm SDK is pip-installed).
    for base in {sys.prefix, getattr(sys, "base_prefix", sys.prefix)}:
        if not base:
            continue
        roots.append(os.path.join(base, "Lib", "site-packages", "_rocm_sdk_devel"))
        roots.append(os.path.join(base, "Lib", "site-packages", "_rocm_sdk_core"))

    # Derive likely roots from hipcc path if available on PATH.
    hipcc = shutil.which("hipcc")
    if hipcc:
        hipcc_dir = os.path.dirname(os.path.abspath(hipcc))
        roots.append(hipcc_dir)
        roots.append(os.path.dirname(hipcc_dir))
        # Common pip layout: <python>\Scripts\hipcc.exe ->
        # <python>\Lib\site-packages\_rocm_sdk_*
        if os.path.basename(hipcc_dir).lower() == "scripts":
            py_root = os.path.dirname(hipcc_dir)
            roots.append(os.path.join(py_root, "Lib", "site-packages", "_rocm_sdk_devel"))
            roots.append(os.path.join(py_root, "Lib", "site-packages", "_rocm_sdk_core"))

    # De-duplicate while preserving order.
    uniq: list[str] = []
    seen: set[str] = set()
    for root in roots:
        try:
            normalized = os.path.normcase(os.path.normpath(str(root).strip().strip('"')))
        except Exception:
            continue
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        uniq.append(root)
    return uniq


def detect_hip_sdk_windows() -> tuple[bool, str | None]:
    if os.name != "nt":
        return False, None
    for root in candidate_hip_roots_windows():
        if has_hip_headers(root):
            return True, root
    return False, None
