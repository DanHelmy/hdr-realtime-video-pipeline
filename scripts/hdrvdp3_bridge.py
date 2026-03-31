#!/usr/bin/env python3
"""
Local HDR-VDP3 bridge for the GUI.

Usage:
    python scripts/hdrvdp3_bridge.py --test <img> --reference <img>

It auto-downloads HDR-VDP3 toolbox into a per-user cache directory (once),
then invokes Octave to compute a quality score and prints:
    HDRVDP3_SCORE=<float>
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


HDRVDP_VERSION = "hdrvdp-3.0.7"
HDRVDP_ZIP_URL = (
    "https://downloads.sourceforge.net/project/hdrvdp/"
    f"hdrvdp/3.0.7/{HDRVDP_VERSION}.zip"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_hdrvdp_cache_dir() -> Path:
    """Return per-user cache directory for HDR-VDP3 toolbox files."""
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        return Path(local_app) / "HDRTVNetCache" / "hdrvdp"
    return Path.home() / ".cache" / "hdrtvnet" / "hdrvdp"


def _safe_extract(zf: zipfile.ZipFile, dst: Path):
    dst_resolved = dst.resolve()
    for member in zf.infolist():
        target = (dst / member.filename).resolve()
        if not str(target).startswith(str(dst_resolved)):
            raise RuntimeError(f"Unsafe zip path: {member.filename}")
        zf.extract(member, dst)


def _ensure_hdrvdp_toolbox(root: Path) -> Path:
    cache_override = str(os.environ.get("HDRTVNET_HDRVDP_CACHE_DIR", "") or "").strip()
    base = Path(cache_override) if cache_override else _default_hdrvdp_cache_dir()
    toolbox = base / HDRVDP_VERSION
    probe = toolbox / "hdrvdp3.m"
    if probe.is_file():
        return toolbox

    base.mkdir(parents=True, exist_ok=True)
    zip_path = base / f"{HDRVDP_VERSION}.zip"
    if not zip_path.is_file() or zip_path.stat().st_size < 1024:
        urllib.request.urlretrieve(HDRVDP_ZIP_URL, zip_path)  # nosec B310

    with zipfile.ZipFile(zip_path, "r") as zf:
        _safe_extract(zf, base)

    if not probe.is_file():
        raise RuntimeError("HDR-VDP3 toolbox extraction failed (hdrvdp3.m missing).")
    return toolbox


def _octave_executable() -> str | None:
    for name in ("octave-cli", "octave", "octave-cli.exe", "octave.exe"):
        p = shutil.which(name)
        if p:
            return p

    # Fallback for Windows users who installed Octave but did not add it to PATH.
    if os.name == "nt":
        patterns = [
            r"C:\Program Files\GNU Octave\Octave-*\mingw64\bin\octave-cli.exe",
            r"C:\Program Files\GNU Octave\Octave-*\mingw64\bin\octave.exe",
            r"C:\Program Files\Octave\Octave-*\mingw64\bin\octave-cli.exe",
            r"C:\Program Files\Octave\Octave-*\mingw64\bin\octave.exe",
        ]
        for pat in patterns:
            hits = sorted(glob.glob(pat), reverse=True)
            for hit in hits:
                if os.path.isfile(hit):
                    return hit
    return None


def _run_octave(test_path: Path, ref_path: Path, toolbox: Path) -> float:
    octave = _octave_executable()
    if not octave:
        raise RuntimeError(
            "GNU Octave not found (PATH/default locations). Install Octave to enable HDR-VDP3."
        )

    m_src = (
        "args = argv();\n"
        "if numel(args) < 3, fprintf(2, 'need test ref root\\n'); exit(2); end\n"
        "test_path = args{1}; ref_path = args{2}; root = args{3};\n"
        "addpath(root); addpath(fullfile(root,'utils'));\n"
        "try, pkg load image; catch, end\n"
        "try, pkg load statistics; catch, end\n"
        "I_test = im2double(imread(test_path));\n"
        "I_ref  = im2double(imread(ref_path));\n"
        "if size(I_test,3)==1, I_test = repmat(I_test,[1 1 3]); end\n"
        "if size(I_ref,3)==1, I_ref = repmat(I_ref,[1 1 3]); end\n"
        "if any(size(I_test,1:2) != size(I_ref,1:2))\n"
        "  I_test = imresize(I_test, [size(I_ref,1) size(I_ref,2)], 'bicubic');\n"
        "end\n"
        "% Approximate display model for compare snapshots.\n"
        "Y_peak=1000; contrast=1000; gamma=2.2; E_ambient=100; ppd=60;\n"
        "L_test = hdrvdp_gog_display_model(I_test, Y_peak, contrast, gamma, E_ambient);\n"
        "L_ref  = hdrvdp_gog_display_model(I_ref,  Y_peak, contrast, gamma, E_ambient);\n"
        "res = hdrvdp3('quality', L_test, L_ref, 'rgb-native', ppd, {'quiet', true});\n"
        "score = NaN;\n"
        "if isfield(res, 'Q_JOD'), score = res.Q_JOD; elseif isfield(res, 'Q'), score = res.Q; end\n"
        "fprintf('HDRVDP3_SCORE=%0.6f\\n', score);\n"
    )

    with tempfile.TemporaryDirectory(prefix="hdrvdp3_bridge_") as td:
        script_path = Path(td) / "run_hdrvdp3.m"
        script_path.write_text(m_src, encoding="utf-8")
        cmd = [
            octave,
            "--quiet",
            "--no-gui",
            str(script_path),
            str(test_path),
            str(ref_path),
            str(toolbox),
        ]
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )

    if cp.returncode != 0:
        stderr = (cp.stderr or "").strip()
        stdout = (cp.stdout or "").strip()
        tail = stderr.splitlines()[-1] if stderr else (stdout.splitlines()[-1] if stdout else "")
        raise RuntimeError(f"Octave HDR-VDP3 failed (rc={cp.returncode}): {tail}")

    merged = f"{cp.stdout or ''}\n{cp.stderr or ''}"
    m = re.search(r"HDRVDP3_SCORE\s*=\s*([-+]?\d+(?:\.\d+)?)", merged, re.IGNORECASE)
    if not m:
        raise RuntimeError("Octave HDR-VDP3 returned no parsable score.")
    return float(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run HDR-VDP3 via local Octave bridge.")
    ap.add_argument("--test", required=True, help="Path to test image")
    ap.add_argument("--reference", required=True, help="Path to reference image")
    args = ap.parse_args()

    test_path = Path(args.test).resolve()
    ref_path = Path(args.reference).resolve()
    if not test_path.is_file():
        print("test image not found", file=sys.stderr)
        return 2
    if not ref_path.is_file():
        print("reference image not found", file=sys.stderr)
        return 2

    try:
        root = _project_root()
        toolbox = _ensure_hdrvdp_toolbox(root)
        score = _run_octave(test_path, ref_path, toolbox)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"HDRVDP3_SCORE={score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


