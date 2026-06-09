"""Generate TensorRT FP8 source checkpoints from original TensorRT sources.

FP8 presets are TensorRT-only. These files are not eager runtime weights; they
reuse the original PTQ/QAT/QAT-Film FP32 source tensors and deployment recipe,
then let ModelOpt Torch emit FP8 Q/DQ during engine build.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import torch


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_WEIGHTS = _REPO_ROOT / "src" / "models" / "weights"
_INT8_TRT = _WEIGHTS / "original" / "tensorrt"
_FP8_TRT = _WEIGHTS / "original" / "tensorrt_fp8"

_TAGS = (
    "mixed_ptq",
    "mixed_qat",
    "mixed_qat_film",
    "full_ptq",
    "full_qat",
    "full_qat_film",
)
_FAMILIES = ("hr", "hr_hg", "hg")


def _fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "name": path.name,
        "size": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def _source_path(family: str, tag: str) -> Path:
    if family == "hr":
        return _INT8_TRT / "hr" / f"HR_original_int8_{tag}.pt"
    if family == "hr_hg":
        return _INT8_TRT / "hr_hg" / f"HR_HG_original_int8_{tag}.pt"
    if family == "hg":
        return _INT8_TRT / "hg" / f"HG_original_int8_{tag}.pt"
    raise ValueError(f"Unsupported family: {family}")


def _output_path(family: str, tag: str) -> Path:
    if family == "hr":
        return _FP8_TRT / "hr" / f"HR_original_fp8_{tag}.pt"
    if family == "hr_hg":
        return _FP8_TRT / "hr_hg" / f"HR_HG_original_fp8_{tag}.pt"
    if family == "hg":
        return _FP8_TRT / "hg" / f"HG_original_fp8_{tag}.pt"
    raise ValueError(f"Unsupported family: {family}")


def _ensure_int8_sources() -> None:
    script = _SCRIPT_DIR / "split_tensorrt_sources.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing TensorRT source split script: {script}")
    subprocess.run(
        [sys.executable, "-u", str(script), "--missing-only"],
        cwd=str(_REPO_ROOT),
        check=True,
    )


def _write_fp8_source(source: Path, output: Path, *, force: bool = False) -> bool:
    if output.is_file() and not force:
        try:
            existing = torch.load(output, map_location="cpu", weights_only=False)
            if (
                isinstance(existing, dict)
                and existing.get("target_backend") == "tensorrt"
                and existing.get("fp8_source_checkpoint") is True
                and existing.get("fp8_source_from") == _fingerprint(source)
            ):
                print(f"Skipped current FP8 TensorRT source: {output}")
                return False
        except Exception:
            pass

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Expected TensorRT source checkpoint: {source}")

    save_data = dict(checkpoint)
    save_data.update(
        {
            "target_backend": "tensorrt",
            "fp8_source_checkpoint": True,
            "fp8_source_schema": "hdrtvnet_tensorrt_fp8_source_v1",
            "fp8_source_from": _fingerprint(source),
            "fp8_source_note": (
                "TensorRT FP8 source: FP32 tensors and the matching "
                "PTQ/QAT/QAT-Film composition are reused; ModelOpt Torch "
                "emits FP8 Q/DQ during engine build."
            ),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, output)
    print(f"Saved FP8 TensorRT source: {output}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate TensorRT FP8 source checkpoints from original TensorRT sources."
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=_FAMILIES,
        help="Family to generate. Repeatable. Default: all families.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        choices=_TAGS,
        help="Preset tag to generate. Repeatable. Default: all tags.",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Skip outputs that are already current.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if the FP8 source appears current.",
    )
    parser.add_argument(
        "--ensure-int8-sources",
        action="store_true",
        default=True,
        help="Generate missing INT8 TensorRT sources before copying.",
    )
    args = parser.parse_args()

    families = tuple(args.family or _FAMILIES)
    tags = tuple(args.tag or _TAGS)

    missing_sources = [
        _source_path(family, tag)
        for family in families
        for tag in tags
        if not _source_path(family, tag).is_file()
    ]
    if missing_sources and args.ensure_int8_sources:
        print(
            "Preparing missing INT8 TensorRT source checkpoint(s) before FP8 copy: "
            f"{len(missing_sources)}"
        )
        _ensure_int8_sources()

    written = 0
    for family in families:
        for tag in tags:
            source = _source_path(family, tag)
            if not source.is_file():
                raise FileNotFoundError(f"Missing source checkpoint: {source}")
            output = _output_path(family, tag)
            force = bool(args.force) or not bool(args.missing_only)
            written += int(_write_fp8_source(source, output, force=force))

    print(f"FP8 TensorRT source generation complete: {written} file(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
