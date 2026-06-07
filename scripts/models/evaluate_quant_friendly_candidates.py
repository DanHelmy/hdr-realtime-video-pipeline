"""Evaluate HR quant-friendly candidates against the original HR+HG teacher."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.hdrtvnet_modules.Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE  # noqa: E402
from models.hdrtvnet_modules.HG_Composite_arch import HG_Composite  # noqa: E402


def _load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"{path} did not contain a checkpoint-like mapping")


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = _load_payload(path)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a state_dict-like mapping")
    return {str(k).replace("module.", ""): v for k, v in payload.items()}


def _arch(path: Path) -> dict:
    payload = _load_payload(path)
    return dict(payload.get("architecture") or {})


def _load_composite(
    base_path: Path,
    hg_path: Path,
    le_arch: str,
    hg_arch: str,
    classifier: str,
    post_correction: str = "",
) -> HG_Composite:
    model = HG_Composite(
        classifier=classifier,
        cond_c=6,
        in_nc=3,
        out_nc=3,
        nf=32,
        act_type="relu",
        weighting_network=False,
        hg_nf=64,
        mask_r=0.75,
        le_arch=le_arch,
        hg_arch=hg_arch,
        post_correction=post_correction,
    )
    model.base.load_state_dict(_load_state(base_path), strict=True)
    model.hg.load_state_dict(_load_state(hg_path), strict=True)
    model.eval()
    return model


def _load_base(
    base_path: Path,
    le_arch: str,
    classifier: str,
    post_correction: str = "",
) -> Ensemble_AGCM_LE:
    model = Ensemble_AGCM_LE(
        classifier=classifier,
        cond_c=6,
        in_nc=3,
        out_nc=3,
        nf=32,
        act_type="relu",
        weighting_network=False,
        le_arch=le_arch,
        post_correction=post_correction,
    )
    model.load_state_dict(_load_state(base_path), strict=True)
    model.eval()
    return model


def _read_rgb(path: Path, max_long_edge: int) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    h, w = img.shape[:2]
    if max_long_edge > 0 and max(h, w) > max_long_edge:
        scale = max_long_edge / max(h, w)
        new_w = max(32, int(round(w * scale / 32)) * 32)
        new_h = max(32, int(round(h * scale / 32)) * 32)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    denom = 65535.0 if img.dtype == np.uint16 else 255.0
    arr = img.astype(np.float32) / denom
    return torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0)


def _prepare_input(sdr: torch.Tensor, device: torch.device, dtype: torch.dtype):
    sdr = sdr.to(device=device, dtype=dtype)
    try:
        cond = F.interpolate(sdr, scale_factor=0.25, mode="bicubic", align_corners=False, antialias=True)
    except TypeError:
        cond = F.interpolate(sdr, scale_factor=0.25, mode="bicubic", align_corners=False)
    return sdr, cond


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float(-10.0 * math.log10(mse + 1e-10))


def _save_png16(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor.detach().float().clamp(0.0, 1.0).squeeze(0).cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    out = np.round(arr * 65535.0).astype(np.uint16)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)


def _paired_paths(sdr_dir: Path, hdr_dir: Path, max_images: int):
    sdr_paths = sorted([p for p in sdr_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    pairs = []
    for sdr_path in sdr_paths:
        hdr_path = hdr_dir / sdr_path.name
        if hdr_path.is_file():
            pairs.append((sdr_path, hdr_path))
        else:
            matches = sorted(hdr_dir.glob(f"{sdr_path.stem}.*"))
            if matches:
                pairs.append((sdr_path, matches[0]))
    if max_images > 0:
        pairs = pairs[:max_images]
    return pairs


def _manifest_pairs(manifest_path: Path, max_images: int):
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise TypeError(f"Manifest must contain a list or an 'items' list: {manifest_path}")

    def _resolve(value: str) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        candidate = (_ROOT / path).resolve()
        if candidate.is_file():
            return candidate
        return (manifest_path.parent / path).resolve()

    pairs = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        sdr_value = item.get("sdr") or item.get("sdr_path")
        hdr_value = item.get("hdr") or item.get("hdr_path")
        if not sdr_value or not hdr_value:
            continue
        sdr = _resolve(str(sdr_value))
        hdr = _resolve(str(hdr_value))
        key = (str(sdr), str(hdr))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((sdr, hdr))
        if max_images > 0 and len(pairs) >= max_images:
            break
    if not pairs:
        raise FileNotFoundError(f"No pairs found in manifest: {manifest_path}")
    return pairs


def parse_candidate(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.stem, path
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


def parse_args() -> argparse.Namespace:
    weights_dir = _SRC / "models" / "weights"
    experiments_dir = weights_dir / "experiments"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-base", default=str(weights_dir / "original" / "HR.pt"))
    parser.add_argument("--teacher-hg", default=str(weights_dir / "original" / "HG.pt"))
    parser.add_argument("--teacher-classifier", default="color_condition")
    parser.add_argument("--student-le-arch", default="")
    parser.add_argument("--student-hg", default=str(weights_dir / "distilled" / "hg" / "HG_qfriendly_directh16_fp32.pt"))
    parser.add_argument("--student-hg-arch", default="")
    parser.add_argument("--student-classifier", default="")
    parser.add_argument("--use-hg", default="1", choices=["1", "0"])
    parser.add_argument("--sdr-dir", default=str(_ROOT / "dataset" / "test_sdr"))
    parser.add_argument("--hdr-dir", default=str(_ROOT / "dataset" / "test_hdr"))
    parser.add_argument("--manifest", default="", help="Optional JSON manifest of SDR/HDR pairs.")
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--max-long-edge", type=int, default=720)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--output-dir", default=str(_ROOT / "outputs" / "quant_friendly_candidates"))
    parser.add_argument("--summary", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate as name=path. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = [parse_candidate(item) for item in args.candidate]
    if not candidates:
        experiments_dir = _SRC / "models" / "weights" / "experiments"
        candidates = [
            ("cleantrunk", experiments_dir / "HR_qfriendly_cleantrunk_init.pt"),
            ("bottleneck_sft", experiments_dir / "HR_qfriendly_bottleneck_sft_init.pt"),
            ("lowres_sft", experiments_dir / "HR_qfriendly_lowres_sft_init.pt"),
            ("downpath_sft", experiments_dir / "HR_qfriendly_downpath_sft_init.pt"),
        ]

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = torch.float16 if args.precision == "fp16" and device.type == "cuda" else torch.float32

    if str(args.manifest or "").strip():
        pairs = _manifest_pairs(Path(args.manifest), args.max_images)
    else:
        pairs = _paired_paths(Path(args.sdr_dir), Path(args.hdr_dir), args.max_images)
    if not pairs:
        raise RuntimeError("No SDR/HDR pairs found")
    print(f"Evaluating {len(candidates)} candidates on {len(pairs)} pair(s), {device}/{dtype}")

    use_hg = str(args.use_hg).strip() != "0"
    if use_hg:
        teacher = _load_composite(
            Path(args.teacher_base),
            Path(args.teacher_hg),
            le_arch="sft",
            hg_arch="pixelshuffle",
            classifier=args.teacher_classifier,
            post_correction="",
        ).to(device=device, dtype=dtype)
    else:
        teacher = _load_base(
            Path(args.teacher_base),
            le_arch="sft",
            classifier=args.teacher_classifier,
            post_correction="",
        ).to(device=device, dtype=dtype)
    teacher.eval()

    output_dir = Path(args.output_dir)
    all_results = []
    for name, path in candidates:
        candidate_arch = _arch(path)
        hg_arch = _arch(Path(args.student_hg))
        le_arch = str(args.student_le_arch or candidate_arch.get("le_arch") or name).strip()
        classifier = str(
            args.student_classifier
            or candidate_arch.get("classifier")
            or "color_condition"
        ).strip()
        post_correction = str(candidate_arch.get("post_correction") or "").strip()
        student_hg_arch = str(
            args.student_hg_arch
            or hg_arch.get("hg_arch")
            or candidate_arch.get("hg_arch")
            or "pixelshuffle"
        ).strip()
        if use_hg:
            student = _load_composite(
                path,
                Path(args.student_hg),
                le_arch=le_arch,
                hg_arch=student_hg_arch,
                classifier=classifier,
                post_correction=post_correction,
            ).to(device=device, dtype=dtype)
        else:
            student = _load_base(
                path,
                le_arch=le_arch,
                classifier=classifier,
                post_correction=post_correction,
            ).to(device=device, dtype=dtype)
        student.eval()

        rows = []
        with torch.inference_mode():
            for idx, (sdr_path, hdr_path) in enumerate(pairs):
                sdr = _read_rgb(sdr_path, args.max_long_edge)
                hdr = _read_rgb(hdr_path, args.max_long_edge).to(device=device, dtype=dtype)
                inp = _prepare_input(sdr, device, dtype)
                teacher_out, _ = teacher(inp)
                student_out, _ = student(inp)
                row = {
                    "name": sdr_path.name,
                    "student_vs_teacher_psnr": _psnr(student_out, teacher_out),
                    "student_vs_teacher_mae": float((student_out.float() - teacher_out.float()).abs().mean().item()),
                    "student_vs_teacher_max_abs": float((student_out.float() - teacher_out.float()).abs().max().item()),
                    "teacher_vs_gt_psnr": _psnr(teacher_out, hdr),
                    "student_vs_gt_psnr": _psnr(student_out, hdr),
                }
                rows.append(row)
                if idx < args.preview_count:
                    stem = f"{idx:03d}_{sdr_path.stem}"
                    _save_png16(sdr.to(device=device, dtype=dtype), output_dir / name / f"{stem}_sdr.png")
                    _save_png16(hdr, output_dir / name / f"{stem}_gt.png")
                    _save_png16(teacher_out, output_dir / name / f"{stem}_teacher.png")
                    _save_png16(student_out, output_dir / name / f"{stem}_student.png")
                    _save_png16((student_out.float() - teacher_out.float()).abs() * 8.0, output_dir / name / f"{stem}_diff_x8.png")

        summary = {
            "candidate": name,
            "path": str(path),
            "count": len(rows),
            "le_arch": le_arch,
            "classifier": classifier,
            "post_correction": post_correction,
            "student_hg_arch": student_hg_arch if use_hg else "",
        }
        for key in (
            "student_vs_teacher_psnr",
            "student_vs_teacher_mae",
            "student_vs_teacher_max_abs",
            "teacher_vs_gt_psnr",
            "student_vs_gt_psnr",
        ):
            vals = [row[key] for row in rows]
            summary[f"{key}_avg"] = float(np.mean(vals))
            summary[f"{key}_min"] = float(np.min(vals))
            summary[f"{key}_max"] = float(np.max(vals))
        all_results.append({"summary": summary, "images": rows})
        print(
            f"{name:16s} teacher PSNR avg={summary['student_vs_teacher_psnr_avg']:.2f} dB "
            f"min={summary['student_vs_teacher_psnr_min']:.2f} dB, "
            f"GT avg={summary['student_vs_gt_psnr_avg']:.2f} dB"
        )
        del student
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "schema": "hdrtvnet_quant_friendly_candidate_eval_v1",
        "teacher_base": args.teacher_base,
        "teacher_hg": args.teacher_hg,
        "student_hg": args.student_hg,
        "use_hg": use_hg,
        "max_long_edge": args.max_long_edge,
        "results": all_results,
    }
    summary_path = Path(args.summary) if args.summary else output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
