"""Train TensorRT-friendly post-correction heads for the distilled HR model.

The fast distilled trunk stays frozen. Only the small post-correction module is
trained, so experiments can improve color/tonal parity without changing the
INT8-friendly compute path that made the TensorRT engines fast.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
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


def _load_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload
    if isinstance(payload, dict):
        return {"state_dict": payload}
    raise TypeError(f"{path} is not a checkpoint/state_dict")


def _clean_state(payload: dict) -> dict[str, torch.Tensor]:
    state = payload.get("state_dict", payload)
    return {str(k).replace("module.", ""): v for k, v in state.items()}


def _load_state_flexible(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    strict: bool = False,
) -> tuple[torch.nn.modules.module._IncompatibleKeys, dict[str, torch.Tensor]]:
    try:
        return model.load_state_dict(state, strict=strict), {}
    except RuntimeError as exc:
        if "size mismatch" not in str(exc):
            raise
        current = model.state_dict()
        filtered = {}
        masks = {}
        partial = []
        dropped = []
        for key, value in state.items():
            if key not in current or not torch.is_tensor(value):
                dropped.append(key)
                continue
            target = current[key]
            if tuple(value.shape) == tuple(target.shape):
                filtered[key] = value
                continue
            if value.ndim == target.ndim and value.ndim in {1, 2, 4}:
                merged = target.clone()
                slices = tuple(
                    slice(0, min(int(dst), int(src)))
                    for dst, src in zip(target.shape, value.shape)
                )
                merged[slices] = value[slices].to(dtype=merged.dtype)
                if target.is_floating_point() and value.ndim in {2, 4} and target.shape[1] > value.shape[1]:
                    out_keep = min(int(target.shape[0]), int(value.shape[0]))
                    in_keep = min(int(target.shape[1]), int(value.shape[1]))
                    merged[:out_keep, in_keep:] = 0
                filtered[key] = merged
                if target.is_floating_point():
                    mask = torch.ones_like(target)
                    mask[slices] = 0
                    masks[key] = mask
                partial.append(key)
                continue
            dropped.append(key)
        print(f"Flexible load dropped {len(dropped)} mismatched/unmapped tensor(s)")
        if partial:
            print(f"  partial-copied {len(partial)} tensor(s): {', '.join(partial[:8])}")
        if dropped:
            print("  first dropped:", ", ".join(dropped[:8]))
        return model.load_state_dict(filtered, strict=False), masks


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


def _paired_paths(sdr_dir: Path, hdr_dir: Path, max_images: int) -> list[tuple[Path, Path]]:
    suffixes = {".png", ".jpg", ".jpeg"}
    sdr_paths = sorted(p for p in sdr_dir.iterdir() if p.suffix.lower() in suffixes)
    pairs: list[tuple[Path, Path]] = []
    for sdr in sdr_paths:
        direct = hdr_dir / sdr.name
        if direct.is_file():
            pairs.append((sdr, direct))
            continue
        matches = sorted(hdr_dir.glob(f"{sdr.stem}.*"))
        if matches:
            pairs.append((sdr, matches[0]))
    if max_images > 0:
        pairs = pairs[:max_images]
    if not pairs:
        raise FileNotFoundError(f"No paired images found in {sdr_dir} + {hdr_dir}")
    return pairs


def _manifest_pairs(manifest_path: Path, max_images: int) -> list[tuple[Path, Path]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise TypeError(f"Manifest must contain a list or an 'items' list: {manifest_path}")
    repo_root = _ROOT
    base_dir = manifest_path.parent
    pairs: list[tuple[Path, Path]] = []

    def _resolve(value: str) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        candidate = (repo_root / path).resolve()
        if candidate.is_file():
            return candidate
        return (base_dir / path).resolve()

    for item in items:
        if not isinstance(item, dict):
            continue
        sdr_value = item.get("sdr") or item.get("sdr_path")
        hdr_value = item.get("hdr") or item.get("hdr_path")
        if not sdr_value or not hdr_value:
            continue
        repeat = max(1, int(item.get("repeat", 1) or 1))
        sdr = _resolve(str(sdr_value))
        hdr = _resolve(str(hdr_value))
        if not sdr.is_file() or not hdr.is_file():
            raise FileNotFoundError(f"Manifest pair missing: {sdr} / {hdr}")
        for _ in range(repeat):
            pairs.append((sdr, hdr))
    if max_images > 0:
        pairs = pairs[:max_images]
    if not pairs:
        raise FileNotFoundError(f"No pairs found in manifest: {manifest_path}")
    return pairs


def _crop_pair(sdr: torch.Tensor, hdr: torch.Tensor, crop_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if crop_size <= 0:
        return sdr, hdr
    _, _, h, w = sdr.shape
    if h <= crop_size or w <= crop_size:
        return sdr, hdr
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return (
        sdr[:, :, top : top + crop_size, left : left + crop_size],
        hdr[:, :, top : top + crop_size, left : left + crop_size],
    )


def _make_input(sdr: torch.Tensor, device: torch.device, dtype: torch.dtype):
    sdr = sdr.to(device=device, dtype=dtype, memory_format=torch.channels_last)
    try:
        cond = F.interpolate(sdr, scale_factor=0.25, mode="bicubic", align_corners=False, antialias=True)
    except TypeError:
        cond = F.interpolate(sdr, scale_factor=0.25, mode="bicubic", align_corners=False)
    return sdr, cond


def _luma(t: torch.Tensor) -> torch.Tensor:
    return (
        t[:, 0:1].float() * 0.2126
        + t[:, 1:2].float() * 0.7152
        + t[:, 2:3].float() * 0.0722
    ).to(dtype=t.dtype)


def _grad_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax = a[:, :, :, 1:] - a[:, :, :, :-1]
    bx = b[:, :, :, 1:] - b[:, :, :, :-1]
    ay = a[:, :, 1:, :] - a[:, :, :-1, :]
    by = b[:, :, 1:, :] - b[:, :, :-1, :]
    return F.l1_loss(ax, bx) + F.l1_loss(ay, by)


def _edge_response(luma: torch.Tensor) -> torch.Tensor:
    dx = F.pad((luma[:, :, :, 1:] - luma[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((luma[:, :, 1:, :] - luma[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return dx + dy


def _soft_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def _masked_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=a.dtype)
    denom = mask.mean().clamp_min(1e-6)
    return ((a - b).abs() * mask).mean() / denom


def _bright_edge_masks(
    reference: torch.Tensor,
    *,
    edge_threshold: float,
    bright_threshold: float,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ref_luma = _luma(reference.clamp(0.0, 1.0))
    edge = _edge_response(ref_luma)
    edge_strength = ((edge - float(edge_threshold)) / max(float(edge_threshold), 1e-6)).clamp(0.0, 1.0)
    bright_context = _soft_dilate(ref_luma, max(1, int(radius)))
    bright_strength = ((bright_context - float(bright_threshold)) / 0.20).clamp(0.0, 1.0)
    edge_mask = (edge_strength * bright_strength).clamp(0.0, 1.0)
    dilated = _soft_dilate(edge_mask, int(radius)).clamp(0.0, 1.0)
    halo = (dilated - edge_mask).clamp(0.0, 1.0)
    return edge_mask, dilated, halo


def _bright_edge_metric(output: torch.Tensor, reference: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    edge_mask, dilated, halo = _bright_edge_masks(
        reference,
        edge_threshold=float(args.bright_edge_threshold),
        bright_threshold=float(args.bright_edge_luma_threshold),
        radius=int(args.bright_edge_radius),
    )
    out_luma = _luma(output.clamp(0.0, 1.0))
    ref_luma = _luma(reference.clamp(0.0, 1.0))
    edge_l1 = _masked_l1(_edge_response(out_luma), _edge_response(ref_luma), dilated)
    halo_l1 = _masked_l1(out_luma, ref_luma, halo)
    return edge_l1, halo_l1


def _loss_terms(
    output: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
    args: argparse.Namespace,
    teacher_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = output.clamp(0.0, 1.0)
    tgt = target.clamp(0.0, 1.0)
    src = source.clamp(0.0, 1.0)

    base_l1 = F.l1_loss(out, tgt)
    base_mse = F.mse_loss(out, tgt)
    luma_l1 = F.l1_loss(_luma(out), _luma(tgt))
    chroma_l1 = F.l1_loss(out - _luma(out), tgt - _luma(tgt))
    grad_l1 = _grad_l1(_luma(out), _luma(tgt))

    loss = (
        float(args.target_weight) * base_l1
        + float(args.mse_weight) * base_mse
        + float(args.luma_weight) * luma_l1
        + float(args.chroma_weight) * chroma_l1
        + float(args.gradient_weight) * grad_l1
    )
    metrics = {
        "l1": float(base_l1.detach().item()),
        "mse": float(base_mse.detach().item()),
        "luma": float(luma_l1.detach().item()),
        "chroma": float(chroma_l1.detach().item()),
        "grad": float(grad_l1.detach().item()),
    }

    target_luma = _luma(tgt)
    hi_mask = ((target_luma - float(args.highlight_threshold)) / 0.12).clamp(0.0, 1.0)
    dark_mask = ((float(args.dark_threshold) - target_luma) / max(float(args.dark_threshold), 1e-6)).clamp(0.0, 1.0)
    sat = (src.amax(dim=1, keepdim=True) - src.amin(dim=1, keepdim=True)).clamp(0.0, 1.0)
    chroma_mask = ((sat - float(args.source_chroma_threshold)) / 0.20).clamp(0.0, 1.0)

    if float(args.highlight_weight) > 0:
        hi = _masked_l1(out, tgt, hi_mask)
        loss = loss + float(args.highlight_weight) * hi
        metrics["highlight"] = float(hi.detach().item())
    if float(args.dark_weight) > 0:
        dark = _masked_l1(out, tgt, dark_mask)
        loss = loss + float(args.dark_weight) * dark
        metrics["dark"] = float(dark.detach().item())
    if float(args.source_chroma_weight) > 0:
        src_luma = _luma(src).clamp_min(1e-4)
        out_ratio = (out / _luma(out).clamp_min(1e-4)).clamp(0.0, 6.0)
        src_ratio = (src / src_luma).clamp(0.0, 6.0)
        src_chroma = _masked_l1(out_ratio, src_ratio, chroma_mask)
        loss = loss + float(args.source_chroma_weight) * src_chroma
        metrics["source_chroma"] = float(src_chroma.detach().item())

    if teacher_output is not None and float(args.teacher_weight) > 0:
        teacher = teacher_output.to(device=out.device, dtype=out.dtype).clamp(0.0, 1.0)
        teacher_l1 = F.l1_loss(out, teacher)
        teacher_mse = F.mse_loss(out, teacher)
        teacher_luma = F.l1_loss(_luma(out), _luma(teacher))
        loss = loss + float(args.teacher_weight) * teacher_l1
        loss = loss + float(args.teacher_mse_weight) * teacher_mse
        loss = loss + float(args.teacher_luma_weight) * teacher_luma
        metrics["teacher_l1"] = float(teacher_l1.detach().item())
        metrics["teacher_mse"] = float(teacher_mse.detach().item())
        metrics["teacher_luma"] = float(teacher_luma.detach().item())

    edge_ref = teacher_output.to(device=out.device, dtype=out.dtype).clamp(0.0, 1.0) if teacher_output is not None else tgt
    if float(args.bright_edge_weight) > 0 or float(args.halo_weight) > 0:
        edge_l1, halo_l1 = _bright_edge_metric(out, edge_ref, args)
        if float(args.bright_edge_weight) > 0:
            loss = loss + float(args.bright_edge_weight) * edge_l1
        if float(args.halo_weight) > 0:
            loss = loss + float(args.halo_weight) * halo_l1
        metrics["bright_edge"] = float(edge_l1.detach().item())
        metrics["halo"] = float(halo_l1.detach().item())

    metrics["total"] = float(loss.detach().item())
    return loss, metrics


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float().clamp(0.0, 1.0) - b.float().clamp(0.0, 1.0)).pow(2).mean().item()
    return float(-10.0 * math.log10(mse + 1e-10))


def _run_hg(composite: HG_Composite, base_out: torch.Tensor) -> torch.Tensor:
    mask = composite._make_mask(base_out, r=composite._mask_r)
    _, _, h, w = base_out.shape
    pad_h = (32 - (h % 32)) % 32
    pad_w = (32 - (w % 32)) % 32
    if pad_h or pad_w:
        base_pad = F.pad(base_out, (0, pad_w, 0, pad_h), mode="reflect")
        mask_pad = F.pad(mask, (0, pad_w, 0, pad_h), mode="reflect")
        return composite.hg((base_pad, mask_pad))[:, :, :h, :w]
    return composite.hg((base_out, mask))


def _split_patterns(text: str) -> tuple[str, ...]:
    return tuple(
        part.strip()
        for chunk in str(text or "").split(";")
        for part in chunk.split(",")
        if part.strip()
    )


def _load_student(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> tuple[HG_Composite, dict]:
    base_payload = _load_checkpoint(Path(args.base_checkpoint))
    arch = dict(base_payload.get("architecture") or {})
    le_arch = str(args.le_arch or arch.get("le_arch") or "conddirecth8wide96x12")
    classifier = str(args.classifier or arch.get("classifier") or "agcm_spatialmixglobalh8wide64x6")
    hg_payload = _load_checkpoint(Path(args.hg_checkpoint))
    hg_arch = str(args.hg_arch or (hg_payload.get("architecture") or {}).get("hg_arch") or arch.get("hg_arch") or "directh16wide64x8")

    model = HG_Composite(
        classifier=classifier,
        cond_c=int(arch.get("cond_c", 6)),
        in_nc=int(arch.get("in_nc", 3)),
        out_nc=int(arch.get("out_nc", 3)),
        nf=int(arch.get("nf", 32)),
        act_type=str(arch.get("act_type", "relu")),
        weighting_network=bool(arch.get("weighting_network", False)),
        hg_nf=int(arch.get("hg_nf", 64)),
        mask_r=float(arch.get("mask_r", 0.75)),
        hg_arch=hg_arch,
        le_arch=le_arch,
        post_correction=args.post_correction,
    )
    load, expanded_masks = _load_state_flexible(model.base, _clean_state(base_payload), strict=False)
    hg_load = model.hg.load_state_dict(_clean_state(hg_payload), strict=True)
    print(f"Student base load: {load}")
    print(f"Student HG load: {hg_load}")

    model.to(device=device, dtype=dtype, memory_format=torch.channels_last)
    for param in model.parameters():
        param.requires_grad = False
    patterns = _split_patterns(getattr(args, "train_patterns", "base.post_correction."))
    freeze_global = bool(getattr(args, "freeze_global", False))
    if bool(getattr(args, "train_expanded_only", False)):
        mask_count = 0
        for name, param in model.named_parameters():
            base_name = name.removeprefix("base.")
            mask = expanded_masks.get(base_name)
            if mask is None:
                continue
            param.requires_grad = True
            mask_device = mask.to(device=param.device, dtype=param.dtype)
            param.register_hook(lambda grad, mask_device=mask_device: grad * mask_device)
            mask_count += 1
        print(f"Expanded-only gradient masks: {mask_count}")
    else:
        for name, param in model.named_parameters():
            if any(name.startswith(pattern) for pattern in patterns):
                if freeze_global and name.startswith("base.post_correction.net."):
                    continue
                param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,} ({';'.join(patterns)})")
    return model, arch | {"le_arch": le_arch, "hg_arch": hg_arch, "classifier": classifier}


def _load_teacher(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> HG_Composite | None:
    teacher_terms = (
        float(args.teacher_weight),
        float(args.teacher_mse_weight),
        float(args.teacher_luma_weight),
        float(args.bright_edge_weight),
        float(args.halo_weight),
    )
    if all(term <= 0 for term in teacher_terms):
        return None
    teacher = HG_Composite(
        classifier=args.teacher_classifier,
        le_arch=args.teacher_le_arch,
        hg_arch=args.teacher_hg_arch,
    )
    teacher.base.load_state_dict(_clean_state(_load_checkpoint(Path(args.teacher_base))), strict=True)
    teacher.hg.load_state_dict(_clean_state(_load_checkpoint(Path(args.teacher_hg))), strict=True)
    teacher.to(device=device, dtype=dtype, memory_format=torch.channels_last).eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def _validate(
    model: HG_Composite,
    teacher: HG_Composite | None,
    pairs: list[tuple[Path, Path]],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    nohg_psnr = []
    hg_psnr = []
    teacher_psnr = []
    teacher_base_match_psnr = []
    teacher_hg_match_psnr = []
    teacher_base_edge_l1 = []
    teacher_base_halo_l1 = []
    teacher_hg_edge_l1 = []
    teacher_hg_halo_l1 = []
    with torch.inference_mode():
        for sdr_path, hdr_path in pairs:
            sdr = _read_rgb(sdr_path, int(args.val_max_long_edge))
            hdr = _read_rgb(hdr_path, int(args.val_max_long_edge)).to(device=device, dtype=dtype, memory_format=torch.channels_last)
            inp = _make_input(sdr, device, dtype)
            base_out, _ = model.base(inp)
            nohg_psnr.append(_psnr(base_out, hdr))
            if float(args.hg_loss_weight) > 0 or bool(args.monitor_hg):
                hg_out = _run_hg(model, base_out)
                hg_psnr.append(_psnr(hg_out, hdr))
            if teacher is not None:
                teacher_base, _ = teacher.base(inp)
                teacher_ref = _run_hg(teacher, teacher_base) if bool(args.monitor_hg) else teacher_base
                teacher_psnr.append(_psnr(teacher_ref, hdr))
                teacher_base_match_psnr.append(_psnr(base_out, teacher_base))
                edge_l1, halo_l1 = _bright_edge_metric(base_out, teacher_base, args)
                teacher_base_edge_l1.append(float(edge_l1.detach().item()))
                teacher_base_halo_l1.append(float(halo_l1.detach().item()))
                if (float(args.hg_loss_weight) > 0 or bool(args.monitor_hg)) and bool(args.monitor_hg):
                    teacher_hg_match_psnr.append(_psnr(hg_out, teacher_ref))
                    edge_l1, halo_l1 = _bright_edge_metric(hg_out, teacher_ref, args)
                    teacher_hg_edge_l1.append(float(edge_l1.detach().item()))
                    teacher_hg_halo_l1.append(float(halo_l1.detach().item()))
    metrics = {
        "nohg_psnr": float(np.mean(nohg_psnr)) if nohg_psnr else 0.0,
        "nohg_min_psnr": float(np.min(nohg_psnr)) if nohg_psnr else 0.0,
        "hg_psnr": float(np.mean(hg_psnr)) if hg_psnr else 0.0,
        "hg_min_psnr": float(np.min(hg_psnr)) if hg_psnr else 0.0,
        "teacher_psnr": float(np.mean(teacher_psnr)) if teacher_psnr else 0.0,
        "teacher_base_match_psnr": float(np.mean(teacher_base_match_psnr)) if teacher_base_match_psnr else 0.0,
        "teacher_hg_match_psnr": float(np.mean(teacher_hg_match_psnr)) if teacher_hg_match_psnr else 0.0,
        "teacher_base_edge_l1": float(np.mean(teacher_base_edge_l1)) if teacher_base_edge_l1 else 0.0,
        "teacher_base_halo_l1": float(np.mean(teacher_base_halo_l1)) if teacher_base_halo_l1 else 0.0,
        "teacher_hg_edge_l1": float(np.mean(teacher_hg_edge_l1)) if teacher_hg_edge_l1 else 0.0,
        "teacher_hg_halo_l1": float(np.mean(teacher_hg_halo_l1)) if teacher_hg_halo_l1 else 0.0,
    }
    score_base = metrics["hg_psnr"] if bool(args.monitor_hg) and hg_psnr else metrics["nohg_psnr"]
    score_min = metrics["hg_min_psnr"] if bool(args.monitor_hg) and hg_psnr else metrics["nohg_min_psnr"]
    edge_key = "teacher_hg_edge_l1" if bool(args.monitor_hg) and teacher_hg_edge_l1 else "teacher_base_edge_l1"
    halo_key = "teacher_hg_halo_l1" if bool(args.monitor_hg) and teacher_hg_halo_l1 else "teacher_base_halo_l1"
    metrics["score"] = float(
        score_base
        + float(args.score_min_weight) * score_min
        - float(args.score_edge_weight) * 100.0 * metrics[edge_key]
        - float(args.score_halo_weight) * 100.0 * metrics[halo_key]
    )
    model.train()
    return metrics


def _save_checkpoint(model: HG_Composite, base_payload: dict, arch: dict, args: argparse.Namespace, path: Path, metrics: dict) -> None:
    state = {k: v.detach().cpu() for k, v in model.base.state_dict().items()}
    recipe = dict(base_payload.get("recipe") or {})
    recipe["post_correction_refit"] = {
        "source_checkpoint": str(Path(args.base_checkpoint)),
        "post_correction": args.post_correction,
        "train_sdr_dir": str(Path(args.train_sdr_dir)),
        "train_hdr_dir": str(Path(args.train_hdr_dir)),
        "epochs": int(args.epochs),
        "crop_size": int(args.crop_size),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "teacher_weight": float(args.teacher_weight),
        "teacher_mse_weight": float(args.teacher_mse_weight),
        "teacher_luma_weight": float(args.teacher_luma_weight),
        "target_weight": float(args.target_weight),
        "mse_weight": float(args.mse_weight),
        "luma_weight": float(args.luma_weight),
        "chroma_weight": float(args.chroma_weight),
        "gradient_weight": float(args.gradient_weight),
        "bright_edge_weight": float(args.bright_edge_weight),
        "halo_weight": float(args.halo_weight),
        "bright_edge_threshold": float(args.bright_edge_threshold),
        "bright_edge_luma_threshold": float(args.bright_edge_luma_threshold),
        "bright_edge_radius": int(args.bright_edge_radius),
        "hg_loss_weight": float(args.hg_loss_weight),
        "score_edge_weight": float(args.score_edge_weight),
        "score_halo_weight": float(args.score_halo_weight),
        "train_patterns": str(args.train_patterns),
        "train_expanded_only": bool(args.train_expanded_only),
        "best_metrics": metrics,
    }
    out_arch = dict(arch)
    out_arch["post_correction"] = args.post_correction
    out_arch["use_hg"] = False
    payload = {
        "schema": base_payload.get("schema", "hdrtvnet++-fp32"),
        "recipe": recipe,
        "architecture": out_arch,
        "state_dict": state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    hg_output = str(getattr(args, "hg_output", "") or "").strip()
    if hg_output:
        hg_payload = {
            "schema": "hdrtvnet_source_checkpoint_v1",
            "recipe": {
                "source": str(Path(args.hg_checkpoint)),
                "training": recipe["post_correction_refit"],
                "best_metrics": metrics,
            },
            "architecture": {"hg_arch": arch.get("hg_arch", args.hg_arch)},
            "state_dict": {k: v.detach().cpu() for k, v in model.hg.state_dict().items()},
        }
        hg_path = Path(hg_output)
        hg_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(hg_payload, hg_path)


def parse_args() -> argparse.Namespace:
    weights = _SRC / "models" / "weights"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", default=str(weights / "distilled" / "hr" / "HR_qfriendly_spatialmixglobal_fp32.pt"))
    parser.add_argument("--hg-checkpoint", default=str(weights / "distilled" / "hg" / "HG_qfriendly_directh16_fp32.pt"))
    parser.add_argument("--teacher-base", default=str(weights / "original" / "HR.pt"))
    parser.add_argument("--teacher-hg", default=str(weights / "original" / "HG.pt"))
    parser.add_argument("--teacher-classifier", default="color_condition")
    parser.add_argument("--teacher-le-arch", default="sft")
    parser.add_argument("--teacher-hg-arch", default="pixelshuffle")
    parser.add_argument("--classifier", default="")
    parser.add_argument("--le-arch", default="")
    parser.add_argument("--hg-arch", default="")
    parser.add_argument("--post-correction", required=True)
    parser.add_argument(
        "--train-patterns",
        default="base.post_correction.",
        help=(
            "Comma/semicolon-separated model parameter prefixes to train. "
            "Examples: base.post_correction. or base.AGCM.,base.LE.,base.post_correction."
        ),
    )
    parser.add_argument(
        "--train-expanded-only",
        action="store_true",
        help="When architecture tensors were width-expanded, update only the non-copied new slices.",
    )
    parser.add_argument(
        "--freeze-global",
        action="store_true",
        help="For stacked global+spatial heads, train only the new spatial branch.",
    )
    parser.add_argument("--train-sdr-dir", default=str(_ROOT / "dataset" / "train_sdr"))
    parser.add_argument("--train-hdr-dir", default=str(_ROOT / "dataset" / "train_hdr"))
    parser.add_argument("--val-sdr-dir", default=str(_ROOT / "dataset" / "test_sdr"))
    parser.add_argument("--val-hdr-dir", default=str(_ROOT / "dataset" / "test_hdr"))
    parser.add_argument(
        "--train-manifest",
        default="",
        help="Optional JSON manifest of SDR/HDR training pairs. Items may include repeat counts.",
    )
    parser.add_argument(
        "--val-manifest",
        default="",
        help="Optional JSON manifest of SDR/HDR validation pairs.",
    )
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=32)
    parser.add_argument("--max-long-edge", type=int, default=1080)
    parser.add_argument("--val-max-long-edge", type=int, default=720)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.0)
    parser.add_argument("--luma-weight", type=float, default=0.65)
    parser.add_argument("--chroma-weight", type=float, default=0.35)
    parser.add_argument("--gradient-weight", type=float, default=0.08)
    parser.add_argument(
        "--bright-edge-weight",
        type=float,
        default=0.0,
        help="Extra edge-response loss on bright/high-contrast teacher or HDR GT boundaries.",
    )
    parser.add_argument(
        "--halo-weight",
        type=float,
        default=0.0,
        help="Extra luma loss in a ring around bright/high-contrast edges to reduce echo/halo artifacts.",
    )
    parser.add_argument("--highlight-weight", type=float, default=0.45)
    parser.add_argument("--dark-weight", type=float, default=0.20)
    parser.add_argument("--source-chroma-weight", type=float, default=0.03)
    parser.add_argument("--teacher-weight", type=float, default=0.18)
    parser.add_argument("--teacher-mse-weight", type=float, default=0.0)
    parser.add_argument("--teacher-luma-weight", type=float, default=0.08)
    parser.add_argument("--hg-loss-weight", type=float, default=0.35)
    parser.add_argument("--highlight-threshold", type=float, default=0.72)
    parser.add_argument("--dark-threshold", type=float, default=0.12)
    parser.add_argument("--source-chroma-threshold", type=float, default=0.16)
    parser.add_argument("--bright-edge-threshold", type=float, default=0.055)
    parser.add_argument("--bright-edge-luma-threshold", type=float, default=0.56)
    parser.add_argument("--bright-edge-radius", type=int, default=2)
    parser.add_argument("--hard-replay-ratio", type=float, default=0.25)
    parser.add_argument("--hard-replay-repeat", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--monitor-hg", action="store_true")
    parser.add_argument(
        "--score-min-weight",
        type=float,
        default=0.08,
        help="Validation score adds this multiple of the monitored minimum-frame PSNR to average PSNR.",
    )
    parser.add_argument(
        "--score-edge-weight",
        type=float,
        default=0.0,
        help="Validation score penalty multiplier for bright-edge L1.",
    )
    parser.add_argument(
        "--score-halo-weight",
        type=float,
        default=0.0,
        help="Validation score penalty multiplier for bright-edge halo L1.",
    )
    parser.add_argument(
        "--score-metric",
        default="",
        help="Optional validation metric name to use directly as the best-checkpoint score.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--hg-output",
        default="",
        help="Optional path to save HG weights when --train-patterns includes hg.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--precision", default="fp32", choices=["fp16", "fp32"])
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Keep FP32 weights but use CUDA autocast/GradScaler for faster training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = torch.float16 if args.precision == "fp16" and device.type == "cuda" else torch.float32

    if str(args.train_manifest or "").strip():
        train_pairs = _manifest_pairs(Path(args.train_manifest), int(args.max_train_images))
    else:
        train_pairs = _paired_paths(Path(args.train_sdr_dir), Path(args.train_hdr_dir), int(args.max_train_images))
    if str(args.val_manifest or "").strip():
        val_pairs = _manifest_pairs(Path(args.val_manifest), int(args.max_val_images))
    else:
        val_pairs = _paired_paths(Path(args.val_sdr_dir), Path(args.val_hdr_dir), int(args.max_val_images))
    print(f"Training post head {args.post_correction} on {len(train_pairs)} pair(s); validating {len(val_pairs)} pair(s)")
    print(f"device={device}, dtype={dtype}, crop={args.crop_size}, batch={args.batch_size}")

    student, arch = _load_student(args, device, dtype)
    teacher = _load_teacher(args, device, dtype)
    base_payload = _load_checkpoint(Path(args.base_checkpoint))

    params = [p for p in student.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable post-correction parameters found")
    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled or (dtype == torch.float16 and device.type == "cuda")))

    best_score = -1e9
    best_metrics: dict[str, float] = {}
    hard_indices: list[int] = []
    output_path = Path(args.output)

    for epoch in range(1, int(args.epochs) + 1):
        start = time.time()
        order = list(range(len(train_pairs)))
        random.shuffle(order)
        if hard_indices and int(args.hard_replay_repeat) > 0:
            replay = hard_indices * int(args.hard_replay_repeat)
            random.shuffle(replay)
            order.extend(replay)
        losses: list[tuple[float, int]] = []
        running: dict[str, float] = {}
        steps = 0

        student.train()
        for offset in range(0, len(order), int(args.batch_size)):
            indices = order[offset : offset + int(args.batch_size)]
            sdr_batch = []
            hdr_batch = []
            for idx in indices:
                sdr_path, hdr_path = train_pairs[idx]
                sdr = _read_rgb(sdr_path, int(args.max_long_edge))
                hdr = _read_rgb(hdr_path, int(args.max_long_edge))
                sdr, hdr = _crop_pair(sdr, hdr, int(args.crop_size))
                sdr_batch.append(sdr)
                hdr_batch.append(hdr)
            sdr_t = torch.cat(sdr_batch, dim=0)
            hdr_t = torch.cat(hdr_batch, dim=0).to(device=device, dtype=dtype, memory_format=torch.channels_last)
            inp = _make_input(sdr_t, device, dtype)
            source = inp[0]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp_enabled or (dtype == torch.float16 and device.type == "cuda")), dtype=torch.float16):
                base_out, _ = student.base(inp)
                teacher_base_out = None
                if teacher is not None:
                    with torch.no_grad():
                        teacher_base_out, _ = teacher.base(inp)
                loss, metrics = _loss_terms(base_out, hdr_t, source, args, teacher_base_out)
                if float(args.hg_loss_weight) > 0:
                    hg_out = _run_hg(student, base_out)
                    hg_teacher = None
                    if teacher is not None and teacher_base_out is not None:
                        with torch.no_grad():
                            hg_teacher = _run_hg(teacher, teacher_base_out)
                    hg_loss, hg_metrics = _loss_terms(hg_out, hdr_t, source, args, hg_teacher)
                    loss = loss + float(args.hg_loss_weight) * hg_loss
                    metrics["hg_total"] = float(hg_loss.detach().item())
                    metrics["total"] = float(loss.detach().item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach().item())
            for idx in indices:
                losses.append((loss_value, idx))
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value)
            steps += 1
            if steps % 50 == 0:
                avg = running.get("total", 0.0) / max(steps, 1)
                print(f"  epoch {epoch} step {steps}: loss={avg:.6f}", flush=True)

        hard_indices = []
        if 0.0 < float(args.hard_replay_ratio) < 1.0 and losses:
            keep = max(1, int(len(train_pairs) * float(args.hard_replay_ratio)))
            hard_indices = [idx for _, idx in sorted(losses, reverse=True)[:keep]]

        avg_running = {k: v / max(steps, 1) for k, v in running.items()}
        validate = int(args.validate_every) > 0 and (epoch % int(args.validate_every) == 0)
        metrics = _validate(student, teacher, val_pairs, args, device, dtype) if validate else {}
        print(
            f"epoch {epoch}/{args.epochs}: train={json.dumps(avg_running, sort_keys=True)} "
            f"val={json.dumps(metrics, sort_keys=True)} "
            f"time={time.time() - start:.1f}s hard={len(hard_indices)}",
            flush=True,
        )
        if metrics:
            score_metric = str(getattr(args, "score_metric", "") or "").strip()
            if score_metric:
                if score_metric not in metrics:
                    raise KeyError(f"--score-metric '{score_metric}' is not available in validation metrics")
                edge_key = "teacher_hg_edge_l1" if bool(args.monitor_hg) else "teacher_base_edge_l1"
                halo_key = "teacher_hg_halo_l1" if bool(args.monitor_hg) else "teacher_base_halo_l1"
                score = float(
                    metrics[score_metric]
                    - float(args.score_edge_weight) * 100.0 * metrics.get(edge_key, 0.0)
                    - float(args.score_halo_weight) * 100.0 * metrics.get(halo_key, 0.0)
                )
            else:
                score = float(metrics.get("score", 0.0))
                if not score:
                    score_base = metrics["hg_psnr"] if bool(args.monitor_hg) else metrics["nohg_psnr"]
                    score_min = metrics["hg_min_psnr"] if bool(args.monitor_hg) else metrics["nohg_min_psnr"]
                    score = float(score_base + float(args.score_min_weight) * score_min)
            metrics["score"] = score
        else:
            score = float(-avg_running.get("total", 1e9))
        if score > best_score:
            best_score = score
            best_metrics = metrics | {"epoch": epoch, "train": avg_running}
            _save_checkpoint(student, base_payload, arch, args, output_path, best_metrics)
            print(f"  saved best -> {output_path} (score={best_score:.4f})", flush=True)

    print(f"Best metrics: {json.dumps(best_metrics, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
