"""ModelOpt Torch QAT for the TensorRT mixed INT8 fast path.

This trains the clean FP32 source weights while ModelOpt Torch Q/DQ simulation
is active, then strips the quantizer wrappers and saves clean HR/HG `.pth`
source checkpoints. TensorRT should still be built with the same ModelOpt
include mask used during training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.hdrtvnet_torch import (  # noqa: E402
    _apply_tensorrt_modelopt_torch_int8_quantization,
    _count_tensorrt_modelopt_torch_quantizers,
)
from scripts.quantize.quantize_int8_mixed_qat import (  # noqa: E402
    _build_fp32_model,
    accumulate_metrics,
    average_metrics,
    compute_loss_terms,
    configure_reproducibility,
    format_metrics,
    load_image_pairs_from_dirs,
    prepare_model_input,
    sample_training_crop_pair,
    select_tone_protected_monitor_pairs,
)


DEFAULT_INCLUDE = (
    "base.AGCM.spatial;base.AGCM.global;base.LE.low_in;"
    "base.LE.recon_trunk3;AGCM.spatial;AGCM.global;"
    "LE.low_in;LE.recon_trunk3;hg.low_in;hg.trunk"
)


@contextmanager
def _env_override(values: dict[str, str]):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is not None:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _model_output(result):
    if isinstance(result, (tuple, list)):
        return result[0]
    return result


def _sample_training_pair(sdr_t: torch.Tensor, hdr_t: torch.Tensor, args: argparse.Namespace):
    """Return a training sample; crop_size <= 0 keeps the loaded full frame."""
    if int(args.crop_size) <= 0:
        return sdr_t, hdr_t
    return sample_training_crop_pair(sdr_t, hdr_t, args)


def _average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    sums: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            sums[key] = sums.get(key, 0.0) + float(value)
    return {key: value / len(rows) for key, value in sums.items()}


def _ohem_reduce(per_sample_loss: torch.Tensor, ratio: float) -> torch.Tensor:
    ratio = float(ratio)
    if per_sample_loss.numel() <= 1 or ratio <= 0.0 or ratio >= 1.0:
        return per_sample_loss.mean()
    keep = max(1, int(math.ceil(per_sample_loss.numel() * ratio)))
    return torch.topk(per_sample_loss, k=keep, largest=True).values.mean()


def _gradient_teacher_loss(output: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if output.shape[-1] < 2 or output.shape[-2] < 2:
        return output.new_zeros(())
    out_dx = output[..., :, 1:] - output[..., :, :-1]
    out_dy = output[..., 1:, :] - output[..., :-1, :]
    teacher_dx = teacher[..., :, 1:] - teacher[..., :, :-1]
    teacher_dy = teacher[..., 1:, :] - teacher[..., :-1, :]
    return 0.5 * (F.l1_loss(out_dx, teacher_dx) + F.l1_loss(out_dy, teacher_dy))


def _multiscale_teacher_loss(
    output: torch.Tensor,
    teacher: torch.Tensor,
    levels: int,
) -> torch.Tensor:
    levels = max(0, int(levels))
    if levels <= 0:
        return output.new_zeros(())
    loss = output.new_zeros(())
    out_cur = output
    teacher_cur = teacher
    used = 0
    for _level in range(levels):
        if out_cur.shape[-1] < 4 or out_cur.shape[-2] < 4:
            break
        out_cur = F.avg_pool2d(out_cur, kernel_size=2, stride=2)
        teacher_cur = F.avg_pool2d(teacher_cur, kernel_size=2, stride=2)
        loss = loss + F.l1_loss(out_cur, teacher_cur)
        used += 1
    if used <= 0:
        return output.new_zeros(())
    return loss / used


def _compute_loss_terms_per_sample(
    output: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
    teacher_output: torch.Tensor | None = None,
    source: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    losses = []
    rows: list[dict[str, float]] = []
    teacher_ref = teacher_output.to(dtype=output.dtype) if teacher_output is not None else None
    for idx in range(output.shape[0]):
        out_i = output[idx:idx + 1]
        target_i = target[idx:idx + 1]
        teacher_i = teacher_ref[idx:idx + 1] if teacher_ref is not None else None
        source_i = source[idx:idx + 1] if source is not None else None
        loss, metrics = compute_loss_terms(
            out_i,
            target_i,
            args,
            teacher_output=teacher_i,
            source=source_i,
        )
        if teacher_i is not None and float(args.gradient_teacher_weight) > 0:
            grad_loss = _gradient_teacher_loss(out_i, teacher_i)
            loss = loss + float(args.gradient_teacher_weight) * grad_loss
            metrics["gradient_teacher"] = float(grad_loss.detach().item())
        if teacher_i is not None and float(args.multiscale_teacher_weight) > 0:
            ms_loss = _multiscale_teacher_loss(
                out_i,
                teacher_i,
                int(args.multiscale_levels),
            )
            loss = loss + float(args.multiscale_teacher_weight) * ms_loss
            metrics["multiscale_teacher"] = float(ms_loss.detach().item())
        metrics["total"] = float(loss.detach().item())
        losses.append(loss)
        rows.append(metrics)
    return torch.stack(losses), rows


def _temporal_loss_for_group(
    qmodel: nn.Module,
    teacher_model: nn.Module,
    pairs,
    group,
    output: torch.Tensor,
    teacher_output: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> torch.Tensor:
    if float(args.temporal_weight) <= 0 or int(args.crop_size) > 0:
        return output.new_zeros(())
    losses = []
    max_gap = max(1, int(args.temporal_max_gap))
    for local_idx, (item_index, sdr_t, _hdr_t) in enumerate(group):
        next_index = int(item_index) + max_gap
        if next_index < 0 or next_index >= len(pairs):
            continue
        next_sdr, _next_hdr = pairs[next_index]
        if tuple(next_sdr.shape[-2:]) != tuple(sdr_t.shape[-2:]):
            continue
        next_inp = prepare_model_input(next_sdr, device, dtype)
        with torch.inference_mode():
            teacher_next = _model_output(teacher_model(next_inp))
            if not torch.isfinite(teacher_next).all():
                teacher_next = torch.nan_to_num(
                    teacher_next,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
        student_next = _model_output(qmodel(next_inp))
        if not torch.isfinite(student_next).all():
            student_next = torch.nan_to_num(
                student_next,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
        losses.append(
            F.l1_loss(
                student_next - output[local_idx:local_idx + 1],
                teacher_next.to(dtype=output.dtype) - teacher_output[local_idx:local_idx + 1],
            )
        )
    if not losses:
        return output.new_zeros(())
    return torch.stack(losses).mean()


def _save_clean_source_weights(qmodel: nn.Module, args: argparse.Namespace) -> dict[str, object]:
    use_hg = str(args.use_hg).strip() != "0"
    clean = _build_fp32_model(args.fp32_model, args.hg_weights, use_hg)
    qstate = qmodel.state_dict()
    native = {}
    missing = []
    for key, value in clean.state_dict().items():
        qvalue = qstate.get(key)
        if qvalue is None or tuple(qvalue.shape) != tuple(value.shape):
            missing.append(key)
            continue
        native[key] = qvalue.detach().float().cpu()
    if missing:
        preview = ", ".join(missing[:16])
        more = "" if len(missing) <= 16 else f", +{len(missing) - 16} more"
        raise RuntimeError(f"Could not strip ModelOpt wrappers; missing {preview}{more}")
    clean.load_state_dict(native, strict=True)

    output_base = Path(args.output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    if use_hg:
        output_hg = Path(args.output_hg)
        output_hg.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clean.base.state_dict(), output_base)
        torch.save(clean.hg.state_dict(), output_hg)
        saved = {"base": str(output_base), "hg": str(output_hg)}
    else:
        torch.save(clean.state_dict(), output_base)
        saved = {"base": str(output_base), "hg": None}

    combined = Path(args.output_combined) if args.output_combined else None
    if combined:
        combined.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clean.state_dict(), combined)
        saved["combined"] = str(combined)

    return saved


def _compute_monitor(
    qmodel: nn.Module,
    monitor_pairs,
    monitor_teacher_outputs,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
):
    if not monitor_pairs:
        return None
    was_training = qmodel.training
    qmodel.eval()
    sums = {}
    count = 0
    with torch.inference_mode():
        per_frame_totals: list[float] = []
        for idx, (sdr_t, hdr_t) in enumerate(monitor_pairs):
            inp = prepare_model_input(sdr_t, device, dtype)
            target = hdr_t.to(device=device, dtype=dtype)
            teacher = monitor_teacher_outputs[idx].to(device=device, dtype=dtype)
            output = _model_output(qmodel(inp))
            if not torch.isfinite(output).all():
                output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
            loss, metrics = compute_loss_terms(
                output,
                target,
                args,
                teacher_output=teacher,
                source=inp[0],
            )
            if torch.isfinite(loss):
                accumulate_metrics(sums, metrics)
                per_frame_totals.append(float(metrics.get("total", loss.detach().item())))
                count += 1
    if was_training:
        qmodel.train()
    if count == 0:
        return None
    metrics = average_metrics(sums, count)
    hard_ratio = float(getattr(args, "monitor_hard_ratio", 0.0))
    if per_frame_totals and 0.0 < hard_ratio < 1.0:
        keep = max(1, int(math.ceil(len(per_frame_totals) * hard_ratio)))
        worst = sorted(per_frame_totals, reverse=True)[:keep]
        hard_total = float(sum(worst) / len(worst))
        hard_weight = max(0.0, float(getattr(args, "monitor_hard_weight", 0.0)))
        metrics["hard_total"] = hard_total
        metrics["score_total"] = (
            (1.0 - hard_weight) * float(metrics.get("total", hard_total))
            + hard_weight * hard_total
        )
    return metrics


def train_modelopt_qat(qmodel, teacher_model, pairs, monitor_pairs, device, dtype, args):
    params = [p for p in qmodel.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=0)
    batch_size = max(1, int(args.batch_size))
    steps_per_epoch = max((len(pairs) + batch_size - 1) // batch_size, 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * steps_per_epoch),
        eta_min=args.lr * 0.01,
    )

    monitor_teacher_outputs = []
    if monitor_pairs:
        teacher_model.eval()
        with torch.inference_mode():
            for sdr_t, _ in monitor_pairs:
                teacher_inp = prepare_model_input(sdr_t, device, dtype)
                teacher_out = _model_output(teacher_model(teacher_inp))
                monitor_teacher_outputs.append(teacher_out.detach().cpu())

    print(
        f"\n  ModelOpt QAT source training: epochs={args.epochs}, "
        f"lr={args.lr:g}, crop={'full' if int(args.crop_size) <= 0 else args.crop_size}, "
        f"batch={batch_size}, "
        f"pairs={len(pairs)}",
        flush=True,
    )
    print(f"  Loss recipe: {args.target_loss_weight:.3f}*L1 "
          f"+ {args.teacher_loss_weight:.3f}*teacher "
          f"+ {args.teacher_luma_weight:.3f}*teacher_luma "
          f"+ {args.teacher_chroma_weight:.3f}*teacher_chroma "
          f"+ {args.highlight_teacher_weight:.3f}*highlight_teacher "
          f"+ {args.dark_teacher_weight:.3f}*dark_teacher "
          f"+ {args.gradient_teacher_weight:.3f}*gradient_teacher "
          f"+ {args.multiscale_teacher_weight:.3f}*multiscale_teacher "
          f"(ohem={args.ohem_ratio:.2f}, temporal={args.temporal_weight:.3f})",
          flush=True)

    best_state = None
    best_score = float("inf")
    best_epoch = 0
    hard_replay_indices: list[int] = []
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        qmodel.train()
        epoch_sums = {}
        samples = 0
        indexed_pairs = list(enumerate(pairs))
        repeat = max(0, int(args.hard_replay_repeat))
        if hard_replay_indices and repeat > 0:
            replay = [
                (idx, pairs[idx])
                for idx in hard_replay_indices
                if 0 <= idx < len(pairs)
            ]
            for _ in range(repeat):
                indexed_pairs.extend(replay)
        if str(args.shuffle).strip() != "0":
            random.shuffle(indexed_pairs)
        sample_losses: list[tuple[int, float]] = []

        for start in range(0, len(indexed_pairs), batch_size):
            batch_items = indexed_pairs[start:start + batch_size]
            batch_pairs = [pair for _idx, pair in batch_items]
            groups = {}
            for item_index, (sdr_t, hdr_t) in batch_items:
                sdr_crop, hdr_crop = _sample_training_pair(sdr_t, hdr_t, args)
                groups.setdefault(tuple(sdr_crop.shape[-2:]), []).append(
                    (item_index, sdr_crop, hdr_crop)
                )

            total = sum(len(group) for group in groups.values())
            if total <= 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            valid = 0

            for group in groups.values():
                sdr_batch = torch.cat([sdr for _idx, sdr, _ in group], dim=0)
                hdr_batch = torch.cat([hdr for _idx, _, hdr in group], dim=0)
                inp = prepare_model_input(sdr_batch, device, dtype)
                target = hdr_batch.to(device=device, dtype=dtype)
                with torch.inference_mode():
                    teacher_out = _model_output(teacher_model(inp))
                    if not torch.isfinite(teacher_out).all():
                        teacher_out = torch.nan_to_num(
                            teacher_out, nan=0.0, posinf=1.0, neginf=0.0
                        )

                output = _model_output(qmodel(inp))
                if not torch.isfinite(output).all():
                    output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
                per_losses, metric_rows = _compute_loss_terms_per_sample(
                    output,
                    target,
                    args,
                    teacher_output=teacher_out,
                    source=inp[0],
                )
                loss = _ohem_reduce(per_losses, args.ohem_ratio)
                temporal_loss = _temporal_loss_for_group(
                    qmodel,
                    teacher_model,
                    pairs,
                    group,
                    output,
                    teacher_out.to(dtype=output.dtype),
                    device,
                    dtype,
                    args,
                )
                if float(args.temporal_weight) > 0:
                    loss = loss + float(args.temporal_weight) * temporal_loss
                if not torch.isfinite(loss):
                    continue
                metrics = _average_metric_rows(metric_rows)
                if float(args.temporal_weight) > 0:
                    metrics["temporal"] = float(temporal_loss.detach().item())
                    metrics["total"] = (
                        float(metrics.get("total", 0.0))
                        + float(args.temporal_weight) * metrics["temporal"]
                    )
                group_size = sdr_batch.shape[0]
                (loss * (group_size / total)).backward()
                accumulate_metrics(
                    epoch_sums,
                    {key: value * group_size for key, value in metrics.items()},
                )
                for (item_index, _sdr, _hdr), row in zip(group, metric_rows):
                    item_loss = float(row.get("teacher_l1", row.get("total", 0.0)))
                    sample_losses.append((int(item_index), item_loss))
                samples += group_size
                valid += group_size

            if valid <= 0:
                continue
            torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

        train_metrics = average_metrics(epoch_sums, max(1, samples))
        monitor_metrics = _compute_monitor(
            qmodel,
            monitor_pairs,
            monitor_teacher_outputs,
            device,
            dtype,
            args,
        )
        score = (
            monitor_metrics.get("score_total", monitor_metrics.get("total", float("inf")))
            if monitor_metrics is not None
            else train_metrics.get("total", float("inf"))
        )
        lr_now = scheduler.get_last_lr()[0]
        if monitor_metrics is None:
            print(f"  Epoch {epoch:3d}/{args.epochs}: "
                  f"{format_metrics(train_metrics)}, lr={lr_now:.2e}", flush=True)
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}: "
                  f"train[{format_metrics(train_metrics)}], "
                  f"monitor[{format_metrics(monitor_metrics)}], "
                  f"lr={lr_now:.2e}", flush=True)

        if args.hard_replay_ratio > 0 and sample_losses:
            best_by_index: dict[int, float] = {}
            for item_index, item_loss in sample_losses:
                old = best_by_index.get(item_index)
                if old is None or item_loss > old:
                    best_by_index[item_index] = item_loss
            keep = max(1, int(len(pairs) * float(args.hard_replay_ratio)))
            hard_replay_indices = [
                item_index
                for item_index, _loss_value in sorted(
                    best_by_index.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:keep]
            ]
            print(
                f"    hard replay next epoch: {len(hard_replay_indices)} frame(s)",
                flush=True,
            )

        if score < best_score - args.early_stop_min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in qmodel.state_dict().items()
            }

    if best_state is not None:
        qmodel.load_state_dict(best_state, strict=True)
    print(
        f"  Selected epoch: {best_epoch} "
        f"(best monitor total={best_score:.6f}, "
        f"{(time.perf_counter() - t0) / 60.0:.1f} min)",
        flush=True,
    )
    return qmodel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-model", required=True)
    parser.add_argument("--hg-weights", required=True)
    parser.add_argument(
        "--teacher-model",
        default="",
        help="Optional FP32 teacher base checkpoint. Defaults to --fp32-model.",
    )
    parser.add_argument(
        "--teacher-hg-weights",
        default="",
        help="Optional FP32 teacher HG checkpoint. Defaults to --hg-weights.",
    )
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--output-hg", default="")
    parser.add_argument("--output-combined", default="")
    parser.add_argument("--metadata", default="")
    parser.add_argument("--use-hg", default="1", choices=["1", "0"])
    parser.add_argument("--classifier", default="agcm_spatialmixglobalh8wide64x6")
    parser.add_argument("--le-arch", default="conddirecth8wide96x12")
    parser.add_argument("--hg-arch", default="directh16wide64x8")
    parser.add_argument("--include", default=DEFAULT_INCLUDE)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--sdr-dir", default=str(_REPO_ROOT / "dataset" / "train_sdr"))
    parser.add_argument("--hdr-dir", default=str(_REPO_ROOT / "dataset" / "train_hdr"))
    parser.add_argument("--val-sdr-dir", default=str(_REPO_ROOT / "dataset" / "test_sdr"))
    parser.add_argument("--val-hdr-dir", default=str(_REPO_ROOT / "dataset" / "test_hdr"))
    parser.add_argument("--max-images", type=int, default=512)
    parser.add_argument("--max-val-images", type=int, default=64)
    parser.add_argument("--max-long-edge", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
        help="Training crop size; <=0 uses each loaded frame without random cropping.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-validate", type=int, default=32)
    parser.add_argument("--shuffle", default="1", choices=["1", "0"])
    parser.add_argument(
        "--hard-replay-ratio",
        type=float,
        default=0.0,
        help="Replay the hardest N fraction of frames next epoch.",
    )
    parser.add_argument(
        "--hard-replay-repeat",
        type=int,
        default=1,
        help="Number of extra copies for hard replay frames.",
    )
    parser.add_argument(
        "--ohem-ratio",
        type=float,
        default=1.0,
        help="Backpropagate only the hardest N fraction within each same-shape group. 1.0 disables OHEM.",
    )
    parser.add_argument(
        "--monitor-hard-ratio",
        type=float,
        default=0.0,
        help="Rank checkpoints by the worst N fraction of monitor frames when between 0 and 1.",
    )
    parser.add_argument(
        "--monitor-hard-weight",
        type=float,
        default=1.0,
        help="Weight of worst-frame monitor loss when --monitor-hard-ratio is active.",
    )
    parser.add_argument(
        "--temporal-weight",
        type=float,
        default=0.0,
        help="Sequential frame-delta loss against teacher. Requires full-frame training with crop-size <= 0.",
    )
    parser.add_argument(
        "--temporal-max-gap",
        type=int,
        default=1,
        help="Index gap for sequential temporal pairs in sorted loaded training data.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--train-dtype",
        default="fp32",
        choices=["fp32", "fp16"],
        help="Torch dtype used for QAT fine-tuning. FP32 is safer for Adam updates.",
    )
    parser.add_argument("--seed", type=int, default=1605)
    parser.add_argument("--deterministic", default="0", choices=["1", "0"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-6)

    parser.add_argument("--target-loss-weight", type=float, default=0.05)
    parser.add_argument("--teacher-loss-weight", type=float, default=3.0)
    parser.add_argument("--teacher-luma-weight", type=float, default=1.5)
    parser.add_argument("--teacher-chroma-weight", type=float, default=1.0)
    parser.add_argument("--gradient-teacher-weight", type=float, default=0.0)
    parser.add_argument("--multiscale-teacher-weight", type=float, default=0.0)
    parser.add_argument("--multiscale-levels", type=int, default=2)
    parser.add_argument("--highlight-loss-weight", type=float, default=0.02)
    parser.add_argument("--highlight-teacher-weight", type=float, default=1.5)
    parser.add_argument("--highlight-chroma-weight", type=float, default=0.12)
    parser.add_argument("--highlight-balance-weight", type=float, default=0.15)
    parser.add_argument("--highlight-threshold", type=float, default=0.75)
    parser.add_argument("--neutral-threshold", type=float, default=0.08)
    parser.add_argument("--dark-loss-weight", type=float, default=0.02)
    parser.add_argument("--dark-teacher-weight", type=float, default=1.0)
    parser.add_argument("--dark-luma-weight", type=float, default=0.12)
    parser.add_argument("--dark-chroma-weight", type=float, default=0.08)
    parser.add_argument("--dark-threshold", type=float, default=0.16)
    parser.add_argument("--dark-floor", type=float, default=0.01)
    parser.add_argument("--dark-crop-weight", type=float, default=1.0)
    parser.add_argument("--dark-monitor-weight", type=float, default=1.0)
    parser.add_argument("--source-chroma-weight", type=float, default=0.0)
    parser.add_argument("--source-chroma-saturation-threshold", type=float, default=0.05)
    parser.add_argument("--source-chroma-luma-floor", type=float, default=0.02)
    parser.add_argument("--source-chroma-ratio-clip", type=float, default=6.0)
    parser.add_argument("--source-chroma-crop-weight", type=float, default=0.5)
    parser.add_argument("--source-chroma-monitor-weight", type=float, default=0.5)
    parser.add_argument("--source-shadow-luma-weight", type=float, default=0.0)
    parser.add_argument("--source-shadow-detail-weight", type=float, default=0.0)
    parser.add_argument("--source-shadow-threshold", type=float, default=0.30)
    parser.add_argument("--source-shadow-floor", type=float, default=0.015)
    parser.add_argument("--source-shadow-luma-scale", type=float, default=0.92)
    parser.add_argument("--source-shadow-detail-kernel", type=int, default=7)
    parser.add_argument("--source-shadow-detail-clip", type=float, default=2.0)
    parser.add_argument("--source-shadow-crop-weight", type=float, default=0.5)
    parser.add_argument("--source-shadow-monitor-weight", type=float, default=0.5)
    parser.add_argument("--highlight-crop-attempts", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.use_hg).strip() != "0" and not args.output_hg:
        raise ValueError("--output-hg is required when --use-hg 1")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = torch.float16 if device.type == "cuda" and args.train_dtype == "fp16" else torch.float32
    seed, deterministic = configure_reproducibility(args)

    env = {
        "HDRTVNET_CLASSIFIER": args.classifier,
        "HDRTVNET_LE_ARCH": args.le_arch,
        "HDRTVNET_HG_ARCH": args.hg_arch,
        "HDRTVNET_TRT_INT8_MODELOPT": "1",
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH": "1",
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_INCLUDE": args.include,
    }
    with _env_override(env):
        print(f"Device: {device}, dtype={dtype}, seed={seed}, deterministic={deterministic}")
        teacher_model_path = args.teacher_model or args.fp32_model
        teacher_hg_path = args.teacher_hg_weights or args.hg_weights
        source = _build_fp32_model(args.fp32_model, args.hg_weights, str(args.use_hg) != "0")
        source = source.to(device=device, dtype=dtype).eval()
        teacher = _build_fp32_model(teacher_model_path, teacher_hg_path, str(args.use_hg) != "0")
        teacher = teacher.to(device=device, dtype=dtype).eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

        qmodel = _apply_tensorrt_modelopt_torch_int8_quantization(
            model=source,
            width=args.width,
            height=args.height,
            dtype=dtype,
            device=device,
            precision="int8-mixed",
        )
        enabled, total = _count_tensorrt_modelopt_torch_quantizers(qmodel)

        print(f"\nLoading SDR/HDR pairs from {args.sdr_dir} + {args.hdr_dir} ...")
        pairs = load_image_pairs_from_dirs(
            args.sdr_dir,
            args.hdr_dir,
            args.max_long_edge,
            args.max_images,
        )
        print(f"  {len(pairs)} training pairs loaded")

        monitor_pairs = []
        monitor_stats = {}
        if args.val_sdr_dir and args.val_hdr_dir:
            print(f"\nLoading monitor pairs from {args.val_sdr_dir} + {args.val_hdr_dir} ...")
            val_pairs = load_image_pairs_from_dirs(
                args.val_sdr_dir,
                args.val_hdr_dir,
                args.max_long_edge,
                args.max_val_images,
            )
            monitor_pairs, monitor_stats = select_tone_protected_monitor_pairs(
                val_pairs,
                args.num_validate,
                args,
            )
            print(
                f"  Monitor pairs: {len(monitor_pairs)} "
                f"(avg hi_cov={monitor_stats.get('avg_highlight_cov', 0.0):.4f}, "
                f"avg dark_cov={monitor_stats.get('avg_dark_cov', 0.0):.4f}, "
                f"avg src_cov={monitor_stats.get('avg_source_chroma_cov', 0.0):.4f})"
            )

        qmodel = train_modelopt_qat(qmodel, teacher, pairs, monitor_pairs, device, dtype, args)
        saved = _save_clean_source_weights(qmodel, args)
        print(f"\nSaved clean ModelOpt-QAT source weights: {saved}")

    metadata = {
        "source": "modelopt_torch_qat_source",
        "saved": saved,
        "architecture": {
            "classifier": args.classifier,
            "le_arch": args.le_arch,
            "hg_arch": args.hg_arch,
            "use_hg": str(args.use_hg).strip() != "0",
            "source_base": args.fp32_model,
            "source_hg": args.hg_weights,
            "teacher_base": args.teacher_model or args.fp32_model,
            "teacher_hg": args.teacher_hg_weights or args.hg_weights,
        },
        "modelopt": {
            "include": args.include,
            "quantizers_enabled": int(enabled),
            "quantizers_total": int(total),
            "width": int(args.width),
            "height": int(args.height),
        },
        "training": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "max_images": int(args.max_images),
            "max_long_edge": int(args.max_long_edge),
            "ohem_ratio": float(args.ohem_ratio),
            "hard_replay_ratio": float(args.hard_replay_ratio),
            "hard_replay_repeat": int(args.hard_replay_repeat),
            "monitor_hard_ratio": float(args.monitor_hard_ratio),
            "monitor_hard_weight": float(args.monitor_hard_weight),
            "temporal_weight": float(args.temporal_weight),
            "temporal_max_gap": int(args.temporal_max_gap),
            "gradient_teacher_weight": float(args.gradient_teacher_weight),
            "multiscale_teacher_weight": float(args.multiscale_teacher_weight),
            "multiscale_levels": int(args.multiscale_levels),
            "seed": int(seed),
        },
        "monitor": monitor_stats,
    }
    if args.metadata:
        path = Path(args.metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"metadata -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
