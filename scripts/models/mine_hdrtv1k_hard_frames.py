"""Mine HDRTV1K hard frames for bright/flat/clipped color replay.

This script keeps the formal dataset story clean: it only reads paired
HDRTV1K-style SDR/HDR image folders and writes local manifests/logs. Movie files
can still be used later for visual stress testing, but not for training data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


_ROOT = Path(__file__).resolve().parents[2]


def _paired_paths(sdr_dir: Path, hdr_dir: Path) -> list[tuple[Path, Path]]:
    suffixes = {".png", ".jpg", ".jpeg"}
    pairs: list[tuple[Path, Path]] = []
    for sdr in sorted(p for p in sdr_dir.iterdir() if p.suffix.lower() in suffixes):
        hdr = hdr_dir / sdr.name
        if not hdr.is_file():
            matches = sorted(hdr_dir.glob(f"{sdr.stem}.*"))
            hdr = matches[0] if matches else hdr
        if hdr.is_file():
            pairs.append((sdr, hdr))
    if not pairs:
        raise FileNotFoundError(f"No SDR/HDR image pairs in {sdr_dir} + {hdr_dir}")
    return pairs


def _read_rgb(path: Path, max_long_edge: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    h, w = img.shape[:2]
    if max_long_edge > 0 and max(h, w) > max_long_edge:
        scale = float(max_long_edge) / float(max(h, w))
        new_w = max(32, int(round(w * scale / 8)) * 8)
        new_h = max(32, int(round(h * scale / 8)) * 8)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    denom = 65535.0 if img.dtype == np.uint16 else 255.0
    return np.clip(img.astype(np.float32) / denom, 0.0, 1.0)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722
    ).astype(np.float32, copy=False)


def _grad_mag(luma: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _pct(mask: np.ndarray) -> float:
    if mask.size <= 0:
        return 0.0
    return float(np.mean(mask.astype(np.float32)))


def _score_pair(sdr_path: Path, hdr_path: Path, max_long_edge: int) -> dict:
    sdr = _read_rgb(sdr_path, max_long_edge)
    hdr = _read_rgb(hdr_path, max_long_edge)
    y = _luma(hdr)
    ys = _luma(sdr)
    grad = _grad_mag(y)
    sat = hdr.max(axis=2) - hdr.min(axis=2)
    maxc = hdr.max(axis=2)
    minc = hdr.min(axis=2)

    bright = y >= 0.70
    very_bright = y >= 0.86
    clipped = maxc >= 0.975
    near_black = y <= 0.06
    flat = grad <= 0.035
    hard_edge = grad >= 0.20
    saturated = (sat >= 0.42) & (maxc >= 0.48)
    saturated_red = saturated & (hdr[:, :, 0] >= maxc - 1e-5)
    saturated_blue = saturated & (hdr[:, :, 2] >= maxc - 1e-5)
    source_bright = ys >= 0.70

    flat_bright_pct = _pct(bright & flat)
    very_bright_flat_pct = _pct(very_bright & flat)
    clipped_pct = _pct(clipped)
    saturated_pct = _pct(saturated)
    saturated_red_blue_pct = _pct(saturated_red | saturated_blue)
    bright_edge_pct = _pct(bright & hard_edge)
    dark_bright_contrast = math.sqrt(max(_pct(near_black), 0.0) * max(_pct(very_bright), 0.0))
    source_target_bright_pct = _pct(source_bright & bright)

    # Prefer hard surfaces/graphics/highlights, but keep enough natural variety.
    score = (
        230.0 * flat_bright_pct
        + 320.0 * very_bright_flat_pct
        + 180.0 * clipped_pct
        + 105.0 * saturated_pct
        + 90.0 * saturated_red_blue_pct
        + 80.0 * bright_edge_pct
        + 28.0 * dark_bright_contrast
        + 30.0 * source_target_bright_pct
    )

    return {
        "sdr": str(sdr_path.relative_to(_ROOT) if sdr_path.is_relative_to(_ROOT) else sdr_path),
        "hdr": str(hdr_path.relative_to(_ROOT) if hdr_path.is_relative_to(_ROOT) else hdr_path),
        "name": sdr_path.name,
        "score": float(score),
        "flat_bright_pct": flat_bright_pct,
        "very_bright_flat_pct": very_bright_flat_pct,
        "clipped_pct": clipped_pct,
        "saturated_pct": saturated_pct,
        "saturated_red_blue_pct": saturated_red_blue_pct,
        "bright_edge_pct": bright_edge_pct,
        "dark_bright_contrast": dark_bright_contrast,
        "source_target_bright_pct": source_target_bright_pct,
        "mean_luma": float(np.mean(y)),
        "p95_luma": float(np.percentile(y, 95)),
    }


def _repeat_for_rank(rank: int, total_hard: int, max_repeat: int) -> int:
    if max_repeat <= 1 or total_hard <= 0:
        return 1
    q = float(rank) / float(max(total_hard - 1, 1))
    if q < 0.15:
        return max_repeat
    if q < 0.40:
        return max(2, max_repeat - 1)
    return 2


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "rank",
        "name",
        "score",
        "repeat",
        "flat_bright_pct",
        "very_bright_flat_pct",
        "clipped_pct",
        "saturated_pct",
        "saturated_red_blue_pct",
        "bright_edge_pct",
        "dark_bright_contrast",
        "source_target_bright_pct",
        "mean_luma",
        "p95_luma",
        "sdr",
        "hdr",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _thumb(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    bgr = np.clip(rgb[:, :, ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)


def _write_contact_sheet(path: Path, rows: list[dict], *, max_items: int, max_long_edge: int) -> None:
    rows = rows[:max_items]
    if not rows:
        return
    tile_w, tile_h = 240, 135
    cols = 4
    label_h = 34
    sheet = np.zeros((math.ceil(len(rows) / cols) * (tile_h + label_h), cols * tile_w, 3), dtype=np.uint8)
    for i, row in enumerate(rows):
        r = i // cols
        c = i % cols
        y0 = r * (tile_h + label_h)
        x0 = c * tile_w
        try:
            thumb = _thumb(_read_rgb(_ROOT / row["hdr"], max_long_edge), tile_w, tile_h)
        except Exception:
            continue
        sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = thumb
        text = f"#{row['rank']} {row['score']:.1f} {Path(row['sdr']).name}"
        cv2.putText(
            sheet,
            text[:34],
            (x0 + 6, y0 + tile_h + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def _manifest(
    *,
    split: str,
    normal_rows: list[dict],
    hard_rows: list[dict],
    hard_repeat: bool,
) -> dict:
    items = []
    for row in normal_rows:
        item = {
            "sdr": row["sdr"],
            "hdr": row["hdr"],
            "source": split,
            "hard_score": row["score"],
            "repeat": 1,
        }
        items.append(item)
    for row in hard_rows:
        repeat = int(row.get("repeat", 1)) if hard_repeat else 1
        if repeat <= 0:
            continue
        items.append({
            "sdr": row["sdr"],
            "hdr": row["hdr"],
            "source": f"{split}_hard",
            "hard_score": row["score"],
            "repeat": repeat,
            "hard_tags": [
                tag
                for tag, value in (
                    ("flat_bright", row.get("flat_bright_pct", 0.0)),
                    ("clipped", row.get("clipped_pct", 0.0)),
                    ("saturated", row.get("saturated_pct", 0.0)),
                    ("bright_edge", row.get("bright_edge_pct", 0.0)),
                )
                if float(value) > 0.01
            ],
        })
    return {
        "schema": "hdrtvnet_hard_frame_manifest_v1",
        "split": split,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "items": items,
    }


def _process_split(
    *,
    split: str,
    sdr_dir: Path,
    hdr_dir: Path,
    out_root: Path,
    max_long_edge: int,
    top: int,
    max_repeat: int,
) -> tuple[list[dict], list[dict]]:
    pairs = _paired_paths(sdr_dir, hdr_dir)
    rows = []
    split_root = out_root / split
    for idx, (sdr, hdr) in enumerate(pairs, 1):
        row = _score_pair(sdr, hdr, max_long_edge)
        rows.append(row)
        if idx % 100 == 0:
            print(f"[{split}] scored {idx}/{len(pairs)}", flush=True)
        if idx % 250 == 0:
            partial = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
            for rank, partial_row in enumerate(partial, 1):
                partial_row["rank"] = rank
                partial_row["repeat"] = _repeat_for_rank(rank - 1, min(top, len(partial)), max_repeat)
            _write_csv(split_root / "scores.partial.csv", partial)
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["repeat"] = _repeat_for_rank(rank - 1, min(top, len(rows)), max_repeat)

    hard = rows[: min(top, len(rows))]
    _write_csv(split_root / "scores.csv", rows)
    _write_csv(split_root / "hard_top.csv", hard)
    _write_contact_sheet(
        split_root / "hard_top_contact.jpg",
        hard,
        max_items=min(48, len(hard)),
        max_long_edge=max_long_edge,
    )
    (split_root / "hard_only_manifest.json").write_text(
        json.dumps(_manifest(split=split, normal_rows=[], hard_rows=hard, hard_repeat=False), indent=2),
        encoding="utf-8",
    )
    return rows, hard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sdr-dir", default=str(_ROOT / "dataset" / "train_sdr"))
    parser.add_argument("--train-hdr-dir", default=str(_ROOT / "dataset" / "train_hdr"))
    parser.add_argument("--test-sdr-dir", default=str(_ROOT / "dataset" / "test_sdr"))
    parser.add_argument("--test-hdr-dir", default=str(_ROOT / "dataset" / "test_hdr"))
    parser.add_argument("--out-root", default=str(_ROOT / "logs" / "hard_frames" / time.strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--max-long-edge", type=int, default=384)
    parser.add_argument("--top-train", type=int, default=256)
    parser.add_argument("--top-test", type=int, default=64)
    parser.add_argument("--max-repeat", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    train_rows, train_hard = _process_split(
        split="train",
        sdr_dir=Path(args.train_sdr_dir),
        hdr_dir=Path(args.train_hdr_dir),
        out_root=out_root,
        max_long_edge=int(args.max_long_edge),
        top=int(args.top_train),
        max_repeat=int(args.max_repeat),
    )
    test_rows, test_hard = _process_split(
        split="test",
        sdr_dir=Path(args.test_sdr_dir),
        hdr_dir=Path(args.test_hdr_dir),
        out_root=out_root,
        max_long_edge=int(args.max_long_edge),
        top=int(args.top_test),
        max_repeat=1,
    )
    (out_root / "train_bright_replay_manifest.json").write_text(
        json.dumps(_manifest(split="train", normal_rows=train_rows, hard_rows=train_hard, hard_repeat=True), indent=2),
        encoding="utf-8",
    )
    (out_root / "test_hard_manifest.json").write_text(
        json.dumps(_manifest(split="test", normal_rows=[], hard_rows=test_hard, hard_repeat=False), indent=2),
        encoding="utf-8",
    )
    summary = {
        "train_pairs": len(train_rows),
        "train_hard": len(train_hard),
        "test_pairs": len(test_rows),
        "test_hard": len(test_hard),
        "out_root": str(out_root),
        "top_train": train_hard[:10],
        "top_test": test_hard[:10],
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_root": str(out_root),
        "train_pairs": len(train_rows),
        "train_hard": len(train_hard),
        "test_pairs": len(test_rows),
        "test_hard": len(test_hard),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
