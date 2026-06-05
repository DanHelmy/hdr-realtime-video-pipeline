from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_INPUT = REPO_ROOT / "logs" / "benchmark_sessions" / "Thesis"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "images" / "thesis_mpv_figures"

FRAME_SOURCES = {
    "sdr": ("sdr.png", False, "SDR"),
    "hdr_gt": ("hdr_gt.tiff", True, "HDR GT"),
    "hdr_convert": ("hdr_convert.tiff", True, "HDRTVNet++"),
}


def _parse_render_size(value: str | None) -> tuple[int, int] | None:
    text = str(value or "").strip().lower()
    if not text or text in {"source", "native", "none"}:
        return None
    for sep in ("x", ",", ":"):
        if sep in text:
            left, right = text.split(sep, 1)
            try:
                w = int(left.strip())
                h = int(right.strip())
            except Exception as exc:
                raise argparse.ArgumentTypeError(
                    "render size must look like 1920x1080"
                ) from exc
            if w <= 0 or h <= 0:
                raise argparse.ArgumentTypeError("render size must be positive")
            return int(w), int(h)
    raise argparse.ArgumentTypeError("render size must look like 1920x1080")


def _parse_kinds(value: str) -> list[str]:
    text = str(value or "all").strip().lower()
    if text in {"all", "*"}:
        return list(FRAME_SOURCES.keys())
    aliases = {
        "gt": "hdr_gt",
        "convert": "hdr_convert",
        "prediction": "hdr_convert",
        "pred": "hdr_convert",
    }
    out: list[str] = []
    for raw in text.replace(";", ",").split(","):
        item = raw.strip().lower().replace("-", "_")
        if not item:
            continue
        item = aliases.get(item, item)
        if item not in FRAME_SOURCES:
            raise argparse.ArgumentTypeError(
                f"unknown frame kind '{raw}'. Use all, sdr, hdr_gt, hdr_convert."
            )
        if item not in out:
            out.append(item)
    if not out:
        raise argparse.ArgumentTypeError("at least one frame kind is required")
    return out


def _safe_relative(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except Exception:
        return Path(path.name)


def _contains_requested_frames(frame_dir: Path, kinds: list[str]) -> bool:
    return any((frame_dir / FRAME_SOURCES[k][0]).is_file() for k in kinds)


def _discover_frame_dirs(input_path: Path, kinds: list[str]) -> tuple[list[Path], Path]:
    input_path = input_path.resolve()
    if input_path.is_file():
        parent = input_path.parent
        return ([parent] if _contains_requested_frames(parent, kinds) else []), parent

    if not input_path.is_dir():
        return [], input_path

    if _contains_requested_frames(input_path, kinds):
        return [input_path], input_path

    parents: set[Path] = set()
    for kind in kinds:
        file_name = FRAME_SOURCES[kind][0]
        for file_path in input_path.rglob(file_name):
            parent = file_path.parent
            if _contains_requested_frames(parent, kinds):
                parents.add(parent)
    return sorted(parents), input_path


def _read_frame(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise RuntimeError(f"Could not read {path}")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise RuntimeError(f"Expected an HxWx3 image: {path}")
    if frame.shape[2] > 3:
        frame = frame[:, :, :3]
    return np.ascontiguousarray(frame)


def _bgr_to_rgb48_bytes(frame: np.ndarray) -> bytes:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected HxWx3 frame.")
    if frame.dtype == np.uint16:
        rgb16 = np.ascontiguousarray(frame[:, :, ::-1])
    elif frame.dtype == np.uint8:
        rgb16 = np.ascontiguousarray(frame[:, :, ::-1].astype(np.uint16) * 257)
    else:
        arr = frame.astype(np.float32, copy=False)
        if arr.max(initial=0.0) <= 1.0:
            arr = arr * 65535.0
        rgb16 = np.ascontiguousarray(
            np.clip(arr[:, :, ::-1], 0.0, 65535.0).astype(np.uint16)
        )
    return rgb16.tobytes()


def _pump_events(app, ms: int) -> None:
    from PyQt6.QtCore import QEventLoop

    deadline = time.monotonic() + max(0, int(ms)) / 1000.0
    while time.monotonic() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        time.sleep(0.01)
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)


def _make_mpv_widget(mpv_lib, mpv_diag: bool):
    from gui_mpv_widget import MpvHDRWidget
    from gui_scaling import (
        BEST_MPV_SCALE,
        FILMGRAIN_SHADER_PATH,
        FSR_SHADER_PATH,
        SSIM_DOWNSCALER_SHADER_PATH,
        SSIM_SUPERRES_SHADER_PATH,
        _ensure_filmgrain_shader,
        _ensure_fsr_shader,
        _ensure_ssim_downscaler_shader,
        _ensure_ssim_superres_shader,
        _normalize_shader_paths,
    )

    return MpvHDRWidget(
        mpv_lib=mpv_lib,
        mpv_diag=mpv_diag,
        normalize_shader_paths=_normalize_shader_paths,
        ensure_fsr_shader=_ensure_fsr_shader,
        ensure_ssim_superres_shader=_ensure_ssim_superres_shader,
        ensure_filmgrain_shader=_ensure_filmgrain_shader,
        ensure_ssim_downscaler_shader=_ensure_ssim_downscaler_shader,
        best_mpv_scale=BEST_MPV_SCALE,
        fsr_shader_path=FSR_SHADER_PATH,
        ssim_superres_shader_path=SSIM_SUPERRES_SHADER_PATH,
        filmgrain_shader_path=FILMGRAIN_SHADER_PATH,
        ssim_downscaler_shader_path=SSIM_DOWNSCALER_SHADER_PATH,
    )


class MpvPreviewRenderer:
    def __init__(
        self,
        *,
        app,
        mpv_lib,
        scale_kernel: str,
        scale_antiring: float | None,
        cas_strength: float | None,
        film_grain: bool,
        wait_ms: int,
        png_depth: int,
        mpv_diag: bool,
    ) -> None:
        self.app = app
        self.widget = _make_mpv_widget(mpv_lib, mpv_diag=mpv_diag)
        self.widget.setWindowTitle("HDRTVNet++ mpv thesis figure renderer")
        self.scale_kernel = str(scale_kernel or "ewa_lanczossharp")
        self.scale_antiring = scale_antiring
        self.cas_strength = cas_strength
        self.film_grain = bool(film_grain)
        self.wait_ms = max(50, int(wait_ms))
        self.png_depth = 16 if int(png_depth) == 16 else 8
        self._active_cfg: tuple[int, int, bool] | None = None

    def close(self) -> None:
        try:
            self.widget.stop_playback()
        except Exception:
            pass
        try:
            self.widget.close()
        except Exception:
            pass
        _pump_events(self.app, 50)

    def _ensure_playback(self, src_w: int, src_h: int, hdr: bool) -> None:
        cfg = (int(src_w), int(src_h), bool(hdr))
        if self._active_cfg == cfg and self.widget._player is not None:
            return
        started = self.widget.start_playback(
            width=int(src_w),
            height=int(src_h),
            fps=1.0,
            scale_kernel=self.scale_kernel,
            scale_antiring=self.scale_antiring,
            cas_strength=self.cas_strength,
            force_hdr_metadata=bool(hdr),
            film_grain=self.film_grain,
        )
        if not started:
            raise RuntimeError(
                getattr(self.widget, "_last_scale_error", None)
                or "mpv preview startup failed."
            )
        self._active_cfg = cfg
        _pump_events(self.app, 150)

    def render(
        self,
        frame: np.ndarray,
        output_path: Path,
        *,
        hdr: bool,
        render_size: tuple[int, int] | None,
        overwrite: bool,
    ) -> None:
        h, w = frame.shape[:2]
        out_w, out_h = render_size or (int(w), int(h))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            if not overwrite:
                return
            output_path.unlink()

        # Offline figure rendering favors correctness over throughput. Reusing a
        # single rawvideo playback session can leave mpv/D3D11 window screenshots
        # one frame behind after sequential feed swaps, so restart playback for
        # every saved render.
        try:
            self.widget.stop_playback()
        except Exception:
            pass
        self._active_cfg = None
        self.widget.resize(int(out_w), int(out_h))
        self.widget.show()
        self.widget.raise_()
        _pump_events(self.app, 100)
        self._ensure_playback(int(w), int(h), bool(hdr))

        payload = _bgr_to_rgb48_bytes(frame)
        self.widget.feed_frame(payload)
        _pump_events(self.app, self.wait_ms)
        # A second feed removes the occasional startup/pipe race in libmpv rawvideo.
        self.widget.feed_frame(payload)
        _pump_events(self.app, max(100, self.wait_ms // 3))

        player = self.widget._player
        if player is None:
            raise RuntimeError("mpv player is not active.")
        try:
            player.command("screenshot-to-file", str(output_path), "window")
            _pump_events(self.app, 250)
        except Exception:
            pass

        if output_path.is_file() and output_path.stat().st_size > 0:
            _normalize_preview_png(output_path, self.png_depth)
            return

        screen = self.widget.screen()
        if screen is None:
            raise RuntimeError("No Qt screen is available for mpv screenshot fallback.")
        pixmap = screen.grabWindow(int(self.widget.winId()), 0, 0, int(out_w), int(out_h))
        if pixmap.isNull() or not pixmap.save(str(output_path), "PNG"):
            raise RuntimeError(f"Could not save mpv preview screenshot: {output_path}")
        _normalize_preview_png(output_path, self.png_depth)


def _normalize_preview_png(path: Path, png_depth: int) -> None:
    if int(png_depth) == 16:
        return
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        return
    if image.shape[2] > 3:
        image = image[:, :, :3]
    if image.dtype == np.uint16:
        image = np.clip((image.astype(np.float32) / 257.0) + 0.5, 0, 255).astype(
            np.uint8
        )
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), np.ascontiguousarray(image))


def _label_bar(width: int, label: str) -> np.ndarray:
    bar = np.zeros((44, int(width), 3), dtype=np.uint8)
    bar[:, :] = (24, 27, 30)
    cv2.putText(
        bar,
        str(label),
        (18, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (220, 228, 235),
        2,
        cv2.LINE_AA,
    )
    return bar


def _write_contact_sheet(rendered: dict[str, Path], output_path: Path) -> None:
    columns: list[np.ndarray] = []
    for kind in ("sdr", "hdr_gt", "hdr_convert"):
        path = rendered.get(kind)
        if path is None or not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] > 3:
            image = image[:, :, :3]
        if image.dtype == np.uint16:
            image = np.clip((image.astype(np.float32) / 257.0) + 0.5, 0, 255).astype(
                np.uint8
            )
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        label = FRAME_SOURCES[kind][2]
        columns.append(np.vstack([_label_bar(image.shape[1], label), image]))
    if not columns:
        return
    target_h = max(col.shape[0] for col in columns)
    padded: list[np.ndarray] = []
    for col in columns:
        if col.shape[0] < target_h:
            pad = np.zeros((target_h - col.shape[0], col.shape[1], 3), dtype=np.uint8)
            col = np.vstack([col, pad])
        padded.append(col)
    separator = np.zeros((target_h, 10, 3), dtype=np.uint8)
    separator[:, :] = (35, 39, 43)
    sheet = padded[0]
    for col in padded[1:]:
        sheet = np.hstack([sheet, separator, col])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render benchmark PNG/TIFF frames through the app's libmpv preview "
            "path and save mpv window screenshots for thesis figures."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Benchmark frame directory or parent tree. Default: logs/benchmark_sessions/Thesis",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output directory. Default: docs/images/thesis_mpv_figures",
    )
    parser.add_argument(
        "--kinds",
        type=_parse_kinds,
        default=_parse_kinds("all"),
        help="Comma list: all, sdr, hdr_gt, hdr_convert. Default: all",
    )
    parser.add_argument(
        "--render-size",
        type=_parse_render_size,
        default=None,
        help="mpv window size, e.g. 1920x1080. Default: source frame size.",
    )
    parser.add_argument(
        "--scale",
        default="ewa_lanczossharp",
        help="mpv scale kernel or shader choice, e.g. ewa_lanczossharp, fsr, ssim_superres.",
    )
    parser.add_argument("--scale-antiring", type=float, default=None)
    parser.add_argument("--cas-strength", type=float, default=None)
    parser.add_argument("--film-grain", action="store_true")
    parser.add_argument("--wait-ms", type=int, default=700)
    parser.add_argument(
        "--png-depth",
        type=int,
        choices=(8, 16),
        default=8,
        help="Saved PNG bit depth after mpv rendering. Default: 8 for thesis/PDF use.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit frame folders; 0 means all.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-contact-sheet", action="store_true")
    parser.add_argument("--mpv-diag", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List frame folders without opening mpv.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    input_path = Path(args.input)
    output_root = Path(args.output)
    frame_dirs, discovery_root = _discover_frame_dirs(input_path, args.kinds)
    if args.limit and args.limit > 0:
        frame_dirs = frame_dirs[: int(args.limit)]

    print(f"Found {len(frame_dirs)} frame folder(s).")
    if args.dry_run:
        for frame_dir in frame_dirs[:50]:
            print(frame_dir)
        if len(frame_dirs) > 50:
            print(f"... {len(frame_dirs) - 50} more")
        return 0
    if not frame_dirs:
        return 1

    sys.path.insert(0, str(SRC_DIR))
    from gui_bootstrap import prepare_runtime_environment

    prepare_runtime_environment(str(SRC_DIR / "gui.py"))

    import mpv as mpv_lib
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(["HDRTVNet++ mpv thesis figure renderer"])

    renderer = MpvPreviewRenderer(
        app=app,
        mpv_lib=mpv_lib,
        scale_kernel=str(args.scale),
        scale_antiring=args.scale_antiring,
        cas_strength=args.cas_strength,
        film_grain=bool(args.film_grain),
        wait_ms=int(args.wait_ms),
        png_depth=int(args.png_depth),
        mpv_diag=bool(args.mpv_diag),
    )

    rendered_count = 0
    manifest_items: list[dict] = []
    try:
        for index, frame_dir in enumerate(frame_dirs, start=1):
            rel_dir = _safe_relative(frame_dir, discovery_root)
            out_dir = output_root / rel_dir
            rendered: dict[str, Path] = {}
            print(f"[{index}/{len(frame_dirs)}] {rel_dir}")
            for kind in args.kinds:
                file_name, hdr, _label = FRAME_SOURCES[kind]
                src_path = frame_dir / file_name
                if not src_path.is_file():
                    continue
                suffix = "mpv_hdr" if hdr else "mpv_sdr"
                out_path = out_dir / f"{kind}_{suffix}.png"
                if out_path.exists() and not args.overwrite:
                    rendered[kind] = out_path
                    continue
                frame = _read_frame(src_path)
                renderer.render(
                    frame,
                    out_path,
                    hdr=hdr,
                    render_size=args.render_size,
                    overwrite=bool(args.overwrite),
                )
                rendered[kind] = out_path
                rendered_count += 1
                print(f"  saved {out_path}")
            if rendered and not args.no_contact_sheet:
                sheet_path = out_dir / "mpv_preview_comparison.png"
                if args.overwrite or not sheet_path.exists():
                    _write_contact_sheet(rendered, sheet_path)
                    print(f"  saved {sheet_path}")
            manifest_items.append(
                {
                    "source_dir": str(frame_dir),
                    "output_dir": str(out_dir),
                    "rendered": {k: str(v) for k, v in rendered.items()},
                }
            )
    finally:
        renderer.close()

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "output": str(output_root),
        "renderer": "MpvHDRWidget screenshot-to-file window",
        "uses_ffmpeg_for_rendering": False,
        "scale": str(args.scale),
        "scale_antiring": args.scale_antiring,
        "cas_strength": args.cas_strength,
        "film_grain": bool(args.film_grain),
        "png_depth": int(args.png_depth),
        "render_size": (
            None if args.render_size is None else [int(args.render_size[0]), int(args.render_size[1])]
        ),
        "wait_ms": int(args.wait_ms),
        "items": manifest_items,
    }
    _write_manifest(output_root / "mpv_preview_manifest.json", manifest)
    print(f"Rendered {rendered_count} image(s).")
    print(f"Manifest: {output_root / 'mpv_preview_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
