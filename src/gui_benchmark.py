from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import re
import shutil
import tempfile
import queue
import threading
import webbrowser
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from PyQt6.QtCore import QObject, QThread, Qt, QDir, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QListView,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from gui_config import (
    DEFAULT_RESOLUTION_KEY,
    PRECISIONS,
    RESOLUTION_SCALES,
    _available_precision_keys,
    _select_model_path,
)
from gui_hdr_io import (
    FFMPEG_WINDOWS_DOWNLOAD_URL,
    hdr_ffmpeg_ready,
    read_hdr_video_frame_rgb16,
    read_image_any,
    write_hdr_tiff,
    tensor_to_bgr_u16,
)
from gui_media_probe import (
    _crop_gt_frame_to_pair_active_area,
    _frame_structure_similarity,
    _map_video_pair_frame_index,
    _probe_video_active_area_info,
    _probe_hdr_input,
    _probe_video_sync_info,
    _probe_video_timing_info,
    _validate_video_timing_compatibility,
)
from gui_objective_metrics import (
    _compute_full_reference_metrics,
    _read_video_frame_at,
)
from gui_compile_cache import _is_compiled, _precision_to_compile_arg
from gui_pipeline_worker_model import _resolve_predequantize_arg
from gui_mpv_widget import MpvHDRWidget
from gui_scaling import (
    BEST_MPV_SCALE,
    FILMGRAIN_SHADER_PATH,
    FSR_SHADER_PATH,
    SSIM_SUPERRES_SHADER_PATH,
    _ensure_filmgrain_shader,
    _ensure_fsr_shader,
    _ensure_ssim_superres_shader,
    _letterbox_bgr,
    _normalize_shader_paths,
)
from gui_widgets import _CompareVideoPane
from models.hdrtvnet_torch import (
    HDRTVNetTensorRT,
    HDRTVNetTorch,
    _HAS_COMPILE,
    _HAS_TRITON,
    _IS_NVIDIA,
    _IS_ROCM,
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".exr"}
_VIDEO_EXTS = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".webm",
    ".flv",
    ".m4v",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".ts",
    ".m2ts",
    ".mts",
    ".ogv",
    ".3gp",
}
_FRAME_RESULT_FILE = "benchmark_frame_result.json"
_BENCHMARK_METRIC_KEYS = [
    "psnr_db",
    "sssim",
    "delta_e_itp",
    "psnr_norm_db",
    "sssim_norm",
    "delta_e_itp_norm",
    "hdr_vdp3",
]
_BENCHMARK_OUTLIER_KEYS = [
    "sssim_norm",
    "sssim",
    "psnr_norm_db",
    "psnr_db",
    "delta_e_itp_norm",
    "delta_e_itp",
    "hdr_vdp3",
]
_BENCHMARK_SHARED_MIN_OVERLAP_RATIO = 0.80

try:
    import mpv as mpv_lib

    _HAS_MPV = True
except (OSError, ImportError):
    mpv_lib = None
    _HAS_MPV = False

_MPV_DIAG = os.environ.get("HDRTVNET_MPV_DIAG", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_BENCHMARK_HDR_GT_MODE = str(
    os.environ.get("HDRTVNET_BENCHMARK_HDR_GT_MODE", "auto")
).strip().lower()
if _BENCHMARK_HDR_GT_MODE not in {"auto", "fast", "exact"}:
    _BENCHMARK_HDR_GT_MODE = "auto"
# Auto mode policy: fast first pass, then exact/local-aligned post verification.
_BENCHMARK_AUTO_POST_VERIFY_ENABLED = str(
    os.environ.get("HDRTVNET_BENCHMARK_AUTO_POST_VERIFY", "1")
).strip().lower() in {"1", "true", "yes", "on"}


def _parse_post_verify_max_items() -> int | None:
    raw = str(
        os.environ.get("HDRTVNET_BENCHMARK_AUTO_POST_VERIFY_MAX_ITEMS", "all")
    ).strip().lower()
    if raw in {"", "all", "full", "unlimited"}:
        return None
    try:
        return max(0, int(raw))
    except Exception:
        return None


_BENCHMARK_AUTO_POST_VERIFY_MAX_ITEMS = _parse_post_verify_max_items()
try:
    _BENCHMARK_AUTO_POST_VERIFY_SUSPECT_SCORE = float(
        os.environ.get("HDRTVNET_BENCHMARK_AUTO_POST_VERIFY_SUSPECT_SCORE", "0.10")
    )
except Exception:
    _BENCHMARK_AUTO_POST_VERIFY_SUSPECT_SCORE = 0.10
try:
    _BENCHMARK_AUTO_POST_VERIFY_IMPROVE_MARGIN = float(
        os.environ.get("HDRTVNET_BENCHMARK_AUTO_POST_VERIFY_IMPROVE_MARGIN", "0.04")
    )
except Exception:
    _BENCHMARK_AUTO_POST_VERIFY_IMPROVE_MARGIN = 0.04
try:
    _BENCHMARK_AUTO_POST_VERIFY_GT_DIFF_SCORE = float(
        os.environ.get("HDRTVNET_BENCHMARK_AUTO_POST_VERIFY_GT_DIFF_SCORE", "0.985")
    )
except Exception:
    _BENCHMARK_AUTO_POST_VERIFY_GT_DIFF_SCORE = 0.985
try:
    _BENCHMARK_AUTO_POST_VERIFY_GT_DIFF_MEAN = float(
        os.environ.get("HDRTVNET_BENCHMARK_AUTO_POST_VERIFY_GT_DIFF_MEAN", "0.0025")
    )
except Exception:
    _BENCHMARK_AUTO_POST_VERIFY_GT_DIFF_MEAN = 0.0025
try:
    _BENCHMARK_GT_LOCAL_SEARCH_FRAMES = max(
        0,
        min(
            120,
            int(os.environ.get("HDRTVNET_BENCHMARK_GT_LOCAL_SEARCH_FRAMES", "0")),
        ),
    )
except Exception:
    _BENCHMARK_GT_LOCAL_SEARCH_FRAMES = 8
try:
    _BENCHMARK_GT_LOCAL_SEARCH_MIN_GAIN = max(
        0.0,
        float(os.environ.get("HDRTVNET_BENCHMARK_GT_LOCAL_SEARCH_MIN_GAIN", "0.035")),
    )
except Exception:
    _BENCHMARK_GT_LOCAL_SEARCH_MIN_GAIN = 0.035
_BENCHMARK_USE_COMPILE_CACHE = str(
    os.environ.get("HDRTVNET_BENCHMARK_USE_COMPILE_CACHE", "1")
).strip().lower() not in {"0", "false", "no", "off"}


def _new_mpv_widget() -> MpvHDRWidget:
    return MpvHDRWidget(
        mpv_lib=mpv_lib,
        mpv_diag=_MPV_DIAG,
        normalize_shader_paths=_normalize_shader_paths,
        ensure_fsr_shader=_ensure_fsr_shader,
        ensure_ssim_superres_shader=_ensure_ssim_superres_shader,
        ensure_filmgrain_shader=_ensure_filmgrain_shader,
        best_mpv_scale=BEST_MPV_SCALE,
        fsr_shader_path=FSR_SHADER_PATH,
        ssim_superres_shader_path=SSIM_SUPERRES_SHADER_PATH,
        filmgrain_shader_path=FILMGRAIN_SHADER_PATH,
    )


def _is_image_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _IMAGE_EXTS


def _is_video_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _VIDEO_EXTS


def _sanitize_name(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text or "").strip())
    safe = re.sub(r"_+", "_", safe).strip("._-")
    return safe or "sample"


def _resolution_dims(res_key: str) -> tuple[int, int]:
    dims = RESOLUTION_SCALES.get(str(res_key or ""))
    if dims is None:
        return 1920, 1080
    return int(dims[0]), int(dims[1])


def _normalize_benchmark_predequantize_mode(mode: str | None) -> str:
    m = str(mode or "auto").strip().lower()
    if m in {"on", "off"}:
        return m
    return "auto"


def _effective_benchmark_predequantize_mode(
    precision_arg: str,
    selected_mode: str | None,
) -> str:
    mode = _normalize_benchmark_predequantize_mode(selected_mode)
    if not str(precision_arg or "").startswith("int8"):
        return mode
    if mode in {"on", "off"}:
        return mode
    if _IS_ROCM:
        return "on"
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            has_int8_tc = (
                props.major > 7
                or (props.major == 7 and props.minor >= 5)
            )
            return "off" if has_int8_tc else "on"
        except Exception:
            return "auto"
    return "auto"


def _benchmark_compile_cache_ready(
    *,
    w: int,
    h: int,
    precision_key: str,
    model_path: str,
    use_hg: bool,
    predequantize_mode: str,
) -> tuple[bool, str, str]:
    """Return whether benchmark can safely use cached max-autotune kernels."""
    if _IS_NVIDIA or not _BENCHMARK_USE_COMPILE_CACHE:
        return False, "disabled", "auto"
    if not (_HAS_COMPILE and _HAS_TRITON and torch.cuda.is_available()):
        return False, "unavailable", "auto"

    precision_arg = _precision_to_compile_arg(precision_key)
    selected_pdq = _normalize_benchmark_predequantize_mode(predequantize_mode)
    effective_pdq = _effective_benchmark_predequantize_mode(
        precision_arg,
        selected_pdq,
    )
    if _is_compiled(
        int(w),
        int(h),
        precision_arg,
        model_path=model_path,
        use_hg=bool(use_hg),
        predequantize_mode=effective_pdq,
    ):
        return True, precision_arg, effective_pdq

    if (
        str(precision_arg).startswith("int8")
        and selected_pdq == "auto"
        and effective_pdq != "auto"
        and _is_compiled(
            int(w),
            int(h),
            precision_arg,
            model_path=model_path,
            use_hg=bool(use_hg),
            predequantize_mode="auto",
        )
    ):
        return True, precision_arg, "auto"

    if str(precision_arg).startswith("int8") and effective_pdq == "on":
        fp16_model = _select_model_path("FP16", bool(use_hg))
        if _is_compiled(
            int(w),
            int(h),
            "fp16",
            model_path=fp16_model,
            use_hg=bool(use_hg),
            predequantize_mode="auto",
        ):
            return True, precision_arg, effective_pdq

    return False, precision_arg, effective_pdq


def _fmt_metric(v, fmt: str = ".4f", suffix: str = "") -> str:
    try:
        fv = float(v)
    except Exception:
        return "-"
    if not np.isfinite(fv):
        return "-"
    return f"{format(fv, fmt)}{suffix}"


def _sorted_media_files(root_dir: str) -> list[str]:
    out: list[str] = []
    for base, _dirs, files in os.walk(root_dir):
        for name in files:
            p = os.path.join(base, name)
            ext = os.path.splitext(name)[1].lower()
            if ext in _IMAGE_EXTS or ext in _VIDEO_EXTS:
                out.append(p)
    out.sort()
    return out


def _read_media_frame(path: str, frame_idx: int | None = None) -> np.ndarray | None:
    if not path or not os.path.isfile(path):
        return None
    if _is_image_path(path):
        frame = read_image_any(path)
        if frame is None:
            return None
        if frame.dtype == np.uint16:
            return np.ascontiguousarray(
                ((frame.astype(np.float32) / 65535.0) * 255.0).astype(np.uint8)
            )
        return np.ascontiguousarray(frame.astype(np.uint8, copy=False))
    if _is_video_path(path):
        idx = max(0, int(frame_idx or 0))
        return _read_video_frame_at(path, idx)
    return None


def _read_media_frame_hdr(path: str, frame_idx: int | None = None) -> np.ndarray | None:
    frame, _mode = _read_media_frame_hdr_with_mode(path, frame_idx=frame_idx)
    return frame


def _read_media_frame_hdr_with_mode(
    path: str,
    frame_idx: int | None = None,
    sdr_reference: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str]:
    if not path or not os.path.isfile(path):
        return None, "missing"
    if _is_image_path(path):
        frame = read_image_any(path)
        if frame is None:
            return None, "missing"
        if frame.dtype == np.uint16:
            return np.ascontiguousarray(frame), "true_hdr_image"
        return None, f"unsupported_image_dtype:{frame.dtype}"
    if _is_video_path(path):
        idx = max(0, int(frame_idx or 0))
        mode = str(_BENCHMARK_HDR_GT_MODE)
        prefer_fast = (mode != "exact")
        decode_mode = "exact"
        if prefer_fast:
            # First pass should stay fast. Auto mode can repair suspicious rows
            # in a deferred exact verification pass at the end of the run.
            rgb16 = read_hdr_video_frame_rgb16(
                path,
                idx,
                prefer_fast_seek=True,
                fast_seek_pts_guard=False,
            )
            decode_mode = "fast"
            if rgb16 is None:
                rgb16 = read_hdr_video_frame_rgb16(path, idx, prefer_fast_seek=False)
                decode_mode = "exact_fallback"
        else:
            rgb16 = read_hdr_video_frame_rgb16(path, idx, prefer_fast_seek=False)
            decode_mode = "exact"
        if rgb16 is not None:
            gt_bgr16 = np.ascontiguousarray(rgb16[:, :, ::-1])
            return gt_bgr16, f"true_hdr_video_{decode_mode}"
        return None, "hdr_video_decode_failed"
    return None, "missing"


def _map_gt_frame_for_sdr(
    sdr_path: str,
    gt_path: str,
    frame_idx: int | None,
) -> int | None:
    if frame_idx is None:
        return None
    if _is_video_path(sdr_path) and _is_video_path(gt_path):
        return _map_video_pair_frame_index(sdr_path, gt_path, int(frame_idx))
    return max(0, int(frame_idx))


def _to_u8_for_alignment_frame(frame: np.ndarray | None) -> np.ndarray | None:
    if not isinstance(frame, np.ndarray):
        return None
    arr = np.ascontiguousarray(frame)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return np.ascontiguousarray(
            ((arr.astype(np.float32) / 65535.0) * 255.0)
            .clip(0.0, 255.0)
            .astype(np.uint8)
        )
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        peak = max(1.0, float(info.max))
        return np.ascontiguousarray(
            (arr.astype(np.float32) * (255.0 / peak))
            .clip(0.0, 255.0)
            .astype(np.uint8)
        )
    arr_f = arr.astype(np.float32)
    peak = float(np.max(arr_f)) if arr_f.size else 1.0
    if peak <= 1.0:
        arr_f = arr_f * 255.0
    return np.ascontiguousarray(arr_f.clip(0.0, 255.0).astype(np.uint8))


def _local_align_gt_frame_for_benchmark(
    *,
    sdr_path: str,
    gt_path: str,
    mapped_gt_frame_idx: int,
    sdr_eval_u8: np.ndarray | None,
    out_w: int,
    out_h: int,
    cancel_check=None,
) -> dict:
    """Find the best nearby GT frame for one benchmark sample."""
    base_idx = max(0, int(mapped_gt_frame_idx or 0))
    info = {
        "canceled": False,
        "frame_idx": int(base_idx),
        "base_frame_idx": int(base_idx),
        "best_frame_idx": int(base_idx),
        "offset_frames": 0,
        "score": None,
        "base_score": None,
        "best_score": None,
        "search_radius_frames": int(_BENCHMARK_GT_LOCAL_SEARCH_FRAMES),
    }
    radius = int(_BENCHMARK_GT_LOCAL_SEARCH_FRAMES)
    if (
        radius <= 0
        or not isinstance(sdr_eval_u8, np.ndarray)
        or not _is_video_path(gt_path)
    ):
        return info

    gt_meta = _probe_video_timing_info(gt_path)
    gt_n = int(gt_meta.get("frame_count", 0) or 0) if isinstance(gt_meta, dict) else 0
    start_idx = max(0, base_idx - radius)
    end_idx = base_idx + radius
    if gt_n > 0:
        end_idx = min(end_idx, gt_n - 1)
    if end_idx < start_idx:
        return info

    cap = cv2.VideoCapture(gt_path)
    if not cap.isOpened():
        cap.release()
        return info

    best_idx = int(base_idx)
    best_score = None
    base_score = None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for gt_idx in range(start_idx, end_idx + 1):

            if callable(cancel_check):
                try:
                    if bool(cancel_check()):
                        info["canceled"] = True
                        break
                except Exception:
                    pass
            
            ok, gt_frame = cap.read()
            if not ok or gt_frame is None:
                break
            try:
                gt_eval = np.ascontiguousarray(
                    _letterbox_bgr(
                        _crop_gt_frame_to_pair_active_area(
                            gt_frame,
                            sdr_path,
                            gt_path,
                        ),
                        out_w,
                        out_h,
                    )
                )
                gt_u8 = _to_u8_for_alignment_frame(gt_eval)
                if gt_u8 is None:
                    continue
                score = _frame_structure_similarity(sdr_eval_u8, gt_u8)
            except Exception:
                score = None
            if score is None or not np.isfinite(float(score)):
                continue
            score_f = float(score)
            if gt_idx == base_idx:
                base_score = score_f
            if best_score is None or score_f > float(best_score):
                best_score = score_f
                best_idx = int(gt_idx)
    finally:
        cap.release()

    selected_idx = int(base_idx)
    selected_score = base_score
    if best_score is not None:
        if best_idx == base_idx:
            selected_idx = int(best_idx)
            selected_score = float(best_score)
        elif base_score is None:
            selected_idx = int(best_idx)
            selected_score = float(best_score)
        elif float(best_score) >= (
            float(base_score) + float(_BENCHMARK_GT_LOCAL_SEARCH_MIN_GAIN)
        ):
            selected_idx = int(best_idx)
            selected_score = float(best_score)

    info.update(
        {
            "frame_idx": int(selected_idx),
            "best_frame_idx": int(best_idx),
            "offset_frames": int(selected_idx) - int(base_idx),
            "score": None if selected_score is None else float(selected_score),
            "base_score": None if base_score is None else float(base_score),
            "best_score": None if best_score is None else float(best_score),
        }
    )
    return info


def _detect_distinct_video_frames(
    video_path: str,
    desired_count: int = 10,
    max_scan_points: int = 260,
) -> list[int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return []

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 1:
            return [1]

        scan_points = max(desired_count * 18, 80)
        scan_points = min(scan_points, max_scan_points, total)
        idxs = np.linspace(0, total - 1, num=scan_points, dtype=np.int64)

        prev_hist = None
        prev_luma = None
        scored: list[tuple[float, int]] = []

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            luma = float(np.mean(gray))
            texture = float(np.std(gray)) / 64.0
            score = texture
            if prev_hist is not None:
                scene = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                luma_jump = abs(luma - float(prev_luma or 0.0)) / 255.0
                score = (scene * 0.78) + (luma_jump * 0.18) + (texture * 0.04)
            scored.append((float(score), int(idx)))
            prev_hist = hist
            prev_luma = luma

        if not scored:
            return [1]

        scored.sort(key=lambda x: (-x[0], x[1]))
        min_spacing = max(1, total // max(2, desired_count * 2))

        chosen: list[int] = []

        def _accept(candidate: int) -> bool:
            for c in chosen:
                if abs(int(candidate) - int(c)) < min_spacing:
                    return False
            return True

        for _score, idx in scored:
            if _accept(idx):
                chosen.append(int(idx))
            if len(chosen) >= desired_count:
                break

        anchors = [0.08, 0.22, 0.38, 0.52, 0.68, 0.82, 0.92]
        for ratio in anchors:
            if len(chosen) >= desired_count:
                break
            idx = int(round((total - 1) * float(ratio)))
            if _accept(idx):
                chosen.append(idx)

        chosen = sorted({max(0, min(total - 1, int(v))) for v in chosen})
        if len(chosen) > desired_count:
            chosen = chosen[:desired_count]
        if not chosen:
            chosen = [0]
        # UI displays 1-based frame numbering.
        return [int(v) + 1 for v in chosen]
    finally:
        cap.release()


@dataclass
class BenchmarkTask:
    task_id: str
    label: str
    sdr_path: str
    gt_path: str
    frame_idx: int | None = None


@dataclass
class BenchmarkRunConfig:
    mode: str
    source_name: str
    precision_key: str
    use_hg: bool
    resolution_key: str
    predequantize_mode: str
    tasks: list[BenchmarkTask]
    session_dir: str
    runtime_mode: str = "auto"


@dataclass
class _QueuedBenchmarkRun:
    config: BenchmarkRunConfig
    title: str


class _BenchmarkWorker(QObject):
    progress = pyqtSignal(int, str)
    sample_ready = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)
    canceled = pyqtSignal(str)

    def __init__(self, config: BenchmarkRunConfig):
        super().__init__()
        self._config = config
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def _is_canceled(self) -> bool:
        return bool(self._cancel.is_set())

    def _run_blocking_with_cancel(
        self,
        fn,
        *args,
        canceled_fallback=None,
        poll_interval_s: float = 0.03,
        **kwargs,
    ):
        done_evt = threading.Event()
        out_q: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def _target() -> None:
            try:
                out_q.put((True, fn(*args, **kwargs)))
            except Exception as exc:
                out_q.put((False, exc))
            finally:
                done_evt.set()

        threading.Thread(target=_target, daemon=True).start()

        while not done_evt.wait(timeout=max(0.005, float(poll_interval_s))):
            if self._is_canceled():
                return canceled_fallback, True

        ok, payload = out_q.get_nowait()
        if not ok:
            raise payload
        return payload, False

    def _choose_unique_dir(self, base_dir: str) -> str:
        if not os.path.exists(base_dir):
            return base_dir
        n = 2
        while True:
            candidate = f"{base_dir}_{n}"
            if not os.path.exists(candidate):
                return candidate
            n += 1

    def _compute_average_from_rows(self, rows: list[dict]) -> dict[str, float | None]:
        averages: dict[str, float | None] = {}
        for key in _BENCHMARK_METRIC_KEYS:
            vals = []
            for r in rows:
                m = r.get("metrics") or {}
                v = m.get(key)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    vals.append(fv)
            averages[key] = float(np.mean(vals)) if vals else None
        return averages

    def _write_session_summaries(self, cfg: BenchmarkRunConfig, rows: list[dict]) -> tuple[str, str, dict[str, float | None]]:
        summary_csv = os.path.join(cfg.session_dir, "benchmark_summary.csv")
        summary_json = os.path.join(cfg.session_dir, "benchmark_summary.json")
        averages = self._compute_average_from_rows(rows)

        try:
            with open(summary_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "item",
                        "frame",
                        "psnr_db",
                        "sssim",
                        "delta_e_itp",
                        "psnr_norm_db",
                        "sssim_norm",
                        "delta_e_itp_norm",
                        "hdr_vdp3",
                        "obj_note",
                        "hdr_vdp3_note",
                        "gt_frame",
                        "gt_alignment_offset_frames",
                        "gt_alignment_score",
                        "sample_dir",
                    ]
                )
                for r in rows:
                    m = r.get("metrics") or {}
                    writer.writerow(
                        [
                            r.get("label"),
                            r.get("frame"),
                            m.get("psnr_db"),
                            m.get("sssim"),
                            m.get("delta_e_itp"),
                            m.get("psnr_norm_db"),
                            m.get("sssim_norm"),
                            m.get("delta_e_itp_norm"),
                            m.get("hdr_vdp3"),
                            m.get("obj_note"),
                            m.get("hdr_vdp3_note"),
                            r.get("gt_frame_idx"),
                            r.get("gt_alignment_offset_frames"),
                            r.get("gt_alignment_score"),
                            r.get("sample_dir"),
                        ]
                    )
                writer.writerow([])
                writer.writerow(["AVERAGE"])
                writer.writerow(
                    [
                        "",
                        "",
                        averages.get("psnr_db"),
                        averages.get("sssim"),
                        averages.get("delta_e_itp"),
                        averages.get("psnr_norm_db"),
                        averages.get("sssim_norm"),
                        averages.get("delta_e_itp_norm"),
                        averages.get("hdr_vdp3"),
                        "",
                        "",
                        "",
                        "",
                        "",
                        cfg.session_dir,
                    ]
                )
        except Exception:
            pass

        try:
            with open(summary_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mode": cfg.mode,
                        "source_name": cfg.source_name,
                        "precision": cfg.precision_key,
                        "use_hg": bool(cfg.use_hg),
                        "resolution": cfg.resolution_key,
                        "predequantize_mode": cfg.predequantize_mode,
                        "benchmark_runtime": cfg.runtime_mode,
                        "session_dir": cfg.session_dir,
                        "average": averages,
                        "results": rows,
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

        return summary_csv, summary_json, averages

    def _write_frame_result_file(self, cfg: BenchmarkRunConfig, row: dict) -> None:
        sample_dir = str(row.get("sample_dir") or "").strip()
        if not sample_dir:
            return
        try:
            os.makedirs(sample_dir, exist_ok=True)
            payload = {
                "source_name": cfg.source_name,
                "precision": cfg.precision_key,
                "resolution": cfg.resolution_key,
                "use_hg": bool(cfg.use_hg),
                "predequantize_mode": cfg.predequantize_mode,
                "benchmark_runtime": cfg.runtime_mode,
                "session_dir": cfg.session_dir,
                "row": dict(row),
            }
            with open(os.path.join(sample_dir, _FRAME_RESULT_FILE), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def _to_u8_for_alignment(frame: np.ndarray | None) -> np.ndarray | None:
        return _to_u8_for_alignment_frame(frame)

    def _compute_metrics_for_pair(
        self,
        pred_eval: np.ndarray,
        gt_eval: np.ndarray,
    ) -> dict:
        return _compute_full_reference_metrics(pred_eval, gt_eval)

    def run(self):
        processor = None
        try:
            cfg = self._config
            if not cfg.tasks:
                self.failed.emit("No benchmark items were provided.")
                return

            model_path = _select_model_path(cfg.precision_key, cfg.use_hg)
            if not model_path or not os.path.isfile(model_path):
                self.failed.emit(f"Model weights not found:\n{model_path}")
                return

            precision_cfg = PRECISIONS.get(cfg.precision_key, {})
            model_precision = str(precision_cfg.get("precision") or "fp16")

            out_w, out_h = _resolution_dims(cfg.resolution_key)
            compile_cache_ready = False
            compile_precision_arg = "auto"
            compile_pdq_mode = "auto"
            if not _IS_NVIDIA:
                (
                    compile_cache_ready,
                    compile_precision_arg,
                    compile_pdq_mode,
                ) = _benchmark_compile_cache_ready(
                    w=out_w,
                    h=out_h,
                    precision_key=cfg.precision_key,
                    model_path=model_path,
                    use_hg=bool(cfg.use_hg),
                    predequantize_mode=str(cfg.predequantize_mode),
                )
            self.progress.emit(
                0,
                "Loading TensorRT engine ..."
                if _IS_NVIDIA
                else (
                    "Loading cached max-autotune model ..."
                    if compile_cache_ready
                    else "Loading model in eager mode ..."
                ),
            )
            if _IS_NVIDIA:
                processor = HDRTVNetTensorRT(
                    model_path,
                    device="auto",
                    precision=model_precision,
                    engine_width=out_w,
                    engine_height=out_h,
                    mode_name=f"{cfg.precision_key}_{'hg' if cfg.use_hg else 'nohg'}",
                    use_hg=bool(cfg.use_hg),
                )
                cfg.runtime_mode = "TensorRT"
            else:
                processor = HDRTVNetTorch(
                    model_path,
                    device="auto",
                    precision=model_precision,
                    compile_model=bool(compile_cache_ready),
                    force_compile=bool(compile_cache_ready),
                    compile_mode="max-autotune" if compile_cache_ready else "default",
                    predequantize=_resolve_predequantize_arg(str(cfg.predequantize_mode)),
                    use_hg=bool(cfg.use_hg),
                )
                if getattr(processor, "_compiled", False):
                    cfg.runtime_mode = (
                        f"cached max-autotune ({compile_precision_arg}, "
                        f"predequantize={compile_pdq_mode})"
                    )
                elif compile_cache_ready:
                    cfg.runtime_mode = "eager (compile cache unavailable at load)"
                else:
                    cfg.runtime_mode = "eager"

            rows: list[dict] = []
            ordered_tasks = sorted(
                cfg.tasks,
                key=lambda task: (
                    str(task.gt_path or "").lower(),
                    str(task.sdr_path or "").lower(),
                    int(task.frame_idx if task.frame_idx is not None else -1),
                    str(task.label or "").lower(),
                ),
            )
            total = max(1, len(ordered_tasks))
            allow_post_verify = (
                str(_BENCHMARK_HDR_GT_MODE) != "exact"
                and _BENCHMARK_AUTO_POST_VERIFY_ENABLED
                and (
                    _BENCHMARK_AUTO_POST_VERIFY_MAX_ITEMS is None
                    or int(_BENCHMARK_AUTO_POST_VERIFY_MAX_ITEMS) > 0
                )
            )
            post_verify_candidates: list[tuple[int, BenchmarkTask, float]] = []

            for i, task in enumerate(ordered_tasks):
                if self._is_canceled():
                    self.canceled.emit("Benchmark canceled.")
                    return

                self.progress.emit(
                    int((100.0 * i) / float(total)),
                    f"Processing {i + 1}/{total}: {task.label}",
                )

                sdr_raw = _read_media_frame(task.sdr_path, frame_idx=task.frame_idx)
                gt_frame_idx = _map_gt_frame_for_sdr(
                    task.sdr_path,
                    task.gt_path,
                    task.frame_idx,
                )
                gt_hdr_raw, gt_hdr_mode = _read_media_frame_hdr_with_mode(
                    task.gt_path,
                    frame_idx=gt_frame_idx,
                    sdr_reference=sdr_raw,
                )
                if sdr_raw is None:
                    note = "Missing frame/image data."
                    row = {
                        "task_id": task.task_id,
                        "label": task.label,
                        "frame": (int(task.frame_idx) + 1) if task.frame_idx is not None else None,
                        "sample_dir": "",
                        "sdr_image": "",
                        "hdr_gt_image": "",
                        "hdr_convert_image": "",
                        "metrics": {
                            "psnr_db": None,
                            "sssim": None,
                            "delta_e_itp": None,
                            "psnr_norm_db": None,
                            "sssim_norm": None,
                            "delta_e_itp_norm": None,
                            "hdr_vdp3": None,
                            "obj_note": note,
                            "hdr_vdp3_note": "",
                        },
                    }
                    rows.append(row)
                    self.sample_ready.emit(row)
                    continue

                if gt_hdr_raw is None:
                    note = (
                        "HDR GT must decode as true uint16 linear data "
                        f"(mode={gt_hdr_mode})."
                    )
                    row = {
                        "task_id": task.task_id,
                        "label": task.label,
                        "frame": (int(task.frame_idx) + 1) if task.frame_idx is not None else None,
                        "sample_dir": "",
                        "sdr_image": "",
                        "hdr_gt_image": "",
                        "hdr_convert_image": "",
                        "metrics": {
                            "psnr_db": None,
                            "sssim": None,
                            "delta_e_itp": None,
                            "psnr_norm_db": None,
                            "sssim_norm": None,
                            "delta_e_itp_norm": None,
                            "hdr_vdp3": None,
                            "obj_note": note,
                            "hdr_vdp3_note": "",
                        },
                    }
                    rows.append(row)
                    self.sample_ready.emit(row)
                    continue

                sdr_eval = np.ascontiguousarray(_letterbox_bgr(sdr_raw, out_w, out_h))
                gt_eval = np.ascontiguousarray(
                    _letterbox_bgr(
                        _crop_gt_frame_to_pair_active_area(
                            gt_hdr_raw,
                            task.sdr_path,
                            task.gt_path,
                        ),
                        out_w,
                        out_h,
                    )
                )
                fast_gt_align_score = None
                if str(gt_hdr_mode).startswith("true_hdr_video_fast"):
                    sdr_u8 = self._to_u8_for_alignment(sdr_eval)
                    gt_u8 = self._to_u8_for_alignment(gt_eval)
                    if sdr_u8 is not None and gt_u8 is not None:
                        try:
                            fast_gt_align_score = _frame_structure_similarity(sdr_u8, gt_u8)
                        except Exception:
                            fast_gt_align_score = None
                try:
                    with torch.inference_mode():
                        pred_tensor, pred_cond = processor.preprocess(sdr_eval)
                        pred_raw = processor.infer((pred_tensor, pred_cond))
                    pred_eval = np.ascontiguousarray(tensor_to_bgr_u16(pred_raw))
                    if (pred_eval.shape[1], pred_eval.shape[0]) != (out_w, out_h):
                        pred_eval = np.ascontiguousarray(
                            cv2.resize(pred_eval, (out_w, out_h), interpolation=cv2.INTER_AREA)
                        )
                    pred_preview = np.ascontiguousarray(pred_eval)
                except Exception as exc:
                    note = f"Inference failed: {exc}"
                    row = {
                        "task_id": task.task_id,
                        "label": task.label,
                        "frame": (int(task.frame_idx) + 1) if task.frame_idx is not None else None,
                        "sample_dir": "",
                        "sdr_image": "",
                        "hdr_gt_image": "",
                        "hdr_convert_image": "",
                        "metrics": {
                            "psnr_db": None,
                            "sssim": None,
                            "delta_e_itp": None,
                            "psnr_norm_db": None,
                            "sssim_norm": None,
                            "delta_e_itp_norm": None,
                            "hdr_vdp3": None,
                            "obj_note": note,
                            "hdr_vdp3_note": "",
                        },
                    }
                    rows.append(row)
                    self.sample_ready.emit(row)
                    continue

                metrics = self._compute_metrics_for_pair(pred_eval, gt_eval)

                frame_tag = (
                    f"frame_{int(task.frame_idx) + 1:06d}"
                    if task.frame_idx is not None
                    else "frame_000000"
                )
                sample_dir_name = frame_tag
                if str(cfg.mode) == "dataset":
                    stem = _sanitize_name(
                        os.path.splitext(os.path.basename(task.sdr_path))[0]
                    )
                    sample_dir_name = f"{frame_tag}_{stem}"
                sample_dir = self._choose_unique_dir(
                    os.path.join(cfg.session_dir, sample_dir_name)
                )
                os.makedirs(sample_dir, exist_ok=True)

                sdr_path = os.path.join(sample_dir, "sdr.png")
                gt_path = os.path.join(sample_dir, "hdr_gt.tiff")
                pred_path = os.path.join(sample_dir, "hdr_convert.tiff")
                cv2.imwrite(sdr_path, sdr_eval)
                write_hdr_tiff(gt_path, gt_eval)
                write_hdr_tiff(pred_path, pred_preview)

                row = {
                    "task_id": task.task_id,
                    "label": task.label,
                    "frame": (int(task.frame_idx) + 1) if task.frame_idx is not None else None,
                    "sample_dir": sample_dir,
                    "sdr_image": sdr_path,
                    "hdr_gt_image": gt_path,
                    "hdr_convert_image": pred_path,
                    "source_sdr_path": task.sdr_path,
                    "source_gt_path": task.gt_path,
                    "source_frame_idx": int(task.frame_idx) if task.frame_idx is not None else None,
                    "gt_frame_idx": int(gt_frame_idx) if gt_frame_idx is not None else None,
                    "gt_alignment_base_frame_idx": int(gt_frame_idx) if gt_frame_idx is not None else None,
                    "gt_alignment_offset_frames": 0,
                    "gt_alignment_score": fast_gt_align_score,
                    "gt_alignment_search_radius_frames": 0,
                    "gt_decode_mode": str(gt_hdr_mode or ""),
                    "fast_gt_align_score": fast_gt_align_score,
                    "benchmark_runtime": cfg.runtime_mode,
                    "metrics": metrics,
                }
                rows.append(row)
                self._write_frame_result_file(cfg, row)
                self.sample_ready.emit(row)
                if allow_post_verify and str(gt_hdr_mode).startswith("true_hdr_video_fast"):
                    if (
                        isinstance(fast_gt_align_score, (int, float))
                        and np.isfinite(float(fast_gt_align_score))
                    ):
                        candidate_score = float(fast_gt_align_score)
                    else:
                        candidate_score = 1.0
                    post_verify_candidates.append((len(rows) - 1, task, candidate_score))

            if allow_post_verify and post_verify_candidates:
                suspects = sorted(post_verify_candidates, key=lambda x: x[2])
                max_verify = _BENCHMARK_AUTO_POST_VERIFY_MAX_ITEMS
                if (
                    max_verify is not None
                    and int(max_verify) > 0
                    and len(suspects) > int(max_verify)
                ):
                    suspects = suspects[: int(max_verify)]
                total_verify = max(1, len(suspects))
                for j, (row_idx, task, fast_score) in enumerate(suspects):
                    if self._is_canceled():
                        self.canceled.emit("Benchmark canceled during post verification.")
                        return

                    verify_pct = 90 + int((9.0 * j) / float(total_verify))
                    self.progress.emit(
                        min(99, max(90, verify_pct)),
                        f"Post-verify {j + 1}/{total_verify}: {task.label}",
                    )

                    row = rows[row_idx]
                    strict_base_gt_frame_idx = _map_gt_frame_for_sdr(
                        task.sdr_path,
                        task.gt_path,
                        task.frame_idx,
                    )
                    sdr_eval_saved = self._to_u8_for_alignment(
                        read_image_any(str(row.get("sdr_image") or ""))
                    )
                    
                    align_info, canceled_now = self._run_blocking_with_cancel(
                        _local_align_gt_frame_for_benchmark,
                        cancel_check=self._is_canceled,
                        sdr_path=task.sdr_path,
                        gt_path=task.gt_path,
                        mapped_gt_frame_idx=max(0, int(strict_base_gt_frame_idx or 0)),
                        sdr_eval_u8=sdr_eval_saved,
                        out_w=out_w,
                        out_h=out_h,
                        canceled_fallback={"canceled": True},
                    )

                    if canceled_now or self._is_canceled() or bool(align_info.get("canceled", False)):
                        self.canceled.emit("Benchmark canceled during post verification.")
                        return
                    
                    strict_gt_frame_idx = int(
                        align_info.get("frame_idx", strict_base_gt_frame_idx or 0) or 0
                    )

                    strict_rgb16, canceled_now = self._run_blocking_with_cancel(
                        read_hdr_video_frame_rgb16,
                        task.gt_path,
                        max(0, int(strict_gt_frame_idx or 0)),
                        prefer_fast_seek=False,
                        canceled_fallback=None,
                    )

                    if canceled_now or self._is_canceled():
                        self.canceled.emit("Benchmark canceled during post verification.")
                        return
                    
                    if strict_rgb16 is None and strict_gt_frame_idx != int(
                        strict_base_gt_frame_idx or 0
                    ):
                        strict_gt_frame_idx = max(0, int(strict_base_gt_frame_idx or 0))

                    strict_rgb16, canceled_now = self._run_blocking_with_cancel(
                        read_hdr_video_frame_rgb16,
                        task.gt_path,
                        strict_gt_frame_idx,
                        prefer_fast_seek=False,
                        canceled_fallback=None,
                    )

                    if canceled_now or self._is_canceled():
                        self.canceled.emit("Benchmark canceled during post verification.")
                        return

                    if strict_rgb16 is None:
                        continue

                    strict_gt_eval = np.ascontiguousarray(
                        _letterbox_bgr(
                            _crop_gt_frame_to_pair_active_area(
                                np.ascontiguousarray(strict_rgb16[:, :, ::-1]),
                                task.sdr_path,
                                task.gt_path,
                            ),
                            out_w,
                            out_h,
                        )
                    )

                    strict_u8 = self._to_u8_for_alignment(strict_gt_eval)
                    strict_score = None
                    if sdr_eval_saved is not None and strict_u8 is not None:
                        try:
                            strict_score = _frame_structure_similarity(
                                sdr_eval_saved,
                                strict_u8,
                            )
                        except Exception:
                            strict_score = None

                    replace_reasons: list[str] = []
                    try:
                        align_offset = int(strict_gt_frame_idx) - int(
                            strict_base_gt_frame_idx or 0
                        )
                    except Exception:
                        align_offset = 0
                    if align_offset:
                        replace_reasons.append(
                            f"local GT alignment {align_offset:+d} frame(s)"
                        )
                    if strict_score is not None:
                        score_gate = max(
                            float(_BENCHMARK_AUTO_POST_VERIFY_SUSPECT_SCORE),
                            float(fast_score)
                            + float(_BENCHMARK_AUTO_POST_VERIFY_IMPROVE_MARGIN),
                        )
                        if float(strict_score) >= score_gate:
                            replace_reasons.append("better alignment")

                    gt_path_saved = str(row.get("hdr_gt_image") or "").strip()
                    fast_gt_saved = read_image_any(gt_path_saved) if gt_path_saved else None
                    if fast_gt_saved is not None:
                        fast_gt_saved = np.ascontiguousarray(fast_gt_saved)
                        if (fast_gt_saved.shape[1], fast_gt_saved.shape[0]) != (out_w, out_h):
                            fast_gt_saved = np.ascontiguousarray(
                                cv2.resize(
                                    fast_gt_saved,
                                    (out_w, out_h),
                                    interpolation=cv2.INTER_AREA,
                                )
                            )
                        fast_u8 = self._to_u8_for_alignment(fast_gt_saved)
                        if fast_u8 is not None:
                            try:
                                gt_diff_score = _frame_structure_similarity(
                                    fast_u8,
                                    strict_u8,
                                )
                            except Exception:
                                gt_diff_score = None
                            if (
                                gt_diff_score is not None
                                and float(gt_diff_score)
                                < float(_BENCHMARK_AUTO_POST_VERIFY_GT_DIFF_SCORE)
                            ):
                                replace_reasons.append(
                                    f"GT frame changed (similarity {float(gt_diff_score):.4f})"
                                )

                        try:
                            fast_f = fast_gt_saved.astype(np.float32)
                            strict_f = strict_gt_eval.astype(np.float32)
                            if fast_gt_saved.dtype == np.uint8:
                                fast_f *= 257.0
                            elif fast_gt_saved.dtype != np.uint16:
                                peak = float(np.max(fast_f)) if fast_f.size else 1.0
                                if peak <= 1.0:
                                    fast_f *= 65535.0
                            gt_mean_abs = float(
                                np.mean(np.abs(fast_f - strict_f)) / 65535.0
                            )
                            if gt_mean_abs >= float(
                                _BENCHMARK_AUTO_POST_VERIFY_GT_DIFF_MEAN
                            ):
                                replace_reasons.append(
                                    f"GT pixel delta {gt_mean_abs:.4f}"
                                )
                        except Exception:
                            pass

                    pred_eval_saved = read_image_any(str(row.get("hdr_convert_image") or ""))

                    if self._is_canceled():
                        self.canceled.emit("Benchmark canceled during post verification.")
                        return

                    if pred_eval_saved is None:
                        continue

                    pred_eval_saved = np.ascontiguousarray(pred_eval_saved)
                    if pred_eval_saved.dtype != np.uint16:
                        if pred_eval_saved.dtype == np.uint8:
                            pred_eval_saved = np.ascontiguousarray(
                                pred_eval_saved.astype(np.uint16) * 257
                            )
                        else:
                            arrf = pred_eval_saved.astype(np.float32)
                            peak = float(np.max(arrf)) if arrf.size else 1.0
                            if peak <= 1.0:
                                arrf = arrf * 65535.0
                            pred_eval_saved = np.ascontiguousarray(
                                np.clip(arrf, 0.0, 65535.0).astype(np.uint16)
                            )
                    if (pred_eval_saved.shape[1], pred_eval_saved.shape[0]) != (out_w, out_h):
                        pred_eval_saved = np.ascontiguousarray(
                            cv2.resize(pred_eval_saved, (out_w, out_h), interpolation=cv2.INTER_AREA)
                        )

                    strict_metrics, canceled_now = self._run_blocking_with_cancel(
                        self._compute_metrics_for_pair,
                        pred_eval_saved,
                        strict_gt_eval,
                        canceled_fallback=None,
                    )
                    if canceled_now or self._is_canceled():
                        self.canceled.emit("Benchmark canceled during post verification.")
                        return
                    verify_reasons = replace_reasons or ["exact verified"]
                    if strict_metrics.get("obj_note"):
                        strict_metrics["obj_note"] = (
                            f"{strict_metrics['obj_note']} [post-verify: {', '.join(verify_reasons)}]"
                        )
                    row["metrics"] = strict_metrics
                    row["gt_decode_mode"] = (
                        "true_hdr_video_exact_aligned"
                        if align_offset
                        else "true_hdr_video_exact_verified"
                    )
                    if strict_score is not None:
                        row["fast_gt_align_score"] = float(strict_score)
                    row["gt_frame_idx"] = int(strict_gt_frame_idx)
                    row["gt_alignment_base_frame_idx"] = int(
                        strict_base_gt_frame_idx or 0
                    )
                    row["gt_alignment_best_frame_idx"] = int(
                        align_info.get("best_frame_idx", strict_gt_frame_idx) or strict_gt_frame_idx
                    )
                    row["gt_alignment_offset_frames"] = int(align_offset)
                    row["gt_alignment_score"] = (
                        None if strict_score is None else float(strict_score)
                    )
                    row["gt_alignment_base_score"] = align_info.get("base_score")
                    row["gt_alignment_best_score"] = align_info.get("best_score")
                    row["gt_alignment_search_radius_frames"] = int(
                        align_info.get("search_radius_frames", _BENCHMARK_GT_LOCAL_SEARCH_FRAMES)
                        or 0
                    )

                    if gt_path_saved:
                        write_hdr_tiff(gt_path_saved, strict_gt_eval)
                    self._write_frame_result_file(cfg, row)
                    self.sample_ready.emit(dict(row))

            summary_csv, summary_json, averages = self._write_session_summaries(cfg, rows)

            self.progress.emit(100, "Benchmark completed.")
            self.finished.emit(
                {
                    "mode": cfg.mode,
                    "source_name": cfg.source_name,
                    "precision": cfg.precision_key,
                    "use_hg": bool(cfg.use_hg),
                    "resolution": cfg.resolution_key,
                    "predequantize_mode": cfg.predequantize_mode,
                    "benchmark_runtime": cfg.runtime_mode,
                    "session_dir": cfg.session_dir,
                    "summary_csv": summary_csv,
                    "summary_json": summary_json,
                    "average": averages,
                    "count": len(rows),
                    "rows": rows,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            try:
                if processor is not None and hasattr(processor, "model"):
                    del processor.model
            except Exception:
                pass
            try:
                del processor
            except Exception:
                pass
            try:
                torch._dynamo.reset()
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
            except Exception:
                pass


class _ImagePreviewLabel(QLabel):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = str(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(260, 160)
        self.setProperty("videoSurface", True)
        self.setWordWrap(True)
        self.setText(self._title)

    def set_image_from_path(self, path: str | None, fallback: str = ""):
        if not path or not os.path.isfile(path):
            self.clear()
            self.setText(fallback or self._title)
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self.clear()
            self.setText(fallback or self._title)
            return
        self.set_image_from_bgr(img, fallback=fallback)

    def set_image_from_bgr(self, bgr: np.ndarray | None, fallback: str = ""):
        if not isinstance(bgr, np.ndarray):
            self.clear()
            self.setText(fallback or self._title)
            return
        img = np.ascontiguousarray(bgr)
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()
        pix = QPixmap.fromImage(qimg)
        self.setPixmap(
            pix.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep existing pixmap responsive to size changes.
        pix = self.pixmap()
        if pix is not None and not pix.isNull():
            self.setPixmap(
                pix.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )


class ModelBenchmarkDialog(QDialog):
    def __init__(
        self,
        *,
        initial_video_path: str | None,
        initial_hdr_gt_path: str | None,
        suggested_dir: str,
        initial_precision_key: str | None,
        initial_use_hg: bool,
        initial_predequantize_mode: str,
        logs_root: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Dialog, True)
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, True)
        self.setSizeGripEnabled(True)
        self.setWindowTitle("Model Quality Benchmark")
        self.setModal(True)
        self.resize(1240, 740)

        self._suggested_dir = suggested_dir if os.path.isdir(suggested_dir) else os.getcwd()
        self._last_source_dir = self._suggested_dir
        self._logs_root = logs_root
        self._dataset_pairs: list[BenchmarkTask] = []
        self._results: list[dict] = []
        self._session_dir: str | None = None
        self._worker_thread: QThread | None = None
        self._worker: _BenchmarkWorker | None = None
        self._result_source_name = ""
        self._result_precision = ""
        self._result_resolution = ""
        self._result_use_hg: bool | None = None
        self._result_predequantize_mode = ""
        self._current_outlier_indices: set[int] = set()
        self._result_sets: list[dict] = []
        self._benchmark_preview_splitter: QSplitter | None = None
        self._benchmark_results_splitter: QSplitter | None = None
        self._expanded_preview_pane: int | None = None
        self._reset_preview_splitter_on_show = True
        self._ffmpeg_hdr_warning_shown = False
        self._benchmark_queue: list[_QueuedBenchmarkRun] = []
        self._queue_running = False
        self._active_queue_total = 0
        self._active_queue_index = 0
        self._preview_processor_cache: dict[tuple[str, bool, str, str], HDRTVNetTorch] = {}
        self._frame_detect_cache_path = os.path.join(self._logs_root, "benchmark_frame_cache.json")
        self._frame_detect_cache = self._load_frame_detect_cache()

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        intro = QLabel(
            "Benchmark model quality with TensorRT on NVIDIA, cached max-autotune on AMD when available, or eager fallback. "
            "You can benchmark a single SDR/HDR-GT video pair or a paired SDR/HDR-GT dataset, "
            "review images (SDR, HDR GT, HDR Convert), and export metrics with images."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        tabs = QTabWidget()
        self._tabs = tabs
        root.addWidget(tabs, 1)

        config_tab = QScrollArea()
        config_tab.setWidgetResizable(True)
        config_tab.setFrameShape(QScrollArea.Shape.NoFrame)
        config_tab.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        config_tab.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        config_body = QWidget()
        config_tab.setWidget(config_body)
        cfg_layout = QVBoxLayout(config_body)
        cfg_layout.setContentsMargins(0, 6, 0, 0)
        cfg_layout.setSpacing(10)
        tabs.addTab(config_tab, "Setup")

        mode_group = QGroupBox("Mode")
        mode_form = QFormLayout(mode_group)
        self._cmb_mode = QComboBox()
        self._cmb_mode.addItem("Video (SDR + HDR GT)", "video")
        self._cmb_mode.addItem("Dataset Folders (SDR + HDR GT)", "dataset")
        mode_form.addRow("Benchmark mode:", self._cmb_mode)
        cfg_layout.addWidget(mode_group)

        self._stack_mode = QStackedWidget()

        video_page = QWidget()
        video_layout = QVBoxLayout(video_page)
        video_layout.setContentsMargins(4, 8, 4, 8)
        video_layout.setSpacing(8)

        video_form = QFormLayout()
        self._txt_video_sdr = QLineEdit()
        self._btn_video_sdr = QPushButton("Browse ...")
        row_sdr = QWidget()
        row_sdr_l = QHBoxLayout(row_sdr)
        row_sdr_l.setContentsMargins(0, 0, 0, 0)
        row_sdr_l.setSpacing(6)
        row_sdr_l.addWidget(self._txt_video_sdr, 1)
        row_sdr_l.addWidget(self._btn_video_sdr)

        self._txt_video_gt = QLineEdit()
        self._btn_video_gt = QPushButton("Browse ...")
        row_gt = QWidget()
        row_gt_l = QHBoxLayout(row_gt)
        row_gt_l.setContentsMargins(0, 0, 0, 0)
        row_gt_l.setSpacing(6)
        row_gt_l.addWidget(self._txt_video_gt, 1)
        row_gt_l.addWidget(self._btn_video_gt)

        video_form.addRow("SDR video:", row_sdr)
        video_form.addRow("HDR GT video:", row_gt)
        video_layout.addLayout(video_form)

        frames_bar = QHBoxLayout()
        self._lbl_detect_frame_count = QLabel("Pool:")
        self._spn_detect_frame_count = QSpinBox()
        self._spn_detect_frame_count.setRange(1, 240)
        self._spn_detect_frame_count.setValue(50)
        self._spn_detect_frame_count.setToolTip(
            "Number of distinct candidate frames to detect. For the same video and pool size, detection is deterministic."
        )
        self._btn_detect_frames = QPushButton("Detect Distinct Frames")
        self._btn_select_all_frames = QPushButton("Check All Frames")
        self._btn_clear_frames = QPushButton("Clear")
        self._btn_select_all_frames.setToolTip(
            "Mark all detected frame checkboxes so every detected frame is benchmarked."
        )
        self._btn_clear_frames.setToolTip(
            "Uncheck all detected frame checkboxes."
        )
        frames_bar.addWidget(self._lbl_detect_frame_count)
        frames_bar.addWidget(self._spn_detect_frame_count)
        frames_bar.addWidget(self._btn_detect_frames)
        frames_bar.addWidget(self._btn_select_all_frames)
        frames_bar.addWidget(self._btn_clear_frames)
        frames_bar.addStretch(1)
        video_layout.addLayout(frames_bar)

        self._lst_video_frames = QListWidget()
        self._lst_video_frames.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._lbl_video_note = QLabel(
            "Detect a candidate pool, then use Average mode to benchmark checked frames or a deterministic subset."
        )
        self._lbl_video_note.setWordWrap(True)
        video_layout.addWidget(self._lst_video_frames, 1)

        video_avg_form = QFormLayout()
        self._cmb_video_avg_mode = QComboBox()
        self._cmb_video_avg_mode.addItem("Average selected frames", "selected")
        self._cmb_video_avg_mode.addItem("Average all detected frames", "all")
        self._cmb_video_avg_mode.addItem("Average deterministic subset", "subset")
        self._cmb_video_avg_mode.setCurrentIndex(2)
        self._spn_video_subset = QSpinBox()
        self._spn_video_subset.setRange(1, 240)
        self._spn_video_subset.setValue(10)
        self._cmb_video_avg_mode.setToolTip(
            "Matches the dataset average-mode workflow. Deterministic subset samples evenly from the checked candidate frames."
        )
        self._spn_video_subset.setToolTip(
            "Number of checked candidate frames to evaluate when Average deterministic subset is selected."
        )
        video_avg_form.addRow("Average mode:", self._cmb_video_avg_mode)
        video_avg_form.addRow("Subset size:", self._spn_video_subset)
        video_layout.addLayout(video_avg_form)

        video_layout.addWidget(self._lbl_video_note)
        self._img_video_setup_preview = _ImagePreviewLabel("Selected SDR Frame Preview")
        self._img_video_setup_preview.setMinimumHeight(120)
        self._img_video_setup_preview.setMaximumHeight(180)
        video_layout.addWidget(self._img_video_setup_preview)

        dataset_page = QWidget()
        dataset_layout = QVBoxLayout(dataset_page)
        dataset_layout.setContentsMargins(4, 8, 4, 8)
        dataset_layout.setSpacing(8)

        dataset_form = QFormLayout()
        self._txt_dataset_sdr = QLineEdit()
        self._btn_dataset_sdr = QPushButton("Browse ...")
        ds_sdr_row = QWidget()
        ds_sdr_layout = QHBoxLayout(ds_sdr_row)
        ds_sdr_layout.setContentsMargins(0, 0, 0, 0)
        ds_sdr_layout.setSpacing(6)
        ds_sdr_layout.addWidget(self._txt_dataset_sdr, 1)
        ds_sdr_layout.addWidget(self._btn_dataset_sdr)

        self._txt_dataset_gt = QLineEdit()
        self._btn_dataset_gt = QPushButton("Browse ...")
        ds_gt_row = QWidget()
        ds_gt_layout = QHBoxLayout(ds_gt_row)
        ds_gt_layout.setContentsMargins(0, 0, 0, 0)
        ds_gt_layout.setSpacing(6)
        ds_gt_layout.addWidget(self._txt_dataset_gt, 1)
        ds_gt_layout.addWidget(self._btn_dataset_gt)

        dataset_form.addRow("SDR folder:", ds_sdr_row)
        dataset_form.addRow("HDR GT folder:", ds_gt_row)
        dataset_layout.addLayout(dataset_form)

        ds_toolbar = QHBoxLayout()
        self._btn_scan_dataset = QPushButton("Scan and Pair Files")
        self._btn_select_all_pairs = QPushButton("Select All")
        self._btn_clear_pairs = QPushButton("Clear")
        self._lbl_pair_count = QLabel("Pairs: 0")
        ds_toolbar.addWidget(self._btn_scan_dataset)
        ds_toolbar.addWidget(self._btn_select_all_pairs)
        ds_toolbar.addWidget(self._btn_clear_pairs)
        ds_toolbar.addWidget(self._lbl_pair_count)
        ds_toolbar.addStretch(1)
        dataset_layout.addLayout(ds_toolbar)

        self._lst_dataset_pairs = QListWidget()
        self._lst_dataset_pairs.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        dataset_layout.addWidget(self._lst_dataset_pairs, 1)

        avg_form = QFormLayout()
        self._cmb_avg_mode = QComboBox()
        self._cmb_avg_mode.addItem("Average selected items", "selected")
        self._cmb_avg_mode.addItem("Average all paired dataset items", "all")
        self._cmb_avg_mode.addItem("Average deterministic subset", "subset")
        self._spn_subset = QSpinBox()
        self._spn_subset.setRange(1, 5000)
        self._spn_subset.setValue(24)
        avg_form.addRow("Average mode:", self._cmb_avg_mode)
        avg_form.addRow("Subset size:", self._spn_subset)
        dataset_layout.addLayout(avg_form)

        self._stack_mode.addWidget(video_page)
        self._stack_mode.addWidget(dataset_page)
        cfg_layout.addWidget(self._stack_mode, 1)

        opt_group = QGroupBox("Benchmark Options")
        opt_form = QFormLayout(opt_group)
        self._cmb_precision = QComboBox()
        self._cmb_precision.addItems(_available_precision_keys())
        if initial_precision_key and initial_precision_key in _available_precision_keys():
            self._cmb_precision.setCurrentText(initial_precision_key)
        self._chk_use_hg = QCheckBox("Use HG highlight refinement")
        self._chk_use_hg.setChecked(bool(initial_use_hg))

        self._cmb_resolution = QComboBox()
        self._cmb_resolution.addItems(list(RESOLUTION_SCALES.keys()))
        if DEFAULT_RESOLUTION_KEY in RESOLUTION_SCALES:
            self._cmb_resolution.setCurrentText(DEFAULT_RESOLUTION_KEY)
        else:
            self._cmb_resolution.setCurrentText("1080p")

        self._cmb_predequantize = QComboBox()
        self._cmb_predequantize.addItem("Auto", "auto")
        self._cmb_predequantize.addItem("Force On", "on")
        self._cmb_predequantize.addItem("Force Off", "off")
        p_mode = str(initial_predequantize_mode or "auto").strip().lower()
        idx = self._cmb_predequantize.findData(p_mode)
        self._cmb_predequantize.setCurrentIndex(idx if idx >= 0 else 0)

        self._txt_session_root = QLineEdit()
        self._txt_session_root.setText(self._logs_root)
        self._btn_session_root = QPushButton("Browse ...")
        root_row = QWidget()
        root_row_l = QHBoxLayout(root_row)
        root_row_l.setContentsMargins(0, 0, 0, 0)
        root_row_l.setSpacing(6)
        root_row_l.addWidget(self._txt_session_root, 1)
        root_row_l.addWidget(self._btn_session_root)

        opt_form.addRow("Precision:", self._cmb_precision)
        opt_form.addRow("", self._chk_use_hg)
        opt_form.addRow("Benchmark resolution:", self._cmb_resolution)
        self._lbl_predequantize = QLabel("INT8 pre-dequantize:")
        opt_form.addRow(self._lbl_predequantize, self._cmb_predequantize)
        if _IS_NVIDIA:
            self._lbl_predequantize.hide()
            self._cmb_predequantize.hide()
        opt_form.addRow("Session logs root:", root_row)
        cfg_layout.addWidget(opt_group)

        queue_group = QGroupBox("Benchmark Queue")
        queue_layout = QGridLayout(queue_group)
        self._lst_benchmark_queue = QListWidget()
        self._lst_benchmark_queue.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._lst_benchmark_queue.setMaximumHeight(96)
        self._lbl_queue_status = QLabel("Queue: empty")
        self._lbl_queue_preview = QLabel("Select a queued run to preview its captured settings.")
        self._lbl_queue_preview.setWordWrap(True)
        self._btn_queue_add = QPushButton("Add Current to Queue")
        self._btn_queue_remove = QPushButton("Remove Selected")
        self._btn_queue_run = QPushButton("Run Queue")
        self._btn_queue_clear = QPushButton("Clear Queue")
        self._btn_queue_run.setProperty("role", "primary")
        self._lst_benchmark_queue.setToolTip(
            "Queued runs use the mode, source/tasks, precision, resolution, HG, and pre-dequantize settings captured when added."
        )
        queue_layout.addWidget(self._lbl_queue_status, 0, 0, 1, 3)
        queue_layout.addWidget(self._lst_benchmark_queue, 1, 0, 1, 3)
        queue_layout.addWidget(self._lbl_queue_preview, 2, 0, 1, 3)
        queue_buttons = QHBoxLayout()
        queue_buttons.setContentsMargins(0, 0, 0, 0)
        queue_buttons.setSpacing(6)
        queue_buttons.addWidget(self._btn_queue_add)
        queue_buttons.addWidget(self._btn_queue_remove)
        queue_buttons.addWidget(self._btn_queue_run)
        queue_buttons.addWidget(self._btn_queue_clear)
        queue_layout.addLayout(queue_buttons, 3, 0, 1, 3)
        cfg_layout.addWidget(queue_group)

        result_tab = QWidget()
        result_layout = QVBoxLayout(result_tab)
        result_layout.setContentsMargins(0, 6, 0, 0)
        result_layout.setSpacing(8)
        tabs.addTab(result_tab, "Results")

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._lbl_progress = QLabel("Ready.")
        result_layout.addWidget(self._progress)
        result_layout.addWidget(self._lbl_progress)

        self._result_sets_bar = QTabBar()
        self._result_sets_bar.setDocumentMode(True)
        self._result_sets_bar.setMovable(True)
        self._result_sets_bar.setTabsClosable(True)
        self._result_sets_bar.setExpanding(False)
        self._result_sets_bar.setDrawBase(False)
        self._result_sets_bar.setToolTip("Opened benchmark result sets")
        result_layout.addWidget(self._result_sets_bar)

        info_group = QGroupBox("Run Info")
        info_form = QFormLayout(info_group)
        self._lbl_result_source = QLabel("-")
        self._lbl_result_config = QLabel("-")
        info_form.addRow("Source:", self._lbl_result_source)
        info_form.addRow("Precision / Resolution:", self._lbl_result_config)
        result_layout.addWidget(info_group)

        preview_opts = QHBoxLayout()
        self._cmb_average_filter = QComboBox()
        self._cmb_average_filter.addItem("Average all rows", "all")
        self._cmb_average_filter.addItem("Exclude outliers in this result", "per_result")
        self._cmb_average_filter.addItem("Exclude shared outliers across open tabs", "shared")
        self._cmb_average_filter.setToolTip(
            "Controls only the displayed preview average. Saved benchmark results and exports are unchanged."
        )
        self._lbl_outlier_status = QLabel("")
        self._lbl_outlier_status.setWordWrap(True)
        preview_opts.addWidget(QLabel("Average filter:"))
        preview_opts.addWidget(self._cmb_average_filter)
        preview_opts.addWidget(self._lbl_outlier_status, 1)
        result_layout.addLayout(preview_opts)

        self._tbl = QTableWidget(0, 9)
        self._tbl.setHorizontalHeaderLabels(
            [
                "Item",
                "PSNR",
                "SSIM",
                "DeltaEITP",
                "PSNR-N",
                "SSIM-N",
                "DeltaEITP-N",
                "HDR-VDP3",
                "Note",
            ]
        )
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tbl.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.horizontalHeader().setStretchLastSection(True)

        previews = QSplitter(Qt.Orientation.Horizontal)
        self._benchmark_preview_splitter = previews
        previews.setChildrenCollapsible(False)
        previews.setHandleWidth(12)
        self._img_sdr = _CompareVideoPane(
            "SDR",
            force_hdr_metadata=False,
            mpv_available=_HAS_MPV,
            mpv_widget_factory=_new_mpv_widget if _HAS_MPV else None,
            best_mpv_scale=BEST_MPV_SCALE,
            preview_scale_kernel="bicubic",
            preview_cas_strength=0.0,
            preview_film_grain=False,

        )
        self._img_gt = _CompareVideoPane(
            "HDR GT",
            force_hdr_metadata=True,
            mpv_available=_HAS_MPV,
            mpv_widget_factory=_new_mpv_widget if _HAS_MPV else None,
            best_mpv_scale=BEST_MPV_SCALE,
            preview_scale_kernel="bicubic",
            preview_cas_strength=0.0,
            preview_film_grain=False,
        )
        self._img_pred = _CompareVideoPane(
            "HDR Convert",
            force_hdr_metadata=True,
            mpv_available=_HAS_MPV,
            mpv_widget_factory=_new_mpv_widget if _HAS_MPV else None,
            best_mpv_scale=BEST_MPV_SCALE,
            preview_scale_kernel="bicubic",
            preview_cas_strength=0.0,
            preview_film_grain=False,
        )
        previews.addWidget(self._img_sdr)
        previews.addWidget(self._img_gt)
        previews.addWidget(self._img_pred)
        previews.setStretchFactor(0, 1)
        previews.setStretchFactor(1, 1)
        previews.setStretchFactor(2, 1)
        self._img_sdr.expand_requested.connect(lambda: self._toggle_benchmark_preview_expand(0))
        self._img_gt.expand_requested.connect(lambda: self._toggle_benchmark_preview_expand(1))
        self._img_pred.expand_requested.connect(lambda: self._toggle_benchmark_preview_expand(2))

        results_splitter = QSplitter(Qt.Orientation.Vertical)
        self._benchmark_results_splitter = results_splitter
        results_splitter.setChildrenCollapsible(False)
        results_splitter.setHandleWidth(10)
        results_splitter.addWidget(self._tbl)
        results_splitter.addWidget(previews)
        results_splitter.setStretchFactor(0, 3)
        results_splitter.setStretchFactor(1, 2)
        result_layout.addWidget(results_splitter, 1)

        export_group = QGroupBox("Export")
        export_layout = QGridLayout(export_group)
        self._lbl_session_dir = QLabel("Session: -")
        self._btn_open_session = QPushButton("Open Session Folder")
        self._btn_load_existing = QPushButton("Load Existing Result(s) ...")
        self._btn_export_selected = QPushButton("Export Selected Result")
        self._btn_export_all = QPushButton("Export All Results")
        self._btn_open_session.setEnabled(False)
        self._btn_export_selected.setEnabled(False)
        self._btn_export_all.setEnabled(False)
        export_layout.addWidget(self._lbl_session_dir, 0, 0, 1, 3)
        export_layout.addWidget(self._btn_open_session, 1, 0)
        export_layout.addWidget(self._btn_load_existing, 1, 1)
        export_layout.addWidget(self._btn_export_selected, 1, 2)
        export_layout.addWidget(self._btn_export_all, 2, 1, 1, 2)
        result_layout.addWidget(export_group)

        status_row = QHBoxLayout()
        self._status_label = QLabel("Status: Ready")
        self._status_progress = QProgressBar()
        self._status_progress.setRange(0, 100)
        self._status_progress.setValue(0)
        self._status_progress.setFixedWidth(220)
        status_row.addWidget(self._status_label, 1)
        status_row.addWidget(self._status_progress)
        root.addLayout(status_row)

        bottom = QHBoxLayout()
        self._btn_run = QPushButton("Run Benchmark")
        self._btn_run.setProperty("role", "primary")
        self._btn_cancel = QPushButton("Cancel Run")
        self._btn_cancel.setEnabled(False)
        self._btn_close = QPushButton("Close")
        bottom.addWidget(self._btn_run)
        bottom.addWidget(self._btn_cancel)
        bottom.addStretch(1)
        bottom.addWidget(self._btn_close)
        root.addLayout(bottom)

        self._cmb_mode.currentIndexChanged.connect(self._sync_mode_ui)
        self._btn_video_sdr.clicked.connect(self._pick_video_sdr)
        self._btn_video_gt.clicked.connect(self._pick_video_gt)
        self._btn_detect_frames.clicked.connect(self._detect_frames)
        self._btn_select_all_frames.clicked.connect(lambda: self._set_all_checked(self._lst_video_frames, True))
        self._btn_clear_frames.clicked.connect(lambda: self._set_all_checked(self._lst_video_frames, False))
        self._lst_video_frames.currentItemChanged.connect(lambda _cur, _prev: self._refresh_video_frame_preview())
        self._cmb_video_avg_mode.currentIndexChanged.connect(self._sync_subset_enabled)

        self._btn_dataset_sdr.clicked.connect(self._pick_dataset_sdr)
        self._btn_dataset_gt.clicked.connect(self._pick_dataset_gt)
        self._btn_scan_dataset.clicked.connect(self._scan_dataset_pairs)
        self._btn_select_all_pairs.clicked.connect(lambda: self._set_all_checked(self._lst_dataset_pairs, True))
        self._btn_clear_pairs.clicked.connect(lambda: self._set_all_checked(self._lst_dataset_pairs, False))
        self._cmb_avg_mode.currentIndexChanged.connect(self._sync_subset_enabled)

        self._btn_session_root.clicked.connect(self._pick_session_root)
        self._btn_run.clicked.connect(self._start_benchmark)
        self._btn_queue_add.clicked.connect(self._add_current_benchmark_to_queue)
        self._btn_queue_remove.clicked.connect(self._remove_selected_benchmark_from_queue)
        self._btn_queue_run.clicked.connect(self._start_benchmark_queue)
        self._btn_queue_clear.clicked.connect(self._clear_benchmark_queue)
        self._lst_benchmark_queue.currentRowChanged.connect(self._update_queue_preview)
        self._btn_cancel.clicked.connect(self._cancel_benchmark)
        self._btn_close.clicked.connect(self._close_dialog)

        self._tbl.itemSelectionChanged.connect(self._on_result_selection_changed)
        self._cmb_average_filter.currentIndexChanged.connect(self._refresh_results_preview_options)
        self._result_sets_bar.currentChanged.connect(self._on_result_set_tab_changed)
        self._result_sets_bar.tabCloseRequested.connect(self._on_result_set_tab_close_requested)
        self._btn_open_session.clicked.connect(self._open_session_folder)
        self._btn_load_existing.clicked.connect(self._load_existing_results_dialog)
        self._btn_export_selected.clicked.connect(lambda: self._export_results(selected_only=True))
        self._btn_export_all.clicked.connect(lambda: self._export_results(selected_only=False))

        if initial_video_path and os.path.isfile(initial_video_path):
            self._txt_video_sdr.setText(initial_video_path)
        if initial_hdr_gt_path and os.path.isfile(initial_hdr_gt_path):
            self._txt_video_gt.setText(initial_hdr_gt_path)

        self._sync_mode_ui()
        self._sync_subset_enabled()
        self._sync_queue_ui()
        self._add_result_set_tab(
            {
                "title": "Current Run",
                "rows": [],
                "average": {},
                "session_dir": None,
                "source_name": "",
                "precision": "",
                "resolution": "",
                "use_hg": None,
                "predequantize_mode": None,
                "selected_row": None,
            },
            activate=True,
        )

    def _load_frame_detect_cache(self) -> dict[str, list[int]]:
        path = str(getattr(self, "_frame_detect_cache_path", "") or "").strip()
        if not path or not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return {}
            out: dict[str, list[int]] = {}
            for key, value in payload.items():
                if not isinstance(key, str) or not isinstance(value, list):
                    continue
                frames: list[int] = []
                for item in value:
                    try:
                        frames.append(int(item))
                    except Exception:
                        continue
                if frames:
                    out[key] = frames
            return out
        except Exception:
            return {}

    def _save_frame_detect_cache(self) -> None:
        path = str(getattr(self, "_frame_detect_cache_path", "") or "").strip()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._frame_detect_cache, f, indent=2)
        except Exception:
            pass

    def _frame_detect_cache_key(self, video_path: str, desired_count: int) -> str:
        try:
            st = os.stat(video_path)
            sig = f"{os.path.abspath(video_path)}|{int(st.st_mtime_ns)}|{int(st.st_size)}"
        except Exception:
            sig = os.path.abspath(video_path) if video_path else str(video_path or "")
        return f"{sig}|count={int(max(1, desired_count))}"

    def last_session_dir(self) -> str | None:
        return self._session_dir

    def last_source_dir(self) -> str | None:
        path = str(self._last_source_dir or "").strip()
        if path and os.path.isdir(path):
            return path
        return None

    def _set_all_checked(self, lst: QListWidget, checked: bool):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(lst.count()):
            item = lst.item(i)
            item.setCheckState(state)

    def _result_tab_payload(
        self,
        *,
        title: str,
        rows: list[dict],
        average: dict | None,
        session_dir: str | None,
        source_name: str | None,
        precision: str | None,
        resolution: str | None,
        use_hg: bool | None = None,
        predequantize_mode: str | None = None,
        selected_row: int | None = None,
    ) -> dict:
        return {
            "title": str(title or "Results").strip() or "Results",
            "rows": list(rows or []),
            "average": dict(average or {}),
            "session_dir": str(session_dir or "").strip() or None,
            "source_name": str(source_name or "").strip(),
            "precision": str(precision or "").strip(),
            "resolution": str(resolution or "").strip(),
            "use_hg": bool(use_hg) if use_hg is not None else None,
            "predequantize_mode": str(predequantize_mode or "").strip() or None,
            "selected_row": (
                int(selected_row)
                if selected_row is not None and int(selected_row) >= 0
                else None
            ),
        }

    def _infer_precision_from_session_dir(self, session_dir: str | None) -> str:
        s = str(session_dir or "").strip()
        if not s:
            return ""
        parts = [p for p in os.path.normpath(s).split(os.sep) if p]
        if len(parts) >= 2 and re.match(r"^\d{8}_\d{6}$", parts[-2]):
            return parts[-1]
        leaf = parts[-1] if parts else ""
        m = re.match(r"^\d{8}_\d{6}__(.+?)(?:__.+)?$", leaf)
        if m:
            return str(m.group(1) or "").strip()
        return ""

    def _result_tab_title(
        self,
        source_name: str | None,
        precision: str | None,
        session_dir: str | None,
    ) -> str:
        src = str(source_name or "").strip() or self._infer_source_name_from_session_dir(session_dir)
        prec = str(precision or "").strip() or self._infer_precision_from_session_dir(session_dir)
        if not src:
            src = "Results"
        return f"{src} [{prec}]" if prec else src

    def _unique_result_tab_title(self, base: str, exclude_idx: int | None = None) -> str:
        title = str(base or "Results").strip() or "Results"
        taken = {
            self._result_sets_bar.tabText(i)
            for i in range(self._result_sets_bar.count())
            if exclude_idx is None or i != int(exclude_idx)
        }
        if title not in taken:
            return title
        n = 2
        while True:
            candidate = f"{title} ({n})"
            if candidate not in taken:
                return candidate
            n += 1

    def _add_result_set_tab(self, payload: dict, activate: bool) -> int:
        title = self._unique_result_tab_title(str(payload.get("title") or "Results"))
        data = dict(payload)
        data["title"] = title
        idx = self._result_sets_bar.addTab(title)
        self._result_sets.append(data)
        if bool(activate):
            self._select_and_apply_result_set_tab(idx)
        return idx

    def _select_and_apply_result_set_tab(self, idx: int):
        i = int(idx)
        if i < 0 or i >= len(self._result_sets):
            self._apply_result_set_tab(i)
            return
        if int(self._result_sets_bar.currentIndex()) != i:
            self._result_sets_bar.setCurrentIndex(i)
        self._apply_result_set_tab(i)

    def _update_result_set_tab(self, idx: int, payload: dict):
        if idx < 0 or idx >= len(self._result_sets):
            self._add_result_set_tab(payload, activate=True)
            return
        title = self._unique_result_tab_title(
            str(payload.get("title") or self._result_sets[idx].get("title") or "Results"),
            exclude_idx=idx,
        )
        data = dict(payload)
        data["title"] = title
        self._result_sets[idx] = data
        self._result_sets_bar.setTabText(idx, title)
        if int(self._result_sets_bar.currentIndex()) == int(idx):
            self._apply_result_set_tab(idx)

    def _apply_result_set_tab(self, idx: int):
        if idx < 0 or idx >= len(self._result_sets):
            self._populate_results_view(
                rows=[],
                average={},
                session_dir=None,
                source_name="",
                precision="",
                resolution="",
                use_hg=None,
                predequantize_mode=None,
                selected_row=None,
            )
            return
        data = self._result_sets[idx]
        self._populate_results_view(
            rows=list(data.get("rows") or []),
            average=data.get("average") or {},
            session_dir=data.get("session_dir"),
            source_name=str(data.get("source_name") or "").strip(),
            precision=str(data.get("precision") or "").strip(),
            resolution=str(data.get("resolution") or "").strip(),
            use_hg=data.get("use_hg"),
            predequantize_mode=data.get("predequantize_mode"),
            selected_row=data.get("selected_row"),
        )

    def _on_result_set_tab_changed(self, idx: int):
        self._apply_result_set_tab(int(idx))

    def _on_result_set_tab_close_requested(self, idx: int):
        if self._worker_thread is not None:
            QMessageBox.information(self, "Benchmark", "Cancel the running benchmark first.")
            return
        if idx < 0 or idx >= len(self._result_sets):
            return
        self._result_sets_bar.blockSignals(True)
        try:
            self._result_sets_bar.removeTab(idx)
            del self._result_sets[idx]
        finally:
            self._result_sets_bar.blockSignals(False)
        if not self._result_sets:
            self._add_result_set_tab(
                {
                    "title": "Current Run",
                    "rows": [],
                    "average": {},
                    "session_dir": None,
                    "source_name": "",
                    "precision": "",
                    "resolution": "",
                    "use_hg": None,
                    "predequantize_mode": None,
                    "selected_row": None,
                },
                activate=True,
            )
            return
        next_idx = min(max(0, idx), len(self._result_sets) - 1)
        self._select_and_apply_result_set_tab(next_idx)

    def _set_status(self, text: str, percent: int | None = None):
        msg = str(text or "").strip() or "Ready"
        self._status_label.setText(f"Status: {msg}")
        if percent is not None:
            self._status_progress.setValue(max(0, min(100, int(percent))))

    def _warn_if_ffmpeg_missing_for_hdr_benchmark(self) -> bool:
        if self._ffmpeg_hdr_warning_shown:
            return False
        if hdr_ffmpeg_ready():
            return True
        self._ffmpeg_hdr_warning_shown = True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("FFmpeg Not Detected")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(
            "FFmpeg and ffprobe were not detected.<br><br>"
            "Benchmark cannot run until FFmpeg and ffprobe are installed and available on PATH.<br><br>"
            f'Download FFmpeg for Windows:<br><a href="{FFMPEG_WINDOWS_DOWNLOAD_URL}">{FFMPEG_WINDOWS_DOWNLOAD_URL}</a>'
        )
        open_btn = box.addButton(
            "Open FFmpeg Download Page",
            QMessageBox.ButtonRole.ActionRole,
        )
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        while True:
            box.exec()
            if box.clickedButton() is open_btn:
                try:
                    webbrowser.open(FFMPEG_WINDOWS_DOWNLOAD_URL, new=2)
                except Exception:
                    pass
                continue
            break
        return False

    def _set_result_run_info(
        self,
        source_name: str | None,
        precision: str | None,
        resolution: str | None,
    ):
        src = str(source_name or "").strip()
        prec = str(precision or "").strip()
        res = str(resolution or "").strip()

        self._result_source_name = src
        self._result_precision = prec
        self._result_resolution = res

        self._lbl_result_source.setText(src or "-")
        if prec and res:
            cfg = f"{prec} @ {res}"
        elif prec:
            cfg = prec
        elif res:
            cfg = res
        else:
            cfg = "-"
        self._lbl_result_config.setText(cfg)

    def _metric_value(self, row: dict, key: str) -> float | None:
        try:
            value = float((row.get("metrics") or {}).get(key))
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return value

    def _result_sample_key(self, row: dict) -> tuple[str, str, str, str]:
        sdr = os.path.normcase(os.path.normpath(str(row.get("source_sdr_path") or "").strip()))
        gt = os.path.normcase(os.path.normpath(str(row.get("source_gt_path") or "").strip()))
        frame_raw = row.get("source_frame_idx")
        if frame_raw is None:
            frame_raw = row.get("frame")
        try:
            frame = str(int(float(frame_raw))) if frame_raw is not None and str(frame_raw).strip() else ""
        except Exception:
            frame = str(frame_raw or "").strip()
        label = str(row.get("label") or row.get("task_id") or "").strip().lower()
        if sdr or gt or frame:
            return (sdr, gt, frame, "")
        return ("", "", "", label)

    def _detect_outlier_indices(self, rows: list[dict]) -> set[int]:
        if len(rows) < 4:
            return set()

        votes: dict[int, int] = {}
        usable_metric_count = 0
        for key in _BENCHMARK_OUTLIER_KEYS:
            indexed_values: list[tuple[int, float]] = []
            for idx, row in enumerate(rows):
                value = self._metric_value(row, key)
                if value is not None:
                    indexed_values.append((idx, value))
            if len(indexed_values) < 4:
                continue

            values = np.asarray([v for _idx, v in indexed_values], dtype=np.float64)
            q1, q3 = np.percentile(values, [25, 75])
            iqr = float(q3 - q1)
            if iqr > 1e-12:
                lower = float(q1 - (1.5 * iqr))
                upper = float(q3 + (1.5 * iqr))
            else:
                median = float(np.median(values))
                mad = float(np.median(np.abs(values - median)))
                if mad <= 1e-12:
                    continue
                span = 3.5 * 1.4826 * mad
                lower = median - span
                upper = median + span

            usable_metric_count += 1
            for idx, value in indexed_values:
                if value < lower or value > upper:
                    votes[idx] = votes.get(idx, 0) + 1

        if usable_metric_count <= 0:
            return set()
        required_votes = 1 if usable_metric_count <= 2 else 2
        return {idx for idx, count in votes.items() if count >= required_votes}

    def _row_sample_keys(self, rows: list[dict]) -> set[tuple[str, str, str, str]]:
        keys: set[tuple[str, str, str, str]] = set()
        for row in rows:
            key = self._result_sample_key(row)
            if any(key):
                keys.add(key)
        return keys

    def _compatible_shared_outlier_keys(
        self,
        active_rows: list[dict],
        active_source_name: str | None,
    ) -> tuple[set[tuple[str, str, str, str]], int]:
        active_keys = self._row_sample_keys(active_rows)
        if not active_keys:
            return set(), 0
        active_source = str(active_source_name or "").strip().lower()

        shared_keys: set[tuple[str, str, str, str]] = set()
        compatible_count = 0
        for data in self._result_sets:
            rows = list(data.get("rows") or [])
            tab_keys = self._row_sample_keys(rows)
            if not tab_keys:
                continue
            tab_source = str(data.get("source_name") or "").strip().lower()
            if active_source and tab_source and active_source != tab_source:
                continue

            overlap = active_keys.intersection(tab_keys)
            smaller_count = max(1, min(len(active_keys), len(tab_keys)))
            overlap_ratio = len(overlap) / float(smaller_count)
            if overlap_ratio < _BENCHMARK_SHARED_MIN_OVERLAP_RATIO:
                continue

            compatible_count += 1
            for idx in self._detect_outlier_indices(rows):
                if 0 <= idx < len(rows):
                    key = self._result_sample_key(rows[idx])
                    if key in active_keys:
                        shared_keys.add(key)

        return shared_keys, compatible_count

    def _preview_average_and_outliers(
        self,
        rows: list[dict],
        saved_average: dict | None,
        source_name: str | None,
    ) -> tuple[dict[str, float | None], set[int], str]:
        filter_mode = str(self._cmb_average_filter.currentData() or "all")
        if filter_mode == "all":
            return dict(saved_average or self._compute_average_from_rows(rows)), set(), "all rows"

        if filter_mode == "shared":
            shared_keys, compatible_count = self._compatible_shared_outlier_keys(rows, source_name)
            outlier_indices = {
                idx for idx, row in enumerate(rows)
                if self._result_sample_key(row) in shared_keys
            }
            if compatible_count > 1:
                mode_label = f"shared compatible tabs ({compatible_count})"
            else:
                mode_label = "this result; no other compatible tab"
        else:
            outlier_indices = self._detect_outlier_indices(rows)
            mode_label = "this result"

        if not outlier_indices:
            return self._compute_average_from_rows(rows), set(), mode_label

        filtered_rows = [
            row for idx, row in enumerate(rows)
            if idx not in outlier_indices
        ]
        return self._compute_average_from_rows(filtered_rows), outlier_indices, mode_label

    def _refresh_results_preview_options(self):
        self._remember_current_result_selection()
        idx = int(self._result_sets_bar.currentIndex())
        if 0 <= idx < len(self._result_sets):
            self._apply_result_set_tab(idx)
            return
        self._populate_results_view(
            rows=self._results,
            average=self._compute_average_from_rows(self._results),
            session_dir=self._session_dir,
            source_name=self._result_source_name,
            precision=self._result_precision,
            resolution=self._result_resolution,
            use_hg=self._result_use_hg,
            predequantize_mode=self._result_predequantize_mode,
            selected_row=self._selected_result_row(),
        )

    def _selected_result_row(self) -> int | None:
        rows = self._tbl.selectionModel().selectedRows() if self._tbl.selectionModel() else []
        if not rows:
            return None
        idx = self._result_row_to_index(rows[0].row())
        if idx is None:
            return None
        return int(idx)

    def _remember_current_result_selection(self):
        idx = int(self._result_sets_bar.currentIndex())
        if idx < 0 or idx >= len(self._result_sets):
            return
        self._result_sets[idx]["selected_row"] = self._selected_result_row()

    def _reset_benchmark_preview_splitter_sizes(self):
        preview_splitter = self._benchmark_preview_splitter
        if preview_splitter is None or preview_splitter.count() < 3:
            return

        total_w = int(preview_splitter.size().width())
        if total_w <= 0:
            total_w = max(3, int(self.size().width()) - 32)
        one = max(1, total_w // 3)
        preview_splitter.setSizes([one, one, max(1, total_w - (2 * one))])

        results_splitter = self._benchmark_results_splitter
        if results_splitter is not None and results_splitter.count() >= 2:
            total_h = int(results_splitter.size().height())
            if total_h <= 0:
                total_h = max(2, int(self.size().height()) - 240)
            preview_h = max(1, int(total_h * 0.42))
            table_h = max(1, total_h - preview_h)
            results_splitter.setSizes([table_h, preview_h])

        self._expanded_preview_pane = None

    def _toggle_benchmark_preview_expand(self, pane_idx: int):
        preview_splitter = self._benchmark_preview_splitter
        if preview_splitter is None or preview_splitter.count() < 3:
            return

        total_w = int(preview_splitter.size().width())
        if total_w <= 0:
            total_w = max(3, int(self.size().width()) - 32)
        one = max(1, total_w // 3)
        preview_splitter.setSizes([one, one, max(1, total_w - (2 * one))])

        results_splitter = self._benchmark_results_splitter
        if results_splitter is None or results_splitter.count() < 2:
            return

        target = int(pane_idx)
        if self._expanded_preview_pane == target:
            self._reset_benchmark_preview_splitter_sizes()
            return

        total_h = int(results_splitter.size().height())
        if total_h <= 0:
            total_h = max(2, int(self.size().height()) - 240)
        preview_h = max(1, int(total_h * 0.72))
        table_h = max(1, total_h - preview_h)
        results_splitter.setSizes([table_h, preview_h])
        self._expanded_preview_pane = target

    def _infer_source_name_from_session_dir(self, session_dir: str | None) -> str:
        s = str(session_dir or "").strip()
        if not s:
            return ""
        parts = [p for p in os.path.normpath(s).split(os.sep) if p]
        if len(parts) >= 3 and re.match(r"^\d{8}_\d{6}$", parts[-2]):
            return parts[-3]
        leaf = parts[-1] if parts else ""
        if re.match(r"^\d{8}_\d{6}__(.+?)(?:__.+)?$", leaf) and len(parts) >= 2:
            return parts[-2]
        return os.path.basename(os.path.normpath(s))

    def _read_preview_image(self, path: str | None) -> np.ndarray | None:
        p = str(path or "").strip()
        if not p or not os.path.isfile(p):
            return None
        return read_image_any(p)

    def _preview_processor_for_current_result(self) -> HDRTVNetTorch | None:
        precision = str(self._result_precision or "").strip()
        use_hg = self._result_use_hg
        predequantize_mode = str(self._result_predequantize_mode or "auto").strip() or "auto"
        resolution_key = str(self._result_resolution or "1080p").strip() or "1080p"
        if not precision or use_hg is None:
            return None
        key = (precision, bool(use_hg), predequantize_mode, resolution_key)
        cached = self._preview_processor_cache.get(key)
        if cached is not None:
            return cached
        model_path = _select_model_path(precision, bool(use_hg))
        if not model_path or not os.path.isfile(model_path):
            return None
        precision_cfg = PRECISIONS.get(precision, {})
        model_precision = str(precision_cfg.get("precision") or "fp16")
        try:
            if _IS_NVIDIA:
                out_w, out_h = _resolution_dims(resolution_key)
                processor = HDRTVNetTensorRT(
                    model_path,
                    device="auto",
                    precision=model_precision,
                    engine_width=out_w,
                    engine_height=out_h,
                    mode_name=f"{precision}_{'hg' if use_hg else 'nohg'}",
                    use_hg=bool(use_hg),
                )
            else:
                out_w, out_h = _resolution_dims(resolution_key)
                compile_ready, _compile_arg, _compile_pdq = _benchmark_compile_cache_ready(
                    w=out_w,
                    h=out_h,
                    precision_key=precision,
                    model_path=model_path,
                    use_hg=bool(use_hg),
                    predequantize_mode=predequantize_mode,
                )
                processor = HDRTVNetTorch(
                    model_path,
                    device="auto",
                    precision=model_precision,
                    compile_model=bool(compile_ready),
                    force_compile=bool(compile_ready),
                    compile_mode="max-autotune" if compile_ready else "default",
                    predequantize=_resolve_predequantize_arg(predequantize_mode),
                    use_hg=bool(use_hg),
                )
        except Exception:
            return None
        self._preview_processor_cache[key] = processor
        return processor

    def _generate_result_previews(self, row: dict) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if not isinstance(row, dict):
            return None, None, None
        sdr_path = str(row.get("source_sdr_path") or "").strip()
        gt_path = str(row.get("source_gt_path") or "").strip()
        frame_raw = row.get("source_frame_idx")
        try:
            frame_idx = int(frame_raw) if frame_raw is not None else None
        except Exception:
            frame_idx = None

        sdr_src = _read_media_frame(sdr_path, frame_idx=frame_idx)
        gt_hdr_src, _gt_mode = _read_media_frame_hdr_with_mode(
            gt_path,
            frame_idx=_map_gt_frame_for_sdr(sdr_path, gt_path, frame_idx),
            sdr_reference=sdr_src,
        )
        if sdr_src is None:
            return None, None, None
        out_w, out_h = _resolution_dims(self._result_resolution or "1080p")
        sdr_preview = np.ascontiguousarray(_letterbox_bgr(sdr_src, out_w, out_h))
        gt_preview = None
        if isinstance(gt_hdr_src, np.ndarray):
            gt_preview = np.ascontiguousarray(
                _letterbox_bgr(
                    _crop_gt_frame_to_pair_active_area(
                        gt_hdr_src,
                        sdr_path,
                        gt_path,
                    ),
                    out_w,
                    out_h,
                )
            )

        processor = self._preview_processor_for_current_result()
        pred_preview = None
        if processor is not None:
            try:
                with torch.inference_mode():
                    pred_tensor, pred_cond = processor.preprocess(sdr_preview)
                    pred_raw = processor.infer((pred_tensor, pred_cond))
                pred_preview = np.ascontiguousarray(tensor_to_bgr_u16(pred_raw))
                if (pred_preview.shape[1], pred_preview.shape[0]) != (out_w, out_h):
                    pred_preview = np.ascontiguousarray(
                        cv2.resize(pred_preview, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    )
            except Exception:
                pred_preview = None
        return sdr_preview, gt_preview, pred_preview

    def _set_result_previews(self, row: dict | None):
        if not isinstance(row, dict):
            self._img_sdr.set_frame(None, "SDR\n(unavailable)")
            self._img_gt.set_frame(None, "HDR GT\n(unavailable)")
            self._img_pred.set_frame(None, "HDR Convert\n(unavailable)")
            return
        sdr = self._read_preview_image(row.get("sdr_image"))
        gt = self._read_preview_image(row.get("hdr_gt_image"))
        pred = self._read_preview_image(row.get("hdr_convert_image"))
        if sdr is None and gt is None and pred is None:
            sdr, gt, pred = self._generate_result_previews(row)
        self._img_sdr.set_frame(sdr, "SDR\n(unavailable)")
        self._img_gt.set_frame(gt, "HDR GT\n(unavailable)")
        self._img_pred.set_frame(pred, "HDR Convert\n(unavailable)")

    def _stop_result_preview_players(self):
        for pane in (self._img_sdr, self._img_gt, self._img_pred):
            try:
                pane.stop()
            except Exception:
                continue

    def _populate_results_view(
        self,
        *,
        rows: list[dict],
        average: dict | None,
        session_dir: str | None,
        source_name: str | None,
        precision: str | None,
        resolution: str | None,
        use_hg: bool | None,
        predequantize_mode: str | None,
        selected_row: int | None,
    ):
        self._results = list(rows or [])
        self._result_use_hg = bool(use_hg) if use_hg is not None else None
        self._result_predequantize_mode = str(predequantize_mode or "").strip()
        self._tbl.blockSignals(True)
        self._tbl.clearSelection()
        self._tbl.setRowCount(0)
        self._tbl.blockSignals(False)

        avg, outlier_indices, filter_label = self._preview_average_and_outliers(
            self._results,
            average,
            source_name,
        )
        self._current_outlier_indices = set(outlier_indices)

        for idx, row in enumerate(self._results):
            self._add_table_row(row, outlier=idx in outlier_indices)

        if self._results:
            r = self._tbl.rowCount()
            self._tbl.insertRow(r)
            included_count = max(0, len(self._results) - len(outlier_indices))
            if outlier_indices:
                avg_text = (
                    f"AVERAGE (N={included_count}, "
                    f"excluded={len(outlier_indices)})"
                )
            else:
                avg_text = f"AVERAGE (N={len(self._results)})"
            avg_label = QTableWidgetItem(avg_text)
            bold = QFont()
            bold.setBold(True)
            avg_label.setFont(bold)
            self._tbl.setItem(r, 0, avg_label)
            avg_values = [
                _fmt_metric(avg.get("psnr_db"), ".2f", " dB"),
                _fmt_metric(avg.get("sssim"), ".4f"),
                _fmt_metric(avg.get("delta_e_itp"), ".2f"),
                _fmt_metric(avg.get("psnr_norm_db"), ".2f", " dB"),
                _fmt_metric(avg.get("sssim_norm"), ".4f"),
                _fmt_metric(avg.get("delta_e_itp_norm"), ".2f"),
                _fmt_metric(avg.get("hdr_vdp3"), ".3f"),
                "",
            ]
            for col, text in enumerate(avg_values, start=1):
                it = QTableWidgetItem(text)
                it.setFont(bold)
                self._tbl.setItem(r, col, it)

        filter_mode = str(self._cmb_average_filter.currentData() or "all")
        if self._results and filter_mode != "all":
            if outlier_indices:
                self._lbl_outlier_status.setText(
                    f"Preview average excludes {len(outlier_indices)} outlier row(s) from {filter_label}."
                )
            else:
                self._lbl_outlier_status.setText(
                    f"No outliers detected for preview average from {filter_label}."
                )
        else:
            self._lbl_outlier_status.setText("")

        self._session_dir = session_dir if session_dir and os.path.isdir(session_dir) else None
        if self._session_dir:
            self._lbl_session_dir.setText(f"Session: {self._session_dir}")
        else:
            self._lbl_session_dir.setText("Session: -")

        self._set_result_run_info(source_name, precision, resolution)

        self._btn_open_session.setEnabled(bool(self._session_dir))
        self._btn_export_selected.setEnabled(bool(self._results))
        self._btn_export_all.setEnabled(bool(self._results))

        if self._results and self._tbl.rowCount() > 0:
            row_to_select = 0
            if selected_row is not None:
                try:
                    row_to_select = max(0, min(int(selected_row), len(self._results) - 1))
                except Exception:
                    row_to_select = 0
            self._tbl.blockSignals(True)
            self._tbl.selectRow(row_to_select)
            self._tbl.blockSignals(False)
            self._on_result_selection_changed()
        else:
            self._set_result_previews(None)

    def _compute_average_from_rows(self, rows: list[dict]) -> dict[str, float | None]:
        avg: dict[str, float | None] = {}
        for key in _BENCHMARK_METRIC_KEYS:
            vals = []
            for row in rows:
                v = (row.get("metrics") or {}).get(key)
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    vals.append(fv)
            avg[key] = float(np.mean(vals)) if vals else None
        return avg

    def _load_results_from_json(
        self,
        json_path: str,
    ) -> tuple[list[dict], dict, str | None, dict[str, object]]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("results")
        if not isinstance(rows, list):
            raise RuntimeError("JSON file does not contain a valid 'results' list.")

        base_dir = os.path.dirname(json_path)
        norm_rows: list[dict] = []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            row = dict(raw)
            m = row.get("metrics")
            if not isinstance(m, dict):
                m = {}
            row["metrics"] = dict(m)

            sample_dir = str(row.get("sample_dir") or "").strip()
            if sample_dir and not os.path.isabs(sample_dir):
                sample_dir = os.path.normpath(os.path.join(base_dir, sample_dir))
            row["sample_dir"] = sample_dir

            for src_key in ("source_sdr_path", "source_gt_path"):
                src_p = str(row.get(src_key) or "").strip()
                if src_p and not os.path.isabs(src_p):
                    src_p = os.path.normpath(os.path.join(base_dir, src_p))
                row[src_key] = src_p

            for key, fname in (
                ("sdr_image", "sdr.png"),
                ("hdr_gt_image", "hdr_gt.tiff"),
                ("hdr_convert_image", "hdr_convert.tiff"),
            ):
                p = str(row.get(key) or "").strip()
                if p and not os.path.isabs(p):
                    p = os.path.normpath(os.path.join(base_dir, p))
                if (not p) and sample_dir:
                    p = os.path.join(sample_dir, fname)
                    if not os.path.isfile(p):
                        legacy_candidates = [
                            os.path.splitext(p)[0] + ".tiff",
                            os.path.splitext(p)[0] + ".png",
                        ]
                        for legacy in legacy_candidates:
                            if os.path.isfile(legacy):
                                p = legacy
                                break
                row[key] = p
            norm_rows.append(row)

        average = data.get("average") if isinstance(data.get("average"), dict) else self._compute_average_from_rows(norm_rows)
        session_dir = str(data.get("session_dir") or "").strip() or base_dir
        if session_dir and not os.path.isabs(session_dir):
            session_dir = os.path.normpath(os.path.join(base_dir, session_dir))
        meta = {
            "source_name": str(data.get("source_name") or "").strip(),
            "precision": str(data.get("precision") or "").strip(),
            "resolution": str(data.get("resolution") or "").strip(),
            "use_hg": data.get("use_hg"),
            "predequantize_mode": str(data.get("predequantize_mode") or "").strip(),
        }
        if not meta["source_name"]:
            meta["source_name"] = self._infer_source_name_from_session_dir(session_dir)
        return norm_rows, average, session_dir, meta

    def _load_results_from_csv(
        self,
        csv_path: str,
    ) -> tuple[list[dict], dict, str | None, dict[str, object]]:
        rows: list[dict] = []
        base_dir = os.path.dirname(csv_path)
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise RuntimeError("CSV file is empty or invalid.")
            for rec in reader:
                label = str(rec.get("item") or "").strip()
                if not label or label.upper().startswith("AVERAGE"):
                    continue
                sample_dir = str(rec.get("sample_dir") or rec.get("exported_folder") or "").strip()
                if sample_dir and not os.path.isabs(sample_dir):
                    sample_dir = os.path.normpath(os.path.join(base_dir, sample_dir))
                metrics = {
                    "psnr_db": rec.get("psnr_db"),
                    "sssim": rec.get("sssim"),
                    "delta_e_itp": rec.get("delta_e_itp"),
                    "psnr_norm_db": rec.get("psnr_norm_db"),
                    "sssim_norm": rec.get("sssim_norm"),
                    "delta_e_itp_norm": rec.get("delta_e_itp_norm"),
                    "hdr_vdp3": rec.get("hdr_vdp3"),
                    "obj_note": rec.get("obj_note") or "",
                    "hdr_vdp3_note": rec.get("hdr_vdp3_note") or "",
                }
                frame_raw = str(rec.get("frame") or "").strip()
                frame_val = None
                if frame_raw:
                    try:
                        frame_val = int(float(frame_raw))
                    except Exception:
                        frame_val = None
                row = {
                    "task_id": _sanitize_name(label),
                    "label": label,
                    "frame": frame_val,
                    "sample_dir": sample_dir,
                    "sdr_image": os.path.join(sample_dir, "sdr.png") if sample_dir else "",
                    "hdr_gt_image": os.path.join(sample_dir, "hdr_gt.tiff") if sample_dir else "",
                    "hdr_convert_image": os.path.join(sample_dir, "hdr_convert.tiff") if sample_dir else "",
                "metrics": metrics,
                }
                if sample_dir:
                    if not os.path.isfile(row["hdr_gt_image"]):
                        exr_path = os.path.join(sample_dir, "hdr_gt.exr")
                        png_path = os.path.join(sample_dir, "hdr_gt.png")
                        row["hdr_gt_image"] = exr_path if os.path.isfile(exr_path) else png_path
                    if not os.path.isfile(row["hdr_convert_image"]):
                        exr_path = os.path.join(sample_dir, "hdr_convert.exr")
                        png_path = os.path.join(sample_dir, "hdr_convert.png")
                        row["hdr_convert_image"] = exr_path if os.path.isfile(exr_path) else png_path
                rows.append(row)
        average = self._compute_average_from_rows(rows)

        meta = {
            "source_name": "",
            "precision": "",
            "resolution": "",
            "use_hg": None,
            "predequantize_mode": "",
        }
        for sidecar_name in ("benchmark_summary.json", "benchmark_export_summary.json"):
            sidecar = os.path.join(base_dir, sidecar_name)
            if not os.path.isfile(sidecar):
                continue
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    sidecar_data = json.load(f)
                if isinstance(sidecar_data, dict):
                    meta["source_name"] = str(sidecar_data.get("source_name") or meta["source_name"]).strip()
                    meta["precision"] = str(sidecar_data.get("precision") or meta["precision"]).strip()
                    meta["resolution"] = str(sidecar_data.get("resolution") or meta["resolution"]).strip()
                    if sidecar_data.get("use_hg") is not None:
                        meta["use_hg"] = bool(sidecar_data.get("use_hg"))
                    meta["predequantize_mode"] = str(
                        sidecar_data.get("predequantize_mode") or meta["predequantize_mode"]
                    ).strip()
                break
            except Exception:
                continue

        if not meta["precision"]:
            parts = [p for p in os.path.normpath(base_dir).split(os.sep) if p]
            if len(parts) >= 2 and re.match(r"^\d{8}_\d{6}$", parts[-2]):
                meta["precision"] = parts[-1]
        if not meta["source_name"]:
            meta["source_name"] = self._infer_source_name_from_session_dir(base_dir)

        return rows, average, base_dir, meta

    def _load_results_from_frame_folders(
        self,
        folder_path: str,
    ) -> tuple[list[dict], dict, str | None, dict[str, object]]:
        base_dir = os.path.normpath(str(folder_path or "").strip())
        if not base_dir or not os.path.isdir(base_dir):
            raise RuntimeError("Result folder does not exist.")

        frame_json_paths: list[str] = []
        target_name = str(_FRAME_RESULT_FILE).lower()
        for root, _dirs, files in os.walk(base_dir):
            for name in files:
                if str(name).lower() == target_name:
                    frame_json_paths.append(os.path.join(root, name))
        frame_json_paths.sort()
        if not frame_json_paths:
            raise RuntimeError("No per-frame result metadata was found.")

        rows: list[dict] = []
        meta: dict[str, object] = {
            "source_name": "",
            "precision": "",
            "resolution": "",
            "use_hg": None,
            "predequantize_mode": "",
        }

        for fp in frame_json_paths:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue

            raw_row = payload.get("row") if isinstance(payload.get("row"), dict) else payload
            if not isinstance(raw_row, dict):
                continue

            row = dict(raw_row)
            m = row.get("metrics")
            row["metrics"] = dict(m) if isinstance(m, dict) else {}

            sample_dir = str(row.get("sample_dir") or "").strip()
            if not sample_dir or sample_dir in {".", "./", ".\\"}:
                sample_dir = os.path.dirname(fp)
            elif not os.path.isabs(sample_dir):
                sample_dir = os.path.normpath(os.path.join(os.path.dirname(fp), sample_dir))
            row["sample_dir"] = sample_dir

            for src_key in ("source_sdr_path", "source_gt_path"):
                src_p = str(row.get(src_key) or "").strip()
                if src_p and not os.path.isabs(src_p):
                    src_p = os.path.normpath(os.path.join(base_dir, src_p))
                row[src_key] = src_p

            for key, fname in (
                ("sdr_image", "sdr.png"),
                ("hdr_gt_image", "hdr_gt.tiff"),
                ("hdr_convert_image", "hdr_convert.tiff"),
            ):
                p = str(row.get(key) or "").strip()
                if p and not os.path.isabs(p):
                    p = os.path.normpath(os.path.join(sample_dir, p))
                if (not p) and sample_dir:
                    p = os.path.join(sample_dir, fname)
                    if not os.path.isfile(p):
                        legacy_candidates = [
                            os.path.splitext(p)[0] + ".tiff",
                            os.path.splitext(p)[0] + ".png",
                            os.path.splitext(p)[0] + ".exr",
                        ]
                        for legacy in legacy_candidates:
                            if os.path.isfile(legacy):
                                p = legacy
                                break
                row[key] = p

            if not str(row.get("label") or "").strip():
                row["label"] = os.path.basename(sample_dir.rstrip("\\/")) or "sample"

            frame_raw = str(row.get("frame") or "").strip()
            if frame_raw:
                try:
                    row["frame"] = int(float(frame_raw))
                except Exception:
                    row["frame"] = None
            else:
                row["frame"] = None

            rows.append(row)

            if not str(meta.get("source_name") or "").strip():
                meta["source_name"] = str(payload.get("source_name") or "").strip()
            if not str(meta.get("precision") or "").strip():
                meta["precision"] = str(payload.get("precision") or "").strip()
            if not str(meta.get("resolution") or "").strip():
                meta["resolution"] = str(payload.get("resolution") or "").strip()
            if meta.get("use_hg") is None and payload.get("use_hg") is not None:
                meta["use_hg"] = bool(payload.get("use_hg"))
            if not str(meta.get("predequantize_mode") or "").strip():
                meta["predequantize_mode"] = str(payload.get("predequantize_mode") or "").strip()

        if not rows:
            raise RuntimeError("No valid per-frame result metadata could be loaded.")

        rows.sort(
            key=lambda r: (
                str(r.get("source_gt_path") or "").lower(),
                str(r.get("source_sdr_path") or "").lower(),
                int(r.get("frame")) if r.get("frame") is not None else 10**12,
                str(r.get("label") or "").lower(),
            )
        )

        average = self._compute_average_from_rows(rows)
        if not str(meta.get("source_name") or "").strip():
            meta["source_name"] = self._infer_source_name_from_session_dir(base_dir)
        if not str(meta.get("precision") or "").strip():
            meta["precision"] = self._infer_precision_from_session_dir(base_dir)

        return rows, average, base_dir, meta

    def _pick_result_folders_multi(self, start_dir: str) -> list[str]:
        dlg = QFileDialog(self, "Select Benchmark Result Folder(s)", start_dir)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setFilter(QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.Drives)
        dlg.setNameFilter("Folders")
        for view in dlg.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for view in dlg.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        if not dlg.exec():
            return []
        out: list[str] = []
        seen: set[str] = set()
        for p in dlg.selectedFiles():
            norm = os.path.normpath(str(p))
            if not os.path.isdir(norm):
                continue
            key = norm.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(norm)
        return out

    def _resolve_result_summary_path(self, path: str) -> str:
        p = os.path.normpath(str(path or "").strip())
        if not p:
            raise RuntimeError("Result path is empty.")
        if os.path.isdir(p):
            candidates = [
                os.path.join(p, "benchmark_summary.json"),
                os.path.join(p, "benchmark_export_summary.json"),
                os.path.join(p, "benchmark_summary.csv"),
                os.path.join(p, "benchmark_export_summary.csv"),
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    return cand
            raise RuntimeError(f"No supported result summary file found in folder: {p}")
        if not os.path.isfile(p):
            raise RuntimeError(f"Path does not exist: {p}")
        ext = os.path.splitext(p)[1].lower()
        if ext not in {".json", ".csv"}:
            raise RuntimeError(f"Unsupported result file type: {p}")
        return p

    def _load_result_source(self, source_path: str) -> tuple[list[dict], dict, str | None, dict[str, object]]:
        raw = os.path.normpath(str(source_path or "").strip())
        if not raw:
            raise RuntimeError("Result path is empty.")

        if os.path.isdir(raw):
            try:
                path = self._resolve_result_summary_path(raw)
                ext = os.path.splitext(path)[1].lower()
                if ext == ".json":
                    return self._load_results_from_json(path)
                if ext == ".csv":
                    return self._load_results_from_csv(path)
                raise RuntimeError("Unsupported result file type. Use JSON or CSV.")
            except Exception as summary_exc:
                try:
                    return self._load_results_from_frame_folders(raw)
                except Exception:
                    raise summary_exc

        path = self._resolve_result_summary_path(raw)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            return self._load_results_from_json(path)
        if ext == ".csv":
            return self._load_results_from_csv(path)
        raise RuntimeError("Unsupported result file type. Use JSON or CSV.")

    def _load_existing_results_dialog(self):
        if self._worker_thread is not None:
            QMessageBox.information(self, "Benchmark", "Cancel the running benchmark first.")
            return

        box = QMessageBox(self)
        box.setWindowTitle("Load Benchmark Results")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText("Load one or more result files or result folders into separate tabs.")
        btn_file = box.addButton("Choose File(s)", QMessageBox.ButtonRole.AcceptRole)
        btn_folder = box.addButton("Choose Folder(s)", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()

        base_start = self._session_dir or self._txt_session_root.text().strip() or self._suggested_dir
        chosen_paths: list[str] = []
        if clicked is btn_file:
            chosen_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Benchmark Result File(s)",
                base_start,
                "Result Files (*.json *.csv);;JSON (*.json);;CSV (*.csv);;All (*)",
            )
        elif clicked is btn_folder:
            chosen_paths = self._pick_result_folders_multi(base_start)
        else:
            return

        normalized_paths: list[str] = []
        seen_paths: set[str] = set()
        for raw in chosen_paths:
            norm = os.path.normpath(str(raw or "").strip())
            if not norm:
                continue
            key = norm.lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            normalized_paths.append(norm)
        chosen_paths = normalized_paths

        if not chosen_paths:
            return

        loaded_count = 0
        loaded_rows = 0
        first_loaded_idx: int | None = None
        failures: list[str] = []

        for src in chosen_paths:
            try:
                rows, average, session_dir, meta = self._load_result_source(src)
                if not rows:
                    raise RuntimeError("Result source has no benchmark rows.")
                source_name = str(meta.get("source_name") or "").strip()
                precision = str(meta.get("precision") or "").strip()
                resolution = str(meta.get("resolution") or "").strip()
                title = self._result_tab_title(source_name, precision, session_dir)
                payload = self._result_tab_payload(
                    title=title,
                    rows=rows,
                    average=average,
                    session_dir=session_dir,
                    source_name=source_name,
                    precision=precision,
                    resolution=resolution,
                    use_hg=meta.get("use_hg"),
                    predequantize_mode=str(meta.get("predequantize_mode") or "").strip(),
                    selected_row=0,
                )
                idx = self._add_result_set_tab(payload, activate=False)
                if first_loaded_idx is None:
                    first_loaded_idx = idx
                loaded_count += 1
                loaded_rows += len(rows)
            except Exception as exc:
                failures.append(f"{src}: {exc}")

        if loaded_count <= 0:
            if failures:
                QMessageBox.warning(
                    self,
                    "Load Benchmark Results",
                    "No result source was loaded.\n\n" + "\n".join(failures[:6]),
                )
            return

        if first_loaded_idx is not None:
            self._select_and_apply_result_set_tab(first_loaded_idx)
        self._set_status(
            f"Loaded {loaded_count} result set(s) ({loaded_rows} rows)",
            percent=100,
        )
        self._lbl_progress.setText("Loaded existing results.")
        self._progress.setValue(100)
        self._tabs.setCurrentIndex(1)

        if failures:
            QMessageBox.warning(
                self,
                "Load Benchmark Results",
                "Some sources could not be loaded:\n\n" + "\n".join(failures[:6]),
            )

    def _sync_mode_ui(self):
        mode = self._cmb_mode.currentData()
        self._stack_mode.setCurrentIndex(0 if mode == "video" else 1)

    def _current_video_frame_1b(self) -> int | None:
        item = self._lst_video_frames.currentItem()
        if item is None and self._lst_video_frames.count() > 0:
            item = self._lst_video_frames.item(0)
        if item is None:
            return None
        try:
            return int(item.data(Qt.ItemDataRole.UserRole))
        except Exception:
            return None

    def _refresh_video_frame_preview(self):
        sdr_path = self._txt_video_sdr.text().strip()
        if not sdr_path or not os.path.isfile(sdr_path):
            self._img_video_setup_preview.set_image_from_bgr(
                None,
                "Selected SDR Frame Preview",
            )
            return
        frame_1b = self._current_video_frame_1b()
        if frame_1b is None:
            self._img_video_setup_preview.set_image_from_bgr(
                None,
                "Select a detected frame to preview.",
            )
            return
        frame = _read_video_frame_at(sdr_path, max(0, int(frame_1b) - 1))
        self._img_video_setup_preview.set_image_from_bgr(
            frame,
            f"Frame {int(frame_1b)} preview unavailable.",
        )

    def _sync_subset_enabled(self):
        video_mode = str(self._cmb_video_avg_mode.currentData() or "selected")
        self._spn_video_subset.setEnabled(video_mode == "subset")
        dataset_mode = str(self._cmb_avg_mode.currentData() or "selected")
        self._spn_subset.setEnabled(dataset_mode == "subset")

    def _pick_video_sdr(self):
        start = self._last_source_dir or self._suggested_dir
        current = self._txt_video_sdr.text().strip()
        if current and os.path.isfile(current):
            start = os.path.dirname(current)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SDR Video",
            start,
            "Video (*.mp4 *.avi *.mkv *.mov *.webm *.flv *.m4v);;All (*)",
        )
        if path:
            hdr_info = _probe_hdr_input(path)
            if bool(hdr_info.get("is_hdr", False)):
                QMessageBox.warning(
                    self,
                    "SDR Video",
                    f"Selected file appears HDR and cannot be used as the SDR source.\n\n{hdr_info.get('reason', 'HDR metadata detected')}",
                )
                return
            self._txt_video_sdr.setText(path)
            self._last_source_dir = os.path.dirname(path) or self._last_source_dir
            self._lst_video_frames.clear()
            self._img_video_setup_preview.set_image_from_bgr(
                None,
                "Detect distinct frames to preview.",
            )

    def _pick_video_gt(self):
        start = self._last_source_dir or self._suggested_dir
        current = self._txt_video_gt.text().strip()
        if current and os.path.isfile(current):
            start = os.path.dirname(current)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDR GT Video",
            start,
            "Video (*.mp4 *.avi *.mkv *.mov *.webm *.flv *.m4v);;All (*)",
        )
        if path:
            hdr_info = _probe_hdr_input(path)
            if not bool(hdr_info.get("is_hdr", False)):
                QMessageBox.warning(
                    self,
                    "HDR GT Video",
                    f"Selected file does not appear HDR and cannot be used as HDR ground truth.\n\n{hdr_info.get('reason', 'HDR metadata not detected')}",
                )
                return
            self._txt_video_gt.setText(path)
            self._last_source_dir = os.path.dirname(path) or self._last_source_dir

    def _pick_dataset_sdr(self):
        start = self._txt_dataset_sdr.text().strip() or self._last_source_dir or self._suggested_dir
        path = QFileDialog.getExistingDirectory(self, "Select SDR Dataset Folder", start)
        if path:
            self._txt_dataset_sdr.setText(path)
            self._last_source_dir = path

    def _pick_dataset_gt(self):
        start = self._txt_dataset_gt.text().strip() or self._last_source_dir or self._suggested_dir
        path = QFileDialog.getExistingDirectory(self, "Select HDR GT Dataset Folder", start)
        if path:
            self._txt_dataset_gt.setText(path)
            self._last_source_dir = path

    def _pick_session_root(self):
        start = self._txt_session_root.text().strip() or self._logs_root
        path = QFileDialog.getExistingDirectory(self, "Select Benchmark Session Root", start)
        if path:
            self._txt_session_root.setText(path)

    def _validate_video_pair(self, sdr_path: str, gt_path: str) -> tuple[bool, str]:
        if not os.path.isfile(sdr_path):
            return False, "SDR video path is invalid."
        if not os.path.isfile(gt_path):
            return False, "HDR GT video path is invalid."

        sdr_hdr = _probe_hdr_input(sdr_path)
        if bool(sdr_hdr.get("is_hdr", False)):
            return False, f"SDR source appears HDR ({sdr_hdr.get('reason', 'HDR metadata detected')})."

        gt_hdr = _probe_hdr_input(gt_path)
        if not bool(gt_hdr.get("is_hdr", False)):
            return False, f"HDR GT does not look HDR ({gt_hdr.get('reason', 'HDR metadata not detected')})."

        sdr_meta = _probe_video_timing_info(sdr_path)
        gt_meta = _probe_video_timing_info(gt_path)
        ok, timing_error, notes = _validate_video_timing_compatibility(
            sdr_meta,
            gt_meta,
            source_label="SDR",
            gt_label="GT",
            metadata_error_message="Could not read video metadata.",
            enforce_sync_tolerance=False,
        )
        if not ok:
            return False, str(timing_error or "Could not read video metadata.")

        sdr_w = int(sdr_meta.get("width") or 0)
        sdr_h = int(sdr_meta.get("height") or 0)
        gt_w = int(gt_meta.get("width") or 0)
        gt_h = int(gt_meta.get("height") or 0)
        if sdr_w > 0 and sdr_h > 0 and gt_w > 0 and gt_h > 0:
            sdr_ar = float(sdr_w) / float(sdr_h)
            gt_ar = float(gt_w) / float(gt_h)
            if abs(sdr_ar - gt_ar) > 0.01:
                sdr_active = _probe_video_active_area_info(sdr_path, sample_count=5)
                gt_active = _probe_video_active_area_info(gt_path, sample_count=5)
                sdr_active_ar = (
                    float(sdr_active.get("active_aspect", 0.0) or 0.0)
                    if isinstance(sdr_active, dict)
                    else 0.0
                )
                gt_active_ar = (
                    float(gt_active.get("active_aspect", 0.0) or 0.0)
                    if isinstance(gt_active, dict)
                    else 0.0
                )
                if (
                    sdr_active_ar > 0.0
                    and gt_active_ar > 0.0
                    and abs(sdr_active_ar - gt_active_ar) <= 0.04
                ):
                    sdr_aw = int(sdr_active.get("active_width", sdr_w))
                    sdr_ah = int(sdr_active.get("active_height", sdr_h))
                    gt_aw = int(gt_active.get("active_width", gt_w))
                    gt_ah = int(gt_active.get("active_height", gt_h))
                    notes.append(
                        "active picture aspect matches after black-bar crop "
                        f"({sdr_aw}x{sdr_ah} vs {gt_aw}x{gt_ah})"
                    )
                else:
                    return False, f"Aspect-ratio mismatch: SDR {sdr_w}x{sdr_h} vs GT {gt_w}x{gt_h}."

        sync_info = _probe_video_sync_info(sdr_path, gt_path, sample_count=3)
        score = sync_info.get("score")
        sampled = int(sync_info.get("sampled", 0) or 0)
        if score is None or sampled < 3:
            return False, "Could not verify content alignment from sampled frames."
        score = float(score)
        if score < 0.34:
            return False, f"Content mismatch (similarity {score:.2f})."
        try:
            sync_offset_frames = int(sync_info.get("offset_frames", 0) or 0)
            sync_offset_s = float(sync_info.get("offset_s", 0.0) or 0.0)
        except Exception:
            sync_offset_frames = 0
            sync_offset_s = 0.0
        if sync_offset_frames:
            notes.append(
                f"GT sync offset {sync_offset_frames:+d} frames ({sync_offset_s:+.3f}s)"
            )

        suffix = ""
        if notes:
            suffix = "; " + "; ".join(notes)
        return True, f"Validated (content similarity {score:.2f}{suffix})."

    def _validate_dataset_pair(self, sdr_path: str, gt_path: str) -> tuple[bool, str]:
        if not os.path.isfile(sdr_path):
            return False, "SDR path is invalid."
        if not os.path.isfile(gt_path):
            return False, "HDR GT path is invalid."

        sdr_is_video = _is_video_path(sdr_path)
        gt_is_video = _is_video_path(gt_path)
        if sdr_is_video != gt_is_video:
            return False, "SDR/GT media types do not match."
        if sdr_is_video and gt_is_video:
            return self._validate_video_pair(sdr_path, gt_path)

        # For image datasets, use bit depth as a coarse guard so obvious
        # SDR-in-GT / HDR-in-SDR folder swaps are rejected.
        sdr_img = read_image_any(sdr_path)
        gt_img = read_image_any(gt_path)
        if sdr_img is None or gt_img is None:
            return False, "Could not read paired dataset image(s)."
        if sdr_img.dtype == np.uint16:
            return False, "SDR dataset image appears HDR/high-bit-depth."
        if gt_img.dtype != np.uint16:
            return False, "HDR GT dataset image does not appear HDR/high-bit-depth."
        return True, "Validated image pair."

    def _detect_frames(self):
        sdr_path = self._txt_video_sdr.text().strip()
        gt_path = self._txt_video_gt.text().strip()
        ok, note = self._validate_video_pair(sdr_path, gt_path)
        if not ok:
            QMessageBox.warning(self, "Video Benchmark", note)
            return

        pool_count = max(1, int(self._spn_detect_frame_count.value()))
        cache_key = self._frame_detect_cache_key(sdr_path, pool_count)
        frames = list(self._frame_detect_cache.get(cache_key) or [])
        used_cache = bool(frames)
        if not frames:
            # Increase scan budget with requested count to keep higher-count runs useful,
            # while keeping deterministic behavior for the same source+count.
            scan_cap = min(4000, max(260, pool_count * 24))
            frames = _detect_distinct_video_frames(
                sdr_path,
                desired_count=pool_count,
                max_scan_points=scan_cap,
            )
            if frames:
                self._frame_detect_cache[cache_key] = list(frames)
                while len(self._frame_detect_cache) > 64:
                    oldest_key = next(iter(self._frame_detect_cache))
                    self._frame_detect_cache.pop(oldest_key, None)
                self._save_frame_detect_cache()
        if not frames:
            QMessageBox.warning(
                self,
                "Video Benchmark",
                "Could not detect representative frames from the selected video.",
            )
            return

        self._lst_video_frames.clear()
        for fidx in frames:
            item = QListWidgetItem(f"Frame {int(fidx)}")
            item.setData(Qt.ItemDataRole.UserRole, int(fidx))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._lst_video_frames.addItem(item)
        if self._lst_video_frames.count() > 0:
            self._lst_video_frames.setCurrentRow(0)
        self._refresh_video_frame_preview()

        self._lbl_video_note.setText(
            f"{len(frames)} deterministic candidate frames detected "
            f"(pool {pool_count}, {'cache hit' if used_cache else 'fresh scan'}). "
            "Average mode controls how many checked candidates are benchmarked. "
            f"{note}"
        )

    def _pair_dataset_files(self, sdr_root: str, gt_root: str) -> list[BenchmarkTask]:
        sdr_files = _sorted_media_files(sdr_root)
        gt_files = _sorted_media_files(gt_root)

        gt_rel_map: dict[str, str] = {}
        gt_stem_map: dict[str, list[str]] = {}

        for p in gt_files:
            rel = os.path.relpath(p, gt_root).replace("\\", "/")
            rel_no_ext = os.path.splitext(rel)[0].lower()
            gt_rel_map[rel_no_ext] = p
            stem = os.path.splitext(os.path.basename(p))[0].lower()
            gt_stem_map.setdefault(stem, []).append(p)

        pairs: list[BenchmarkTask] = []
        invalid_notes: list[str] = []
        for sdr_path in sdr_files:
            rel = os.path.relpath(sdr_path, sdr_root).replace("\\", "/")
            rel_no_ext = os.path.splitext(rel)[0].lower()
            gt_path = gt_rel_map.get(rel_no_ext)
            if gt_path is None:
                stem = os.path.splitext(os.path.basename(sdr_path))[0].lower()
                cands = sorted(gt_stem_map.get(stem, []))
                if cands:
                    gt_path = cands[0]
            if gt_path is None:
                continue

            ok, note = self._validate_dataset_pair(sdr_path, gt_path)
            if not ok:
                invalid_notes.append(f"{rel}: {note}")
                continue

            task = BenchmarkTask(
                task_id=_sanitize_name(rel_no_ext),
                label=rel,
                sdr_path=sdr_path,
                gt_path=gt_path,
                frame_idx=0 if _is_video_path(sdr_path) else None,
            )
            pairs.append(task)

        pairs.sort(key=lambda t: t.label.lower())
        self._last_dataset_pair_warnings = invalid_notes
        return pairs

    def _scan_dataset_pairs(self):
        sdr_root = self._txt_dataset_sdr.text().strip()
        gt_root = self._txt_dataset_gt.text().strip()
        if not os.path.isdir(sdr_root):
            QMessageBox.warning(self, "Dataset Benchmark", "Choose a valid SDR dataset folder.")
            return
        if not os.path.isdir(gt_root):
            QMessageBox.warning(self, "Dataset Benchmark", "Choose a valid HDR GT dataset folder.")
            return

        pairs = self._pair_dataset_files(sdr_root, gt_root)
        self._dataset_pairs = pairs
        self._lst_dataset_pairs.clear()
        for task in pairs:
            item = QListWidgetItem(task.label)
            item.setData(Qt.ItemDataRole.UserRole, task.task_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._lst_dataset_pairs.addItem(item)

        self._lbl_pair_count.setText(f"Pairs: {len(pairs)}")
        if not pairs:
            QMessageBox.information(
                self,
                "Dataset Benchmark",
                "No SDR/HDR GT pairs were found. Pairing uses relative path first, then filename stem fallback.",
            )
        elif getattr(self, "_last_dataset_pair_warnings", None):
            notes = list(self._last_dataset_pair_warnings[:8])
            extra = len(self._last_dataset_pair_warnings) - len(notes)
            if extra > 0:
                notes.append(f"... and {extra} more invalid pair(s).")
            QMessageBox.warning(
                self,
                "Dataset Benchmark",
                "Some dataset pairs were skipped because benchmark only allows SDR source -> HDR GT pairs:\n\n"
                + "\n".join(notes),
            )

    def _build_video_tasks(self) -> list[BenchmarkTask]:
        sdr_path = self._txt_video_sdr.text().strip()
        gt_path = self._txt_video_gt.text().strip()
        ok, note = self._validate_video_pair(sdr_path, gt_path)
        if not ok:
            raise RuntimeError(note)

        checked_frames: list[int] = []
        all_frames: list[int] = []
        for i in range(self._lst_video_frames.count()):
            item = self._lst_video_frames.item(i)
            try:
                frame_1b = int(item.data(Qt.ItemDataRole.UserRole))
            except Exception:
                continue
            all_frames.append(frame_1b)
            if item.checkState() == Qt.CheckState.Checked:
                checked_frames.append(frame_1b)
        if not all_frames:
            raise RuntimeError("Detect distinct frames first.")

        avg_mode = str(self._cmb_video_avg_mode.currentData() or "selected")
        if avg_mode == "all":
            frames = list(all_frames)
        elif avg_mode == "subset":
            if not checked_frames:
                raise RuntimeError(
                    "Check at least one candidate frame before using deterministic subset mode."
                )
            subset_n = max(1, int(self._spn_video_subset.value()))
            frame_tasks = [
                BenchmarkTask(
                    task_id=f"frame_{int(frame_1b)}",
                    label=f"Frame {int(frame_1b)}",
                    sdr_path=sdr_path,
                    gt_path=gt_path,
                    frame_idx=max(0, int(frame_1b) - 1),
                )
                for frame_1b in sorted(set(checked_frames))
            ]
            return self._deterministic_subset(frame_tasks, subset_n)
        else:
            frames = list(checked_frames)
        if not frames:
            raise RuntimeError("Select at least one frame for video benchmarking.")

        tasks: list[BenchmarkTask] = []
        for frame_1b in sorted(set(frames)):
            frame_0b = max(0, int(frame_1b) - 1)
            task_id = f"frame_{int(frame_1b)}"
            tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    label=f"Frame {int(frame_1b)}",
                    sdr_path=sdr_path,
                    gt_path=gt_path,
                    frame_idx=frame_0b,
                )
            )
        return tasks

    def _deterministic_subset(self, tasks: list[BenchmarkTask], count: int) -> list[BenchmarkTask]:
        if count <= 0 or len(tasks) <= count:
            return list(tasks)
        if count == 1:
            return [tasks[len(tasks) // 2]]
        step = float(len(tasks) - 1) / float(count - 1)
        idxs = sorted({int(round(i * step)) for i in range(count)})
        return [tasks[max(0, min(len(tasks) - 1, idx))] for idx in idxs]

    def _build_dataset_tasks(self) -> list[BenchmarkTask]:
        if not self._dataset_pairs:
            raise RuntimeError("Scan dataset pairs first.")

        checked_ids: set[str] = set()
        for i in range(self._lst_dataset_pairs.count()):
            item = self._lst_dataset_pairs.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                checked_ids.add(str(item.data(Qt.ItemDataRole.UserRole) or ""))

        selected = [p for p in self._dataset_pairs if p.task_id in checked_ids]
        if not selected:
            selected = list(self._dataset_pairs)

        avg_mode = str(self._cmb_avg_mode.currentData() or "selected")
        if avg_mode == "all":
            return list(self._dataset_pairs)
        if avg_mode == "subset":
            subset_n = max(1, int(self._spn_subset.value()))
            base = selected if selected else list(self._dataset_pairs)
            base_sorted = sorted(base, key=lambda x: x.label.lower())
            return self._deterministic_subset(base_sorted, subset_n)
        return selected

    def _derive_source_name(self, mode: str, tasks: list[BenchmarkTask]) -> str:
        if mode == "video":
            path = self._txt_video_sdr.text().strip()
            if path:
                return os.path.splitext(os.path.basename(path))[0]
        else:
            root = self._txt_dataset_sdr.text().strip()
            if root:
                return os.path.basename(root.rstrip("\\/"))

        if tasks:
            first = tasks[0]
            base = os.path.splitext(os.path.basename(first.sdr_path))[0]
            if base:
                return base
        return "benchmark_source"

    def _new_session_dir(self, source_name: str, task_count: int) -> str:
        root_dir = self._txt_session_root.text().strip() or self._logs_root
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        pkey = _sanitize_name(self._cmb_precision.currentText().lower().replace(" ", "_"))
        rkey = _sanitize_name(self._cmb_resolution.currentText().lower().replace(" ", "_"))
        source_key = _sanitize_name(source_name)
        session_name = f"{ts}__{pkey}"
        if rkey:
            session_name = f"{session_name}__{rkey}"
        session_name = f"{session_name}__n{max(0, int(task_count))}"
        session_root = os.path.join(root_dir, source_key)
        session = os.path.join(session_root, session_name)
        suffix = 2
        while os.path.exists(session):
            session = os.path.join(session_root, f"{session_name}_{suffix}")
            suffix += 1
        os.makedirs(session, exist_ok=True)
        return session

    def _build_current_run_config(self) -> BenchmarkRunConfig:
        mode = str(self._cmb_mode.currentData() or "video")
        if mode == "video":
            tasks = self._build_video_tasks()
        else:
            tasks = self._build_dataset_tasks()
        source_name = self._derive_source_name(mode, tasks)
        session_dir = self._new_session_dir(source_name, len(tasks))
        return BenchmarkRunConfig(
            mode=mode,
            source_name=source_name,
            precision_key=self._cmb_precision.currentText(),
            use_hg=bool(self._chk_use_hg.isChecked()),
            resolution_key=self._cmb_resolution.currentText(),
            predequantize_mode=str(self._cmb_predequantize.currentData() or "auto"),
            tasks=tasks,
            session_dir=session_dir,
        )

    def _queue_title_for_config(self, cfg: BenchmarkRunConfig) -> str:
        hg_label = "HG" if bool(cfg.use_hg) else "no-HG"
        pdq = _normalize_benchmark_predequantize_mode(cfg.predequantize_mode)
        mode_label = "video" if cfg.mode == "video" else "dataset"
        return (
            f"{cfg.source_name} | {cfg.precision_key} | {cfg.resolution_key} | "
            f"{hg_label} | predeq={pdq} | {mode_label} n={len(cfg.tasks)}"
        )

    def _queue_detail_for_config(self, cfg: BenchmarkRunConfig) -> str:
        hg_label = "on" if bool(cfg.use_hg) else "off"
        pdq = _normalize_benchmark_predequantize_mode(cfg.predequantize_mode)
        mode_label = "Video" if cfg.mode == "video" else "Dataset"
        first_task = cfg.tasks[0] if cfg.tasks else None
        first_label = str(first_task.label) if first_task is not None else "-"
        if len(cfg.tasks) > 1:
            first_label = f"{first_label} (+{len(cfg.tasks) - 1} more)"
        return (
            f"{mode_label} benchmark: {cfg.source_name} | "
            f"precision={cfg.precision_key}, resolution={cfg.resolution_key}, "
            f"HG={hg_label}, predequantize={pdq}, items={len(cfg.tasks)} | "
            f"first item={first_label} | session={cfg.session_dir}"
        )

    def _sync_queue_ui(self):
        if not hasattr(self, "_lbl_queue_status"):
            return
        running = self._worker_thread is not None
        queued = len(self._benchmark_queue)
        if self._queue_running:
            remaining = queued
            total = max(1, int(self._active_queue_total))
            current = max(1, int(self._active_queue_index))
            self._lbl_queue_status.setText(
                f"Queue: running {current}/{total} ({remaining} remaining)"
            )
        else:
            self._lbl_queue_status.setText(
                f"Queue: {queued} run{'s' if queued != 1 else ''}"
            )
        selected_row = -1
        if hasattr(self, "_lst_benchmark_queue"):
            selected_row = int(self._lst_benchmark_queue.currentRow())
        can_remove = (not running) and 0 <= selected_row < queued
        self._btn_queue_add.setEnabled(not running)
        self._btn_queue_run.setEnabled((not running) and queued > 0)
        self._btn_queue_remove.setEnabled(can_remove)
        self._btn_queue_clear.setEnabled((not running) and queued > 0)

    def _refresh_queue_list(self):
        previous_row = self._lst_benchmark_queue.currentRow()
        self._lst_benchmark_queue.clear()
        for idx, queued in enumerate(self._benchmark_queue, start=1):
            item = QListWidgetItem(f"{idx}. {queued.title}")
            item.setToolTip(self._queue_detail_for_config(queued.config))
            self._lst_benchmark_queue.addItem(item)
        if self._benchmark_queue:
            next_row = min(max(previous_row, 0), len(self._benchmark_queue) - 1)
            self._lst_benchmark_queue.setCurrentRow(next_row)
        else:
            self._update_queue_preview(-1)
        self._sync_queue_ui()

    def _update_queue_preview(self, row: int):
        if not hasattr(self, "_lbl_queue_preview"):
            return
        if row < 0 or row >= len(self._benchmark_queue):
            self._lbl_queue_preview.setText(
                "Select a queued run to preview its captured settings."
            )
            return
        queued = self._benchmark_queue[row]
        self._lbl_queue_preview.setText(self._queue_detail_for_config(queued.config))

    def _add_current_benchmark_to_queue(self):
        if self._worker_thread is not None:
            return
        try:
            cfg = self._build_current_run_config()
        except Exception as exc:
            QMessageBox.warning(self, "Benchmark Queue", str(exc))
            return
        title = self._queue_title_for_config(cfg)
        self._benchmark_queue.append(_QueuedBenchmarkRun(config=cfg, title=title))
        self._refresh_queue_list()
        self._set_status(f"Queued benchmark: {title}")

    def _remove_selected_benchmark_from_queue(self):
        if self._worker_thread is not None:
            return
        row = int(self._lst_benchmark_queue.currentRow())
        if row < 0 or row >= len(self._benchmark_queue):
            return
        removed = self._benchmark_queue.pop(row)
        self._refresh_queue_list()
        self._set_status(f"Removed queued benchmark: {removed.title}")

    def _clear_benchmark_queue(self):
        if self._worker_thread is not None:
            return
        self._benchmark_queue.clear()
        self._queue_running = False
        self._active_queue_total = 0
        self._active_queue_index = 0
        self._refresh_queue_list()
        self._set_status("Benchmark queue cleared")

    def _start_benchmark_queue(self):
        if self._worker_thread is not None or not self._benchmark_queue:
            return
        if not self._warn_if_ffmpeg_missing_for_hdr_benchmark():
            return
        self._queue_running = True
        self._active_queue_total = len(self._benchmark_queue)
        self._active_queue_index = 0
        self._start_next_queued_benchmark()

    def _start_next_queued_benchmark(self):
        if self._worker_thread is not None:
            return
        if not self._benchmark_queue:
            self._queue_running = False
            self._active_queue_total = 0
            self._active_queue_index = 0
            self._lbl_progress.setText("Benchmark queue completed.")
            self._set_status("Benchmark queue completed", percent=100)
            self._sync_queue_ui()
            return
        queued = self._benchmark_queue.pop(0)
        self._active_queue_index += 1
        self._refresh_queue_list()
        self._start_benchmark_config(queued.config, queue_title=queued.title)

    def _set_running_state(self, running: bool):
        self._btn_run.setEnabled(not running)
        self._btn_cancel.setEnabled(running)
        self._btn_close.setEnabled(not running)
        for w in (
            self._cmb_mode,
            self._btn_video_sdr,
            self._btn_video_gt,
            self._spn_detect_frame_count,
            self._btn_detect_frames,
            self._cmb_video_avg_mode,
            self._spn_video_subset,
            self._btn_dataset_sdr,
            self._btn_dataset_gt,
            self._btn_scan_dataset,
            self._cmb_precision,
            self._chk_use_hg,
            self._cmb_resolution,
            self._cmb_predequantize,
            self._btn_session_root,
            self._cmb_avg_mode,
            self._spn_subset,
            self._btn_select_all_frames,
            self._btn_clear_frames,
            self._btn_select_all_pairs,
            self._btn_clear_pairs,
            self._btn_load_existing,
            self._cmb_average_filter,
            self._result_sets_bar,
        ):
            w.setEnabled(not running)
        if hasattr(self, "_lst_benchmark_queue"):
            self._lst_benchmark_queue.setEnabled(not running)
        self._sync_queue_ui()

    def _start_benchmark(self):
        if self._worker_thread is not None:
            return

        try:
            run_cfg = self._build_current_run_config()
        except Exception as exc:
            QMessageBox.warning(self, "Benchmark", str(exc))
            return

        if not self._warn_if_ffmpeg_missing_for_hdr_benchmark():
            return
        self._queue_running = False
        self._active_queue_total = 0
        self._active_queue_index = 0
        self._start_benchmark_config(run_cfg)

    def _start_benchmark_config(
        self,
        run_cfg: BenchmarkRunConfig,
        *,
        queue_title: str | None = None,
    ):
        source_name = run_cfg.source_name
        session_dir = run_cfg.session_dir
        run_title = self._result_tab_title(
            source_name,
            run_cfg.precision_key,
            session_dir,
        )
        run_payload = self._result_tab_payload(
            title=run_title,
            rows=[],
            average={},
            session_dir=session_dir,
            source_name=source_name,
            precision=run_cfg.precision_key,
            resolution=run_cfg.resolution_key,
            use_hg=bool(run_cfg.use_hg),
            predequantize_mode=str(run_cfg.predequantize_mode),
            selected_row=0,
        )
        self._add_result_set_tab(run_payload, activate=True)

        self._session_dir = session_dir
        self._lbl_session_dir.setText(f"Session: {session_dir}")
        if self._queue_running:
            self._lbl_progress.setText(
                f"Benchmark queue {self._active_queue_index}/{self._active_queue_total}: "
                f"{queue_title or run_title}"
            )
        else:
            self._lbl_progress.setText("Benchmark started ...")
        self._progress.setValue(0)
        status_message = (
            f"Benchmark queue {self._active_queue_index}/{self._active_queue_total} started"
            if self._queue_running else "Benchmark started"
        )
        self._set_status(status_message, percent=0)
        self._set_result_run_info(
            source_name,
            run_cfg.precision_key,
            run_cfg.resolution_key,
        )
        self._tabs.setCurrentIndex(1)

        self._results = []
        self._tbl.setRowCount(0)
        self._set_result_previews(None)

        thread = QThread(self)
        worker = _BenchmarkWorker(run_cfg)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_worker_progress)
        worker.sample_ready.connect(self._on_worker_sample_ready)
        worker.finished.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        worker.canceled.connect(self._on_worker_canceled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.canceled.connect(thread.quit)
        thread.finished.connect(self._cleanup_worker)

        self._worker_thread = thread
        self._worker = worker
        self._set_running_state(True)
        self._btn_open_session.setEnabled(False)
        self._btn_export_selected.setEnabled(False)
        self._btn_export_all.setEnabled(False)
        thread.start()

    def _cancel_benchmark(self):
        if self._worker is not None:
            self._worker.cancel()
            if self._queue_running:
                self._lbl_progress.setText("Canceling benchmark queue ...")
                self._set_status("Canceling benchmark queue ...")
            else:
                self._lbl_progress.setText("Canceling benchmark ...")
                self._set_status("Canceling benchmark ...")

    def _cleanup_worker(self):
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
            self._worker_thread = None
        self._set_running_state(False)
        if self._queue_running:
            self._start_next_queued_benchmark()

    def _on_worker_progress(self, percent: int, message: str):
        self._progress.setValue(max(0, min(100, int(percent))))
        self._lbl_progress.setText(str(message or ""))
        self._set_status(str(message or "Running"), percent=percent)

    def _add_table_row(self, row: dict, outlier: bool = False):
        metrics = row.get("metrics") or {}
        r = self._tbl.rowCount()
        self._tbl.insertRow(r)
        note = str(metrics.get("obj_note") or metrics.get("hdr_vdp3_note") or "")
        if outlier:
            outlier_note = "Outlier (excluded from preview average)"
            note = f"{note}; {outlier_note}" if note else outlier_note

        values = [
            str(row.get("label") or "-"),
            _fmt_metric(metrics.get("psnr_db"), ".2f", " dB"),
            _fmt_metric(metrics.get("sssim"), ".4f"),
            _fmt_metric(metrics.get("delta_e_itp"), ".2f"),
            _fmt_metric(metrics.get("psnr_norm_db"), ".2f", " dB"),
            _fmt_metric(metrics.get("sssim_norm"), ".4f"),
            _fmt_metric(metrics.get("delta_e_itp_norm"), ".2f"),
            _fmt_metric(metrics.get("hdr_vdp3"), ".3f"),
            note,
        ]
        for c, value in enumerate(values):
            item = QTableWidgetItem(value)
            self._tbl.setItem(r, c, item)

    def _on_worker_sample_ready(self, row: dict):
        self._results.append(row)
        self._add_table_row(row)
        self._tbl.scrollToBottom()
        if self._tbl.rowCount() == 1:
            self._tbl.blockSignals(True)
            self._tbl.selectRow(0)
            self._tbl.blockSignals(False)
            self._on_result_selection_changed()

    def _on_worker_finished(self, payload: dict):
        avg = payload.get("average") or {}
        final_rows = payload.get("rows")
        if isinstance(final_rows, list):
            self._results = list(final_rows)
        session_dir = str(payload.get("session_dir") or self._session_dir or "")
        source_name = str(payload.get("source_name") or "").strip()
        precision = str(payload.get("precision") or "").strip()
        resolution = str(payload.get("resolution") or "").strip()
        use_hg = payload.get("use_hg")
        predequantize_mode = str(payload.get("predequantize_mode") or "").strip()
        selected_row = self._selected_result_row()
        idx = int(self._result_sets_bar.currentIndex())
        self._update_result_set_tab(
            idx,
            self._result_tab_payload(
                title=self._result_tab_title(source_name, precision, session_dir),
                rows=self._results,
                average=avg,
                session_dir=session_dir,
                source_name=source_name,
                precision=precision,
                resolution=resolution,
                use_hg=use_hg,
                predequantize_mode=predequantize_mode,
                selected_row=selected_row,
            ),
        )
        self._progress.setValue(100)
        self._lbl_progress.setText("Benchmark completed.")
        self._set_status("Benchmark completed", percent=100)
        self._populate_results_view(
            rows=self._results,
            average=avg,
            session_dir=session_dir,
            source_name=source_name,
            precision=precision,
            resolution=resolution,
            use_hg=use_hg,
            predequantize_mode=predequantize_mode,
            selected_row=selected_row,
        )

    def _on_worker_failed(self, message: str):
        self._queue_running = False
        self._active_queue_total = 0
        self._active_queue_index = 0
        self._sync_queue_ui()
        self._lbl_progress.setText("Benchmark failed.")
        self._set_status("Benchmark failed")
        QMessageBox.warning(self, "Benchmark Failed", str(message or "Unknown error."))

    def _on_worker_canceled(self, message: str):
        was_queue_running = bool(self._queue_running)
        self._queue_running = False
        self._active_queue_total = 0
        self._active_queue_index = 0
        self._sync_queue_ui()
        avg = self._compute_average_from_rows(self._results)
        selected_row = self._selected_result_row()
        idx = int(self._result_sets_bar.currentIndex())
        self._update_result_set_tab(
            idx,
            self._result_tab_payload(
                title=self._result_tab_title(
                    self._result_source_name,
                    self._result_precision,
                    self._session_dir,
                ),
                rows=self._results,
                average=avg,
                session_dir=self._session_dir,
                source_name=self._result_source_name,
                precision=self._result_precision,
                resolution=self._result_resolution,
                use_hg=self._result_use_hg,
                predequantize_mode=self._result_predequantize_mode,
                selected_row=selected_row,
            ),
        )
        self._populate_results_view(
            rows=self._results,
            average=avg,
            session_dir=self._session_dir,
            source_name=self._result_source_name,
            precision=self._result_precision,
            resolution=self._result_resolution,
            use_hg=self._result_use_hg,
            predequantize_mode=self._result_predequantize_mode,
            selected_row=selected_row,
        )
        if was_queue_running:
            self._lbl_progress.setText("Benchmark queue canceled.")
            self._set_status("Benchmark queue canceled")
        else:
            self._lbl_progress.setText("Benchmark canceled.")
            self._set_status("Benchmark canceled")
        if message:
            QMessageBox.information(self, "Benchmark", message)

    def _result_row_to_index(self, row_idx: int) -> int | None:
        if row_idx < 0:
            return None
        if row_idx >= len(self._results):
            return None
        return row_idx

    def _on_result_selection_changed(self):
        rows = self._tbl.selectionModel().selectedRows() if self._tbl.selectionModel() else []
        if not rows:
            return
        idx = self._result_row_to_index(rows[0].row())
        if idx is None:
            return
        self._remember_current_result_selection()
        row = self._results[idx]
        self._set_result_previews(row)

    def _open_session_folder(self):
        if not self._session_dir or not os.path.isdir(self._session_dir):
            return
        try:
            os.startfile(self._session_dir)
        except Exception as exc:
            QMessageBox.warning(self, "Open Session Folder", f"Could not open folder:\n{exc}")

    def _selected_results(self) -> list[dict]:
        rows = self._tbl.selectionModel().selectedRows() if self._tbl.selectionModel() else []
        if not rows:
            return []
        out = []
        for m_idx in rows:
            idx = self._result_row_to_index(m_idx.row())
            if idx is None:
                continue
            out.append(self._results[idx])
        return out

    def _write_export_summary(self, folder: str, rows: list[dict]):
        summary_csv = os.path.join(folder, "benchmark_export_summary.csv")
        summary_json = os.path.join(folder, "benchmark_export_summary.json")

        avg = {}
        for key in _BENCHMARK_METRIC_KEYS:
            vals = []
            for r in rows:
                v = (r.get("metrics") or {}).get(key)
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    vals.append(fv)
            avg[key] = float(np.mean(vals)) if vals else None

        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "item",
                    "frame",
                    "psnr_db",
                    "sssim",
                    "delta_e_itp",
                    "psnr_norm_db",
                    "sssim_norm",
                    "delta_e_itp_norm",
                    "hdr_vdp3",
                    "obj_note",
                    "hdr_vdp3_note",
                    "exported_folder",
                ]
            )
            for r in rows:
                m = r.get("metrics") or {}
                writer.writerow(
                    [
                        r.get("label"),
                        r.get("frame"),
                        m.get("psnr_db"),
                        m.get("sssim"),
                        m.get("delta_e_itp"),
                        m.get("psnr_norm_db"),
                        m.get("sssim_norm"),
                        m.get("delta_e_itp_norm"),
                        m.get("hdr_vdp3"),
                        m.get("obj_note"),
                        m.get("hdr_vdp3_note"),
                        r.get("sample_dir"),
                    ]
                )
            writer.writerow([])
            writer.writerow(["AVERAGE"])
            writer.writerow(
                [
                    "",
                    "",
                    avg.get("psnr_db"),
                    avg.get("sssim"),
                    avg.get("delta_e_itp"),
                    avg.get("psnr_norm_db"),
                    avg.get("sssim_norm"),
                    avg.get("delta_e_itp_norm"),
                    avg.get("hdr_vdp3"),
                    "",
                    "",
                    folder,
                ]
            )

        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_name": self._result_source_name,
                    "precision": self._result_precision,
                    "resolution": self._result_resolution,
                    "average": avg,
                    "results": rows,
                },
                f,
                indent=2,
            )

    def _copy_sample_folder(self, src: str, dest_root: str) -> str:
        base = os.path.basename(src.rstrip("\\/"))
        if not base:
            base = "sample"
        dst = os.path.join(dest_root, base)
        n = 2
        while os.path.exists(dst):
            dst = os.path.join(dest_root, f"{base}_{n}")
            n += 1
        shutil.copytree(src, dst)
        return dst

    def _export_results(self, selected_only: bool):
        if not self._results:
            QMessageBox.information(self, "Export", "No benchmark results are available yet.")
            return

        rows = self._selected_results() if selected_only else list(self._results)
        if not rows:
            QMessageBox.information(self, "Export", "Select a result row first.")
            return

        start = self._session_dir or self._suggested_dir
        export_root = QFileDialog.getExistingDirectory(
            self,
            "Select Export Folder",
            start,
        )
        if not export_root:
            return

        exported: list[dict] = []
        for row in rows:
            src = str(row.get("sample_dir") or "")
            if not src or not os.path.isdir(src):
                continue
            try:
                dst = self._copy_sample_folder(src, export_root)
            except Exception:
                continue
            copied = dict(row)
            copied["sample_dir"] = dst
            exported.append(copied)

        if not exported:
            exported = [dict(r) for r in rows]

        try:
            self._write_export_summary(export_root, exported)
        except Exception:
            pass

        if any(str(r.get("sample_dir") or "").strip() for r in exported):
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(exported)} result folder(s) to:\n{export_root}",
            )
        else:
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported summary only to:\n{export_root}",
            )

    def _close_dialog(self):
        if self._worker_thread is not None:
            QMessageBox.information(
                self,
                "Benchmark Running",
                "Cancel the running benchmark first.",
            )
            return
        self._stop_result_preview_players()
        self._preview_processor_cache.clear()
        self._reset_preview_splitter_on_show = True
        self.accept()

    def closeEvent(self, event):
        if self._worker_thread is not None:
            event.ignore()
            QMessageBox.information(
                self,
                "Benchmark Running",
                "Cancel the running benchmark first.",
            )
            return
        self._stop_result_preview_players()
        self._preview_processor_cache.clear()
        self._reset_preview_splitter_on_show = True
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if bool(getattr(self, "_reset_preview_splitter_on_show", False)):
            self._reset_preview_splitter_on_show = False
            self._reset_benchmark_preview_splitter_sizes()
