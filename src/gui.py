"""
HDRTVNet++ Real-Time Video Pipeline — PyQt6 GUI

Usage:
    python src/gui.py

Features:
    - Browse or drag-and-drop any video file
    - Switch precision (FP16 / FP32 / INT8) at any time, even mid-playback
    - Play / Pause / Stop controls
    - Side-by-side SDR input vs HDR output (or single view)
    - Toggle real-time metrics panel (FPS, latency, GPU/CPU memory, model size)
    - Dark theme
"""

import os
import sys
import time
import json
import re
import shutil
import subprocess
import threading
import queue as _queue
import numpy as np
import cv2
import psutil

# ── Inductor/Triton cache dirs (must be set BEFORE importing torch) ───
# Pin caches to a stable user path to avoid Temp churn/permissions.
def _default_cache_root():
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        return os.path.join(local_app, "HDRTVNetCache")
    return os.path.join(os.path.expanduser("~"), ".cache", "hdrtvnet")

_cache_root = os.environ.get("HDRTVNET_CACHE_DIR", _default_cache_root())
try:
    os.makedirs(_cache_root, exist_ok=True)
except Exception:
    pass

os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(_cache_root, "torchinductor"),
)
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    os.path.join(_cache_root, "triton"),
)

# ── Inductor FX-graph cache ──────────────────────────────────────────
# Ensures that torch.compile autotune decisions are persisted to disk
# and shared across processes, so the worker never re-benchmarks kernels
# that the subprocess already compiled.
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

# Import torch BEFORE PyQt6 — on ROCm-Windows the ROCm SDK DLLs must
# be loaded first; PyQt6 loads its own DLLs which can conflict if torch
# hasn't initialised ROCm yet.
import torch
import torch.nn.functional as F

from collections import deque

# ── libmpv for HDR display (optional) ────────────────────────
# libmpv-2.dll ships alongside this file; add to PATH so ctypes can find it.
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] = _HERE + os.pathsep + os.environ.get("PATH", "")

try:
    import mpv as mpv_lib
    _HAS_MPV = True
except (OSError, ImportError):
    _HAS_MPV = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QFileDialog, QCheckBox,
    QGroupBox, QSplitter, QDialog, QMessageBox, QProgressBar,
    QProgressDialog,
    QTextEdit, QSlider, QStyle, QStackedWidget, QTabWidget,
    QGraphicsOpacityEffect,
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QProcess, QTimer, QRect, QEvent, QPoint, QObject,
    qInstallMessageHandler, QPropertyAnimation, QEasingCurve,
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QCursor
from PyQt6.QtCore import QUrl
try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    _HAS_QT_AUDIO = True
except ImportError:
    QMediaPlayer = None
    QAudioOutput = None
    _HAS_QT_AUDIO = False

from models.hdrtvnet_torch import HDRTVNetTorch
from video_source import VideoSource

# ── Paths ────────────────────────────────────────────────────
_ROOT = os.path.dirname(_HERE)
_MPV_DIAG = os.environ.get("HDRTVNET_MPV_DIAG", "1").strip().lower() in {"1", "true", "yes", "on"}
_PREFS_PATH = os.path.join(_ROOT, ".gui_prefs.json")
_HG_WEIGHTS_PATH = os.path.join(
    _ROOT, "src", "models", "weights", "HG_weights.pth"
)
# Keep wall-clock playback close to target by dropping decode frames when
# inference falls behind. This favors cadence over full-frame coverage.
_REALTIME_CATCHUP_ENABLED = True
_REALTIME_SKIP_LAG_FRAMES = 1.1
_REALTIME_MAX_CATCHUP_SKIP = 6


def _weight(name):
    return os.path.join(_HERE, "models", "weights", name)


def _probe_hdr_input(video_path: str) -> dict:
    """Detect whether the input stream appears to be HDR."""
    info = {
        "is_hdr": False,
        "transfer": "unknown",
        "primaries": "unknown",
        "pix_fmt": "unknown",
        "bits": 0,
        "reason": "",
    }
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        info["reason"] = "ffprobe not found; assuming SDR input"
        return info

    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=color_transfer,color_primaries,pix_fmt,bits_per_raw_sample",
        "-of", "json",
        video_path,
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(cp.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            info["reason"] = "no video stream metadata found; assuming SDR input"
            return info

        stream = streams[0]
        trc = str(stream.get("color_transfer") or "unknown").lower()
        prim = str(stream.get("color_primaries") or "unknown").lower()
        pix_fmt = str(stream.get("pix_fmt") or "unknown").lower()

        bits = 0
        raw_bits = stream.get("bits_per_raw_sample")
        if raw_bits is not None and str(raw_bits).isdigit():
            bits = int(raw_bits)
        if bits == 0:
            m = re.search(r"p(10|12|16)", pix_fmt)
            if m:
                bits = int(m.group(1))

        info.update({
            "transfer": trc,
            "primaries": prim,
            "pix_fmt": pix_fmt,
            "bits": bits,
        })

        hdr_trc = trc in {"smpte2084", "arib-std-b67"}
        hdr_bt2020_inferred = (
            prim.startswith("bt2020")
            and bits >= 10
            and trc not in {"bt709", "unknown", "iec61966-2-1"}
        )
        info["is_hdr"] = bool(hdr_trc or hdr_bt2020_inferred)
        if info["is_hdr"]:
            if hdr_trc:
                info["reason"] = f"HDR transfer detected ({trc})"
            else:
                info["reason"] = f"BT.2020 {bits}-bit stream detected"
        else:
            info["reason"] = "no HDR transfer metadata detected"
    except Exception as exc:
        info["reason"] = f"ffprobe failed ({exc}); assuming SDR input"
    return info


def _probe_audio_streams(video_path: str) -> list[dict]:
    """Return detected audio streams with basic metadata via ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return []
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "a",
        "-show_entries",
        "stream=index,codec_name,channels:stream_tags=language,title:stream_disposition=default",
        "-of", "json",
        video_path,
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(cp.stdout or "{}")
        streams = payload.get("streams") or []
    except Exception:
        return []

    out = []
    for i, s in enumerate(streams):
        tags = s.get("tags") or {}
        disp = s.get("disposition") or {}
        lang = str(tags.get("language") or "und").strip().lower()
        title = str(tags.get("title") or "").strip()
        codec = str(s.get("codec_name") or "audio").strip().lower()
        ch = s.get("channels")
        try:
            ch = int(ch)
        except Exception:
            ch = 0
        out.append({
            "ordinal": i,
            "stream_index": int(s.get("index", i)),
            "language": lang,
            "title": title,
            "codec": codec,
            "channels": ch,
            "is_default": bool(disp.get("default", 0)),
        })
    return out


def _fit_with_aspect(src_w: int, src_h: int, max_w: int, max_h: int) -> tuple[int, int]:
    """Fit source resolution into a bounding box while preserving aspect ratio."""
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    max_w = max(1, int(max_w))
    max_h = max(1, int(max_h))
    scale = min(max_w / src_w, max_h / src_h, 1.0)
    out_w = max(2, int(round(src_w * scale)))
    out_h = max(2, int(round(src_h * scale)))
    # Keep even dimensions for encoder/decoder friendliness.
    out_w -= out_w % 2
    out_h -= out_h % 2
    return max(2, out_w), max(2, out_h)


def _limited_playback_fps(src_fps: float) -> float:
    """Limit playback FPS by halving high-FPS sources (e.g., 50->25, 60->30)."""
    fps = float(src_fps) if src_fps and src_fps > 0 else 30.0
    while fps > 30.0:
        fps *= 0.5
    return max(1.0, fps)


def _select_hdr_scale_kernel(proc_w: int, proc_h: int,
                             out_w: int, out_h: int,
                             upscale_choice: str | None = None) -> str:
    """Pick mpv scale kernel for HDR output."""
    if proc_w == out_w and proc_h == out_h:
        return "bicubic"
    if upscale_choice:
        return _normalize_upscale_choice(upscale_choice)
    return BEST_MPV_SCALE


def _select_hdr_scale_antiring(proc_w: int, proc_h: int,
                               out_w: int, out_h: int,
                               scale_kernel: str | None = None) -> float:
    """Tune antiring strength by processing resolution and kernel."""
    if proc_w >= out_w and proc_h >= out_h:
        return 0.0
    k = str(scale_kernel or "").strip().lower()
    if k == "fsr":
        return 0.0
    if "ssim" in k:
        return 0.0
    if proc_h <= 540 or proc_w <= 960:
        base = 0.30
    elif proc_h <= 720 or proc_w <= 1280:
        base = 0.22
    else:
        base = 0.10
    if "lanczossharp" in k or k == "ewa_lanczos":
        return max(0.0, base - 0.05)
    return base


def _select_mpv_cas_strength(proc_w: int, proc_h: int,
                             out_w: int, out_h: int,
                             using_fsr: bool = False,
                             scale_kernel: str | None = None) -> float:
    """Select CAS sharpening strength for HDR upscale."""
    if proc_w >= out_w and proc_h >= out_h:
        return 0.0
    if using_fsr:
        return 0.0
    k = str(scale_kernel or "").strip().lower()
    if "ssim" in k:
        return 0.0
    if proc_h <= 540 or proc_w <= 960:
        base = 0.22
    elif proc_h <= 720 or proc_w <= 1280:
        base = 0.20
    else:
        base = 0.16
    k = str(scale_kernel or "").strip().lower()
    if "lanczossharp" in k or k == "ewa_lanczos":
        return base + 0.02
    return base


BEST_UPSCALE_MODE = "On (Best)"
BEST_MPV_SCALE = "ewa_lanczossharp"
BEST_CV2_INTERP = cv2.INTER_LANCZOS4
UPSCALE_SHARPEN_STRENGTH = 0.0
UPSCALE_SHARPEN_SIGMA = 1.0
FSR_SHADER_PATH = os.path.join(
    _ROOT, "assets", "shaders", "FSR.glsl"
)
FSR_SHADER_URL = (
    "https://gist.githubusercontent.com/agyild/"
    "82219c545228d70c5604f865ce0b0ce5/raw/"
    "2623d743b9c23f500ba086f05b385dcb1557e15d/FSR.glsl"
)
FILMGRAIN_SHADER_PATH = os.path.join(
    _ROOT, "assets", "shaders", "filmgrain.glsl"
)
FILMGRAIN_SHADER_URL = (
    "https://raw.githubusercontent.com/haasn/gentoo-conf/"
    "xor/home/nand/.mpv/shaders/filmgrain.glsl"
)
SSIM_SUPERRES_SHADER_PATH = os.path.join(
    _ROOT, "assets", "shaders", "SSimSuperRes.glsl"
)
SSIM_SUPERRES_SHADER_URL = (
    "https://gist.githubusercontent.com/igv/"
    "2364ffa6e81540f29cb7ab4c9bc05b6b/raw/"
    "15d93440d0a24fc4b8770070be6a9fa2af6f200b/SSimSuperRes.glsl"
)
UPSCALER_CHOICES = ["EWA LanczosSharp", "FSR", "SSimSuperRes"]
DEFAULT_UPSCALER = "EWA LanczosSharp"


def _normalize_upscale_choice(choice: str) -> str:
    c = str(choice or "").strip().lower()
    if "fsr" in c:
        return "fsr"
    if "ssim" in c:
        return "ssim_superres"
    return BEST_MPV_SCALE


def _ensure_fsr_shader() -> bool:
    """Ensure the FSR shader exists on disk (download on demand)."""
    if os.path.isfile(FSR_SHADER_PATH):
        return True
    try:
        os.makedirs(os.path.dirname(FSR_SHADER_PATH), exist_ok=True)
        import urllib.request
        with urllib.request.urlopen(FSR_SHADER_URL, timeout=10) as resp:
            data = resp.read()
        if data:
            with open(FSR_SHADER_PATH, "wb") as f:
                f.write(data)
            return True
    except Exception as exc:
        print(f"[fsr] download failed: {exc}")
    return os.path.isfile(FSR_SHADER_PATH)


def _ensure_filmgrain_shader() -> bool:
    """Ensure the film grain shader exists on disk (download on demand)."""
    if os.path.isfile(FILMGRAIN_SHADER_PATH):
        return True
    try:
        os.makedirs(os.path.dirname(FILMGRAIN_SHADER_PATH), exist_ok=True)
        import urllib.request
        with urllib.request.urlopen(FILMGRAIN_SHADER_URL, timeout=10) as resp:
            data = resp.read()
        if data:
            with open(FILMGRAIN_SHADER_PATH, "wb") as f:
                f.write(data)
            return True
    except Exception as exc:
        print(f"[filmgrain] download failed: {exc}")
    return os.path.isfile(FILMGRAIN_SHADER_PATH)


def _ensure_ssim_superres_shader() -> bool:
    """Ensure the SSimSuperRes shader exists on disk (download on demand)."""
    if os.path.isfile(SSIM_SUPERRES_SHADER_PATH):
        return True
    try:
        os.makedirs(os.path.dirname(SSIM_SUPERRES_SHADER_PATH), exist_ok=True)
        import urllib.request
        with urllib.request.urlopen(SSIM_SUPERRES_SHADER_URL, timeout=10) as resp:
            data = resp.read()
        if data:
            with open(SSIM_SUPERRES_SHADER_PATH, "wb") as f:
                f.write(data)
            return True
    except Exception as exc:
        print(f"[ssim] download failed: {exc}")
    return os.path.isfile(SSIM_SUPERRES_SHADER_PATH)


def _normalize_shader_paths(paths: list[str]) -> list[str]:
    return [p for p in paths if p]


def _letterbox_bgr(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize with preserved aspect ratio and black padding to exact output size."""
    h, w = frame.shape[:2]
    if w == out_w and h == out_h:
        return frame

    scale = min(out_w / max(w, 1), out_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((out_h, out_w, 3), dtype=frame.dtype)
    x = (out_w - new_w) // 2
    y = (out_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def _apply_upscale_sharpen(img: np.ndarray,
                           strength: float = UPSCALE_SHARPEN_STRENGTH,
                           sigma: float = UPSCALE_SHARPEN_SIGMA) -> np.ndarray:
    """Mild unsharp mask after upscaling."""
    if strength <= 0.0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)


# ── Precision configurations ─────────────────────────────────
PRECISIONS = {
    "FP16": {
        "precision": "fp16",
        "model": _weight("Ensemble_AGCM_LE.pth"),
    },
    "FP32": {
        "precision": "fp32",
        "model": _weight("Ensemble_AGCM_LE.pth"),
    },
    "INT8 Mixed (PTQ)": {
        "precision": "int8-mixed",
        "model": _weight("Ensemble_AGCM_LE_int8_mixed.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_mixed_nohg.pt"),
    },
    "INT8 Mixed (QAT)": {
        "precision": "int8-mixed",
        "model": _weight("Ensemble_AGCM_LE_int8_mixed_qat.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_mixed_qat_nohg.pt"),
    },
    "INT8 Full (PTQ)": {
        "precision": "int8-full",
        "model": _weight("Ensemble_AGCM_LE_int8_full.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_full_nohg.pt"),
    },
    "INT8 Full (QAT)": {
        "precision": "int8-full",
        "model": _weight("Ensemble_AGCM_LE_int8_full_qat.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_full_qat_nohg.pt"),
    },
}


def _select_model_path(precision_key: str, use_hg: bool) -> str:
    cfg = PRECISIONS.get(precision_key, {})
    model_path = cfg.get("model", "")
    if cfg.get("precision", "").startswith("int8") and not use_hg:
        model_path = cfg.get("model_nohg") or model_path
    return model_path


def _precision_is_available(precision_key: str) -> bool:
    cfg = PRECISIONS.get(precision_key, {})
    model_path = cfg.get("model")
    if model_path and os.path.isfile(model_path):
        return True
    if cfg.get("precision", "").startswith("int8"):
        alt_path = cfg.get("model_nohg")
        if alt_path and os.path.isfile(alt_path):
            return True
    return False


def _available_precision_keys() -> list[str]:
    keys = [k for k in PRECISIONS.keys() if _precision_is_available(k)]
    return keys or list(PRECISIONS.keys())

MAX_W, MAX_H = 1920, 1080

# ── Resolution-scale presets (process lower resolution) ──
RESOLUTION_SCALES = {
    "1080p": None,            # full output resolution path (no upscale stage)
    "720p":  (1280, 720),
    "540p":  (960,  540),
}



# ╔═══════════════════════════════════════════════════════════════╗
# ║  mpv HDR Display (named-pipe writer + embedded widget)        ║
# ╚═══════════════════════════════════════════════════════════════╝

class MpvHDRWidget(QWidget):
    """QWidget that embeds an mpv player for real-time HDR frame display.

    Feeds raw RGB48LE frames to mpv through a **Windows named pipe** so
    that the heavy I/O runs inside mpv/FFmpeg's own threads — completely
    off the Python GIL.  The previous stream-protocol approach routed
    every ``read()`` through Python, causing ~200 GIL round-trips per
    frame and destroying throughput.

    Architecture
    ------------
    ::

        feeder thread                   mpv (internal threads)
        ────────────                    ──────────────────────
        hdr_queue.get()  ──▶  pipe  ──▶  FFmpeg rawvideo demuxer
                                       ──▶  gpu vo  ──▶  D3D11 HDR

    * Tags input as BT.2020 / PQ so mpv can either pass-through on
      HDR displays or tone-map automatically on SDR displays.
    """

    hdr_info_ready = pyqtSignal(dict)    # emitted once VO params are populated

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        self.setStyleSheet("background: #111;")
        self._player = None
        self._pipe_name: str | None = None
        self._pipe_handle = None      # win32 HANDLE
        self._feeder: threading.Thread | None = None
        self._queue: _queue.Queue | None = None
        self._shutdown = threading.Event()
        self._fps = 30.0
        self._force_hdr_metadata = True
        self._diag_enabled = _MPV_DIAG
        self._last_playback_cfg: dict | None = None
        self._seek_warned = False

    def _build_hdr_info_snapshot(self, p) -> tuple[dict, dict, dict]:
        """Collect current mpv HDR-related properties for UI/diagnostics."""
        vp = {}
        vop = {}
        try:
            vp = p.video_params or {}
        except Exception:
            pass
        try:
            vop = p.video_out_params or {}
        except Exception:
            pass

        def _g(d, *keys):
            for k in keys:
                v = d.get(k)
                if v is not None:
                    return str(v)
            return '?'

        def _prop(name):
            try:
                v = getattr(p, name.replace('-', '_'), None)
                return str(v) if v is not None else '?'
            except Exception:
                return '?'

        out_prim = _g(vop, 'primaries', 'colormatrix-primaries')
        out_trc = _g(vop, 'gamma', 'transfer')
        out_lvl = _g(vop, 'levels', 'colorlevels')
        t_trc = _prop('target_trc')
        t_prim = _prop('target_prim')

        hdr_metadata_forced = ('pq' in t_trc and 'bt.2020' in t_prim)
        hdr_vo_confirmed = ('bt.2020' in out_prim and 'pq' in out_trc)
        hdr_vo_unknown = (out_prim == '?' or out_trc == '?')
        hdr_info = {
            "primaries": out_prim,
            "transfer": out_trc,
            "levels": out_lvl,
            "sig_peak": _g(vop, 'sig-peak', 'sig_peak'),
            "max_cll": _g(vop, 'max-cll', 'max_cll'),
            "max_fall": _g(vop, 'max-fall', 'max_fall'),
            "vo": _prop('current_vo'),
            "gpu_api": "d3d11",
            "hdr_metadata_forced": hdr_metadata_forced,
            "hdr_vo_confirmed": hdr_vo_confirmed,
            "hdr_vo_unknown": hdr_vo_unknown,
            "hdr_active": hdr_metadata_forced and (hdr_vo_confirmed or hdr_vo_unknown),
            "target_trc": t_trc,
            "target_prim": t_prim,
        }
        aux = {"vp": vp, "vop": vop}
        props = {"t_trc": t_trc, "t_prim": t_prim}
        return hdr_info, aux, props

    def _set_glsl_shaders(self, shader_paths: list[str]) -> bool:
        """Apply a list of GLSL shaders to mpv with a robust setter."""
        p = self._player
        if p is None:
            return False
        paths = _normalize_shader_paths(shader_paths)
        try:
            try:
                p.glsl_shaders = paths
                return True
            except Exception:
                pass
            joined = os.pathsep.join(paths)
            p.command("set", "glsl-shaders", joined)
            return True
        except Exception as exc:
            self._last_scale_error = str(exc)
            print(f"[mpv] glsl-shaders set failed: {exc}")
            return False

    @staticmethod
    def _kernel_antiring(scale_kernel: str) -> tuple[str, float]:
        """Return normalized kernel name and anti-ringing strength."""
        k = str(scale_kernel or "bicubic").strip().lower()
        if not k:
            k = "bicubic"
        if k in {"fsr"}:
            return "fsr", 0.0
        if k in {"ssim_superres", "ssim"}:
            # SSimSuperRes is a shader; use a neutral base scaler.
            return "spline36", 0.0
        # Slightly stronger antiring for lanczossharp to tame halos.
        if k in {"ewa_lanczossharp", "ewa_lanczos"}:
            return "ewa_lanczossharp", 0.20
        return k, 0.0

    def _attach_audio_async(self, audio_path: str):
        """Attach external audio after mpv is fully initialized."""
        def _worker():
            import time as _t
            last_exc = None
            for _ in range(15):  # ~3 seconds total
                if self._shutdown.is_set():
                    return
                p = self._player
                if p is None:
                    return
                try:
                    p.command("audio-add", audio_path, "select")
                    # Give mpv a moment to update selected track state.
                    _t.sleep(0.05)
                    aid = getattr(p, "aid", None)
                    if aid not in (None, "no"):
                        print(f"[mpv audio] attached (aid={aid})")
                        return
                except Exception as exc:
                    last_exc = exc
                _t.sleep(0.2)
            if last_exc is not None:
                print(f"[mpv audio] attach failed after retries: {last_exc}")
            else:
                print("[mpv audio] attach failed: no active audio track")
        threading.Thread(target=_worker, daemon=True).start()

    # ── pipe writer (runs on dedicated thread, no GIL contention) ──

    @staticmethod
    def _pipe_feeder_fn(pipe_name: str, frame_queue: _queue.Queue,
                        shutdown: threading.Event):
        """Open the named pipe server, accept mpv's connection, and
        shovel frame bytes into it.  Runs off the main / inference
        threads entirely."""
        import ctypes
        import ctypes.wintypes as wt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        PIPE_ACCESS_OUTBOUND = 0x00000002
        PIPE_TYPE_BYTE       = 0x00000000
        PIPE_WAIT            = 0x00000000
        INVALID_HANDLE_VALUE = wt.HANDLE(-1).value
        PIPE_BUF             = 1 << 22       # 4 MB kernel buffer

        h = kernel32.CreateNamedPipeW(
            pipe_name,
            PIPE_ACCESS_OUTBOUND,
            PIPE_TYPE_BYTE | PIPE_WAIT,
            1,            # max instances
            PIPE_BUF,     # out buffer
            0,            # in buffer
            0,            # default timeout
            None,         # default security
        )
        if h == INVALID_HANDLE_VALUE:
            return

        # Block until mpv connects (it will open the pipe as a file)
        kernel32.ConnectNamedPipe(h, None)

        written = wt.DWORD(0)
        while not shutdown.is_set():
            try:
                item = frame_queue.get(timeout=0.2)
            except _queue.Empty:
                continue
            if item is None:
                break
            buf = bytes(item) if not isinstance(item, bytes) else item
            off = 0
            while off < len(buf):
                chunk = buf[off:off + PIPE_BUF]
                ok = kernel32.WriteFile(
                    h,
                    chunk,
                    len(chunk),
                    ctypes.byref(written),
                    None,
                )
                if not ok:
                    shutdown.set()
                    break
                off += written.value

        kernel32.FlushFileBuffers(h)
        kernel32.DisconnectNamedPipe(h)
        kernel32.CloseHandle(h)

    # ── public API ───────────────────────────────────────────

    def start_playback(self, width: int, height: int, fps: float = 30.0,
                       scale_kernel: str = "bicubic",
                       scale_antiring: float | None = None,
                       cas_strength: float | None = None,
                       audio_path: str | None = None,
                       film_grain: bool = False,
                       force_hdr_metadata: bool = True):
        """Create an mpv instance and begin reading frames."""
        self.stop_playback()
        self._shutdown.clear()
        self._queue = _queue.Queue(maxsize=1)
        self._fps = float(fps) if fps and fps > 0 else 30.0
        self._force_hdr_metadata = bool(force_hdr_metadata)
        # Keep verbose diagnostics only for HDR-tagged output path.
        self._diag_enabled = bool(_MPV_DIAG and self._force_hdr_metadata)
        kernel_name, antiring = self._kernel_antiring(scale_kernel)
        if scale_antiring is not None:
            antiring = max(0.0, min(1.0, float(scale_antiring)))

        self._last_playback_cfg = {
            "width": int(width),
            "height": int(height),
            "fps": float(self._fps),
            "scale_kernel": str(kernel_name),
            "scale_antiring": None if scale_antiring is None else float(scale_antiring),
            "cas_strength": None if cas_strength is None else float(cas_strength),
            "audio_path": audio_path,
            "film_grain": bool(film_grain),
            "force_hdr_metadata": self._force_hdr_metadata,
        }

        # Unique per-widget named pipe
        pipe_id = id(self)
        self._pipe_name = rf"\\.\pipe\hdrtvnet_mpv_{pipe_id}"

        # Start pipe server thread FIRST so it's listening when mpv opens
        self._feeder = threading.Thread(
            target=self._pipe_feeder_fn,
            args=(self._pipe_name, self._queue, self._shutdown),
            daemon=True,
        )
        self._feeder.start()

        wid = str(int(self.winId()))
        pipe_url = f"lavf://file:{self._pipe_name}"

        # Keep demuxer buffer at ~1 frame to minimize display latency.
        # Must be >0 or mpv can't buffer even 1 frame.
        frame_bytes = width * height * 6          # RGB48LE
        max_demux = str(frame_bytes)

        use_fsr = (kernel_name == "fsr" and _ensure_fsr_shader())
        if kernel_name == "fsr" and not use_fsr:
            print(f"[mpv] FSR shader unavailable (download failed). "
                  f"Falling back to {BEST_MPV_SCALE}.")
            kernel_name = BEST_MPV_SCALE
            if self._last_playback_cfg is not None:
                self._last_playback_cfg["scale_kernel"] = str(kernel_name)
        use_ssim = (kernel_name == "ssim_superres" and _ensure_ssim_superres_shader())
        if kernel_name == "ssim_superres" and not use_ssim:
            print("[mpv] SSimSuperRes shader unavailable (download failed). "
                  f"Falling back to {BEST_MPV_SCALE}.")
            kernel_name = BEST_MPV_SCALE
            if self._last_playback_cfg is not None:
                self._last_playback_cfg["scale_kernel"] = str(kernel_name)
        use_fsr = (kernel_name == "fsr")
        use_ssim = (kernel_name == "ssim_superres")
        use_film_grain = bool(film_grain and _ensure_filmgrain_shader())
        if film_grain and not use_film_grain:
            print("[mpv] film grain shader unavailable (download failed).")
            if self._last_playback_cfg is not None:
                self._last_playback_cfg["film_grain"] = False

        mpv_kwargs = dict(
            wid=wid,
            vo="gpu",
            gpu_api="d3d11",                # d3d11 required for HDR on Windows
            demuxer="rawvideo",
            demuxer_rawvideo_w=str(width),
            demuxer_rawvideo_h=str(height),
            demuxer_rawvideo_mp_format="rgb48le",
            demuxer_rawvideo_fps=str(self._fps),
            untimed=True,                   # present frames as soon as fed
            audio="auto",
            audio_file_auto="no",
            osc="no",
            input_default_bindings="no",
            input_vo_keyboard="no",
            # ── Minimal-buffer passthrough ──
            cache="no",
            demuxer_max_bytes=max_demux,    # ~2 frames (not 0!)
            demuxer_readahead_secs=0,
            video_sync="desync",            # avoid VO clock drift vs worker
            scale="bilinear" if use_fsr else str(kernel_name),
            cscale="bilinear" if use_fsr else str(kernel_name),
            scale_antiring=str(antiring),
            cscale_antiring=str(antiring),
        )
        shader_paths = []
        if use_fsr:
            shader_paths.append(FSR_SHADER_PATH)
        if use_ssim:
            shader_paths.append(SSIM_SUPERRES_SHADER_PATH)
        if use_film_grain:
            shader_paths.append(FILMGRAIN_SHADER_PATH)
        if self._force_hdr_metadata:
            vf_chain = "format=colorlevels=full:primaries=bt.2020:gamma=pq"
            if cas_strength is not None and cas_strength > 0.0:
                vf_chain += f",cas={cas_strength}"
            mpv_kwargs.update(
                # HDR path: declare input as BT.2020/PQ, but do not force
                # output target. mpv will auto-detect display capabilities.
                target_colorspace_hint="no",
                vf=vf_chain,
            )
        else:
            vf_chain = "format=colorlevels=full:primaries=bt.709:gamma=bt.1886"
            if cas_strength is not None and cas_strength > 0.0:
                vf_chain += f",cas={cas_strength}"
            # SDR path: tag input as Rec.709 and let mpv choose output target.
            mpv_kwargs.update(
                target_colorspace_hint="no",
                vf=vf_chain,
            )
        use_external_audio = bool(audio_path and os.path.isfile(audio_path))
        if not use_external_audio:
            mpv_kwargs["audio"] = "no"

        player = mpv_lib.MPV(**mpv_kwargs)

        self._player = player
        player.play(pipe_url)
        if shader_paths:
            self._set_glsl_shaders(shader_paths)
        if use_external_audio:
            # Some python-mpv builds don't expose --audio-file as an init
            # option; runtime audio-add is broadly compatible.
            self._attach_audio_async(audio_path)

        # Poll HDR state continuously for UI updates and fallback recovery.
        def _hdr_monitor():
            import time as _t
            printed_once = False
            while not self._shutdown.is_set():
                _t.sleep(0.5)
                p = self._player
                if p is None:
                    return
                try:
                    hdr_info, aux, props = self._build_hdr_info_snapshot(p)
                    self.hdr_info_ready.emit(hdr_info)

                    if self._diag_enabled and not printed_once and len(aux.get("vop", {})) > 2:
                        print("\n╔══════════ mpv HDR diagnostic ══════════╗")
                        print(f"║  video-params keys : {list(aux['vp'].keys())}")
                        print(f"║  video-out-params  : {list(aux['vop'].keys())}")
                        print(f"║  VO output prims   : {hdr_info.get('primaries', '?')}")
                        print(f"║  VO output TRC     : {hdr_info.get('transfer', '?')}")
                        print(f"║  VO output levels  : {hdr_info.get('levels', '?')}")
                        print(f"║  target-trc        : {props.get('t_trc', '?')}")
                        print(f"║  target-prim       : {props.get('t_prim', '?')}")
                        print(f"║  target-peak       : {getattr(p, 'target_peak', '?')}")
                        print(f"║  current-vo        : {getattr(p, 'current_vo', '?')}")
                        print("║  gpu-api           : d3d11 (forced)")
                        hint_mode = "yes (forced)" if self._force_hdr_metadata else "no"
                        print(f"║  colorspace-hint   : {hint_mode}")
                        print("╠════════════════════════════════════════╣")
                        if hdr_info.get("hdr_vo_confirmed", False):
                            print("║  ✓ VO confirms BT.2020 + PQ output")
                        else:
                            print("║  ⚠ VO not yet confirming BT.2020 + PQ")
                        print("╚════════════════════════════════════════╝\n")
                        printed_once = True
                except Exception:
                    pass

        threading.Thread(target=_hdr_monitor, daemon=True).start()

    def feed_frame(self, rgb48_bytes):
        """Push one frame (raw RGB48LE bytes or buffer) for display."""
        q = self._queue
        if q is None:
            return
        try:
            q.put_nowait(rgb48_bytes)
        except _queue.Full:
            try:
                q.get_nowait()
            except _queue.Empty:
                pass
            try:
                q.put_nowait(rgb48_bytes)
            except _queue.Full:
                pass

    def set_paused(self, paused: bool):
        """Pause/resume mpv playback (audio + timeline)."""
        p = self._player
        if p is None:
            return
        try:
            p.pause = bool(paused)
        except Exception:
            pass

    def seek_seconds(self, sec: float):
        """Seek playback timeline to absolute seconds."""
        p = self._player
        if p is None:
            return
        try:
            p.command("seek", max(0.0, float(sec)), "absolute")
        except Exception:
            if not self._seek_warned:
                self._seek_warned = True
                # Seek may fail briefly during mpv init; ignore silently.
                pass

    def set_muted(self, muted: bool):
        """Mute/unmute mpv audio."""
        p = self._player
        if p is None:
            return
        try:
            p.mute = bool(muted)
        except Exception:
            try:
                p.command("set", "mute", "yes" if muted else "no")
            except Exception:
                pass

    def set_volume_percent(self, volume_percent: int):
        """Set mpv audio volume in 0..100 range."""
        p = self._player
        if p is None:
            return
        v = max(0, min(100, int(volume_percent)))
        try:
            p.volume = v
        except Exception:
            try:
                p.command("set", "volume", str(v))
            except Exception:
                pass

    def set_audio_track_ordinal(self, ordinal: int) -> bool:
        """Select Nth audio track (0-based by appearance order) in mpv."""
        p = self._player
        if p is None:
            return False
        try:
            tracks = getattr(p, "track_list", None)
        except Exception:
            tracks = None
        if isinstance(tracks, list):
            aud = [t for t in tracks if str(t.get("type", "")).lower() == "audio"]
            if aud:
                idx = max(0, min(int(ordinal), len(aud) - 1))
                aid = aud[idx].get("id")
                if aid is not None:
                    try:
                        p.command("set", "aid", str(aid))
                        return True
                    except Exception:
                        pass
        # Fallback: mpv audio IDs are usually 1-based in track order.
        try:
            p.command("set", "aid", str(max(1, int(ordinal) + 1)))
            return True
        except Exception:
            return False

    def set_scale_kernel(self, scale_kernel: str,
                         scale_antiring: float | None = None) -> bool:
        """Update mpv luma/chroma scaling kernels at runtime."""
        p = self._player
        if p is None:
            return False
        self._last_scale_error = None
        requested_kernel = str(scale_kernel or "").strip().lower()
        kernel, antiring = self._kernel_antiring(scale_kernel)
        if scale_antiring is not None:
            antiring = max(0.0, min(1.0, float(scale_antiring)))
        film_on = False
        if self._last_playback_cfg is not None:
            film_on = bool(self._last_playback_cfg.get("film_grain", False))
        try:
            use_fsr = (kernel == "fsr" and _ensure_fsr_shader())
            if kernel == "fsr" and not use_fsr:
                kernel = BEST_MPV_SCALE
            use_ssim = (requested_kernel == "ssim_superres" and _ensure_ssim_superres_shader())
            if requested_kernel == "ssim_superres" and not use_ssim:
                use_ssim = False
            if use_fsr:
                p.command("set", "scale", "bilinear")
                p.command("set", "cscale", "bilinear")
            else:
                p.command("set", "scale", kernel)
                p.command("set", "cscale", kernel)
            use_film_grain = bool(film_on and _ensure_filmgrain_shader())
            if film_on and not use_film_grain:
                print("[mpv] film grain shader unavailable (download failed).")
            shader_paths = []
            if use_fsr:
                shader_paths.append(FSR_SHADER_PATH)
            if use_ssim:
                shader_paths.append(SSIM_SUPERRES_SHADER_PATH)
            if use_film_grain:
                shader_paths.append(FILMGRAIN_SHADER_PATH)
            if not self._set_glsl_shaders(shader_paths):
                raise RuntimeError("Failed to set glsl-shaders.")
            p.command("set", "scale-antiring", str(antiring))
            p.command("set", "cscale-antiring", str(antiring))
            if self._last_playback_cfg is not None:
                if use_ssim:
                    self._last_playback_cfg["scale_kernel"] = "ssim_superres"
                else:
                    self._last_playback_cfg["scale_kernel"] = str(kernel)
                self._last_playback_cfg["scale_antiring"] = float(antiring)
                self._last_playback_cfg["film_grain"] = bool(use_film_grain)
            return True
        except Exception as exc:
            self._last_scale_error = str(exc)
            print(f"[mpv] scale hot-swap failed: {exc}")
            return False

    def set_cas_strength(self, cas_strength: float | None) -> bool:
        """Update mpv CAS filter at runtime."""
        p = self._player
        if p is None:
            return False
        try:
            cas_val = float(cas_strength or 0.0)
        except Exception:
            cas_val = 0.0
        if self._force_hdr_metadata:
            vf_chain = "format=colorlevels=full:primaries=bt.2020:gamma=pq"
        else:
            vf_chain = "format=colorlevels=full:primaries=bt.709:gamma=bt.1886"
        if cas_val > 0.0:
            vf_chain += f",cas={cas_val}"
        try:
            p.command("set", "vf", vf_chain)
            if self._last_playback_cfg is not None:
                self._last_playback_cfg["cas_strength"] = float(cas_val)
            return True
        except Exception as exc:
            self._last_scale_error = str(exc)
            print(f"[mpv] cas hot-swap failed: {exc}")
            return False

    def set_film_grain(self, enabled: bool) -> bool:
        """Enable/disable film grain shader at runtime."""
        p = self._player
        if p is None:
            return False
        use_film = bool(enabled and _ensure_filmgrain_shader())
        if enabled and not use_film:
            print("[mpv] film grain shader unavailable (download failed).")
        use_fsr = False
        if self._last_playback_cfg is not None:
            k = str(self._last_playback_cfg.get("scale_kernel", "")).lower()
            use_fsr = (k == "fsr" and _ensure_fsr_shader())
        shader_paths = []
        if use_fsr:
            shader_paths.append(FSR_SHADER_PATH)
        if use_film:
            shader_paths.append(FILMGRAIN_SHADER_PATH)
        try:
            if not self._set_glsl_shaders(shader_paths):
                raise RuntimeError("Failed to set glsl-shaders.")
            if self._last_playback_cfg is not None:
                self._last_playback_cfg["film_grain"] = bool(use_film)
            return True
        except Exception as exc:
            self._last_scale_error = str(exc)
            print(f"[mpv] film grain hot-swap failed: {exc}")
            return False

    def is_fsr_active(self) -> bool:
        """Return True if FSR shader is currently active in mpv."""
        p = self._player
        if p is None:
            return False
        try:
            val = getattr(p, "glsl_shaders", None)
            if not val:
                return False
            if isinstance(val, (list, tuple)):
                return any("fsr" in str(v).lower() for v in val)
            return "fsr" in str(val).lower()
        except Exception:
            return False

    def get_time_seconds(self) -> float | None:
        """Return mpv playback clock (seconds), if available."""
        p = self._player
        if p is None:
            return None
        for prop in ("time_pos", "playback_time"):
            try:
                v = getattr(p, prop, None)
                if v is not None:
                    return float(v)
            except Exception:
                pass
        return None

    def stop_playback(self):
        """Terminate mpv and clean up."""
        self._shutdown.set()
        q = self._queue
        if q is not None:
            try:
                q.put_nowait(None)          # EOF sentinel
            except _queue.Full:
                pass
        if self._player is not None:
            try:
                self._player.terminate()
            except Exception:
                pass
            self._player = None
        if self._feeder is not None:
            self._feeder.join(timeout=3)
            self._feeder = None
        self._queue = None

    def refresh_surface(self):
        """Recreate mpv VO using the last playback config after window-state changes."""
        cfg = self._last_playback_cfg
        if not cfg:
            return
        # Defensive copy: ignore stale/unknown keys from older runs.
        allowed = {"width", "height", "fps", "scale_kernel", "scale_antiring",
                   "cas_strength",
                   "audio_path", "force_hdr_metadata", "film_grain"}
        safe_cfg = {k: v for k, v in cfg.items() if k in allowed}
        self.start_playback(**safe_cfg)


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Pipeline Worker Thread                                       ║
# ╚═══════════════════════════════════════════════════════════════╝

class PipelineWorker(QThread):
    """Runs SDR→HDR inference in a background thread.

    Signals
    -------
    frame_ready(sdr_bgr, hdr_bgr)
        Emitted per processed frame with numpy uint8 BGR arrays.
    metrics_updated(dict)
        Periodic performance counters.
    status_message(str)
        Human-readable status for the status bar.
    playback_finished()
        Video EOF reached or stopped.
    """

    frame_ready = pyqtSignal(np.ndarray, np.ndarray)
    metrics_updated = pyqtSignal(dict)
    status_message = pyqtSignal(str)
    playback_finished = pyqtSignal()
    compile_ready = pyqtSignal()          # emitted after warmup_compile finishes
    position_updated = pyqtSignal(int, int)  # (current_frame, total_frames)
    seek_frame_ready = pyqtSignal(int)    # emitted after first rendered frame post-seek

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path = None
        self._precision_key = "FP16"
        self._use_hg = True
        self._processor = None
        self._stop_flag = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused
        self._pending_precision = None
        self._mpv_widget: MpvHDRWidget | None = None
        self._sdr_mpv_widget: MpvHDRWidget | None = None
        self._hdr_queue: _queue.Queue | None = None
        self._sdr_queue: _queue.Queue | None = None
        self._hdr_thread: threading.Thread | None = None
        self._sdr_thread: threading.Thread | None = None
        self._proc_w: int = MAX_W
        self._proc_h: int = MAX_H
        self._output_w: int = MAX_W       # display / mpv resolution
        self._output_h: int = MAX_H
        self._sdr_visible: bool = True   # toggled by main thread
        self._seek_frame: int | None = None   # pending seek request
        self._user_paused: bool = False          # True while user has paused
        self._source: VideoSource | None = None  # ref for seeking
        self._pending_resolution: tuple[int, int] | None = None
        self._enh_prev_luma: torch.Tensor | None = None
        self._enh_temporal_detail: torch.Tensor | None = None
        self._sobel_x: torch.Tensor | None = None
        self._sobel_y: torch.Tensor | None = None
        self._input_is_hdr: bool = False
        self._sdr_delay_frame: np.ndarray | None = None
        self._hdr_drop_until_frame: int = 0
        self._sdr_drop_until_frame: int = 0
        self._frame_idx: int = 0
        self._hold_until_t: float = 0.0
        # App-level dedicated GPU memory (VRAM) from Windows perf counters.
        self._app_vram_mb: float = 0.0
        self._app_vram_poll_stop = threading.Event()
        self._app_vram_poll_thread: threading.Thread | None = None
        self._realtime_drop_frames: int = 0

    # ── public API (called from main thread) ──

    def configure(self, video_path, precision_key, proc_w=MAX_W, proc_h=MAX_H,
                  output_w=MAX_W, output_h=MAX_H, input_is_hdr=False,
                  use_hg=True):
        self._video_path = video_path
        self._precision_key = precision_key
        self._proc_w = proc_w
        self._proc_h = proc_h
        self._output_w = output_w
        self._output_h = output_h
        self._reset_enhance_history()
        self._input_is_hdr = bool(input_is_hdr)
        self._use_hg = bool(use_hg)
        self._realtime_drop_frames = 0
        # Drop a couple of frames on startup to avoid mpv buffer lag.
        self._hdr_drop_until_frame = 2
        self._sdr_drop_until_frame = 2
        self._hold_until_t = time.perf_counter() + 0.5

    def request_precision_change(self, key):
        self._pending_precision = key

    def request_resolution_change(self, proc_w: int, proc_h: int):
        """Request a processing-resolution change (thread-safe)."""
        self._pending_resolution = (proc_w, proc_h)

    def flush_hdr_queue(self, drop_frames: int = 2):
        """Flush mpv queues and drop a couple of frames to re-align output."""
        if self._hdr_queue is not None:
            try:
                while True:
                    self._hdr_queue.get_nowait()
            except _queue.Empty:
                pass
        if self._sdr_queue is not None:
            try:
                while True:
                    self._sdr_queue.get_nowait()
            except _queue.Empty:
                pass
        self._hdr_drop_until_frame = max(self._hdr_drop_until_frame,
                                         self._frame_idx + max(0, int(drop_frames)))
        self._sdr_drop_until_frame = max(self._sdr_drop_until_frame,
                                         self._frame_idx + max(0, int(drop_frames)))

    def _reset_enhance_history(self):
        self._enh_prev_luma = None
        self._enh_temporal_detail = None

    @staticmethod
    def _box_blur(x: torch.Tensor, k: int = 3) -> torch.Tensor:
        p = k // 2
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=p)

    def _ensure_sobel_kernels(self, device: torch.device, dtype: torch.dtype):
        if self._sobel_x is not None and self._sobel_x.device == device and self._sobel_x.dtype == dtype:
            return
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)
        self._sobel_x = kx
        self._sobel_y = ky

    def _enhance_best_gpu(self, t_rgb: torch.Tensor) -> torch.Tensor:
        """GPU enhancement pass on luma only (preserves chroma/hue)."""
        linear = torch.clamp(t_rgb, 0.0, 1.0)
        gray = (
            0.2126 * linear[:, 0:1, :, :]
            + 0.7152 * linear[:, 1:2, :, :]
            + 0.0722 * linear[:, 2:3, :, :]
        )
        # Tiny pre-blur (sigma-like ~0.35-0.45) to suppress aliasing before directional enhancement.
        gray_pre = gray * 0.72 + self._box_blur(gray, 3) * 0.28

        self._ensure_sobel_kernels(linear.device, linear.dtype)
        gx = F.conv2d(gray_pre, self._sobel_x, padding=1)
        gy = F.conv2d(gray_pre, self._sobel_y, padding=1)
        edge_strength = torch.sqrt(gx * gx + gy * gy + 1e-8)
        edge_strength = edge_strength / (edge_strength + 0.08)

        low = self._box_blur(gray_pre, 3)
        high_freq = (gray_pre - low).abs()
        high_freq = high_freq / (high_freq + 0.03)
        adapt = torch.clamp(0.7 * edge_strength + 0.3 * high_freq, 0.0, 1.0)

        # Channel-wise clamp envelope preserves hue/chroma much better than luma-only clamp.
        min_c = -F.max_pool2d(-linear, kernel_size=3, stride=1, padding=1)
        max_c = F.max_pool2d(linear, kernel_size=3, stride=1, padding=1)

        detail = gray_pre - self._box_blur(gray_pre, 3)
        luma = gray_pre
        if self._enh_prev_luma is None or self._enh_prev_luma.shape != luma.shape:
            accum = detail
        else:
            motion = (luma - self._enh_prev_luma).abs()
            motion_w = torch.clamp(1.0 - motion * 6.0, 0.0, 1.0)
            if self._enh_temporal_detail is None or self._enh_temporal_detail.shape != detail.shape:
                accum = detail
            else:
                accum = self._enh_temporal_detail * motion_w + detail * (1.0 - motion_w)
        self._enh_prev_luma = luma.detach()
        self._enh_temporal_detail = accum.detach()

        base = self._box_blur(gray_pre, 3)
        grain = gray_pre - base
        lap = gray_pre - self._box_blur(gray_pre, 5)

        y_out = gray_pre + 0.075 * accum + grain * (0.040 + 0.030 * adapt) + (0.042 * adapt) * lap
        y_min = -F.max_pool2d(-gray, kernel_size=3, stride=1, padding=1)
        y_max = F.max_pool2d(gray, kernel_size=3, stride=1, padding=1)
        y_out = torch.minimum(torch.maximum(y_out, y_min), y_max)
        # Gentle edge anti-aliasing to reduce stair-step text edges.
        y_smooth = self._box_blur(y_out, 3)
        aa_mask = torch.clamp((edge_strength - 0.045) / 0.155, 0.0, 1.0)
        y_out = y_out * (1.0 - 0.58 * aa_mask) + y_smooth * (0.58 * aa_mask)
        # Extra smoothing for text-like, high-contrast near-binary edges only.
        local_contrast = torch.clamp(y_max - y_min, 0.0, 1.0)
        near_extreme = 1.0 - torch.clamp(torch.minimum(gray, 1.0 - gray) / 0.22, 0.0, 1.0)
        text_like = torch.clamp((edge_strength - 0.10) / 0.22, 0.0, 1.0)
        text_like = text_like * torch.clamp((local_contrast - 0.18) / 0.25, 0.0, 1.0) * near_extreme
        y_text = self._box_blur(y_out, 3)
        y_out = y_out * (1.0 - 0.47 * text_like) + y_text * (0.47 * text_like)
        # Recover a bit of high-contrast text edge clarity after AA smoothing.
        text_mask = torch.clamp((edge_strength - 0.10) / 0.22, 0.0, 1.0)
        text_mask = text_mask * torch.clamp((local_contrast - 0.18) / 0.25, 0.0, 1.0)
        text_mask = text_mask * (1.0 - 0.85 * text_like)
        text_unsharp = y_out - self._box_blur(y_out, 3)
        y_out = y_out + 0.07 * text_mask * text_unsharp
        y_out = torch.minimum(torch.maximum(y_out, y_min), y_max)
        # Conservative blend to keep colorimetry close to model output.
        y_out = gray_pre * 0.83 + y_out * 0.17
        y_out = torch.clamp(y_out, 0.0, 1.0)

        # Apply luminance change as additive delta (safer for text halos than gain scaling).
        delta_y = y_out - gray
        # Suppress highlight bloom: reduce enhancement near near-white luma.
        hi = torch.clamp((gray - 0.75) / 0.20, 0.0, 1.0)
        delta_y = delta_y * (1.0 - 0.88 * hi)
        out = torch.clamp(linear + delta_y, 0.0, 1.0)
        # Final per-channel safety clamp.
        out = torch.minimum(torch.maximum(out, min_c), max_c)
        return out

    def set_mpv_widget(self, widget):
        """Set the MpvHDRWidget reference for feeding HDR frames."""
        self._mpv_widget = widget

    def set_sdr_mpv_widget(self, widget):
        """Set the MpvHDRWidget reference for feeding SDR frames."""
        self._sdr_mpv_widget = widget

    def set_sdr_visible(self, visible: bool):
        """Tell the worker whether the SDR QLabel is on-screen."""
        self._sdr_visible = visible

    def request_seek(self, frame_number: int):
        """Request a seek to a specific frame (thread-safe)."""
        self._seek_frame = frame_number
        self._pause_event.set()  # unblock so the loop can process the seek

    def pause(self):
        self._user_paused = True
        self._pause_event.clear()

    def resume(self):
        self._user_paused = False
        self._pause_event.set()

    def stop(self):
        self._stop_flag = True
        self._user_paused = False
        self._pause_event.set()  # unblock if paused

    @property
    def is_paused(self):
        return self._user_paused

    # ── model management ──

    @staticmethod
    def _silent_warmup(processor, w, h):
        """Run warmup_compile with **all** output suppressed.

        PyTorch's AUTOTUNE prints via C-level I/O that bypasses
        Python's ``sys.stdout``.  We redirect the real OS file
        descriptors (fd 1 / fd 2) to ``os.devnull`` **and** swap
        the Python-level ``sys.stdout`` / ``sys.stderr`` so that
        both C-level and Python-level writes succeed on Windows.
        """
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        try:
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
            processor.warmup_compile(w, h)
        finally:
            sys.stdout.close()
            sys.stderr.close()
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr
            os.close(devnull_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

    def _load_model(self, key):
        cfg = PRECISIONS.get(key, {})
        path = _select_model_path(key, self._use_hg)
        if not os.path.isfile(path):
            if key in PRECISIONS and PRECISIONS[key].get("precision", "").startswith("int8"):
                alt = PRECISIONS[key].get("model_nohg" if self._use_hg else "model")
                note = ""
                if alt and alt != path:
                    note = f" (alt: {alt})"
                self.status_message.emit(
                    f"ERROR: weights not found — {path}{note}"
                )
            else:
                self.status_message.emit(f"ERROR: weights not found — {path}")
            return False

        cw, ch = self._proc_w, self._proc_h

        self.status_message.emit(f"Loading model: {key} ...")

        # Free previous model
        if self._processor is not None:
            del self._processor
            self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Reset ALL torch.compile / Inductor / Dynamo state ──────
        # torch.compile assigns each generated Triton kernel a
        # globally-incrementing name (triton_xxx_N).  On the second
        # model load the counter is already high, so kernel source
        # hashes differ from those the subprocess wrote to the
        # autotune local cache (.best_config files) → full AUTOTUNE
        # re-benchmarking in-process.
        # Resetting Dynamo brings the counter back to zero so the
        # hashes match the subprocess cache exactly.
        try:
            torch._dynamo.reset()
        except (AssertionError, RuntimeError):
            # QThread may lack the CUDA-graph TLS keys that reset()
            # tries to clean up — harmless, no graphs were created.
            pass

        self._processor = HDRTVNetTorch(
            path,
            device="auto",
            precision=cfg["precision"],
            compile_model=True,
            compile_mode="max-autotune",
            predequantize="auto",
            use_hg=self._use_hg,
        )

        # ── Silent in-process warmup ───────────────────────────────
        # The subprocess (compile_kernels.py) already ran max-autotune
        # and cached .best_config files on disk.  After the dynamo
        # reset above, the kernel naming counter starts from zero —
        # matching the subprocess — so the autotune local cache hits
        # and no GPU benchmarking is needed (just code-gen, ~5-10 s).
        # Output is suppressed at the OS fd level so nothing leaks
        # even from C-level AUTOTUNE prints.
        # Runs BEFORE compile_ready (before mpv), so GPU is clean.
        self.status_message.emit(
            f"Warming up kernels for {cw}×{ch} ({key}) …"
        )
        self._silent_warmup(self._processor, cw, ch)

        self._precision_key = key
        self.status_message.emit(f"Ready — {key}")
        return True

    @staticmethod
    def _tensor_mb(t: torch.Tensor | None) -> float:
        if t is None:
            return 0.0
        try:
            return (t.numel() * t.element_size()) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def _model_memory_breakdown_mb(self) -> tuple[float, float, float]:
        """Return (total_mb, cpu_mb, gpu_mb) for loaded model tensors only.

        This reflects the active quantization model footprint (parameters +
        registered buffers), excluding runtime working/staging tensors.
        """
        p = self._processor
        if p is None:
            return 0.0, 0.0, 0.0

        model = getattr(p, "model", None)
        if model is None:
            return 0.0, 0.0, 0.0
        model = getattr(model, "_orig_mod", model)

        cpu_mb = 0.0
        gpu_mb = 0.0
        seen = set()

        def _accumulate_tensor(t: torch.Tensor | None):
            nonlocal cpu_mb, gpu_mb
            if t is None:
                return
            tid = id(t)
            if tid in seen:
                return
            seen.add(tid)
            mb = self._tensor_mb(t)
            if mb <= 0:
                return
            try:
                dev_type = t.device.type
            except Exception:
                dev_type = "cpu"
            if dev_type == "cuda":
                gpu_mb += mb
            else:
                cpu_mb += mb

        try:
            for t in model.parameters(recurse=True):
                _accumulate_tensor(t)
            for t in model.buffers(recurse=True):
                _accumulate_tensor(t)
        except Exception:
            pass

        total_mb = cpu_mb + gpu_mb
        return total_mb, cpu_mb, gpu_mb

    @staticmethod
    def _query_app_vram_mb_windows(pid: int) -> float | None:
        """Return dedicated GPU process memory (MB) for this process on Windows.

        Uses the Windows GPU performance counter category:
        \\GPU Process Memory(*)\\Dedicated Usage
        """
        if os.name != "nt":
            return None
        try:
            pid_i = int(pid)
        except Exception:
            return None
        if pid_i <= 0:
            return None

        ps_cmd = (
            "$tag='pid_" + str(pid_i) + "'; "
            "$d=(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' "
            "| Select-Object -ExpandProperty CounterSamples "
            "| Where-Object { $_.InstanceName -like \"*$tag*\" } "
            "| Measure-Object -Property CookedValue -Sum).Sum; "
            "$td=if ($null -eq $d) { 0 } else { [double]$d }; "
            "[string]$td"
        )
        try:
            cp = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            raw = (cp.stdout or "").strip()
            if not raw:
                return 0.0
            bytes_used = float(raw)
            return max(0.0, bytes_used / (1024.0 * 1024.0))
        except Exception:
            return None

    def _start_app_vram_poll(self):
        """Poll app VRAM off the render loop to avoid FPS stalls."""
        if os.name != "nt":
            return
        self._app_vram_poll_stop.clear()

        def _poll():
            pid = os.getpid()
            while not self._app_vram_poll_stop.is_set():
                q = self._query_app_vram_mb_windows(pid)
                if q is not None:
                    self._app_vram_mb = q
                self._app_vram_poll_stop.wait(0.5)

        self._app_vram_poll_thread = threading.Thread(target=_poll, daemon=True)
        self._app_vram_poll_thread.start()

    def _stop_app_vram_poll(self):
        self._app_vram_poll_stop.set()
        t = self._app_vram_poll_thread
        if t is not None:
            t.join(timeout=1.0)
        self._app_vram_poll_thread = None

    # ── background HDR feeder (runs on its own thread) ──

    def _hdr_feeder_fn(self, hdr_q: _queue.Queue, mpv_widget: MpvHDRWidget):
        """Drains *hdr_q*, converts GPU fp16 tensors → uint16 RGB48LE,
        and feeds them to the mpv widget.  Runs entirely off the
        inference thread so it never blocks GPU work."""
        while True:
            try:
                item = hdr_q.get(timeout=0.2)
            except _queue.Empty:
                continue
            if item is None:                 # poison pill → exit
                break
            # If producer is ahead, skip stale entries and keep latest frame.
            while True:
                try:
                    newer = hdr_q.get_nowait()
                    if newer is None:
                        item = None
                        break
                    item = newer
                except _queue.Empty:
                    break
            if item is None:
                break
            if isinstance(item, tuple) and len(item) == 2:
                present_t, tensor = item
            else:
                present_t, tensor = None, item
            # tensor is a GPU fp16 tensor (1,3,H,W) — all heavy work here
            with torch.inference_mode():
                raw_cpu = (tensor.squeeze(0)
                           .clamp_(0.0, 1.0)
                           .permute(1, 2, 0)
                           .contiguous()
                           .cpu()
                           .numpy())            # float16 HWC on CPU
            hdr_u16 = (raw_cpu.astype(np.float32).__imul__(65535)
                       ).__add__(0.5)
            np.clip(hdr_u16, 0, 65535, out=hdr_u16)
            if present_t is not None:
                now = time.perf_counter()
                if now < present_t:
                    time.sleep(present_t - now)
            mpv_widget.feed_frame(hdr_u16.astype(np.uint16).data)

    def _sdr_feeder_fn(self, sdr_q: _queue.Queue, mpv_widget: MpvHDRWidget):
        """Drains *sdr_q*, converts uint8 BGR → uint16 RGB48LE,
        and feeds them to the mpv widget on the shared presentation clock."""
        while True:
            try:
                item = sdr_q.get(timeout=0.2)
            except _queue.Empty:
                continue
            if item is None:                 # poison pill → exit
                break
            # If producer is ahead, skip stale entries and keep latest frame.
            while True:
                try:
                    newer = sdr_q.get_nowait()
                    if newer is None:
                        item = None
                        break
                    item = newer
                except _queue.Empty:
                    break
            if item is None:
                break
            if isinstance(item, tuple) and len(item) == 2:
                present_t, frame = item
            else:
                present_t, frame = None, item
            try:
                rgb16 = np.ascontiguousarray(
                    frame[:, :, ::-1].astype(np.uint16) * 257
                )
                if present_t is not None:
                    now = time.perf_counter()
                    if now < present_t:
                        time.sleep(present_t - now)
                mpv_widget.feed_frame(rgb16.data)
            except Exception:
                pass

    def _start_hdr_feeder(self):
        """Spin up background thread for async HDR extraction."""
        self._hdr_queue = _queue.Queue(maxsize=1)
        self._hdr_thread = threading.Thread(
            target=self._hdr_feeder_fn,
            args=(self._hdr_queue, self._mpv_widget),
            daemon=True,
        )
        self._hdr_thread.start()

    def _stop_hdr_feeder(self):
        """Shut down the HDR feeder thread."""
        q = self._hdr_queue
        if q is not None:
            try:
                q.put_nowait(None)      # poison pill
            except _queue.Full:
                try:
                    q.get_nowait()
                except _queue.Empty:
                    pass
                q.put(None)
        t = self._hdr_thread
        if t is not None:
            t.join(timeout=3)
        self._hdr_queue = None
        self._hdr_thread = None

    def _start_sdr_feeder(self):
        """Spin up background thread for async SDR feeding."""
        if self._sdr_mpv_widget is None or self._sdr_queue is not None:
            return
        self._sdr_queue = _queue.Queue(maxsize=1)
        self._sdr_thread = threading.Thread(
            target=self._sdr_feeder_fn,
            args=(self._sdr_queue, self._sdr_mpv_widget),
            daemon=True,
        )
        self._sdr_thread.start()

    def _stop_sdr_feeder(self):
        """Shut down the SDR feeder thread."""
        q = self._sdr_queue
        if q is not None:
            try:
                q.put_nowait(None)      # poison pill
            except _queue.Full:
                try:
                    q.get_nowait()
                except _queue.Empty:
                    pass
                q.put(None)
        t = self._sdr_thread
        if t is not None:
            t.join(timeout=3)
        self._sdr_queue = None
        self._sdr_thread = None

    # ── main loop ──

    def run(self):
        self._stop_flag = False
        self._pending_precision = None

        if self._input_is_hdr:
            # No SDR->HDR inference when source is already HDR.
            if self._processor is not None:
                del self._processor
                self._processor = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.status_message.emit(
                "HDR input detected: bypassing SDR→HDR model conversion."
            )
        else:
            if not self._load_model(self._precision_key):
                self.playback_finished.emit()
                return

        # Signal main thread that compile is done — safe to start mpv now
        self.compile_ready.emit()

        if not self._video_path:
            self.status_message.emit("No video selected.")
            self.playback_finished.emit()
            return

        # Wait a moment for mpv to be started by the main thread
        import time as _t
        for _ in range(20):                  # up to 1 s
            if self._mpv_widget is not None or self._stop_flag:
                break
            _t.sleep(0.05)

        # Start async feeders if mpv is active
        mpv_w = self._mpv_widget
        if mpv_w is not None and not self._input_is_hdr:
            self._start_hdr_feeder()
        if self._sdr_mpv_widget is not None:
            self._start_sdr_feeder()
        self._start_app_vram_poll()

        # Keep prefetch shallow to improve frame-accurate seek behavior.
        source = VideoSource(self._video_path, prefetch=1)
        self._source = source
        total_frames = source.frame_count
        src_fps = source.fps if source.fps and source.fps > 0 else 30.0
        out_fps = _limited_playback_fps(src_fps)
        frame_stride = max(1, int(round(src_fps / out_fps)))
        frame_interval_s = 1.0 / src_fps
        next_frame_t = time.perf_counter()
        frame_times = deque(maxlen=30)
        presented_times = deque(maxlen=30)
        metrics_warmup_frames = 0
        frame_idx = 0
        force_position_emit = False
        seek_frame_ready_pending = False
        process = psutil.Process(os.getpid())
        use_cuda = torch.cuda.is_available()

        # Local copies of resolution settings (may change mid-playback)
        proc_w, proc_h = self._proc_w, self._proc_h
        out_w, out_h = self._output_w, self._output_h
        lower_res_processing = (proc_w != out_w or proc_h != out_h)

        while not self._stop_flag:
            if self._sdr_mpv_widget is not None and self._sdr_queue is None:
                self._start_sdr_feeder()
            # Hot-swap precision
            pending = self._pending_precision
            if (not self._input_is_hdr) and pending and pending != self._precision_key:
                self._pending_precision = None
                if not self._load_model(pending):
                    continue

            # Hot-swap processing resolution
            pending_res = self._pending_resolution
            if (not self._input_is_hdr) and pending_res is not None:
                self._pending_resolution = None
                new_pw, new_ph = pending_res
                if (new_pw, new_ph) != (proc_w, proc_h):
                    self.status_message.emit(
                        f"Switching to {new_pw}×{new_ph} …")
                    self._proc_w, self._proc_h = new_pw, new_ph
                    proc_w, proc_h = new_pw, new_ph
                    lower_res_processing = (proc_w != out_w or proc_h != out_h)
                    self._reset_enhance_history()
                    self._silent_warmup(self._processor, proc_w, proc_h)
                    self.status_message.emit(
                        f"Ready — {self._precision_key} @ {proc_w}×{proc_h}")

                self._reset_enhance_history()

            # Seek gate
            seek_to = self._seek_frame
            if seek_to is not None:
                self._seek_frame = None
                source.seek(seek_to)
                frame_idx = max(0, seek_to - 1)
                force_position_emit = True
                seek_frame_ready_pending = True
                next_frame_t = time.perf_counter()
                self._sdr_delay_frame = None
                if seek_to <= 1:
                    self._hdr_drop_until_frame = 2
                    self._sdr_drop_until_frame = 2
                    self._hold_until_t = time.perf_counter() + 0.5
                # Discard stale FPS history so metrics/auto-mute re-lock quickly.
                frame_times.clear()
                presented_times.clear()
                metrics_warmup_frames = 4

            # Pause gate
            paused_before_wait = not self._pause_event.is_set()
            self._pause_event.wait()
            if self._stop_flag:
                break
            if paused_before_wait:
                next_frame_t = time.perf_counter()
                # Pauses create a long timestamp gap that pollutes FPS windows.
                frame_times.clear()
                presented_times.clear()
                metrics_warmup_frames = 4

            # Playback pacing: cap processing to the video FPS.
            now = time.perf_counter()
            lag_s = 0.0
            if now < next_frame_t:
                time.sleep(next_frame_t - now)
            else:
                lag_s = now - next_frame_t
            if self._hold_until_t and now < self._hold_until_t:
                time.sleep(min(frame_interval_s, self._hold_until_t - now))

            ret, frame = source.read()
            if not ret:
                break

            frame_idx += 1
            self._frame_idx = frame_idx

            # Real-time catch-up: if inference is behind wall clock, drop a few
            # decode frames and process the newest one to preserve cadence.
            if (
                _REALTIME_CATCHUP_ENABLED
                and (not seek_frame_ready_pending)
                and lag_s > (frame_interval_s * _REALTIME_SKIP_LAG_FRAMES)
            ):
                skip_n = min(
                    _REALTIME_MAX_CATCHUP_SKIP,
                    max(0, int(lag_s / frame_interval_s)),
                )
                while skip_n > 0:
                    ret_skip, frame_skip = source.read()
                    if not ret_skip:
                        ret = False
                        break
                    frame = frame_skip
                    frame_idx += 1
                    self._frame_idx = frame_idx
                    self._realtime_drop_frames += 1
                    next_frame_t += frame_interval_s
                    skip_n -= 1
                if not ret:
                    break

            # FPS limiter via frame skipping (keeps wall-clock speed).
            if (not seek_frame_ready_pending) and frame_stride > 1 and (frame_idx % frame_stride) != 0:
                next_frame_t += frame_interval_s
                continue

            present_t = max(next_frame_t, time.perf_counter())

            t0 = time.perf_counter()

            # Match display/output resolution without stretch (black bars as needed).
            display_frame = _letterbox_bgr(frame, out_w, out_h)

            # Downscale for model if processing at lower resolution
            if lower_res_processing:
                model_inp = _letterbox_bgr(display_frame, proc_w, proc_h)
            else:
                model_inp = display_frame

            if self._input_is_hdr:
                # Pass-through mode for HDR input: do not run SDR->HDR model.
                need_hdr_cpu = False
                output = display_frame
                if (output.shape[1], output.shape[0]) != (out_w, out_h):
                    output = cv2.resize(output, (out_w, out_h),
                                        interpolation=cv2.INTER_AREA)
                if mpv_w is not None:
                    rgb16 = np.ascontiguousarray(
                        output[:, :, ::-1].astype(np.uint16) * 257
                    )
                    if present_t is not None:
                        now_t = time.perf_counter()
                        if now_t < present_t:
                            time.sleep(present_t - now_t)
                    mpv_w.feed_frame(rgb16.data)
                if (
                    self._sdr_mpv_widget is not None
                    and self._sdr_queue is not None
                    and self._sdr_visible
                ):
                    if frame_idx < self._sdr_drop_until_frame:
                        # Let mpv settle at the start; skip early SDR frames.
                        pass
                    else:
                        self._sdr_drop_until_frame = 0
                    if self._sdr_drop_until_frame == 0:
                        try:
                            self._sdr_queue.put_nowait((present_t, display_frame))
                        except _queue.Full:
                            try:
                                self._sdr_queue.get_nowait()
                            except _queue.Empty:
                                pass
                            try:
                                self._sdr_queue.put_nowait((present_t, display_frame))
                            except _queue.Full:
                                pass
                if self._sdr_visible:
                    need_hdr_cpu = True
            else:
                # ── Fast path: preprocess → infer → mpv ─────────────────────
                # When mpv handles HDR display, skip postprocess entirely:
                # the GPU→CPU D2H transfer + numpy conversion would be wasted
                # since the QLabel HDR fallback is not active.  This is the
                # single largest per-frame overhead saved (~1 ms at 1080p).
                with torch.inference_mode():
                    tensor, cond = self._processor.preprocess(model_inp)
                    raw_out = self._processor.infer((tensor, cond))

                if mpv_w is not None and self._hdr_queue is not None:
                    t_raw = (raw_out[0] if isinstance(raw_out, (tuple, list))
                             else raw_out)
                    if lower_res_processing:
                        t_raw = self._enhance_best_gpu(t_raw)
                    if frame_idx < self._hdr_drop_until_frame:
                        # Let mpv settle at the start; skip early HDR frames.
                        pass
                    else:
                        self._hdr_drop_until_frame = 0
                    if self._hdr_drop_until_frame == 0:
                        try:
                            self._hdr_queue.put_nowait((present_t, t_raw.clone()))
                        except _queue.Full:
                            # Keep newest frame to avoid persistent HDR lag buildup.
                            try:
                                self._hdr_queue.get_nowait()
                            except _queue.Empty:
                                pass
                            try:
                                self._hdr_queue.put_nowait((present_t, t_raw.clone()))
                            except _queue.Full:
                                pass

                if (
                    self._sdr_mpv_widget is not None
                    and self._sdr_queue is not None
                    and self._sdr_visible
                ):
                    if frame_idx < self._sdr_drop_until_frame:
                        # Let mpv settle at the start; skip early SDR frames.
                        pass
                    else:
                        self._sdr_drop_until_frame = 0
                    if self._sdr_drop_until_frame == 0:
                        try:
                            self._sdr_queue.put_nowait((present_t, display_frame))
                        except _queue.Full:
                            # Keep newest frame to avoid persistent SDR lag buildup.
                            try:
                                self._sdr_queue.get_nowait()
                            except _queue.Empty:
                                pass
                            try:
                                self._sdr_queue.put_nowait((present_t, display_frame))
                            except _queue.Full:
                                pass

                # Only run the expensive postprocess (GPU→CPU D2H) when
                # the QLabel HDR fallback is the display path.
                need_hdr_cpu = (mpv_w is None)
                if need_hdr_cpu:
                    output = self._processor.postprocess(raw_out)
                    # postprocess() already calls stream.synchronize()
                    if (output.shape[1], output.shape[0]) != (out_w, out_h):
                        if out_w > output.shape[1] or out_h > output.shape[0]:
                            interp = BEST_CV2_INTERP
                            output = cv2.resize(output, (out_w, out_h), interpolation=interp)
                            output = _apply_upscale_sharpen(output)
                        else:
                            output = cv2.resize(output, (out_w, out_h), interpolation=cv2.INTER_AREA)
                elif use_cuda:
                    # Sync for timing only (no D2H to wait for)
                    torch.cuda.synchronize()

            t1 = time.perf_counter()
            next_frame_t += frame_interval_s
            frame_ms = (t1 - t0) * 1000.0
            frame_times.append(frame_ms)
            presented_times.append(t1)
            if metrics_warmup_frames > 0:
                metrics_warmup_frames -= 1

            # Position update (every 10 frames, plus immediately after seeks,
            # and every frame while paused for accurate paused-frame actions).
            if self._user_paused or force_position_emit or (frame_idx % 10 == 0):
                self.position_updated.emit(frame_idx, total_frames)
                force_position_emit = False
            if seek_frame_ready_pending:
                self.seek_frame_ready.emit(frame_idx)
                seek_frame_ready_pending = False

            # Emit only what the UI actually needs
            if need_hdr_cpu:
                # QLabel fallback — both SDR + HDR frames needed
                self.frame_ready.emit(display_frame, output)
            elif self._sdr_visible:
                # mpv handles HDR; SDR QLabel still visible.
                self.frame_ready.emit(display_frame, display_frame)
            # else: both outputs are handled directly by mpv panes

            if self._hold_until_t and time.perf_counter() >= self._hold_until_t:
                self._hold_until_t = 0.0

            # Re-pause: if user paused and no further seek pending,
            # block again now that this seek frame has been emitted.
            if self._user_paused and self._seek_frame is None:
                self._pause_event.clear()

            # Metrics every 2 frames for responsive dashboard/audio policy.
            if frame_idx % 2 == 0 and frame_times and metrics_warmup_frames == 0:
                avg = sum(frame_times) / len(frame_times)
                if len(presented_times) >= 2:
                    dt = presented_times[-1] - presented_times[0]
                    fps = ((len(presented_times) - 1) / dt) if dt > 0 else 0.0
                else:
                    fps = 1000.0 / avg if avg > 0 else 0.0
                cpu_mb = process.memory_info().rss / (1024 * 1024)
                gpu_mb = self._app_vram_mb
                if gpu_mb <= 0.0 and use_cuda:
                    # Fallback when Windows counters are unavailable:
                    # allocator reservation is closer to "reserved VRAM"
                    # than memory_allocated() (live tensor bytes only).
                    gpu_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                if self._input_is_hdr:
                    model_mb = 0.0
                    prec_label = "Bypass (HDR input)"
                else:
                    model_path = _select_model_path(self._precision_key, self._use_hg)
                    model_mb = os.path.getsize(model_path) / (1024 * 1024)
                    if self._use_hg and self._precision_key in ("FP16", "FP32"):
                        if os.path.isfile(_HG_WEIGHTS_PATH):
                            model_mb += os.path.getsize(_HG_WEIGHTS_PATH) / (1024 * 1024)
                    prec_label = self._precision_key
                self.metrics_updated.emit({
                    "fps": fps,
                    "latency_ms": avg,
                    "frame": frame_idx,
                    "cpu_mb": cpu_mb,
                    "gpu_mb": gpu_mb,
                    "model_mb": model_mb,
                    "precision": prec_label,
                    "proc_res": f"{proc_w}\u00d7{proc_h}",
                    "realtime_drops": int(self._realtime_drop_frames),
                })

        source.release()
        self._source = None
        self._stop_hdr_feeder()
        self._stop_sdr_feeder()
        self._stop_app_vram_poll()
        self.playback_finished.emit()
        self.status_message.emit("Playback finished.")


class _KernelCacheClearWorker(QObject):
    finished = pyqtSignal(bool)

    def __init__(self, dirs):
        super().__init__()
        self._dirs = dirs

    def run(self):
        import shutil
        ok = True
        for d in self._dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                ok = False
        self.finished.emit(ok)


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Video Display Widget                                         ║
# ╚═══════════════════════════════════════════════════════════════╝

class VideoDisplay(QLabel):
    """QLabel that efficiently renders a BGR numpy frame."""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self._title = title
        self.setMinimumSize(320, 180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background: #111; color: #555; "
                           "font-size: 14px; }")
        self.setText(title)

    def update_frame(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        # Use BGR888 directly to avoid per-frame color conversion cost.
        qimg = QImage(
            bgr.data, w, h, 3 * w, QImage.Format.Format_BGR888
        ).copy()
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.setPixmap(scaled)

    def clear_display(self):
        self.clear()
        self.setText(self._title)


class DetachedVideoWindow(QWidget):
    """Floating host window for a video pane (SDR/HDR)."""

    closed = pyqtSignal(str)

    def __init__(self, key: str, title: str, parent=None):
        super().__init__(parent)
        self._key = str(key)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, True)
        self.setWindowTitle(title)
        self.resize(960, 540)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

    def set_content(self, widget: QWidget):
        self._layout.addWidget(widget, 1)

    def closeEvent(self, event):
        self.closed.emit(self._key)
        super().closeEvent(event)


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Compile Progress Dialog (inline, for loading screen)          ║
# ╚═══════════════════════════════════════════════════════════════╝

class _CompileDialog(QDialog):
    """Non-modal dialog shown while Triton kernels are being compiled
    in-process (loading screen during playback start)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compiling Kernels")
        self.setFixedSize(460, 160)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        title = QLabel("\u2699  Compiling optimized GPU kernels \u2026")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(title)

        detail = QLabel(
            "First run at this resolution may take 2\u20135 minutes.\n"
            "Subsequent runs load from cache in seconds."
        )
        detail.setStyleSheet("color: #aaa;")
        layout.addWidget(detail)

        bar = QProgressBar()
        bar.setRange(0, 0)                 # indeterminate / busy animation
        bar.setFixedHeight(6)
        bar.setTextVisible(False)
        layout.addWidget(bar)

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color: #6af;")
        layout.addWidget(self._lbl_status)

    def set_status(self, text: str):
        self._lbl_status.setText(text)


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Pre-compile Dialog (subprocess-based, clean GPU)              ║
# ╚═══════════════════════════════════════════════════════════════╝

class _PrecompileOptionsDialog(QDialog):
    """Collect precision + resolution choices before launching compile."""

    def __init__(
        self,
        initial_precision: str,
        initial_resolution: str,
        precision_keys: list[str],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Pre-compile Kernels")
        self.setMinimumSize(420, 260)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        self._precision_combo = QComboBox()
        self._precision_combo.addItems(precision_keys)
        if initial_precision in precision_keys:
            self._precision_combo.setCurrentText(initial_precision)

        self._res_combo = QComboBox()
        res_options = [
            ("1080p (1920x1080)", f"{MAX_W}x{MAX_H}", "1080p"),
            ("720p (1280x720)", "1280x720", "720p"),
            ("540p (960x540)", "960x540", "540p"),
        ]
        for label, res_value, _res_key in res_options:
            self._res_combo.addItem(label, res_value)
        if initial_resolution:
            for i, (_label, _value, res_key) in enumerate(res_options):
                if initial_resolution == res_key:
                    self._res_combo.setCurrentIndex(i)
                    break

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Precision:"))
        layout.addWidget(self._precision_combo)

        layout.addWidget(QLabel("Resolution to compile:"))
        layout.addWidget(self._res_combo)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_ok = QPushButton("Start")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

    def selected_resolutions(self) -> list[str]:
        return [str(self._res_combo.currentData())]

    def selected_precision(self) -> str:
        return self._precision_combo.currentText()


class _PrecompileDialog(QDialog):
    """Modal dialog that launches ``compile_kernels.py`` in a separate
    process (zero GPU interference) and streams stdout into a log view."""

    def __init__(self, resolutions: list[str], precision: str = "fp16",
                 model_path: str | None = None, use_hg: bool = True,
                 hg_weights: str | None = None, clear_cache: bool = False,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pre-compile Kernels")
        self.setMinimumSize(540, 340)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        self._resolutions = resolutions
        self._precision = precision
        self._model_path = model_path
        self._use_hg = bool(use_hg)
        self._hg_weights = hg_weights
        self._clear_cache = clear_cache
        self._process: QProcess | None = None
        self._finished_ok = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        title = QLabel("\u2699  Compiling optimised GPU kernels \u2026")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(title)

        detail = QLabel(
            f"Resolutions: {', '.join(resolutions)}  |  Precision: {precision}\n"
            "This runs in a clean process for best kernel quality.\n"
            "First compilation may take 2\u20135 minutes per resolution."
        )
        detail.setStyleSheet("color: #aaa;")
        layout.addWidget(detail)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)
        self._bar.setFixedHeight(6)
        self._bar.setTextVisible(False)
        layout.addWidget(self._bar)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setStyleSheet(
            "QTextEdit { background: #111; color: #ccc; border: 1px solid #333; }"
        )
        layout.addWidget(self._log, 1)

        btn_row = QHBoxLayout()
        self._btn_close = QPushButton("Close")
        self._btn_close.setFixedSize(90, 28)
        self._btn_close.setEnabled(False)
        self._btn_close.clicked.connect(self.accept)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_close)
        layout.addLayout(btn_row)

        # Start compilation immediately
        self._start()

    def _start(self):
        script = os.path.join(_HERE, "compile_kernels.py")
        args = [script] + self._resolutions
        args += ["--precision", self._precision]
        if self._model_path:
            args += ["--model", self._model_path]
        args += ["--use-hg", "1" if self._use_hg else "0"]
        if self._hg_weights:
            args += ["--hg-weights", self._hg_weights]
        if self._clear_cache:
            args += ["--clear-cache"]

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        # Force UTF-8 so Unicode chars (→, ×, etc.) in model print() don't
        # crash on Windows cp1252 console encoding.
        env = self._process.processEnvironment()
        if env.isEmpty():
            from PyQt6.QtCore import QProcessEnvironment
            env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        if os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
            env.insert("TORCHINDUCTOR_CACHE_DIR", os.environ["TORCHINDUCTOR_CACHE_DIR"])
        if os.environ.get("TRITON_CACHE_DIR"):
            env.insert("TRITON_CACHE_DIR", os.environ["TRITON_CACHE_DIR"])
        self._process.setProcessEnvironment(env)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_finished)

        self._log.append(f"$ python {' '.join(os.path.basename(a) for a in args)}\n")
        self._process.start(sys.executable, args)

    def _on_stdout(self):
        data = self._process.readAllStandardOutput()
        text = bytes(data).decode("utf-8", errors="replace").rstrip()
        if text:
            self._log.append(text)
            # Auto-scroll
            sb = self._log.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _on_finished(self, exit_code, _status):
        self._bar.setRange(0, 1)
        self._bar.setValue(1)
        self._btn_close.setEnabled(True)

        if exit_code == 0:
            self._finished_ok = True
            self._log.append("\n\u2705  Done — kernels cached to disk.")
            self._log.append("Starting playback ...")
            # Auto-close after a brief pause so the user sees the result
            QTimer.singleShot(800, self.accept)
        else:
            self._log.append(f"\n\u274c  Process exited with code {exit_code}.")
            self._log.append("Check the log above for errors.")

        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
            self._process.waitForFinished(3000)
        super().closeEvent(event)

    @property
    def succeeded(self) -> bool:
        return self._finished_ok


# ── Kernel-cache marker ──────────────────────────────────────
# A lightweight text file inside the Triton cache dir that records which
# (resolution, precision) pairs have been compiled via a clean subprocess.
# Cleared automatically when the user clears kernel caches.
import pathlib as _pathlib
import hashlib as _hashlib

_TRITON_CACHE = (
    _pathlib.Path(os.environ.get("TRITON_CACHE_DIR", _pathlib.Path.home() / ".triton"))
    / "cache"
)


def _compiled_marker_path() -> _pathlib.Path:
    return _TRITON_CACHE / "hdrtvnet_compiled.txt"


_MODEL_HASH_CACHE: dict[str, str] = {}


def _model_hash(path: str) -> str:
    if not path:
        return "missing"
    cached = _MODEL_HASH_CACHE.get(path)
    if cached:
        return cached
    try:
        data = _pathlib.Path(path).read_bytes()
    except Exception:
        digest = "missing"
    else:
        digest = _hashlib.sha256(data).hexdigest()
    _MODEL_HASH_CACHE[path] = digest
    return digest


def _compiled_key(
    w: int,
    h: int,
    precision: str,
    model_path: str,
    use_hg: bool,
    compile_mode: str = "max-autotune",
) -> str:
    mh = _model_hash(model_path)
    return f"{w}x{h}_{precision}_hg{int(use_hg)}_{compile_mode}_{mh}"


def _is_compiled(
    w: int,
    h: int,
    precision: str,
    model_path: str,
    use_hg: bool,
    compile_mode: str = "max-autotune",
) -> bool:
    """Check if clean-compiled kernels exist for this config."""
    mp = _compiled_marker_path()
    if mp.is_file():
        key = _compiled_key(w, h, precision, model_path, use_hg, compile_mode)
        return key in mp.read_text(encoding="utf-8").splitlines()
    return False


def _mark_compiled(
    w: int,
    h: int,
    precision: str,
    model_path: str,
    use_hg: bool,
    compile_mode: str = "max-autotune",
):
    """Record that kernels for this config were compiled cleanly."""
    mp = _compiled_marker_path()
    mp.parent.mkdir(parents=True, exist_ok=True)
    key = _compiled_key(w, h, precision, model_path, use_hg, compile_mode)
    existing = set()
    if mp.is_file():
        existing = set(mp.read_text(encoding="utf-8").splitlines())
    existing.add(key)
    mp.write_text("\n".join(sorted(existing)) + "\n", encoding="utf-8")


def _precision_to_compile_arg(gui_precision: str) -> str:
    """Map GUI precision label to compile/precompile precision argument."""
    return {
        "FP16": "fp16",
        "FP32": "fp32",
        "INT8 Mixed (PTQ)": "int8-mixed",
        "INT8 Mixed (QAT)": "int8-mixed",
        "INT8 Full (PTQ)": "int8-full",
        "INT8 Full (QAT)": "int8-full",
    }.get(str(gui_precision), "fp16")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Main Window                                                  ║
# ╚═══════════════════════════════════════════════════════════════╝

class MainWindow(QMainWindow):
    def __init__(self, initial_video=None, initial_resolution=None,
                 initial_precision=None, initial_view=None,
                 initial_use_hg=None,
                 initial_autoplay=False, initial_start_frame=None,
                 initial_upscale=None, initial_film_grain=None):
        super().__init__()
        self.setWindowTitle("HDRTVNet++ — Real-Time SDR → HDR Pipeline")
        self.setMinimumSize(1024, 600)
        self.resize(1600, 900)
        self.setAcceptDrops(True)

        self._worker = PipelineWorker()
        self._video_path = None
        self._playing = False
        self._compile_dlg = None
        self._pending_mpv_start = None
        self._pending_sdr_mpv_start = None
        self._last_res = None          # (pw, ph) of last played video
        self._active_precision = None
        self._active_resolution = None
        self._active_use_mpv = False
        self._sdr_mpv_feed_from_worker = False
        self._active_use_hg = True
        self._active_film_grain = False
        self._source_hdr_info = {"is_hdr": False, "reason": "unknown"}
        self._last_seek_frame = 0
        self._post_seek_resync_frames = 0
        self._pending_seek_on_resume: int | None = None
        self._audio_last_hard_sync_t = 0.0
        self._audio_seek_guard_until = 0.0
        self._audio_track_lock_until = 0.0
        self._audio_resync_pending = False
        self._audio_fps_recovered = True
        self._audio_drift_check_stride = 10
        self._startup_audio_gate_active = False
        self._startup_audio_gate_count = 0
        self._user_pause_override_startup = False
        self._first_seek_done = False
        self._ui_resync_gate_strict = False
        self._target_playback_fps = 24.0
        self._resume_audio_after_seek = False
        self._seek_resume_target = 0
        self._seek_resume_started_t = 0.0
        self._precision_swap_pending: str | None = None
        self._precision_pause_armed = False
        self._precision_swap_timer: QTimer | None = None
        self._last_user_pause_t = 0.0
        self._audio_player = None
        self._audio_output = None
        self._audio_available = _HAS_QT_AUDIO
        self._audio_tracks: list[dict] = []
        self._selected_audio_track = 0
        self._audio_apply_token = 0
        self._volume_percent = 100
        self._auto_muted_low_fps = False
        self._scrub_muted = False
        self._scrub_unmute_seq = 0
        self._low_fps_count = 0
        self._high_fps_count = 0
        self._audio_fade_timer: QTimer | None = None
        self._audio_fade_steps = 8
        self._audio_fade_step_idx = 0
        self._proc_priority_saved = None
        self._app_active = True
        self._deferred_mpv_refresh = False
        self._cursor_idle_timer: QTimer | None = None
        self._cursor_idle_enabled = True
        self._cursor_hidden = False
        self._cursor_idle_ms = 1500
        self._startup_sync_pending = False
        self._last_sdr_frame: np.ndarray | None = None
        self._source_proc_dims: tuple[int, int] | None = None
        self._borderless_full_window = False
        self._mpv_start_resync_t = 0.0
        self._ui_hidden = False
        self._ui_overlay_btn: QPushButton | None = None
        self._ui_overlay_timer: QTimer | None = None
        self._ui_overlay_hide_ms = 1200
        self._ui_anim_effects: dict[QWidget, QGraphicsOpacityEffect] = {}
        self._ui_anim_running: dict[QWidget, QPropertyAnimation] = {}
        self._ui_anim_duration_ms = 0
        self._layout_freeze_depth = 0
        self._saved_window_geometry: QRect | None = None
        self._saved_window_state = Qt.WindowState.WindowNoState
        self._window_toggle_last_t = 0.0
        self._window_toggle_cooldown_s = 0.25
        self._window_refresh_timer: QTimer | None = None
        self._window_refresh_soft_only = False
        self._overlay_reposition_timer: QTimer | None = None
        self._ui_pause_timer: QTimer | None = None
        self._ui_pause_duration_ms = 180
        self._periodic_relock_timer: QTimer | None = None
        self._periodic_relock_ms = 3000
        self._act_borderless_full_window = None
        self._root_layout: QVBoxLayout | None = None
        self._immersive_saved_margins: tuple[int, int, int, int] | None = None
        self._immersive_saved_spacing: int | None = None
        self._immersive_saved_view_mode: str | None = None
        self._immersive_saved_vis: dict[str, bool] = {}
        self._video_tabs: QTabWidget | None = None
        self._sdr_tab_host: QWidget | None = None
        self._hdr_tab_host: QWidget | None = None
        self._side_tab_host: QWidget | None = None
        self._side_sdr_host: QWidget | None = None
        self._side_hdr_host: QWidget | None = None
        self._sdr_float_window: DetachedVideoWindow | None = None
        self._hdr_float_window: DetachedVideoWindow | None = None
        self._ui_closing = False
        try:
            self._startup_seek_frame = (
                int(initial_start_frame) if initial_start_frame is not None else None
            )
        except (TypeError, ValueError):
            self._startup_seek_frame = None

        self._build_ui()
        self._connect_signals()
        self._init_audio_backend()
        self._load_user_settings(
            initial_resolution,
            initial_precision,
            initial_view,
            initial_use_hg,
            initial_upscale,
            initial_film_grain,
        )
        self._init_cursor_idle_tracking()

        # Auto-open video passed via --video (used by restart)
        if initial_video and os.path.isfile(initial_video):
            def _boot_open():
                if initial_precision in PRECISIONS:
                    self._cmb_prec.setCurrentText(initial_precision)
                legacy_resolution_map = {
                    "Native": "1080p",
                }
                mapped_resolution = legacy_resolution_map.get(initial_resolution, initial_resolution)
                if mapped_resolution in RESOLUTION_SCALES or mapped_resolution == "Source":
                    self._cmb_res.setCurrentText(mapped_resolution)
                if isinstance(initial_upscale, str) and initial_upscale in UPSCALER_CHOICES:
                    if hasattr(self, "_cmb_upscale"):
                        self._cmb_upscale.setCurrentText(initial_upscale)
                if initial_view == "Tabbed":
                    self._cmb_view.setCurrentText("Tabbed")
                self._set_video(initial_video, auto_play=bool(initial_autoplay))
            QTimer.singleShot(200, _boot_open)

    # ── UI construction ──────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        self._root_layout = root
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(6)

        # ---- Menu bar ----
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction("\U0001F4C2  &Open Video \u2026", self._open_file)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close)

        tools_menu = menu_bar.addMenu("&Tools")
        tools_menu.addAction(
            "\u2699  &Pre-compile Kernels \u2026", self._precompile_kernels
        )
        tools_menu.addSeparator()
        tools_menu.addAction(
            "\U0001F5D1  Clear &Kernel Cache \u2026", self._clear_kernel_cache
        )

        view_menu = menu_bar.addMenu("&View")
        self._act_borderless_full_window = view_menu.addAction(
            "Borderless Full Window\tF11",
            self._toggle_borderless_full_window,
        )
        self._act_borderless_full_window.setCheckable(True)

        # ---- Row 1: file + precision + view ----
        self._row1_widget = QWidget()
        row1 = QHBoxLayout(self._row1_widget)
        row1.setContentsMargins(0, 0, 0, 0)

        self._btn_file = QPushButton("📂  Open Video ...")
        self._btn_file.setFixedHeight(32)
        self._lbl_file = QLabel("No video selected")
        self._lbl_file.setStyleSheet("color: #999; padding-left: 8px;")
        row1.addWidget(self._btn_file)
        row1.addWidget(self._lbl_file, 1)

        row1.addWidget(QLabel("Precision:"))
        self._cmb_prec = QComboBox()
        self._cmb_prec.addItems(_available_precision_keys())
        self._cmb_prec.setFixedWidth(170)
        row1.addWidget(self._cmb_prec)
        self._chk_hg = QCheckBox("Use HG")
        self._chk_hg.setChecked(True)
        self._chk_hg.setToolTip("Enable highlight refinement (HG).")
        row1.addWidget(self._chk_hg)

        row1.addWidget(QLabel("Resolution:"))
        self._cmb_res = QComboBox()
        self._cmb_res.addItems(RESOLUTION_SCALES.keys())
        self._cmb_res.setFixedWidth(100)
        row1.addWidget(self._cmb_res)

        row1.addWidget(QLabel("Upscale:"))
        self._cmb_upscale = QComboBox()
        self._cmb_upscale.addItems(UPSCALER_CHOICES)
        self._cmb_upscale.setFixedWidth(130)
        self._cmb_upscale.setToolTip(
            "Upscale kernel for 540p/720p. 1080p stays native (no upscale)."
        )
        row1.addWidget(self._cmb_upscale)

        self._chk_film_grain = QCheckBox("Film Grain")
        self._chk_film_grain.setToolTip("Restore film grain using mpv shader.")
        if not _HAS_MPV:
            self._chk_film_grain.setEnabled(False)
            self._chk_film_grain.setToolTip("Requires mpv (libmpv-2.dll).")
        row1.addWidget(self._chk_film_grain)

        self._btn_apply_settings = QPushButton("Apply")
        self._btn_apply_settings.setFixedWidth(90)
        self._btn_apply_settings.setEnabled(False)
        row1.addWidget(self._btn_apply_settings)

        self._cmb_view = QComboBox()
        self._cmb_view.addItems(["Tabbed"])
        self._cmb_view.setCurrentText("Tabbed")
        self._btn_pop_sdr = QPushButton("Pop SDR")
        self._btn_pop_sdr.setFixedWidth(88)
        self._btn_pop_hdr = QPushButton("Pop HDR")
        self._btn_pop_hdr.setFixedWidth(88)
        row1.addWidget(self._btn_pop_sdr)
        row1.addWidget(self._btn_pop_hdr)
        self._btn_toggle_ui = QPushButton("Hide UI")
        self._btn_toggle_ui.setFixedWidth(90)
        self._btn_toggle_ui.setEnabled(False)
        row1.addWidget(self._btn_toggle_ui)

        root.addWidget(self._row1_widget)

        # ---- Row 2: playback controls ----
        self._row2_widget = QWidget()
        row2 = QHBoxLayout(self._row2_widget)
        row2.setContentsMargins(0, 0, 0, 0)

        self._btn_play = QPushButton("▶  Play")
        self._btn_pause = QPushButton("⏸  Pause")
        self._btn_stop = QPushButton("⏹  Stop")
        for b in (self._btn_play, self._btn_pause, self._btn_stop):
            b.setFixedSize(100, 30)
        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)

        self._chk_metrics = QCheckBox("Show Metrics")
        self._chk_metrics.setChecked(True)
        self._chk_hide_cursor = QCheckBox("Hide Cursor")
        self._chk_hide_cursor.setChecked(True)
        self._lbl_volume = QLabel("Volume:")
        self._sld_volume = QSlider(Qt.Orientation.Horizontal)
        self._sld_volume.setRange(0, 100)
        self._sld_volume.setValue(100)
        self._sld_volume.setFixedWidth(140)
        self._sld_volume.setToolTip("Master volume")
        self._lbl_volume_val = QLabel("100%")
        self._lbl_volume_val.setFixedWidth(42)
        self._lbl_audio_track = QLabel("Audio:")
        self._cmb_audio_track = QComboBox()
        self._cmb_audio_track.setFixedWidth(260)
        self._cmb_audio_track.setEnabled(False)
        self._cmb_audio_track.setToolTip("Load a video with multiple audio tracks.")

        row2.addWidget(self._btn_play)
        row2.addWidget(self._btn_pause)
        row2.addWidget(self._btn_stop)
        row2.addStretch()
        row2.addWidget(self._lbl_volume)
        row2.addWidget(self._sld_volume)
        row2.addWidget(self._lbl_volume_val)
        row2.addWidget(self._lbl_audio_track)
        row2.addWidget(self._cmb_audio_track)
        row2.addWidget(self._chk_hide_cursor)
        row2.addWidget(self._chk_metrics)
        root.addWidget(self._row2_widget)

        # ---- Row 3: seek bar ----
        self._row3_widget = QWidget()
        row3 = QHBoxLayout(self._row3_widget)
        row3.setContentsMargins(0, 0, 0, 0)

        self._lbl_time = QLabel("0:00")
        self._lbl_time.setFixedWidth(50)
        self._lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._lbl_time.setFont(QFont("Consolas", 9))

        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 0)
        self._seek_slider.setEnabled(False)
        self._seek_slider.setTracking(True)    # emit valueChanged while dragging

        self._lbl_duration = QLabel("0:00")
        self._lbl_duration.setFixedWidth(50)
        self._lbl_duration.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._lbl_duration.setFont(QFont("Consolas", 9))

        row3.addWidget(self._lbl_time)
        row3.addWidget(self._seek_slider, 1)
        row3.addWidget(self._lbl_duration)
        root.addWidget(self._row3_widget)

        # ---- Video displays ----
        if _HAS_MPV:
            self._disp_sdr_mpv = MpvHDRWidget()
            self._disp_sdr_cpu = VideoDisplay("SDR Input")
            self._disp_sdr_stack = QStackedWidget()
            self._disp_sdr_stack.addWidget(self._disp_sdr_mpv)
            self._disp_sdr_stack.addWidget(self._disp_sdr_cpu)
            self._disp_sdr = self._disp_sdr_stack
        else:
            self._disp_sdr_mpv = None
            self._disp_sdr_cpu = VideoDisplay("SDR Input")
            self._disp_sdr = self._disp_sdr_cpu
        if _HAS_MPV:
            self._disp_hdr_mpv = MpvHDRWidget()
            self._disp_hdr_cpu = VideoDisplay("HDR Output")
            self._disp_hdr_stack = QStackedWidget()
            self._disp_hdr_stack.addWidget(self._disp_hdr_mpv)
            self._disp_hdr_stack.addWidget(self._disp_hdr_cpu)
            self._disp_hdr = self._disp_hdr_stack
        else:
            self._disp_hdr_mpv = None
            self._disp_hdr_cpu = VideoDisplay("HDR Output")
            self._disp_hdr = self._disp_hdr_cpu

        self._video_tabs = QTabWidget()
        self._video_tabs.setDocumentMode(True)
        self._sdr_tab_host = QWidget()
        sdr_tab_layout = QVBoxLayout(self._sdr_tab_host)
        sdr_tab_layout.setContentsMargins(0, 0, 0, 0)
        sdr_tab_layout.setSpacing(0)
        sdr_tab_layout.addWidget(self._disp_sdr)
        self._hdr_tab_host = QWidget()
        hdr_tab_layout = QVBoxLayout(self._hdr_tab_host)
        hdr_tab_layout.setContentsMargins(0, 0, 0, 0)
        hdr_tab_layout.setSpacing(0)
        hdr_tab_layout.addWidget(self._disp_hdr)
        self._side_tab_host = QWidget()
        side_tab_layout = QVBoxLayout(self._side_tab_host)
        side_tab_layout.setContentsMargins(0, 0, 0, 0)
        side_tab_layout.setSpacing(0)
        side_split = QSplitter(Qt.Orientation.Horizontal)
        self._side_sdr_host = QWidget()
        side_sdr_layout = QVBoxLayout(self._side_sdr_host)
        side_sdr_layout.setContentsMargins(0, 0, 0, 0)
        side_sdr_layout.setSpacing(0)
        self._side_hdr_host = QWidget()
        side_hdr_layout = QVBoxLayout(self._side_hdr_host)
        side_hdr_layout.setContentsMargins(0, 0, 0, 0)
        side_hdr_layout.setSpacing(0)
        side_split.addWidget(self._side_sdr_host)
        side_split.addWidget(self._side_hdr_host)
        side_split.setStretchFactor(0, 1)
        side_split.setStretchFactor(1, 1)
        side_tab_layout.addWidget(side_split, 1)
        self._video_tabs.addTab(self._sdr_tab_host, "SDR")
        self._video_tabs.addTab(self._hdr_tab_host, "HDR")
        self._video_tabs.addTab(self._side_tab_host, "Side by Side")
        # Default to HDR tab unless user prefs override it later.
        self._video_tabs.setCurrentIndex(1)
        if _HAS_MPV and self._disp_sdr_mpv is not None:
            self._disp_sdr_stack.setCurrentWidget(self._disp_sdr_mpv)
        if _HAS_MPV and self._disp_hdr_mpv is not None:
            self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_mpv)
        root.addWidget(self._video_tabs, 1)

        # Apply the view mode immediately
        self._on_view(self._cmb_view.currentText())

        # ---- Metrics panel ----
        self._grp_metrics = QGroupBox("Metrics")
        ml = QHBoxLayout(self._grp_metrics)
        ml.setContentsMargins(12, 4, 12, 4)

        self._m = {}
        mono = QFont("Consolas", 9)
        for key in ("fps", "latency", "frame", "res", "gpu", "cpu", "model", "prec", "upscale"):
            lbl = QLabel(f"{key}: —")
            lbl.setFont(mono)
            ml.addWidget(lbl)
            self._m[key] = lbl

        root.addWidget(self._grp_metrics)

        # ---- HDR Info panel ----
        self._grp_hdr = QGroupBox("HDR Output")
        hl = QHBoxLayout(self._grp_hdr)
        hl.setContentsMargins(12, 4, 12, 4)
        self._hdr_labels = {}
        mono = QFont("Consolas", 9)
        for key, default in [
            ("status", "HDR: waiting…"),
            ("primaries", "Primaries: —"),
            ("transfer", "Transfer: —"),
            ("peak", "Peak: —"),
            ("vo", "VO: —"),
        ]:
            lbl = QLabel(default)
            lbl.setFont(mono)
            hl.addWidget(lbl)
            self._hdr_labels[key] = lbl
        root.addWidget(self._grp_hdr)

        # ---- Status bar ----
        self.statusBar().showMessage(
            "Ready — open a video file to begin.  "
            "You can also drag-and-drop a video onto this window."
        )
        self._sync_upscale_controls()

        # ---- Overlay UI toggle (shown on mouse move when UI hidden) ----
        self._ui_overlay_btn = QPushButton("Show UI", self)
        self._ui_overlay_btn.setFixedSize(110, 30)
        self._ui_overlay_btn.setStyleSheet(
            "QPushButton { background: rgba(30,30,30,210); color: #eee; "
            "border: 1px solid #666; border-radius: 6px; }"
            "QPushButton:hover { background: rgba(50,50,50,230); }"
        )
        self._ui_overlay_btn.clicked.connect(self._toggle_ui_visibility)
        self._ui_overlay_btn.hide()


    # ── Signal wiring ────────────────────────────────────────

    def _connect_signals(self):
        app = QApplication.instance()
        if app is not None:
            app.applicationStateChanged.connect(self._on_app_state_changed)

        self._btn_file.clicked.connect(self._open_file)
        self._btn_play.clicked.connect(self._play)
        self._btn_pause.clicked.connect(self._toggle_pause)
        self._btn_stop.clicked.connect(self._stop_and_restart)
        self._btn_apply_settings.clicked.connect(self._apply_runtime_settings)
        self._chk_metrics.toggled.connect(
            lambda on: self._grp_metrics.setVisible(on))
        self._chk_metrics.toggled.connect(lambda _on: self._save_user_settings())
        self._chk_hide_cursor.toggled.connect(self._on_hide_cursor_toggled)
        self._chk_hide_cursor.toggled.connect(lambda _on: self._save_user_settings())
        self._sld_volume.valueChanged.connect(self._on_volume_changed)
        self._cmb_audio_track.currentIndexChanged.connect(self._on_audio_track_changed)
        self._cmb_prec.currentTextChanged.connect(self._on_precision)
        self._chk_hg.stateChanged.connect(self._on_hg_toggle)
        self._cmb_prec.currentTextChanged.connect(lambda _v: self._save_user_settings())
        self._cmb_res.currentTextChanged.connect(self._on_resolution)
        self._cmb_res.currentTextChanged.connect(lambda _v: self._save_user_settings())
        self._cmb_upscale.currentTextChanged.connect(self._on_upscale_changed)
        self._cmb_upscale.currentTextChanged.connect(lambda _v: self._save_user_settings())
        if hasattr(self, "_chk_film_grain") and self._chk_film_grain is not None:
            self._chk_film_grain.stateChanged.connect(self._on_film_grain_changed)
        self._cmb_view.currentTextChanged.connect(self._on_view)
        self._cmb_view.currentTextChanged.connect(lambda _v: self._save_user_settings())
        self._btn_pop_sdr.clicked.connect(self._toggle_sdr_popout)
        self._btn_pop_hdr.clicked.connect(self._toggle_hdr_popout)
        self._btn_toggle_ui.clicked.connect(self._toggle_ui_visibility)
        if self._video_tabs is not None:
            self._video_tabs.currentChanged.connect(self._on_video_tab_changed)

        self._worker.frame_ready.connect(self._on_frame)
        self._worker.metrics_updated.connect(self._on_metrics)
        self._worker.status_message.connect(self._on_status_message)
        self._worker.playback_finished.connect(self._on_finished)
        self._worker.compile_ready.connect(self._on_compile_ready)
        self._worker.position_updated.connect(self._on_position)
        self._seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self._seek_slider.sliderMoved.connect(self._on_seek)
        self._seek_slider.sliderReleased.connect(self._on_seek_released)

        # HDR info from mpv
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.hdr_info_ready.connect(self._on_hdr_info)

    def _sync_upscale_controls(self):
        """Enable/disable upscale selector based on resolution preset."""
        if not hasattr(self, "_cmb_upscale") or self._cmb_upscale is None:
            return
        scale_key = self._cmb_res.currentText()
        allow = (scale_key in {"540p", "720p"})
        self._cmb_upscale.blockSignals(True)
        if allow:
            self._cmb_upscale.setEnabled(True)
            if self._cmb_upscale.currentText() not in UPSCALER_CHOICES:
                self._cmb_upscale.setCurrentText(DEFAULT_UPSCALER)
        else:
            self._cmb_upscale.setCurrentText(DEFAULT_UPSCALER)
            self._cmb_upscale.setEnabled(False)
        self._cmb_upscale.blockSignals(False)

    def _refresh_resolution_options_for_video(self, path: str):
        """Show only processing presets that do not exceed source resolution."""
        options = list(RESOLUTION_SCALES.keys())
        source_dims = None
        current = self._cmb_res.currentText()
        self._source_proc_dims = None

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if vw > 0 and vh > 0:
                source_dims = (vw, vh)
                filtered = []
                for key, dims in RESOLUTION_SCALES.items():
                    tw, th = (MAX_W, MAX_H) if dims is None else dims
                    if tw <= vw and th <= vh:
                        filtered.append(key)
                if filtered:
                    options = filtered
                    self._source_proc_dims = None
                else:
                    sw = max(2, vw - (vw % 2))
                    sh = max(2, vh - (vh % 2))
                    self._source_proc_dims = (sw, sh)
                    options = ["Source"]
        else:
            cap.release()

        self._cmb_res.blockSignals(True)
        self._cmb_res.clear()
        self._cmb_res.addItems(options)
        if current in options:
            self._cmb_res.setCurrentText(current)
        else:
            self._cmb_res.setCurrentIndex(0)
        self._cmb_res.blockSignals(False)
        self._sync_upscale_controls()

        if source_dims is not None and source_dims[1] < MAX_H:
            self.statusBar().showMessage(
                f"Source is {source_dims[0]}x{source_dims[1]}: hiding higher processing presets."
            )

    def _current_screen_geometry(self) -> QRect:
        """Get full monitor geometry (not availableGeometry) to cover taskbar."""
        win_handle = self.windowHandle()
        screen = win_handle.screen() if win_handle is not None else None
        if screen is None:
            screen = self.screen()
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is None:
            return QRect(0, 0, 1600, 900)
        return screen.geometry()

    def _safe_borderless_geometry(self) -> QRect:
        """Near-fullscreen rect to avoid exclusive/fullscreen optimizations."""
        g = self._current_screen_geometry()
        # Keep a 1px inset so Windows does not treat it like true fullscreen.
        if g.width() > 4 and g.height() > 4:
            return g.adjusted(1, 1, -1, -1)
        return g

    def _set_view_mode_silently(self, mode: str):
        prev = self._cmb_view.blockSignals(True)
        self._cmb_view.setCurrentText(mode)
        self._cmb_view.blockSignals(prev)
        self._on_view(mode)

    def _set_process_priority(self, high: bool):
        try:
            p = psutil.Process(os.getpid())
            if high:
                if self._proc_priority_saved is None:
                    self._proc_priority_saved = p.nice()
                p.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                if self._proc_priority_saved is not None:
                    p.nice(self._proc_priority_saved)
                self._proc_priority_saved = None
        except Exception:
            pass

    @staticmethod
    def _rehost_widget(widget: QWidget, host_layout: QVBoxLayout):
        old_parent = widget.parentWidget()
        if old_parent is not None and old_parent.layout() is not None:
            old_parent.layout().removeWidget(widget)
        widget.setParent(None)
        host_layout.addWidget(widget)

    def _on_video_tab_changed(self, index: int):
        if self._video_tabs is None or index < 0:
            return
        self._pause_for_ui_transition()
        def _apply_tab_switch():
            label = self._video_tabs.tabText(index)
            if label == "Side by Side":
                # Side-by-side needs both panes docked in the split hosts.
                if self._sdr_float_window is not None:
                    self._dock_video_pane("sdr")
                if self._hdr_float_window is not None:
                    self._dock_video_pane("hdr")
                if self._side_sdr_host is not None and self._disp_sdr.parentWidget() is not self._side_sdr_host:
                    self._rehost_widget(self._disp_sdr, self._side_sdr_host.layout())
                if self._side_hdr_host is not None and self._disp_hdr.parentWidget() is not self._side_hdr_host:
                    self._rehost_widget(self._disp_hdr, self._side_hdr_host.layout())
            else:
                # Single tabs own their respective panes when not popped out.
                if self._sdr_float_window is None and self._sdr_tab_host is not None:
                    if self._disp_sdr.parentWidget() is not self._sdr_tab_host:
                        self._rehost_widget(self._disp_sdr, self._sdr_tab_host.layout())
                if self._hdr_float_window is None and self._hdr_tab_host is not None:
                    if self._disp_hdr.parentWidget() is not self._hdr_tab_host:
                        self._rehost_widget(self._disp_hdr, self._hdr_tab_host.layout())
            self._save_user_settings()
        self._with_layout_freeze(_apply_tab_switch, refresh_delay=40)
        if self._playing:
            self._relock_timeline(delay_ms=140, drop_frames=3)

    def _on_app_state_changed(self, state: Qt.ApplicationState):
        active = (state == Qt.ApplicationState.ApplicationActive)
        if self._app_active == active:
            return
        self._app_active = active
        if self._playing:
            self._set_process_priority(True)
            if active:
                self._pause_for_ui_transition()

    def _toggle_sdr_popout(self):
        if self._sdr_float_window is not None:
            self._dock_video_pane("sdr")
            return
        def _apply_pop():
            win = DetachedVideoWindow("sdr", "SDR View")
            win.closed.connect(self._on_video_window_closed)
            self._rehost_widget(self._disp_sdr, win.layout())
            win.move(self.frameGeometry().topLeft() + QPoint(40, 40))
            win.show()
            win.raise_()
            win.activateWindow()
            self._sdr_float_window = win
            self._btn_pop_sdr.setText("Dock SDR")
        self._with_layout_freeze(_apply_pop, refresh_delay=40)
        if self._playing:
            self._relock_timeline(delay_ms=160, drop_frames=3)

    def _toggle_hdr_popout(self):
        if self._hdr_float_window is not None:
            self._dock_video_pane("hdr")
            return
        def _apply_pop():
            win = DetachedVideoWindow("hdr", "HDR View")
            win.closed.connect(self._on_video_window_closed)
            self._rehost_widget(self._disp_hdr, win.layout())
            win.move(self.frameGeometry().topLeft() + QPoint(80, 80))
            win.show()
            win.raise_()
            win.activateWindow()
            self._hdr_float_window = win
            self._btn_pop_hdr.setText("Dock HDR")
        self._with_layout_freeze(_apply_pop, refresh_delay=40)
        if self._playing:
            self._relock_timeline(delay_ms=160, drop_frames=3)
            # Follow-up relock after next position update to prevent stale-frame audio lag.
            QTimer.singleShot(520, lambda: self._relock_timeline(drop_frames=1))

    def _dock_video_pane(self, key: str, from_signal: bool = False):
        def _apply_dock():
            side_mode = (
                self._video_tabs is not None
                and self._video_tabs.currentIndex() >= 0
                and self._video_tabs.tabText(self._video_tabs.currentIndex()) == "Side by Side"
            )
            if key == "sdr":
                win = self._sdr_float_window
                if win is None:
                    return
                if side_mode and self._side_sdr_host is not None:
                    self._rehost_widget(self._disp_sdr, self._side_sdr_host.layout())
                elif self._sdr_tab_host is not None:
                    self._rehost_widget(self._disp_sdr, self._sdr_tab_host.layout())
                if not from_signal:
                    try:
                        win.closed.disconnect(self._on_video_window_closed)
                    except Exception:
                        pass
                    win.close()
                self._sdr_float_window = None
                self._btn_pop_sdr.setText("Pop SDR")
            elif key == "hdr":
                win = self._hdr_float_window
                if win is None:
                    return
                if side_mode and self._side_hdr_host is not None:
                    self._rehost_widget(self._disp_hdr, self._side_hdr_host.layout())
                elif self._hdr_tab_host is not None:
                    self._rehost_widget(self._disp_hdr, self._hdr_tab_host.layout())
                if not from_signal:
                    try:
                        win.closed.disconnect(self._on_video_window_closed)
                    except Exception:
                        pass
                    win.close()
                self._hdr_float_window = None
                self._btn_pop_hdr.setText("Pop HDR")
        self._with_layout_freeze(_apply_dock, refresh_delay=40)
        if self._playing:
            self._relock_timeline(delay_ms=160, drop_frames=3)
            if str(key).lower() == "hdr":
                # Follow-up relock after next position update to prevent stale-frame audio lag.
                QTimer.singleShot(520, lambda: self._relock_timeline(drop_frames=1))

    def _on_video_window_closed(self, key: str):
        if self._ui_closing:
            return
        self._dock_video_pane(str(key), from_signal=True)

    @staticmethod
    def _without_fullscreen(state: Qt.WindowState) -> Qt.WindowState:
        return state & ~Qt.WindowState.WindowFullScreen

    def _set_immersive_video_ui(self, enabled: bool):
        """Hide controls/panels and make video surface edge-to-edge."""
        if self._root_layout is None:
            return

        targets = {
            "row1": self._row1_widget,
            "row2": self._row2_widget,
            "metrics": self._grp_metrics,
            "hdr": self._grp_hdr,
        }

        def _ensure_effect(widget: QWidget) -> QGraphicsOpacityEffect:
            eff = self._ui_anim_effects.get(widget)
            if eff is None:
                eff = QGraphicsOpacityEffect(widget)
                eff.setOpacity(1.0)
                widget.setGraphicsEffect(eff)
                self._ui_anim_effects[widget] = eff
            return eff

        def _animate_widget(widget: QWidget, show: bool):
            if widget is None:
                return
            if self._ui_anim_duration_ms <= 0:
                widget.setVisible(bool(show))
                eff = self._ui_anim_effects.get(widget)
                if eff is not None:
                    eff.setOpacity(1.0)
                return
            anim = self._ui_anim_running.get(widget)
            if anim is not None:
                anim.stop()
            eff = _ensure_effect(widget)
            anim = QPropertyAnimation(eff, b"opacity", self)
            anim.setDuration(max(60, int(self._ui_anim_duration_ms)))
            anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            if show:
                widget.setVisible(True)
                eff.setOpacity(0.0)
                anim.setStartValue(0.0)
                anim.setEndValue(1.0)
            else:
                eff.setOpacity(1.0)
                anim.setStartValue(1.0)
                anim.setEndValue(0.0)

            def _done():
                if not show:
                    widget.setVisible(False)
                    eff.setOpacity(1.0)

            anim.finished.connect(_done)
            self._ui_anim_running[widget] = anim
            anim.start()

        if enabled:
            self._immersive_saved_vis = {k: w.isVisible() for k, w in targets.items()}
            m = self._root_layout.contentsMargins()
            self._immersive_saved_margins = (m.left(), m.top(), m.right(), m.bottom())
            self._immersive_saved_spacing = self._root_layout.spacing()
            self._immersive_saved_view_mode = self._cmb_view.currentText()

            for w in targets.values():
                if w is not None and w.isVisible():
                    _animate_widget(w, False)
            self._root_layout.setContentsMargins(0, 0, 0, 0)
            self._root_layout.setSpacing(0)
            if self._video_tabs is not None:
                self._video_tabs.tabBar().setVisible(False)
            return

        for key, w in targets.items():
            want = self._immersive_saved_vis.get(key, True)
            if w is not None:
                _animate_widget(w, bool(want))

        if self._immersive_saved_margins is not None:
            l, t, r, b = self._immersive_saved_margins
            self._root_layout.setContentsMargins(l, t, r, b)
        if self._immersive_saved_spacing is not None:
            self._root_layout.setSpacing(self._immersive_saved_spacing)
        if self._video_tabs is not None:
            self._video_tabs.tabBar().setVisible(True)
        if self._immersive_saved_view_mode:
            self._set_view_mode_silently(self._immersive_saved_view_mode)

    def _position_ui_overlay(self):
        if self._ui_overlay_btn is None:
            return
        margin = 16
        w = self._ui_overlay_btn.width()
        h = self._ui_overlay_btn.height()
        x = max(margin, self.width() - w - margin)
        y = margin
        self._ui_overlay_btn.move(x, y)

    def _show_ui_overlay_temporarily(self):
        if self._ui_overlay_btn is None:
            return
        if not self._ui_hidden:
            self._ui_overlay_btn.hide()
            return
        if self._row3_widget is not None:
            self._row3_widget.setVisible(True)
        self._position_ui_overlay()
        self._ui_overlay_btn.show()
        self._ui_overlay_btn.raise_()
        if self._ui_overlay_timer is None:
            self._ui_overlay_timer = QTimer(self)
            self._ui_overlay_timer.setSingleShot(True)
            self._ui_overlay_timer.timeout.connect(self._hide_ui_overlay)
        self._ui_overlay_timer.stop()
        self._ui_overlay_timer.start(int(self._ui_overlay_hide_ms))

    def _hide_ui_overlay(self):
        if self._ui_overlay_btn is not None:
            self._ui_overlay_btn.hide()

    def _toggle_ui_visibility(self):
        if not self._playing:
            return
        self._pause_for_ui_transition()
        self._ui_hidden = not self._ui_hidden
        self._set_immersive_video_ui(self._ui_hidden)
        if self._ui_hidden:
            self.menuBar().setVisible(False)
            self.statusBar().setVisible(False)
            self._show_ui_overlay_temporarily()
        else:
            self.menuBar().setVisible(True)
            self.statusBar().setVisible(True)
            if self._row3_widget is not None:
                self._row3_widget.setVisible(True)
            if self._ui_overlay_btn is not None:
                self._ui_overlay_btn.hide()
        if self._btn_toggle_ui is not None:
            self._btn_toggle_ui.setText("Show UI" if self._ui_hidden else "Hide UI")
        # Layout changes can momentarily stall video presentation.
        # Re-anchor audio to the current timeline after the UI toggle.
        if self._playing:
            self._relock_timeline(delay_ms=120, drop_frames=2)

    def _set_pause_button_labels(self, paused: bool):
        if paused:
            self._btn_pause.setText("▶  Resume")
        else:
            self._btn_pause.setText("⏸  Pause")

    def _ensure_window_refresh_timer(self):
        if self._window_refresh_timer is not None:
            return
        self._window_refresh_timer = QTimer(self)
        self._window_refresh_timer.setSingleShot(True)
        self._window_refresh_timer.timeout.connect(self._refresh_mpv_after_window_state_change)

    def _begin_layout_freeze(self):
        return

    def _end_layout_freeze(self, refresh_delay: int | None = 40,
                           refresh_soft_only: bool = False):
        if refresh_delay is not None:
            self._schedule_window_state_refresh(
                refresh_delay, soft_only=refresh_soft_only
            )

    def _with_layout_freeze(self, fn, refresh_delay: int | None = 40,
                            refresh_soft_only: bool = False):
        fn()
        if refresh_delay is not None:
            self._schedule_window_state_refresh(
                refresh_delay, soft_only=refresh_soft_only
            )

    def _schedule_window_state_refresh(self, delay_ms: int = 140,
                                       soft_only: bool = False):
        self._ensure_window_refresh_timer()
        self._window_refresh_soft_only = bool(soft_only)
        self._window_refresh_timer.stop()
        self._window_refresh_timer.start(max(0, int(delay_ms)))

    def _ensure_overlay_reposition_timer(self):
        if self._overlay_reposition_timer is not None:
            return
        self._overlay_reposition_timer = QTimer(self)
        self._overlay_reposition_timer.setSingleShot(True)
        self._overlay_reposition_timer.timeout.connect(self._position_ui_overlay)

    def _schedule_overlay_position(self, delay_ms: int = 16):
        self._position_ui_overlay()

    def _ensure_ui_pause_timer(self):
        if self._ui_pause_timer is not None:
            return
        self._ui_pause_timer = QTimer(self)
        self._ui_pause_timer.setSingleShot(True)

    def _ensure_periodic_relock_timer(self):
        if self._periodic_relock_timer is not None:
            return
        self._periodic_relock_timer = QTimer(self)
        self._periodic_relock_timer.setSingleShot(False)
        self._periodic_relock_timer.timeout.connect(self._periodic_relock_tick)

    def _periodic_relock_tick(self):
        if not self._playing:
            return
        if self._worker is None or self._worker.is_paused:
            return
        if self._startup_sync_pending:
            return
        if not self._active_use_mpv:
            return
        # Light-touch resync to keep HDR/SDR aligned over time (video-only).
        fps = getattr(self, "_vid_fps", 30.0)
        target_sec = float(self._last_seek_frame) / max(fps, 1e-6)
        if self._audio_available and self._audio_player is not None:
            try:
                target_sec = float(self._audio_player.position()) / 1000.0
            except Exception:
                target_sec = float(self._last_seek_frame) / max(fps, 1e-6)
        if self._worker is not None:
            self._worker.flush_hdr_queue(drop_frames=1)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.seek_seconds(target_sec)
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.seek_seconds(target_sec)

    def _start_periodic_relock(self):
        self._ensure_periodic_relock_timer()
        if self._periodic_relock_timer is None:
            return
        self._periodic_relock_timer.stop()
        self._periodic_relock_timer.start(int(self._periodic_relock_ms))

    def _stop_periodic_relock(self):
        if self._periodic_relock_timer is not None:
            self._periodic_relock_timer.stop()

    def _pause_for_ui_transition(self, duration_ms: int | None = None,
                                 wait_for_stable: bool = True):
        if not self._playing:
            return
        if self._worker is not None and self._worker.is_paused:
            return
        if self._ui_pause_timer is not None and self._ui_pause_timer.isActive():
            self._ui_pause_timer.stop()
        delay = int(duration_ms if duration_ms is not None else self._ui_pause_duration_ms)
        # FSR shader refresh can stall mpv a bit longer during UI changes.
        if self._active_mpv_scale_kernel == "fsr":
            delay += 140
        delay = max(60, delay)

        if self._worker is not None:
            self._worker.pause()
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.set_paused(True)
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.set_paused(True)
        if self._audio_available:
            self._set_audio_paused(True)

        self._ensure_ui_pause_timer()
        if self._ui_pause_timer is None:
            return

        def _resume():
            if not self._playing:
                return
            if self._worker is not None and self._worker.is_paused:
                self._worker.resume()
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_mpv.set_paused(False)
            if self._disp_sdr_mpv is not None:
                self._disp_sdr_mpv.set_paused(False)
            if self._audio_available and not self._startup_sync_pending:
                # Keep audio paused until FPS stabilizes after the UI change.
                self._startup_audio_gate_active = True
                self._startup_audio_gate_count = 0
                self._ui_resync_gate_strict = True
                now_t = time.perf_counter()
                self._audio_seek_guard_until = max(self._audio_seek_guard_until, now_t + 0.8)

        try:
            self._ui_pause_timer.timeout.disconnect()
        except Exception:
            pass
        self._ui_pause_timer.timeout.connect(_resume)
        self._ui_pause_timer.start(delay)

    def _resync_audio_to_current_timeline(self):
        if not self._playing:
            return
        fps = getattr(self, "_vid_fps", 30.0)
        sec = float(self._last_seek_frame) / max(fps, 1e-6)
        if self._audio_available:
            self._force_audio_seek(sec)
            if not self._worker.is_paused and not self._startup_sync_pending:
                self._set_audio_paused(False)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.seek_seconds(sec)
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.seek_seconds(sec)

    def _relock_timeline(self, delay_ms: int = 0, drop_frames: int = 2):
        """Force a timeline relock (audio + HDR queue) after UI actions."""
        if not self._playing:
            return
        def _do():
            if not self._playing:
                return
            if self._worker is not None:
                self._worker.flush_hdr_queue(drop_frames=drop_frames)
            self._resync_audio_to_current_timeline()
        if delay_ms > 0:
            QTimer.singleShot(int(delay_ms), _do)
        else:
            _do()

    def _force_audio_seek(self, sec: float):
        """Aggressive audio resync: double-seek to reduce drift after UI changes."""
        if not self._audio_available:
            return
        self._seek_audio_seconds(sec)
        def _second_seek():
            self._seek_audio_seconds(sec)
        QTimer.singleShot(20, _second_seek)

    def _schedule_precision_audio_resync(self, delay_ms: int = 240):
        if not self._playing:
            return
        # Precision swaps can stall inference; pause briefly, then re-anchor audio.
        self._audio_resync_pending = True
        self._audio_fps_recovered = False
        if self._audio_available:
            self._set_audio_paused(True)
        now_t = time.perf_counter()
        self._audio_seek_guard_until = max(self._audio_seek_guard_until, now_t + 0.6)

        def _finish():
            if not self._playing:
                return
            self._audio_fps_recovered = True
            self._resync_audio_to_current_timeline()

        QTimer.singleShot(max(0, int(delay_ms)), _finish)

    def _pause_for_precision_swap(self, key: str, timeout_ms: int = 20000):
        if not self._playing or self._worker is None:
            return
        # Track which precision we are waiting to stabilize.
        self._precision_swap_pending = key
        # Only auto-resume if we weren't already paused by the user.
        self._precision_pause_armed = not self._worker.is_paused

        if self._precision_pause_armed:
            self._worker.pause()
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_mpv.set_paused(True)
            if self._disp_sdr_mpv is not None:
                self._disp_sdr_mpv.set_paused(True)
            if self._audio_available:
                self._set_audio_paused(True)

        # Safety timeout in case we miss the "Ready" status message.
        if self._precision_swap_timer is None:
            self._precision_swap_timer = QTimer(self)
            self._precision_swap_timer.setSingleShot(True)
        else:
            self._precision_swap_timer.stop()

        def _timeout_resume():
            self._resume_after_precision_swap(force=True)

        try:
            self._precision_swap_timer.timeout.disconnect()
        except Exception:
            pass
        self._precision_swap_timer.timeout.connect(_timeout_resume)
        self._precision_swap_timer.start(int(timeout_ms))

    def _resume_after_precision_swap(self, force: bool = False):
        if self._precision_swap_pending is None:
            return
        if self._precision_swap_timer is not None:
            self._precision_swap_timer.stop()
        if self._precision_pause_armed:
            if self._worker is not None and self._worker.is_paused:
                self._worker.resume()
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_mpv.set_paused(False)
            if self._disp_sdr_mpv is not None:
                self._disp_sdr_mpv.set_paused(False)
            if self._audio_available and not self._startup_sync_pending:
                # Keep audio paused until FPS stabilizes after the precision swap.
                self._startup_audio_gate_active = True
                self._startup_audio_gate_count = 0
        self._precision_swap_pending = None
        self._precision_pause_armed = False

    def _toggle_borderless_full_window(self):
        now_t = time.perf_counter()
        if (now_t - self._window_toggle_last_t) < self._window_toggle_cooldown_s:
            return
        self._window_toggle_last_t = now_t
        if self._borderless_full_window:
            self._exit_borderless_full_window()
        else:
            self._enter_borderless_full_window()

    def _enter_borderless_full_window(self):
        if self._borderless_full_window:
            return
        self._pause_for_ui_transition()
        def _apply_full():
            self._saved_window_geometry = self.geometry()
            self._saved_window_state = self._without_fullscreen(self.windowState())
            # Borderless fullscreen geometry: hides taskbar/top bar without
            # using true fullscreen (keeps "zoom" feel).
            self.menuBar().setVisible(True)
            self.statusBar().setVisible(True)
            self._ui_hidden = False
            self._set_immersive_video_ui(False)
            if self._btn_toggle_ui is not None:
                self._btn_toggle_ui.setText("Hide UI")
            self.setWindowState(self._without_fullscreen(self.windowState()))
            self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
            self.show()
            self.setGeometry(self._current_screen_geometry())

            self._borderless_full_window = True
            if self._act_borderless_full_window is not None:
                self._act_borderless_full_window.setChecked(True)
            self._schedule_overlay_position(0)
        self._with_layout_freeze(
            _apply_full, refresh_delay=140, refresh_soft_only=True
        )
        self._relock_timeline(delay_ms=180, drop_frames=3)

    def _exit_borderless_full_window(self):
        if not self._borderless_full_window:
            return
        self._pause_for_ui_transition()
        def _apply_exit():
            restore_state = self._without_fullscreen(self._saved_window_state)
            restore_geom = self._saved_window_geometry
            self._borderless_full_window = False
            if self._act_borderless_full_window is not None:
                self._act_borderless_full_window.setChecked(False)

            self.setWindowState(self._without_fullscreen(self.windowState()))
            self.setWindowFlag(Qt.WindowType.FramelessWindowHint, False)
            self.show()
            self.menuBar().setVisible(True)
            self.statusBar().setVisible(True)
            self._set_immersive_video_ui(False)
            self._ui_hidden = False
            if self._btn_toggle_ui is not None:
                self._btn_toggle_ui.setText("Hide UI")
            if restore_geom is not None:
                self.setGeometry(restore_geom)
            if bool(restore_state & Qt.WindowState.WindowMaximized):
                self.showMaximized()
            else:
                self.showNormal()
            self._schedule_overlay_position(0)
        self._with_layout_freeze(
            _apply_exit, refresh_delay=140, refresh_soft_only=True
        )
        self._relock_timeline(delay_ms=180, drop_frames=3)

    def _should_use_mpv_pipeline(self) -> bool:
        return self._disp_hdr_mpv is not None

    def _load_user_settings(self, initial_resolution, initial_precision,
                            initial_view, initial_use_hg,
                            initial_upscale, initial_film_grain):
        """Load persisted GUI preferences unless explicitly overridden by CLI."""
        data = {}
        if os.path.isfile(_PREFS_PATH):
            try:
                with open(_PREFS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        if initial_precision is None:
            p = data.get("precision")
            if p in _available_precision_keys():
                self._cmb_prec.setCurrentText(p)
        if initial_resolution is None:
            r = data.get("resolution")
            if r in RESOLUTION_SCALES or r == "Source":
                self._cmb_res.setCurrentText(r)
        if hasattr(self, "_cmb_upscale"):
            if isinstance(initial_upscale, str) and initial_upscale in UPSCALER_CHOICES:
                self._cmb_upscale.setCurrentText(initial_upscale)
            else:
                um = data.get("upscale_mode")
                if isinstance(um, str) and um in UPSCALER_CHOICES:
                    self._cmb_upscale.setCurrentText(um)
        if hasattr(self, "_chk_film_grain") and self._chk_film_grain is not None:
            if initial_film_grain is None:
                fg = data.get("film_grain")
                if isinstance(fg, bool):
                    self._chk_film_grain.setChecked(fg)
            else:
                self._chk_film_grain.setChecked(str(initial_film_grain).strip() == "1")
        if initial_view is None:
            v = data.get("view")
            if v == "Tabbed":
                self._cmb_view.setCurrentText("Tabbed")
        # Restore last active tab (defaults to HDR if missing).
        if self._video_tabs is not None:
            saved_tab = data.get("active_tab", "HDR")
            for i in range(self._video_tabs.count()):
                if self._video_tabs.tabText(i) == saved_tab:
                    self._video_tabs.setCurrentIndex(i)
                    break
            else:
                # If saved tab not found, default to HDR.
                for i in range(self._video_tabs.count()):
                    if self._video_tabs.tabText(i) == "HDR":
                        self._video_tabs.setCurrentIndex(i)
                        break
        self._chk_hg.setChecked(bool(data.get("use_hg", True)))
        if initial_use_hg is not None:
            self._chk_hg.setChecked(str(initial_use_hg).strip() == "1")

        m = data.get("show_metrics")
        if isinstance(m, bool):
            self._chk_metrics.setChecked(m)
            self._grp_metrics.setVisible(m)

        vol = data.get("volume_percent")
        if isinstance(vol, int):
            self._sld_volume.setValue(max(0, min(100, vol)))
        aidx = data.get("audio_track")
        if isinstance(aidx, int):
            self._selected_audio_track = max(0, int(aidx))
        hc = data.get("hide_cursor_idle")
        if isinstance(hc, bool):
            self._chk_hide_cursor.setChecked(hc)

        self._sync_upscale_controls()

    def _save_user_settings(self):
        data = {
            "precision": self._cmb_prec.currentText(),
            "resolution": self._cmb_res.currentText(),
            "upscale_mode": self._cmb_upscale.currentText() if hasattr(self, "_cmb_upscale") else DEFAULT_UPSCALER,
            "view": self._cmb_view.currentText(),
            "show_metrics": self._chk_metrics.isChecked(),
            "use_hg": self._chk_hg.isChecked(),
            "film_grain": bool(getattr(self, "_chk_film_grain", None) and self._chk_film_grain.isChecked()),
            "volume_percent": int(self._volume_percent),
            "audio_track": int(self._selected_audio_track),
            "hide_cursor_idle": bool(self._cursor_idle_enabled),
        }
        if self._video_tabs is not None and self._video_tabs.currentIndex() >= 0:
            data["active_tab"] = self._video_tabs.tabText(self._video_tabs.currentIndex())
        try:
            with open(_PREFS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _init_audio_backend(self):
        """Initialize Qt audio backend (used for seekable timeline audio)."""
        if not _HAS_QT_AUDIO:
            self._audio_available = False
            return
        try:
            self._audio_player = QMediaPlayer(self)
            self._audio_output = QAudioOutput(self)
            self._audio_output.setVolume(self._volume_percent / 100.0)
            self._audio_player.setAudioOutput(self._audio_output)
            self._audio_available = True
        except Exception:
            self._audio_player = None
            self._audio_output = None
            self._audio_available = False

    @staticmethod
    def _format_audio_track_label(track: dict, fallback_idx: int) -> str:
        n = int(track.get("ordinal", fallback_idx)) + 1
        lang = str(track.get("language") or "und").upper()
        codec = str(track.get("codec") or "audio").upper()
        ch = int(track.get("channels") or 0)
        ch_sfx = f" {ch}ch" if ch > 0 else ""
        title = str(track.get("title") or "").strip()
        default_sfx = " (Default)" if bool(track.get("is_default", False)) else ""
        base = f"{n}. {lang} {codec}{ch_sfx}"
        if title:
            return f"{base} - {title}{default_sfx}"
        return f"{base}{default_sfx}"

    def _refresh_audio_tracks_for_video(self, path: str):
        tracks = _probe_audio_streams(path)
        self._audio_tracks = tracks
        self._cmb_audio_track.blockSignals(True)
        self._cmb_audio_track.clear()
        if not tracks:
            self._cmb_audio_track.addItem("No audio tracks detected")
            self._cmb_audio_track.setEnabled(False)
            self._selected_audio_track = 0
            self._cmb_audio_track.blockSignals(False)
            return

        default_idx = 0
        for i, t in enumerate(tracks):
            self._cmb_audio_track.addItem(self._format_audio_track_label(t, i), i)
            if bool(t.get("is_default", False)):
                default_idx = i

        preferred = self._selected_audio_track
        if preferred < 0 or preferred >= len(tracks):
            preferred = default_idx
        self._selected_audio_track = preferred
        self._cmb_audio_track.setCurrentIndex(preferred)
        self._cmb_audio_track.setEnabled(len(tracks) > 1)
        if len(tracks) > 1:
            self._cmb_audio_track.setToolTip("Choose audio stream from source file.")
        else:
            self._cmb_audio_track.setToolTip("Single audio stream detected.")
        self._cmb_audio_track.blockSignals(False)

    def _apply_selected_audio_track_qt_async(self):
        if not self._audio_available or self._audio_player is None:
            return
        if not self._audio_tracks:
            return
        target = max(0, min(int(self._selected_audio_track), len(self._audio_tracks) - 1))
        self._audio_apply_token += 1
        token = self._audio_apply_token

        def _try_apply(attempt: int = 0):
            if token != self._audio_apply_token:
                return
            p = self._audio_player
            if p is None:
                return
            try:
                qtracks = p.audioTracks()
            except Exception:
                qtracks = []
            if qtracks:
                idx = max(0, min(target, len(qtracks) - 1))
                try:
                    p.setActiveAudioTrack(idx)
                    return
                except Exception:
                    pass
            if attempt < 20:
                QTimer.singleShot(120, lambda: _try_apply(attempt + 1))

        _try_apply(0)

    def _ensure_selected_audio_track_qt(self):
        """Re-assert selected Qt audio track if backend switched tracks after seek/rebuffer."""
        if not self._audio_available or self._audio_player is None or not self._audio_tracks:
            return
        p = self._audio_player
        try:
            qtracks = p.audioTracks()
        except Exception:
            qtracks = []
        if not qtracks:
            return
        target = max(0, min(int(self._selected_audio_track), len(qtracks) - 1))
        try:
            active = int(p.activeAudioTrack())
        except Exception:
            active = None
        if active != target:
            try:
                p.setActiveAudioTrack(target)
            except Exception:
                pass

    def _apply_selected_audio_track_mpv_async(self):
        if self._audio_available:
            return
        if self._disp_hdr_mpv is None or not self._audio_tracks:
            return
        target = max(0, min(int(self._selected_audio_track), len(self._audio_tracks) - 1))
        self._audio_apply_token += 1
        token = self._audio_apply_token

        def _try_apply(attempt: int = 0):
            if token != self._audio_apply_token:
                return
            if self._disp_hdr_mpv is None:
                return
            if self._disp_hdr_mpv.set_audio_track_ordinal(target):
                return
            if attempt < 20:
                QTimer.singleShot(150, lambda: _try_apply(attempt + 1))

        _try_apply(0)

    def _apply_volume_to_backends(self):
        """Apply current volume/mute policy to Qt audio and mpv fallback audio."""
        muted = (self._auto_muted_low_fps or self._scrub_muted)
        if muted and self._audio_fade_timer is not None:
            self._audio_fade_timer.stop()
        if self._audio_output is not None:
            self._audio_output.setVolume(0.0 if muted else (self._volume_percent / 100.0))
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.set_muted(muted)
            if not muted:
                self._disp_hdr_mpv.set_volume_percent(self._volume_percent)

    def _start_audio_restore_fade(self, duration_ms: int = 140):
        """Smoothly restore audio level after auto-mute release."""
        if self._auto_muted_low_fps:
            return
        if self._audio_fade_timer is None:
            self._audio_fade_timer = QTimer(self)
            self._audio_fade_timer.timeout.connect(self._on_audio_fade_tick)
        self._audio_fade_step_idx = 0
        if self._audio_output is not None:
            self._audio_output.setVolume(0.0)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.set_muted(False)
            self._disp_hdr_mpv.set_volume_percent(0)
        step_ms = max(10, int(duration_ms / max(1, self._audio_fade_steps)))
        self._audio_fade_timer.start(step_ms)

    def _on_audio_fade_tick(self):
        if self._auto_muted_low_fps:
            if self._audio_fade_timer is not None:
                self._audio_fade_timer.stop()
            return
        self._audio_fade_step_idx += 1
        ratio = min(1.0, self._audio_fade_step_idx / max(1, self._audio_fade_steps))
        target = max(0.0, min(1.0, self._volume_percent / 100.0))
        if self._audio_output is not None:
            self._audio_output.setVolume(target * ratio)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.set_volume_percent(int(round(self._volume_percent * ratio)))
        if ratio >= 1.0 and self._audio_fade_timer is not None:
            self._audio_fade_timer.stop()
            self._apply_volume_to_backends()

    def _set_low_fps_mute(self, enabled: bool):
        enabled = bool(enabled)
        if self._auto_muted_low_fps == enabled:
            return
        self._auto_muted_low_fps = enabled
        if enabled:
            # Arm one-shot A/V re-sync for when playback recovers.
            self._audio_resync_pending = True
            self._audio_fps_recovered = False
            # Keep audio clock from running ahead while muted.
            if self._audio_available:
                self._set_audio_paused(True)
            self._apply_volume_to_backends()
            self.statusBar().showMessage("Audio auto-muted (FPS below 20).")
        else:
            self._audio_fps_recovered = True
            # Re-anchor audio to current video position before unmuting.
            if self._audio_available and self._playing:
                fps = getattr(self, "_vid_fps", 30.0)
                self._seek_audio_seconds(float(self._last_seek_frame) / max(fps, 1e-6))
                if not self._worker.is_paused and not self._startup_sync_pending:
                    self._set_audio_paused(False)
            if self._scrub_muted and (not self._seek_slider.isSliderDown()):
                self._scrub_muted = False
                self._apply_volume_to_backends()
            self._start_audio_restore_fade()
            self.statusBar().showMessage("Audio restored.")

    def _arm_mute_until_fps_recovery(self):
        """Force mute now; unmute only via measured FPS recovery logic."""
        self._low_fps_count = 0
        self._high_fps_count = 0
        self._audio_resync_pending = True
        self._audio_fps_recovered = False
        if not self._auto_muted_low_fps:
            self._set_low_fps_mute(True)

    def _update_auto_mute_from_fps(self, fps_value: float):
        """Sustained low-FPS auto-mute with heavy debounce to prevent flapping."""
        fps = float(fps_value)
        low_trip = 19.5
        high_trip = 22.5

        if fps < low_trip:
            self._low_fps_count += 1
            self._high_fps_count = 0
        elif fps >= high_trip:
            self._high_fps_count += 1
            self._low_fps_count = 0
        else:
            self._low_fps_count = max(0, self._low_fps_count - 1)
            self._high_fps_count = max(0, self._high_fps_count - 1)

        # metrics update every 2 frames; at 24fps this is ~12 samples/sec.
        # Require ~2.5s low fps before muting, ~0.25s healthy fps before unmuting.
        if not self._auto_muted_low_fps and self._low_fps_count >= 30:
            self._set_low_fps_mute(True)
            self._low_fps_count = 0
            self._high_fps_count = 0
        elif self._auto_muted_low_fps and self._high_fps_count >= 2:
            self._set_low_fps_mute(False)
            self._low_fps_count = 0
            self._high_fps_count = 0

    def _on_volume_changed(self, value: int):
        self._volume_percent = int(value)
        self._lbl_volume_val.setText(f"{self._volume_percent}%")
        if self._audio_fade_timer is not None and self._audio_fade_timer.isActive():
            # Fade tick uses current _volume_percent as target.
            self._save_user_settings()
            return
        self._apply_volume_to_backends()
        self._save_user_settings()

    def _show_cursor(self):
        if self._cursor_hidden:
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass
            self._cursor_hidden = False

    def _hide_cursor_if_idle(self):
        if not self._cursor_idle_enabled or not self._playing:
            return
        if self._cursor_hidden:
            return
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.BlankCursor))
        self._cursor_hidden = True

    def _arm_cursor_idle_timer(self):
        if self._cursor_idle_timer is None:
            return
        self._cursor_idle_timer.stop()
        if self._cursor_idle_enabled and self._playing:
            self._cursor_idle_timer.start(int(self._cursor_idle_ms))

    def _on_hide_cursor_toggled(self, enabled: bool):
        self._cursor_idle_enabled = bool(enabled)
        if not self._cursor_idle_enabled:
            if self._cursor_idle_timer is not None:
                self._cursor_idle_timer.stop()
            self._show_cursor()
        else:
            self._arm_cursor_idle_timer()

    def _init_cursor_idle_tracking(self):
        self._cursor_idle_timer = QTimer(self)
        self._cursor_idle_timer.setSingleShot(True)
        self._cursor_idle_timer.timeout.connect(self._hide_cursor_if_idle)
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        et = event.type()
        if et == QEvent.Type.Resize:
            self._position_ui_overlay()
        if et in (
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.Wheel,
            QEvent.Type.KeyPress,
        ):
            self._show_cursor()
            self._arm_cursor_idle_timer()
            if self._ui_hidden:
                self._show_ui_overlay_temporarily()
        return super().eventFilter(obj, event)

    def _on_audio_track_changed(self, index: int):
        if index < 0:
            return
        self._selected_audio_track = int(index)
        self._save_user_settings()
        if not self._playing:
            return
        if self._audio_available:
            self._apply_selected_audio_track_qt_async()
        else:
            self._apply_selected_audio_track_mpv_async()

    def _start_audio_playback(self, path: str):
        if not self._audio_available or self._audio_player is None:
            return
        try:
            self._audio_player.stop()
            self._audio_player.setSource(QUrl.fromLocalFile(path))
            self._audio_player.setPosition(0)
            self._audio_player.setPlaybackRate(1.0)
            self._audio_player.play()
            self._apply_selected_audio_track_qt_async()
            self._apply_volume_to_backends()
        except Exception as exc:
            self.statusBar().showMessage(f"Qt audio unavailable: {exc}")
            self._audio_available = False

    def _stop_audio_playback(self):
        if self._audio_player is not None:
            try:
                self._audio_player.stop()
            except Exception:
                pass

    def _set_audio_paused(self, paused: bool):
        if not self._audio_available or self._audio_player is None:
            return
        try:
            if paused:
                self._audio_player.pause()
            else:
                self._audio_player.setPlaybackRate(1.0)
                self._audio_player.play()
        except Exception:
            pass

    def _seek_audio_seconds(self, sec: float):
        if not self._audio_available or self._audio_player is None:
            return
        try:
            self._audio_player.setPosition(int(max(0.0, sec) * 1000.0))
        except Exception:
            pass

    def _release_startup_sync(self):
        """Unpause worker/mpv/audio together after startup warm sync."""
        if not self._playing or not self._startup_sync_pending:
            return
        if self._user_pause_override_startup:
            # User explicitly paused during startup; do not auto-resume.
            self._startup_sync_pending = False
            return
        self._startup_sync_pending = False
        if self._audio_available:
            self._seek_audio_seconds(0.0)
            # Keep audio paused until startup FPS gate opens.
            self._set_audio_paused(True)
        self._worker.resume()
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.set_paused(False)
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.set_paused(False)
        if self._startup_audio_gate_active:
            self._arm_mute_until_fps_recovery()
        # Force an initial timeline relock so audio isn't ahead if user never seeks.
        self._relock_timeline(delay_ms=160, drop_frames=3)
        # Follow-up relock after startup UI interactions to keep HDR aligned.
        QTimer.singleShot(520, lambda: self._relock_timeline(drop_frames=2))

    def _has_pending_setting_changes(self) -> bool:
        if not self._playing:
            return False
        upscale_changed = False
        if hasattr(self, "_cmb_upscale") and self._cmb_res.currentText() in {"540p", "720p"}:
            upscale_changed = (self._cmb_upscale.currentText() != self._active_upscale_mode)
        film_grain_changed = False
        if hasattr(self, "_chk_film_grain") and self._chk_film_grain is not None:
            film_grain_changed = (self._chk_film_grain.isChecked() != self._active_film_grain)
        return (
            self._cmb_prec.currentText() != self._active_precision
            or self._cmb_res.currentText() != self._active_resolution
            or self._chk_hg.isChecked() != self._active_use_hg
            or upscale_changed
            or film_grain_changed
        )

    def _update_apply_button_state(self):
        self._btn_apply_settings.setEnabled(self._has_pending_setting_changes())

    def _choose_preview_frame(self, path: str) -> np.ndarray | None:
        """Pick a representative frame (non-black, high-detail) for idle preview."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                probe_ids = [
                    int(total * 0.08),
                    int(total * 0.22),
                    int(total * 0.38),
                    int(total * 0.55),
                    int(total * 0.72),
                ]
            else:
                probe_ids = [0]

            best = None
            best_score = -1.0
            for fid in probe_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fid))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = float(gray.std())
                if score > best_score:
                    best = frame
                    best_score = score

            if best is not None:
                return best

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    def _show_idle_preview(self, path: str):
        """Render selected-video preview without starting playback."""
        preview = self._choose_preview_frame(path)
        if preview is not None:
            if self._disp_sdr_cpu is not None:
                self._disp_sdr_cpu.update_frame(preview)
            if self._disp_hdr_cpu is not None:
                self._disp_hdr_cpu.update_frame(preview)
            if self._disp_sdr_mpv is not None:
                self._disp_sdr_stack.setCurrentWidget(self._disp_sdr_cpu)
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_cpu)
        else:
            if self._disp_sdr_cpu is not None:
                self._disp_sdr_cpu.clear_display()
            if self._disp_hdr_cpu is not None:
                self._disp_hdr_cpu.clear_display()

    def _prepare_idle_timeline(self, path: str):
        """Populate duration labels/slider in idle state."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._seek_slider.setEnabled(False)
            self._seek_slider.setValue(0)
            self._lbl_time.setText("0:00")
            self._lbl_duration.setText("0:00")
            return
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()

        self._vid_fps = fps if fps > 0 else 30.0
        self._seek_slider.setRange(0, max(0, total_frames - 1))
        self._seek_slider.setValue(0)
        self._seek_slider.setEnabled(False)
        self._lbl_time.setText("0:00")
        dur_secs = total_frames / self._vid_fps if self._vid_fps > 0 else 0.0
        self._lbl_duration.setText(self._fmt_time(dur_secs))

    # ── Slots: file / tools ───────────────────────────────────

    def _on_compile_ready(self):
        """Called on main thread after Triton compile finishes.
        Safe to start mpv now — GPU is free from autotuning."""
        if self._compile_dlg is not None:
            self._compile_dlg.close()
            self._compile_dlg.deleteLater()
            self._compile_dlg = None

        pending = getattr(self, '_pending_mpv_start', None)
        if pending and self._disp_hdr_mpv is not None:
            (pw, ph, fps, scale_kernel, scale_antiring,
             cas_strength, audio_path, film_grain, force_hdr_metadata) = pending
            self._disp_hdr_mpv.start_playback(
                pw, ph, fps=fps, scale_kernel=scale_kernel,
                scale_antiring=scale_antiring,
                cas_strength=cas_strength,
                audio_path=audio_path,
                film_grain=film_grain,
                force_hdr_metadata=force_hdr_metadata,
            )
            # Anchor mpv timeline at 0 on startup to avoid initial drift.
            self._disp_hdr_mpv.seek_seconds(0.0)
            if not self._audio_available:
                self._apply_selected_audio_track_mpv_async()
            self._apply_volume_to_backends()
            if self._startup_sync_pending:
                self._disp_hdr_mpv.set_paused(True)
            self._worker.set_mpv_widget(self._disp_hdr_mpv)
            self._pending_mpv_start = None
            if self._startup_sync_pending:
                QTimer.singleShot(250, self._release_startup_sync)
        pending_sdr = getattr(self, '_pending_sdr_mpv_start', None)
        if pending_sdr and self._disp_sdr_mpv is not None:
            pw, ph, fps, scale_kernel = pending_sdr
            self._disp_sdr_mpv.start_playback(
                pw, ph, fps=fps, scale_kernel=scale_kernel, audio_path=None,
                force_hdr_metadata=False,
            )
            if self._startup_sync_pending:
                self._disp_sdr_mpv.set_paused(True)
            self._worker.set_sdr_mpv_widget(self._disp_sdr_mpv)
            self._sdr_mpv_feed_from_worker = True
            self._pending_sdr_mpv_start = None

    def _precompile_kernels(self):
        """Open the pre-compile dialog — runs compile_kernels.py as a
        completely separate process with zero GPU interference."""

        opts = _PrecompileOptionsDialog(
            initial_precision=self._cmb_prec.currentText(),
            initial_resolution=self._cmb_res.currentText(),
            precision_keys=_available_precision_keys(),
            parent=self,
        )
        if opts.exec() != QDialog.DialogCode.Accepted:
            return

        gui_prec = opts.selected_precision()
        resolutions = opts.selected_resolutions()
        prec_arg = _precision_to_compile_arg(gui_prec)
        model_path = _select_model_path(gui_prec, self._chk_hg.isChecked())

        dlg = _PrecompileDialog(
            resolutions, precision=prec_arg, model_path=model_path,
            use_hg=self._chk_hg.isChecked(),
            hg_weights=_HG_WEIGHTS_PATH if os.path.isfile(_HG_WEIGHTS_PATH) else None,
            clear_cache=False, parent=self,
        )
        dlg.exec()          # modal — blocks until user closes
        if dlg.succeeded:
            for r in resolutions:
                w, h = r.lower().split("x")
                _mark_compiled(
                    int(w), int(h), prec_arg,
                    model_path=model_path,
                    use_hg=self._chk_hg.isChecked(),
                )

    def _clear_kernel_cache(self):
        """Delete cached Triton / TorchInductor kernels."""
        import pathlib, getpass, tempfile

        dirs = []
        triton_root = pathlib.Path(
            os.environ.get("TRITON_CACHE_DIR", pathlib.Path.home() / ".triton")
        )
        triton_dir = triton_root / "cache"
        if triton_dir.is_dir() and any(triton_dir.iterdir()):
            dirs.append(triton_dir)

        inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if not inductor_dir:
            inductor_dir = os.path.join(
                tempfile.gettempdir(),
                f"torchinductor_{getpass.getuser()}",
            )
        inductor_path = pathlib.Path(inductor_dir)
        if inductor_path.is_dir() and any(inductor_path.iterdir()):
            dirs.append(inductor_path)

        if not dirs:
            QMessageBox.information(
                self, "Kernel Cache",
                "No Triton / Inductor kernel cache found.",
            )
            return

        if self._worker is not None and self._worker.isRunning():
            msg = (
                "Kernel cache cannot be cleared while playback is running.\n\n"
                "Stop playback and clear cache now?"
            )
            btn = QMessageBox.question(
                self, "Clear Kernel Cache", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if btn != QMessageBox.StandardButton.Yes:
                return
            self._worker.stop()
            self._worker.wait(10000)

        msg = (
            "This will delete cached Triton and TorchInductor kernels:\n\n"
            + "\n".join(f"  {d}" for d in dirs)
            + "\n\nKernels will be recompiled on next playback.\n"
              "Continue?"
        )
        btn = QMessageBox.question(
            self, "Clear Kernel Cache", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if btn == QMessageBox.StandardButton.Yes:
            dlg = QProgressDialog(
                "Clearing kernel cache...", None, 0, 0, self
            )
            dlg.setWindowTitle("Clear Kernel Cache")
            dlg.setMinimumDuration(0)
            dlg.setCancelButton(None)
            dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
            dlg.show()

            self._cache_clear_thread = QThread(self)
            self._cache_clear_worker = _KernelCacheClearWorker(dirs)
            self._cache_clear_worker.moveToThread(self._cache_clear_thread)

            def _on_finished(ok: bool):
                dlg.close()
                self._cache_clear_worker = None
                self._cache_clear_thread = None
                if ok:
                    QMessageBox.information(
                        self, "Kernel Cache",
                        "Cache cleared. Kernels will recompile on next play.",
                    )
                else:
                    QMessageBox.warning(
                        self, "Kernel Cache",
                        "Cache clear completed with errors. "
                        "Some cache files may remain.",
                    )

            self._cache_clear_thread.started.connect(self._cache_clear_worker.run)
            self._cache_clear_worker.finished.connect(_on_finished)
            self._cache_clear_worker.finished.connect(self._cache_clear_thread.quit)
            self._cache_clear_worker.finished.connect(self._cache_clear_worker.deleteLater)
            self._cache_clear_thread.finished.connect(self._cache_clear_thread.deleteLater)
            self._cache_clear_thread.start()

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", _ROOT,
            "Video (*.mp4 *.avi *.mkv *.mov *.webm *.flv);;All (*)",
        )
        if path:
            self._set_video(path)

    def _set_video(self, path, auto_play: bool = False):
        # Stop current playback if running
        if self._playing:
            self._stop()

        # If playback has already been started once in this process,
        # always restart before loading another video to keep compile/
        # runtime state clean and deterministic.
        if self._last_res is not None:
            self._restart_with_video(
                path,
                resolution=self._cmb_res.currentText(),
                precision=self._cmb_prec.currentText(),
                view=self._cmb_view.currentText(),
                upscale=self._cmb_upscale.currentText() if hasattr(self, "_cmb_upscale") else None,
                film_grain=self._chk_film_grain.isChecked() if hasattr(self, "_chk_film_grain") else None,
                autoplay=auto_play,
            )
            return

        self._video_path = path
        self._refresh_resolution_options_for_video(path)
        self._refresh_audio_tracks_for_video(path)
        self._lbl_file.setText(os.path.basename(path))
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_apply_settings.setEnabled(False)
        self._prepare_idle_timeline(path)
        self._show_idle_preview(path)
        self.setWindowTitle(
            f"HDRTVNet++ — {os.path.basename(path)}")
        self.statusBar().showMessage(
            f"Selected: {path} - preview loaded. Press Play to start.")
        if auto_play:
            QTimer.singleShot(100, self._play)

    def _restart_with_video(self, path, resolution=None, precision=None, view=None,
                            use_hg=None, upscale=None, film_grain=None,
                            autoplay=False, start_frame=None):
        """Restart the GUI process with a new video.

        A fresh process avoids stale torch.compile/dynamo state that
        causes slow in-process re-tracing when the resolution changes.
        """
        self.statusBar().showMessage("Restarting for new resolution …")
        # Clean shutdown
        if self._playing:
            self._worker.stop()
            self._worker.wait(5000)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.stop_playback()
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.stop_playback()

        # Hide the parent window so the user doesn't see two GUIs
        self.hide()
        QApplication.instance().processEvents()

        # Re-exec with --video
        # The parent must wait for the child so the shell stays blocked
        # the entire time.  Otherwise the prompt appears immediately and
        # the child's output overwrites it, leaving no usable prompt
        # after the child exits.
        import subprocess as _sp
        args = [sys.executable, sys.argv[0], "--video", path]
        if resolution in RESOLUTION_SCALES:
            args += ["--resolution", resolution]
        elif resolution == "Source":
            args += ["--resolution", "Source"]
        if precision in PRECISIONS:
            args += ["--precision", precision]
        if view == "Tabbed":
            args += ["--view", view]
        if use_hg is not None:
            args += ["--use-hg", "1" if use_hg else "0"]
        if isinstance(upscale, str) and upscale in UPSCALER_CHOICES:
            args += ["--upscale", upscale]
        if film_grain is not None:
            args += ["--film-grain", "1" if film_grain else "0"]
        if autoplay:
            args += ["--autoplay", "1"]
        if start_frame is not None:
            args += ["--start-frame", str(max(0, int(start_frame)))]
        rc = _sp.call(args)
        sys.exit(rc)

    # ── Slots: playback ──────────────────────────────────────

    def _play(self):
        if self._playing or not self._video_path:
            return

        self._source_hdr_info = _probe_hdr_input(self._video_path)

        # Determine processing resolution
        cap = cv2.VideoCapture(self._video_path)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vfps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Output (display) resolution: always 1080p (letterbox handles aspect).
        ow, oh = MAX_W, MAX_H
        self._cur_output_w, self._cur_output_h = ow, oh

        # Processing resolution from scale selector (fixed preset; letterbox handles aspect).
        scale_key = self._cmb_res.currentText()
        scale_dims = RESOLUTION_SCALES.get(scale_key)
        if scale_key == "Source" and self._source_proc_dims is not None:
            scale_dims = self._source_proc_dims
        if scale_dims is not None and (scale_dims[0] < ow or scale_dims[1] < oh):
            pw, ph = scale_dims
        else:
            pw, ph = ow, oh

        # Select upscale kernel choice (only allowed for 540p/720p presets)
        upscale_choice = DEFAULT_UPSCALER
        if scale_key in {"540p", "720p"}:
            upscale_choice = self._cmb_upscale.currentText() or DEFAULT_UPSCALER
        # Resolve FSR availability up-front so we don't report it if it can't load.
        if _normalize_upscale_choice(upscale_choice) == "fsr" and not _ensure_fsr_shader():
            self.statusBar().showMessage(
                "FSR shader unavailable (download failed). Falling back to EWA LanczosSharp."
            )
            upscale_choice = "EWA LanczosSharp"

        # Set up seek slider
        self._vid_fps = vfps if vfps > 0 else 30.0
        display_fps = _limited_playback_fps(self._vid_fps)
        self._target_playback_fps = float(display_fps)
        self._seek_slider.setRange(0, max(0, total_frames - 1))
        self._seek_slider.setValue(0)
        self._seek_slider.setEnabled(True)
        self._seek_slider.setToolTip("Seek while paused is queued and applied on Resume.")
        self._lbl_time.setText("0:00")
        dur_secs = total_frames / self._vid_fps if self._vid_fps > 0 else 0
        self._lbl_duration.setText(self._fmt_time(dur_secs))

        # Map GUI precision to compile arg
        gui_prec = self._cmb_prec.currentText()
        prec_arg = _precision_to_compile_arg(gui_prec)

        source_is_hdr = bool(self._source_hdr_info.get("is_hdr", False))
        if source_is_hdr:
            self.statusBar().showMessage(
                "HDR input detected. Using model path (OpenCV decode is 8-bit)."
            )

        # Always compile via a clean subprocess — this ensures autotune
        # benchmarks have zero GPU interference from Qt / D3D11 / mpv.
        # If the Triton + Inductor cache is already warm from a previous
        # compile, the subprocess finishes in seconds and auto-closes.
        model_path = _select_model_path(gui_prec, self._chk_hg.isChecked())
        dlg = _PrecompileDialog(
            [f"{pw}x{ph}"], precision=prec_arg,
            model_path=model_path,
            use_hg=self._chk_hg.isChecked(),
            hg_weights=_HG_WEIGHTS_PATH if os.path.isfile(_HG_WEIGHTS_PATH) else None,
            parent=self,
        )
        dlg.exec()                        # modal — blocks until done
        if dlg.succeeded:
            _mark_compiled(
                pw, ph, prec_arg,
                model_path=model_path,
                use_hg=self._chk_hg.isChecked(),
            )
        else:
            # Compile failed or user closed early — don't start playback
            return

        # ── Start playback ──
        self._last_res = (pw, ph)
        self._playing = True
        self._active_precision = self._cmb_prec.currentText()
        self._active_resolution = self._cmb_res.currentText()
        self._active_use_hg = self._chk_hg.isChecked()
        self._active_upscale_mode = upscale_choice
        # Keep mpv initialized whenever available so view switches are UI-only.
        use_mpv_pipeline = (self._disp_hdr_mpv is not None)
        self._active_use_mpv = use_mpv_pipeline
        self._startup_sync_pending = bool(use_mpv_pipeline)
        self._mpv_start_resync_t = 0.0
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_stack.setCurrentWidget(
                self._disp_sdr_mpv if use_mpv_pipeline else self._disp_sdr_cpu
            )
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_stack.setCurrentWidget(
                self._disp_hdr_mpv if use_mpv_pipeline else self._disp_hdr_cpu
            )
        self._update_apply_button_state()
        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_stop.setEnabled(True)
        self._btn_file.setEnabled(False)
        if self._btn_toggle_ui is not None:
            self._btn_toggle_ui.setEnabled(True)
        self._cmb_prec.setEnabled(True)
        self._set_pause_button_labels(False)

        # Start mpv HDR display AFTER compile finishes (via signal)
        # so that mpv's D3D11 GPU usage doesn't pollute Triton autotuning.
        # mpv receives frames at processing resolution; GPU scaling happens in mpv.
        self._pending_mpv_start = None
        self._pending_sdr_mpv_start = None
        if use_mpv_pipeline and self._disp_hdr_mpv is not None:
            mpv_audio_path = None if self._audio_available else self._video_path
            self._active_mpv_scale_kernel = _select_hdr_scale_kernel(
                pw, ph, ow, oh, upscale_choice
            )
            self._active_mpv_scale_antiring = _select_hdr_scale_antiring(
                pw, ph, ow, oh, self._active_mpv_scale_kernel
            )
            using_fsr = (self._active_mpv_scale_kernel == "fsr")
            self._active_mpv_cas = _select_mpv_cas_strength(
                pw, ph, ow, oh, using_fsr, self._active_mpv_scale_kernel
            )
            self._active_film_grain = bool(
                hasattr(self, "_chk_film_grain") and self._chk_film_grain.isChecked()
            )
            if self._active_film_grain and not _ensure_filmgrain_shader():
                self.statusBar().showMessage(
                    "Film grain shader unavailable (download failed)."
                )
                self._active_film_grain = False
                if hasattr(self, "_chk_film_grain") and self._chk_film_grain is not None:
                    self._chk_film_grain.blockSignals(True)
                    self._chk_film_grain.setChecked(False)
                    self._chk_film_grain.blockSignals(False)
            self._pending_mpv_start = (
                pw, ph, float(display_fps), self._active_mpv_scale_kernel,
                self._active_mpv_scale_antiring,
                self._active_mpv_cas,
                mpv_audio_path,
                self._active_film_grain,
                True,
            )
            if self._disp_sdr_mpv is not None:
                self._pending_sdr_mpv_start = (
                    ow, oh, float(display_fps), "bicubic"
                )
        else:
            self._worker.set_mpv_widget(None)
            self._worker.set_sdr_mpv_widget(None)
            self._sdr_mpv_feed_from_worker = False

        self._worker.configure(
            self._video_path, self._cmb_prec.currentText(),
            proc_w=pw, proc_h=ph,
            output_w=ow, output_h=oh,
            input_is_hdr=False,
            use_hg=self._chk_hg.isChecked(),
        )

        # Show loading dialog (in-process model load + cache warmup is fast
        # since subprocess already compiled the kernels)
        self._compile_dlg = _CompileDialog(self)
        self._compile_dlg.show()

        self._worker.start()
        self._set_process_priority(True)
        if pw != ow or ph != oh:
            upscale_backend = "mpv GPU" if use_mpv_pipeline else "CPU fallback"
            self.statusBar().showMessage(
                f"Upscale active: {pw}×{ph} -> {ow}×{oh} via {BEST_UPSCALE_MODE} ({upscale_backend})"
            )
        else:
            self.statusBar().showMessage(
                f"No upscale stage: processing at {ow}×{oh}."
            )
        self._post_seek_resync_frames = 0
        self._pending_seek_on_resume = None
        if self._startup_sync_pending:
            self._worker.pause()
        # Startup audio gate: release audio only after FPS reaches target.
        self._startup_audio_gate_active = True
        self._startup_audio_gate_count = 0
        self._scrub_muted = True
        self._arm_mute_until_fps_recovery()
        if self._audio_available:
            self._start_audio_playback(self._video_path)
            self._set_audio_paused(True)
        else:
            self.statusBar().showMessage(
                "Qt audio backend unavailable; using mpv audio fallback (seek sync may be limited)."
            )

        # Restore timeline position after process restart (resolution change).
        if self._startup_seek_frame is not None:
            target = max(0, min(int(self._startup_seek_frame), self._seek_slider.maximum()))
            self._worker.request_seek(target)
            self._seek_slider.setValue(target)
            self._lbl_time.setText(self._fmt_time(target / max(self._vid_fps, 1e-6)))
            self._seek_audio_seconds(target / max(self._vid_fps, 1e-6))
            if self._disp_hdr_mpv is not None and not self._audio_available:
                self._disp_hdr_mpv.seek_seconds(target / max(self._vid_fps, 1e-6))
        self._startup_seek_frame = None
        self._arm_cursor_idle_timer()
        self._start_periodic_relock()

    def _toggle_pause(self):
        if not self._playing:
            return
        if self._worker.is_paused:
            self._user_pause_override_startup = False
            queued = self._pending_seek_on_resume
            if queued is not None:
                self._worker.request_seek(int(queued))
                fps = getattr(self, '_vid_fps', 30.0)
                self._seek_audio_seconds(int(queued) / max(fps, 1e-6))
                if self._disp_hdr_mpv is not None and not self._audio_available:
                    self._disp_hdr_mpv.seek_seconds(int(queued) / max(fps, 1e-6))
                now_t = time.perf_counter()
                self._audio_seek_guard_until = now_t + 1.0
                self._audio_track_lock_until = now_t + 0.45
                self._audio_resync_pending = True
                self._audio_fps_recovered = False
                self._post_seek_resync_frames = 120
                self._resume_audio_after_seek = bool(self._audio_available)
                self._seek_resume_target = int(queued)
                self._seek_resume_started_t = time.perf_counter()
                if self._audio_available:
                    QTimer.singleShot(420, self._ensure_selected_audio_track_qt)
                self._pending_seek_on_resume = None
            self._worker.resume()
            self._set_pause_button_labels(False)
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_mpv.set_paused(False)
            if self._disp_sdr_mpv is not None:
                self._disp_sdr_mpv.set_paused(False)
            if not self._resume_audio_after_seek:
                self._set_audio_paused(False)
            if self._active_use_mpv:
                self._relock_timeline(delay_ms=60, drop_frames=2)
                if queued is None:
                    def _resume_video_resync():
                        if not self._playing or self._worker.is_paused:
                            return
                        fps = getattr(self, '_vid_fps', 30.0)
                        target_frame = int(self._last_seek_frame)
                        resume_dt = time.perf_counter() - float(self._last_user_pause_t or 0.0)
                        if self._audio_available and self._audio_player is not None:
                            try:
                                have_ms = int(self._audio_player.position())
                                target_frame = int(round((have_ms / 1000.0) * max(fps, 1e-6)))
                            except Exception:
                                target_frame = int(self._last_seek_frame)
                        if resume_dt < 0.6:
                            return
                        if abs(target_frame - int(self._last_seek_frame)) >= 6:
                            self._worker.request_seek(int(target_frame))
                            if self._disp_hdr_mpv is not None and not self._audio_available:
                                self._disp_hdr_mpv.seek_seconds(int(target_frame) / max(fps, 1e-6))
                            self._audio_seek_guard_until = time.perf_counter() + 1.0
                            self._audio_resync_pending = True
                            self._audio_fps_recovered = False
                            self._post_seek_resync_frames = 60
                    QTimer.singleShot(80, _resume_video_resync)
            self._arm_cursor_idle_timer()
        else:
            self._worker.pause()
            self._last_user_pause_t = time.perf_counter()
            self._set_pause_button_labels(True)
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_mpv.set_paused(True)
            if self._disp_sdr_mpv is not None:
                self._disp_sdr_mpv.set_paused(True)
            self._set_audio_paused(True)
            if self._startup_sync_pending:
                self._user_pause_override_startup = True
            if self._cursor_idle_timer is not None:
                self._cursor_idle_timer.stop()
            self._show_cursor()

    def _stop(self):
        self._worker.stop()
        self._worker.wait(10000)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.stop_playback()
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.stop_playback()
        self._stop_audio_playback()
        self._set_process_priority(False)
        # Clear the current video so user must choose a file again.
        self._video_path = None
        self._source_hdr_info = {"is_hdr": False, "reason": "unknown"}
        if self._lbl_file is not None:
            self._lbl_file.setText("No video selected")
        if self._lbl_duration is not None:
            self._lbl_duration.setText("0:00")
        self._reset_controls()

    def _restart_app_clean(self):
        self.statusBar().showMessage("Restarting app …")
        self._save_user_settings()
        # Clean shutdown
        if self._playing:
            self._worker.stop()
            self._worker.wait(5000)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.stop_playback()
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.stop_playback()
        self._stop_audio_playback()

        # Hide the parent window so the user doesn't see two GUIs
        self.hide()
        QApplication.instance().processEvents()

        import subprocess as _sp
        args = [sys.executable, sys.argv[0]]
        rc = _sp.call(args)
        sys.exit(rc)

    def _stop_and_restart(self):
        self._stop()
        self._restart_app_clean()

    # ── Slots: settings ──────────────────────────────────────

    def _on_precision(self, key):
        self._chk_hg.setEnabled(True)
        if self._playing:
            self._update_apply_button_state()

    def _on_resolution(self, scale_key):
        self._sync_upscale_controls()
        if self._playing:
            self._update_apply_button_state()

    def _on_upscale_changed(self, _mode: str):
        self._sync_upscale_controls()
        if self._playing:
            self._update_apply_button_state()

    def _on_film_grain_changed(self, _state):
        if self._playing:
            self._update_apply_button_state()
        self._save_user_settings()

    def _on_hg_toggle(self, _state):
        if self._playing:
            self._update_apply_button_state()

    def _apply_runtime_settings(self):
        if not self._playing or not self._video_path:
            return
        if not self._has_pending_setting_changes():
            self.statusBar().showMessage("No pending setting changes.")
            return

        new_prec = self._cmb_prec.currentText()
        new_res = self._cmb_res.currentText()
        current_upscale = self._cmb_upscale.currentText() if hasattr(self, "_cmb_upscale") else DEFAULT_UPSCALER
        new_upscale = DEFAULT_UPSCALER
        if new_res in {"540p", "720p"}:
            new_upscale = current_upscale or DEFAULT_UPSCALER
        upscale_changed = (new_upscale != self._active_upscale_mode)
        film_grain_changed = False
        if hasattr(self, "_chk_film_grain") and self._chk_film_grain is not None:
            film_grain_changed = (self._chk_film_grain.isChecked() != self._active_film_grain)
        needs_restart = (new_res != self._active_resolution)
        if self._chk_hg.isChecked() != self._active_use_hg:
            needs_restart = True
        notices: list[str] = []

        def _apply_mpv_hot_swap(action, pause_ms: int = 260,
                                relock_ms: int = 260,
                                relock_drop: int = 2) -> bool:
            if self._worker is not None and not self._worker.is_paused:
                self._pause_for_ui_transition(duration_ms=pause_ms, wait_for_stable=True)
            ok = bool(action())
            if ok and self._playing:
                self._relock_timeline(delay_ms=relock_ms, drop_frames=relock_drop)
            return ok

        if needs_restart:
            self._save_user_settings()
            self._restart_with_video(
                self._video_path,
                resolution=new_res,
                precision=new_prec,
                view=self._cmb_view.currentText(),
                use_hg=self._chk_hg.isChecked(),
                upscale=current_upscale,
                film_grain=self._chk_film_grain.isChecked() if hasattr(self, "_chk_film_grain") else None,
                autoplay=True,
                start_frame=int(self._seek_slider.value()),
            )
            return

        def _apply_film_grain_toggle() -> bool:
            if not film_grain_changed:
                return True
            if not self._active_use_mpv or self._disp_hdr_mpv is None:
                self.statusBar().showMessage(
                    "Film grain requires mpv; keeping previous setting."
                )
                if hasattr(self, "_chk_film_grain") and self._chk_film_grain is not None:
                    self._chk_film_grain.blockSignals(True)
                    self._chk_film_grain.setChecked(self._active_film_grain)
                    self._chk_film_grain.blockSignals(False)
                return False
            enabled = bool(self._chk_film_grain.isChecked())
            if enabled and not _ensure_filmgrain_shader():
                self.statusBar().showMessage(
                    "Film grain shader unavailable (download failed)."
                )
                if hasattr(self, "_chk_film_grain") and self._chk_film_grain is not None:
                    self._chk_film_grain.blockSignals(True)
                    self._chk_film_grain.setChecked(False)
                    self._chk_film_grain.blockSignals(False)
                return False
            # Match upscale behavior: brief pause + relock around shader swap.
            ok = _apply_mpv_hot_swap(
                lambda: self._disp_hdr_mpv.set_film_grain(enabled)
            )
            if ok:
                self._active_film_grain = enabled
                self._save_user_settings()
                self.statusBar().showMessage(
                    "Film grain enabled." if enabled else "Film grain disabled."
                )
            else:
                self.statusBar().showMessage(
                    "Film grain hot-swap failed; keeping previous setting."
                )
                if hasattr(self, "_chk_film_grain") and self._chk_film_grain is not None:
                    self._chk_film_grain.blockSignals(True)
                    self._chk_film_grain.setChecked(self._active_film_grain)
                    self._chk_film_grain.blockSignals(False)
            return ok

        if upscale_changed:
            if not self._active_use_mpv or self._disp_hdr_mpv is None:
                self.statusBar().showMessage(
                    "Upscale mode change requires mpv GPU pipeline; keeping previous setting."
                )
                if hasattr(self, "_cmb_upscale"):
                    self._cmb_upscale.blockSignals(True)
                    self._cmb_upscale.setCurrentText(self._active_upscale_mode)
                    self._cmb_upscale.blockSignals(False)
                return
            # Pause both SDR/HDR + audio briefly so the hot-swap doesn't
            # desync the two pipelines.
            cur_pw, cur_ph = self._last_res if self._last_res else (MAX_W, MAX_H)
            ow, oh = self._cur_output_w, self._cur_output_h
            kernel = _select_hdr_scale_kernel(cur_pw, cur_ph, ow, oh, new_upscale)
            antiring = _select_hdr_scale_antiring(cur_pw, cur_ph, ow, oh, kernel)
            def _apply_upscale_hot_swap() -> bool:
                if not self._disp_hdr_mpv.set_scale_kernel(kernel, antiring):
                    return False
                self._active_mpv_scale_kernel = kernel
                self._active_mpv_scale_antiring = antiring
                self._active_mpv_cas = _select_mpv_cas_strength(
                    cur_pw, cur_ph, ow, oh, using_fsr=(kernel == "fsr"),
                    scale_kernel=kernel
                )
                self._disp_hdr_mpv.set_cas_strength(self._active_mpv_cas)
                self._active_upscale_mode = new_upscale
                def _announce():
                    mode_label = str(self._active_upscale_mode or "")
                    using_shader = (kernel == "fsr" or "ssim" in str(kernel).lower())
                    self.statusBar().showMessage(
                        f"Upscale hot-swap: {mode_label} ({'shader active' if using_shader else 'kernel active'})"
                    )
                QTimer.singleShot(80, _announce)
                self._save_user_settings()
                return True

            if not _apply_mpv_hot_swap(_apply_upscale_hot_swap):
                err = getattr(self._disp_hdr_mpv, "_last_scale_error", None)
                if err:
                    self.statusBar().showMessage(f"Upscale hot-swap failed: {err}")
                else:
                    self.statusBar().showMessage("Upscale hot-swap failed; keeping previous setting.")
            _apply_film_grain_toggle()
            self._update_apply_button_state()
            return

        if new_prec != self._active_precision:
            cur_pw, cur_ph = self._last_res if self._last_res else (MAX_W, MAX_H)
            target_prec_arg = _precision_to_compile_arg(new_prec)
            target_model_path = _select_model_path(new_prec, self._chk_hg.isChecked())
            if not _is_compiled(
                cur_pw, cur_ph, target_prec_arg,
                model_path=target_model_path,
                use_hg=self._chk_hg.isChecked(),
            ):
                self.statusBar().showMessage(
                    f"Precision {new_prec} not precompiled at {cur_pw}x{cur_ph}; restarting for clean compile."
                )
                self._save_user_settings()
                self._restart_with_video(
                    self._video_path,
                    resolution=new_res,
                    precision=new_prec,
                    view=self._cmb_view.currentText(),
                    use_hg=self._chk_hg.isChecked(),
                    upscale=current_upscale,
                    film_grain=self._chk_film_grain.isChecked() if hasattr(self, "_chk_film_grain") else None,
                    autoplay=True,
                    start_frame=int(self._seek_slider.value()),
                )
                return
            self._pause_for_precision_swap(new_prec)
            self._worker.request_precision_change(new_prec)
            if self._playing:
                self._schedule_precision_audio_resync()
            notices.append(f"Applying precision change: {new_prec}")
            self._active_precision = new_prec

        _apply_film_grain_toggle()

        self._active_use_hg = self._chk_hg.isChecked()
        self._active_upscale_mode = new_upscale

        self._active_resolution = new_res
        self._save_user_settings()
        self._update_apply_button_state()
        if notices:
            self.statusBar().showMessage(" | ".join(notices))

    def _on_view(self, mode):
        if self._video_tabs is not None:
            self._video_tabs.tabBar().setVisible(True)
        if _HAS_MPV and self._disp_sdr_mpv is not None:
            if self._playing:
                self._disp_sdr_stack.setCurrentWidget(self._disp_sdr_mpv)
            else:
                self._disp_sdr_stack.setCurrentWidget(self._disp_sdr_cpu)
        if _HAS_MPV and self._disp_hdr_mpv is not None:
            # Show textual placeholder when idle; switch to mpv only during playback.
            if self._playing:
                self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_mpv)
            else:
                self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_cpu)
        # Let the worker skip unnecessary copies / postprocess
        if self._playing:
            # Keep SDR path running so tab switches stay instantaneous.
            self._worker.set_sdr_visible(True)
            if self._last_sdr_frame is not None:
                if self._disp_sdr_cpu is not None:
                    self._disp_sdr_cpu.update_frame(self._last_sdr_frame)
            # Any view change can desync; relock the timeline.
            self._relock_timeline(delay_ms=140, drop_frames=3)

    def _refresh_mpv_after_window_state_change(self):
        if not self._playing:
            return
        soft_only = bool(self._window_refresh_soft_only)
        self._window_refresh_soft_only = False
        paused = bool(self._worker is not None and self._worker.is_paused)
        if paused:
            # Defer full mpv refresh while paused to avoid blackscreen.
            self._deferred_mpv_refresh = True
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_mpv.set_cas_strength(self._active_mpv_cas)
                self._disp_hdr_mpv.set_film_grain(self._active_film_grain)
            if self._disp_sdr_mpv is not None:
                self._disp_sdr_mpv.set_cas_strength(0.0)
            return
        self._deferred_mpv_refresh = False
        if (not soft_only) and self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.refresh_surface()
            if paused:
                self._disp_hdr_mpv.set_paused(True)
        if (not soft_only) and self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.refresh_surface()
            if paused:
                self._disp_sdr_mpv.set_paused(True)
        # Re-assert colorspace/vf after UI reparent/fullscreen to avoid grey output.
        def _reapply_colorspace(attempts_left: int = 3):
            ok = True
            if self._disp_hdr_mpv is not None:
                ok = bool(self._disp_hdr_mpv.set_cas_strength(self._active_mpv_cas)) and ok
                ok = bool(self._disp_hdr_mpv.set_film_grain(self._active_film_grain)) and ok
            if self._disp_sdr_mpv is not None:
                ok = bool(self._disp_sdr_mpv.set_cas_strength(0.0)) and ok
            if (not ok) and attempts_left > 0:
                QTimer.singleShot(120, lambda: _reapply_colorspace(attempts_left - 1))
        _reapply_colorspace(3)
        # Relock video/audio after any window/layout refresh.
        self._relock_timeline(drop_frames=3)

    # ── Slots: seek / position ────────────────────────────────

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Format seconds as M:SS or H:MM:SS."""
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def _on_position(self, current_frame: int, total_frames: int):
        """Update seek slider + time labels from worker."""
        self._last_seek_frame = int(current_frame)
        if not self._seek_slider.isSliderDown():
            self._seek_slider.setValue(current_frame)
        fps = getattr(self, '_vid_fps', 30.0)
        self._lbl_time.setText(self._fmt_time(current_frame / fps))

        # On startup or when returning to the beginning, re-anchor mpv to 0
        # to eliminate initial HDR lag.
        if self._disp_hdr_mpv is not None and current_frame <= 1:
            now_t = time.perf_counter()
            if (now_t - self._mpv_start_resync_t) > 0.8:
                self._mpv_start_resync_t = now_t
                self._disp_hdr_mpv.seek_seconds(0.0)

        if self._resume_audio_after_seek and not self._worker.is_paused:
            settled = abs(int(current_frame) - int(self._seek_resume_target)) <= 1
            timed_out = (time.perf_counter() - self._seek_resume_started_t) > 0.45
            if settled or timed_out:
                self._set_audio_paused(False)
                self._resume_audio_after_seek = False

        if (
            self._playing
            and self._audio_available
            and self._audio_player is not None
            and not self._seek_slider.isSliderDown()
        ):
            want_ms = int((current_frame / max(fps, 1e-6)) * 1000.0)
            have_ms = int(self._audio_player.position())
            drift_signed_ms = float(have_ms - want_ms)  # + => audio ahead, - => audio behind
            drift_ms = abs(drift_signed_ms)
            now_t = time.perf_counter()
            in_seek_guard = (now_t < self._audio_seek_guard_until)

            # Post-seek stabilization window: hold neutral rate, avoid correction thrash.
            if self._post_seek_resync_frames > 0:
                self._post_seek_resync_frames -= 1
                self._audio_player.setPlaybackRate(1.0)
            elif (current_frame % self._audio_drift_check_stride == 0) and (not self._worker.is_paused):
                # One-shot sync only after FPS recovery. Do not keep correcting
                # until another low-FPS mute/seek event re-arms this gate.
                if (
                    self._audio_resync_pending
                    and self._audio_fps_recovered
                    and (not in_seek_guard)
                    and drift_ms > 220
                ):
                    self._audio_player.setPosition(max(0, want_ms))
                    self._audio_last_hard_sync_t = now_t
                    self._audio_resync_pending = False
                elif (not in_seek_guard) and drift_ms > 2200 and (now_t - self._audio_last_hard_sync_t) > 10.0:
                    # Emergency recovery only.
                    self._audio_player.setPosition(max(0, want_ms))
                    self._audio_last_hard_sync_t = now_t
                    self._audio_player.setPlaybackRate(1.0)
                elif drift_signed_ms > 320:
                    self._audio_player.setPlaybackRate(0.9997)
                elif drift_signed_ms < -320:
                    self._audio_player.setPlaybackRate(1.0003)
                else:
                    self._audio_player.setPlaybackRate(1.0)

    def _on_seek_pressed(self):
        if not self._playing:
            return
        self._scrub_unmute_seq += 1
        if self._audio_available:
            self._set_audio_paused(True)
        self._arm_mute_until_fps_recovery()
        if not self._scrub_muted:
            self._scrub_muted = True
            self._apply_volume_to_backends()

    def _on_seek(self, frame_number: int):
        if not self._playing:
            return
        # Fast click-seek can race sliderPressed; enforce mute/pause here too.
        if self._audio_available:
            self._set_audio_paused(True)
        self._arm_mute_until_fps_recovery()
        if not self._scrub_muted:
            self._scrub_muted = True
            self._apply_volume_to_backends()
        self._last_seek_frame = int(frame_number)
        fps = getattr(self, '_vid_fps', 30.0)
        self._lbl_time.setText(self._fmt_time(frame_number / fps))

    def _on_seek_released(self):
        if not self._playing:
            return
        # Defensive gate for very fast release where press handler may lag.
        if self._audio_available:
            self._set_audio_paused(True)
        self._arm_mute_until_fps_recovery()
        if not self._scrub_muted:
            self._scrub_muted = True
            self._apply_volume_to_backends()
        target_frame = int(self._seek_slider.value())
        self._last_seek_frame = target_frame
        fps = getattr(self, '_vid_fps', 30.0)

        if self._worker.is_paused:
            self._pending_seek_on_resume = target_frame
            self._lbl_time.setText(self._fmt_time(target_frame / max(fps, 1e-6)))
            self.statusBar().showMessage(
                f"Seek queued to {self._fmt_time(target_frame / max(fps, 1e-6))}. Press Resume to apply."
            )
            return

        self._worker.request_seek(target_frame)
        self._seek_audio_seconds(target_frame / max(fps, 1e-6))
        now_t = time.perf_counter()
        self._audio_seek_guard_until = now_t + 1.0
        self._audio_track_lock_until = now_t + 0.45
        self._audio_resync_pending = True
        self._audio_fps_recovered = False
        if self._audio_available:
            QTimer.singleShot(420, self._ensure_selected_audio_track_qt)
        self._post_seek_resync_frames = 120
        self._resume_audio_after_seek = bool(self._audio_available)
        self._seek_resume_target = int(target_frame)
        self._seek_resume_started_t = time.perf_counter()
        if self._disp_hdr_mpv is not None and not self._audio_available:
            self._disp_hdr_mpv.seek_seconds(target_frame / max(fps, 1e-6))
        if not getattr(self, "_first_seek_done", False):
            self._first_seek_done = True
            def _second_seek():
                if not self._playing:
                    return
                self._worker.request_seek(target_frame)
                self._seek_audio_seconds(target_frame / max(fps, 1e-6))
                if self._disp_hdr_mpv is not None and not self._audio_available:
                    self._disp_hdr_mpv.seek_seconds(target_frame / max(fps, 1e-6))
            QTimer.singleShot(90, _second_seek)

    def _on_hdr_info(self, info: dict):
        """Update the HDR Info panel from mpv metadata."""
        vo_confirmed = bool(info.get("hdr_vo_confirmed", False))
        metadata_forced = bool(info.get("hdr_metadata_forced", False))
        if vo_confirmed:
            self._hdr_labels["status"].setText("HDR: ✓ Active (VO confirmed)")
            self._hdr_labels["status"].setStyleSheet("color: #00e676;")
        elif metadata_forced:
            self._hdr_labels["status"].setText("HDR: ~ Metadata tagged (VO unconfirmed)")
            self._hdr_labels["status"].setStyleSheet("color: #ffca28;")
        else:
            self._hdr_labels["status"].setText("HDR: ✗ Inactive")
            self._hdr_labels["status"].setStyleSheet("color: #ff5252;")
        self._hdr_labels["primaries"].setText(
            f"Primaries: {info.get('primaries', '?')}")
        self._hdr_labels["transfer"].setText(
            f"Transfer: {info.get('transfer', '?')}")
        peak_raw = info.get("sig_peak", "?")
        try:
            nits = float(peak_raw) * 203  # sig-peak is relative to ref white
            peak_str = f"{nits:.0f} nits (sig-peak {float(peak_raw):.1f})"
        except (ValueError, TypeError):
            peak_str = str(peak_raw)
        self._hdr_labels["peak"].setText(f"Peak: {peak_str}")
        self._hdr_labels["vo"].setText(
            f"VO: {info.get('vo', '?')}/{info.get('gpu_api', '?')}")

    # ── Slots: worker signals ────────────────────────────────

    def _on_frame(self, sdr, hdr):
        sdr_show = sdr
        self._last_sdr_frame = sdr_show
        if (
            self._disp_sdr_mpv is not None
            and self._playing
            and self._active_use_mpv
            and not self._sdr_mpv_feed_from_worker
        ):
            try:
                rgb16 = np.ascontiguousarray(
                    sdr_show[:, :, ::-1].astype(np.uint16) * 257
                )
                self._disp_sdr_mpv.feed_frame(rgb16.data)
            except Exception:
                pass
        if self._disp_sdr_cpu is not None and self._disp_sdr_cpu.isVisible():
            self._disp_sdr_cpu.update_frame(sdr_show)
        # QLabel fallbacks (mpv panes are fed directly from the worker)
        if self._disp_hdr_cpu is not None and self._disp_hdr_cpu.isVisible():
            self._disp_hdr_cpu.update_frame(hdr)

    def _on_metrics(self, m):
        self._m["fps"].setText(f"FPS: {m['fps']:.1f}")
        self._m["latency"].setText(f"Latency: {m['latency_ms']:.1f} ms")
        self._m["frame"].setText(f"Frame: {m['frame']}")
        self._m["res"].setText(f"Res: {m['proc_res']}")
        self._m["gpu"].setText(f"VRAM: {m['gpu_mb']:.0f} MB")
        self._m["cpu"].setText(f"CPU: {m['cpu_mb']:.0f} MB")
        self._m["model"].setText(f"Model: {m['model_mb']:.2f} MB")
        self._m["prec"].setText(f"Prec: {m['precision']}")
        upscale_label = "—"
        if self._last_res and self._cur_output_w and self._cur_output_h:
            proc_w, proc_h = self._last_res
            if proc_w == self._cur_output_w and proc_h == self._cur_output_h:
                upscale_label = "None (native)"
            else:
                fsr_active = False
                if self._disp_hdr_mpv is not None and self._active_use_mpv:
                    fsr_active = self._disp_hdr_mpv.is_fsr_active()
                if fsr_active:
                    upscale_label = "FSR"
                elif self._active_mpv_scale_kernel:
                    if str(self._active_mpv_scale_kernel).lower().startswith("ssim"):
                        upscale_label = "SSimSuperRes"
                    else:
                        upscale_label = str(self._active_mpv_scale_kernel)
        self._m["upscale"].setText(f"Upscale: {upscale_label}")
        fps_now = float(m.get("fps", 0.0))
        self._update_auto_mute_from_fps(fps_now)
        if self._startup_audio_gate_active:
            gate_trip = max(1.0, self._target_playback_fps - 0.3)
            gate_need = 4 if self._ui_resync_gate_strict else 2
            if fps_now >= gate_trip:
                self._startup_audio_gate_count += 1
            else:
                self._startup_audio_gate_count = 0
            if self._startup_audio_gate_count >= gate_need:
                self._startup_audio_gate_active = False
                self._startup_audio_gate_count = 0
                self._ui_resync_gate_strict = False
                if self._scrub_muted:
                    self._scrub_muted = False
                    self._apply_volume_to_backends()
                if self._audio_available and not self._worker.is_paused and not self._startup_sync_pending:
                    fps = getattr(self, "_vid_fps", 30.0)
                    cur_frame = int(m.get("frame", self._last_seek_frame))
                    sec = float(cur_frame) / max(fps, 1e-6)
                    self._force_audio_seek(sec)
                    self._set_audio_paused(False)
    def _on_status_message(self, text: str):
        """Forward worker status to status bar *and* compile dialog."""
        self.statusBar().showMessage(text)
        if self._compile_dlg is not None:
            self._compile_dlg.set_status(text)
        if self._precision_swap_pending is not None and text.startswith("Ready — "):
            ready_key = text.split("Ready — ", 1)[-1].strip()
            if ready_key == self._precision_swap_pending:
                self._resume_after_precision_swap()
        if self._precision_swap_pending is not None and text.startswith("ERROR:"):
            self._resume_after_precision_swap(force=True)

    def _on_finished(self):
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.stop_playback()
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.stop_playback()
        self._stop_audio_playback()
        self._set_process_priority(False)
        self._reset_controls()
        self.statusBar().showMessage("Playback finished.")
        if self._disp_sdr_cpu is not None:
            self._disp_sdr_cpu.clear_display()
        if self._disp_hdr_cpu is not None:
            self._disp_hdr_cpu.clear_display()

    # ── UI helpers ───────────────────────────────────────────

    def _reset_controls(self):
        self._playing = False
        self._active_use_mpv = False
        self._sdr_mpv_feed_from_worker = False
        self._active_mpv_scale_kernel = BEST_MPV_SCALE
        self._active_mpv_scale_antiring = 0.15
        self._active_mpv_cas = 0.0
        self._active_upscale_mode = DEFAULT_UPSCALER
        self._active_film_grain = False
        self._startup_sync_pending = False
        self._auto_muted_low_fps = False
        self._scrub_muted = False
        self._ui_hidden = False
        self._low_fps_count = 0
        self._high_fps_count = 0
        self._user_pause_override_startup = False
        self._deferred_mpv_refresh = False
        if self._audio_fade_timer is not None:
            self._audio_fade_timer.stop()
        self._apply_volume_to_backends()
        self._post_seek_resync_frames = 0
        self._pending_seek_on_resume = None
        self._resume_audio_after_seek = False
        self._seek_resume_target = 0
        self._seek_resume_started_t = 0.0
        self._audio_seek_guard_until = 0.0
        self._audio_track_lock_until = 0.0
        self._audio_resync_pending = False
        self._audio_fps_recovered = True
        self._startup_audio_gate_active = False
        self._startup_audio_gate_count = 0
        if self._window_refresh_timer is not None:
            self._window_refresh_timer.stop()
        if self._cursor_idle_timer is not None:
            self._cursor_idle_timer.stop()
        if self._periodic_relock_timer is not None:
            self._periodic_relock_timer.stop()
        if self._periodic_relock_timer is not None:
            self._periodic_relock_timer.stop()
        self._show_cursor()
        if self._compile_dlg is not None:
            self._compile_dlg.close()
            self._compile_dlg.deleteLater()
            self._compile_dlg = None
        self._btn_play.setEnabled(bool(self._video_path))
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_file.setEnabled(True)
        if self._btn_toggle_ui is not None:
            self._btn_toggle_ui.setEnabled(False)
            self._btn_toggle_ui.setText("Hide UI")
        if self._ui_overlay_btn is not None:
            self._ui_overlay_btn.hide()
        if self._row3_widget is not None:
            self._row3_widget.setVisible(True)
        self._set_pause_button_labels(False)
        self._btn_apply_settings.setEnabled(False)
        if _HAS_MPV and self._disp_sdr_mpv is not None:
            self._disp_sdr_stack.setCurrentWidget(self._disp_sdr_cpu)
        if _HAS_MPV and self._disp_hdr_mpv is not None:
            self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_cpu)
        self._seek_slider.setEnabled(False)
        self._seek_slider.setValue(0)
        self._lbl_time.setText("0:00")
        # Reset HDR panel
        self._hdr_labels["status"].setText("HDR: waiting\u2026")
        self._hdr_labels["status"].setStyleSheet("")
        self._hdr_labels["primaries"].setText("Primaries: \u2014")
        self._hdr_labels["transfer"].setText("Transfer: \u2014")
        self._hdr_labels["peak"].setText("Peak: \u2014")
        self._hdr_labels["vo"].setText("VO: \u2014")

    # ── Drag-and-drop ────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        exts = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv")
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(exts):
                self._set_video(path)
                break

    # ── Clean shutdown ───────────────────────────────────────

    def closeEvent(self, event):
        self._ui_closing = True
        self._save_user_settings()
        if self._window_refresh_timer is not None:
            self._window_refresh_timer.stop()
        if self._cursor_idle_timer is not None:
            self._cursor_idle_timer.stop()
        self._show_cursor()
        self._dock_video_pane("sdr")
        self._dock_video_pane("hdr")
        if self._playing:
            self._worker.stop()
            self._worker.wait(10000)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.stop_playback()
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.stop_playback()
        self._stop_audio_playback()
        self._set_process_priority(False)
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F11:
            self._toggle_borderless_full_window()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space and self._playing:
            self._toggle_pause()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self._borderless_full_window:
            self._exit_borderless_full_window()
            event.accept()
            return
        super().keyPressEvent(event)


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Dark Theme                                                   ║
# ╚═══════════════════════════════════════════════════════════════╝

def _apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(30, 30, 30))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(208, 208, 208))
    p.setColor(QPalette.ColorRole.Base,            QColor(22, 22, 22))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(35, 35, 35))
    p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(40, 40, 40))
    p.setColor(QPalette.ColorRole.ToolTipText,     QColor(208, 208, 208))
    p.setColor(QPalette.ColorRole.Text,            QColor(208, 208, 208))
    p.setColor(QPalette.ColorRole.Button,          QColor(45, 45, 45))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(208, 208, 208))
    p.setColor(QPalette.ColorRole.BrightText,      QColor(255, 51, 51))
    p.setColor(QPalette.ColorRole.Link,            QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.Highlight,       QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(p)


def _install_qt_log_filter():
    """Suppress noisy, non-actionable Qt FFmpeg teardown warnings."""
    noisy_prefix = (
        "QObject::disconnect: wildcard call disconnects from destroyed signal of QFFmpeg::"
    )

    def _handler(_msg_type, _context, message):
        text = str(message or "")
        if text.startswith(noisy_prefix):
            return
        try:
            print(text)
        except Exception:
            pass

    qInstallMessageHandler(_handler)


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Entry Point                                                  ║
# ╚═══════════════════════════════════════════════════════════════╝

def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video", default=None,
                        help="Auto-open this video on launch")
    parser.add_argument("--resolution", default=None,
                        help="Initial resolution preset (1080p/720p/540p/Source)")
    parser.add_argument("--precision", default=None,
                        help="Initial precision preset (GUI label)")
    parser.add_argument("--use-hg", default=None,
                        help="Enable HG (1/0). Default from saved settings.")
    parser.add_argument("--view", default=None,
                        help="Initial view mode (Tabbed)")
    parser.add_argument("--autoplay", default="0",
                        help="Auto-start playback after loading video (0/1)")
    parser.add_argument("--start-frame", default=None,
                        help="Initial frame index to seek to after startup")
    parser.add_argument("--upscale", default=None,
                        help="Initial upscale mode (GUI label)")
    parser.add_argument("--film-grain", default=None,
                        help="Enable film grain shader (0/1)")
    args, _unknown = parser.parse_known_args()

    os.chdir(_ROOT)
    _install_qt_log_filter()
    app = QApplication(sys.argv)
    _apply_dark_theme(app)
    win = MainWindow(
        initial_video=args.video,
        initial_resolution=args.resolution,
        initial_precision=args.precision,
        initial_view=args.view,
        initial_use_hg=args.use_hg,
        initial_autoplay=(str(args.autoplay).strip() == "1"),
        initial_start_frame=args.start_frame,
        initial_upscale=args.upscale,
        initial_film_grain=args.film_grain,
    )
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
