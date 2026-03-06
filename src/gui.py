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

# ── Inductor FX-graph cache (must be set BEFORE importing torch) ─────
# Ensures that torch.compile autotune decisions are persisted to disk
# and shared across processes, so the worker never re-benchmarks kernels
# that the subprocess already compiled.
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

# Import torch BEFORE PyQt6 — on ROCm-Windows the ROCm SDK DLLs must
# be loaded first; PyQt6 loads its own DLLs which can conflict if torch
# hasn't initialised ROCm yet.
import torch

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
    QTextEdit, QInputDialog, QSlider, QStyle, QStackedWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QProcess, QTimer, QRect
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
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
    },
    "INT8 Mixed (QAT)": {
        "precision": "int8-mixed",
        "model": _weight("Ensemble_AGCM_LE_int8_mixed_qat.pt"),
    },
    "INT8 Full (W8A8)": {
        "precision": "int8-full",
        "model": _weight("Ensemble_AGCM_LE_int8_full.pt"),
    },
}

MAX_W, MAX_H = 1920, 1080

# ── Resolution-scale presets (process lower resolution) ──
RESOLUTION_SCALES = {
    "1080p": None,            # full output resolution path (no upscale stage)
    "720p":  (1280, 720),
    "540p":  (960,  540),
}

UPSCALE_MODES = [
    "Bicubic (GPU)",
    "Lanczos (GPU)",
    "Spline36 (GPU)",
]

UPSCALE_MODE_TO_MPV_SCALE = {
    "Bicubic (GPU)": "bicubic",
    "Lanczos (GPU)": "lanczos",
    "Spline36 (GPU)": "spline36",
}

UPSCALE_MODE_TO_CV2_INTERP = {
    "Bicubic (GPU)": cv2.INTER_CUBIC,
    "Lanczos (GPU)": cv2.INTER_LANCZOS4,
    # Closest OpenCV equivalent for CPU fallback.
    "Spline36 (GPU)": cv2.INTER_LANCZOS4,
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

    @staticmethod
    def _kernel_antiring(scale_kernel: str) -> tuple[str, float]:
        """Return normalized kernel name and anti-ringing strength."""
        k = str(scale_kernel or "bicubic").strip().lower()
        if not k:
            k = "bicubic"
        # Lanczos/Spline can ring on high-contrast text edges; clamp it.
        if k in {"lanczos", "spline36"}:
            return k, 0.80
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
                       audio_path: str | None = None,
                       force_hdr_metadata: bool = True):
        """Create an mpv instance and begin reading frames."""
        self.stop_playback()
        self._shutdown.clear()
        self._queue = _queue.Queue(maxsize=1)
        self._fps = float(fps) if fps and fps > 0 else 30.0
        self._force_hdr_metadata = bool(force_hdr_metadata)
        kernel_name, antiring = self._kernel_antiring(scale_kernel)
        if scale_antiring is not None:
            antiring = max(0.0, min(1.0, float(scale_antiring)))

        self._last_playback_cfg = {
            "width": int(width),
            "height": int(height),
            "fps": float(self._fps),
            "scale_kernel": str(kernel_name),
            "audio_path": audio_path,
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
            scale=str(kernel_name),         # GPU scaler kernel in mpv
            cscale=str(kernel_name),
            scale_antiring=str(antiring),
            cscale_antiring=str(antiring),
        )
        if self._force_hdr_metadata:
            mpv_kwargs.update(
                target_colorspace_hint="yes",
                target_trc="pq",
                target_prim="bt.2020",
                vf="format=colorlevels=full:primaries=bt.2020:gamma=pq",
            )
        else:
            mpv_kwargs.update(target_colorspace_hint="no")
        use_external_audio = bool(audio_path and os.path.isfile(audio_path))
        if not use_external_audio:
            mpv_kwargs["audio"] = "no"

        player = mpv_lib.MPV(**mpv_kwargs)

        self._player = player
        player.play(pipe_url)
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

                    if _MPV_DIAG and not printed_once and len(aux.get("vop", {})) > 2:
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
                print("[mpv] seek command failed; keeping video-only seek.")

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

    def set_scale_kernel(self, scale_kernel: str) -> bool:
        """Update mpv luma/chroma scaling kernels at runtime."""
        p = self._player
        if p is None:
            return False
        kernel, antiring = self._kernel_antiring(scale_kernel)
        try:
            p.command("set", "scale", kernel)
            p.command("set", "cscale", kernel)
            p.command("set", "scale-antiring", str(antiring))
            p.command("set", "cscale-antiring", str(antiring))
            return True
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
                   "audio_path", "force_hdr_metadata"}
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
        self._processor = None
        self._stop_flag = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused
        self._pending_precision = None
        self._mpv_widget: MpvHDRWidget | None = None
        self._hdr_queue: _queue.Queue | None = None
        self._hdr_thread: threading.Thread | None = None
        self._proc_w: int = MAX_W
        self._proc_h: int = MAX_H
        self._output_w: int = MAX_W       # display / mpv resolution
        self._output_h: int = MAX_H
        self._sdr_visible: bool = True   # toggled by main thread
        self._seek_frame: int | None = None   # pending seek request
        self._user_paused: bool = False          # True while user has paused
        self._source: VideoSource | None = None  # ref for seeking
        self._pending_resolution: tuple[int, int] | None = None
        self._upscale_mode: str = "Bicubic (GPU)"
        self._pending_upscale_mode: str | None = None
        self._input_is_hdr: bool = False
        # App-level dedicated GPU memory (VRAM) from Windows perf counters.
        self._app_vram_mb: float = 0.0
        self._app_vram_poll_stop = threading.Event()
        self._app_vram_poll_thread: threading.Thread | None = None

    # ── public API (called from main thread) ──

    def configure(self, video_path, precision_key, proc_w=MAX_W, proc_h=MAX_H,
                  output_w=MAX_W, output_h=MAX_H, upscale_mode="Bicubic (GPU)",
                  input_is_hdr=False):
        self._video_path = video_path
        self._precision_key = precision_key
        self._proc_w = proc_w
        self._proc_h = proc_h
        self._output_w = output_w
        self._output_h = output_h
        self._upscale_mode = upscale_mode
        self._input_is_hdr = bool(input_is_hdr)

    def request_precision_change(self, key):
        self._pending_precision = key

    def request_resolution_change(self, proc_w: int, proc_h: int):
        """Request a processing-resolution change (thread-safe)."""
        self._pending_resolution = (proc_w, proc_h)

    def request_upscale_change(self, upscale_mode: str):
        """Request display upscale-kernel change (thread-safe)."""
        self._pending_upscale_mode = str(upscale_mode)

    def set_mpv_widget(self, widget):
        """Set the MpvHDRWidget reference for feeding HDR frames."""
        self._mpv_widget = widget

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
        cfg = PRECISIONS[key]
        path = cfg["model"]
        if not os.path.isfile(path):
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

    @staticmethod
    def _hdr_feeder_fn(hdr_q: _queue.Queue, mpv_widget: MpvHDRWidget):
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
            # item is a GPU fp16 tensor (1,3,H,W) — all heavy work here
            with torch.inference_mode():
                raw_cpu = (item.squeeze(0)
                           .clamp_(0.0, 1.0)
                           .permute(1, 2, 0)
                           .contiguous()
                           .cpu()
                           .numpy())            # float16 HWC on CPU
            hdr_u16 = (raw_cpu.astype(np.float32).__imul__(65535)
                       ).__add__(0.5)
            np.clip(hdr_u16, 0, 65535, out=hdr_u16)
            mpv_widget.feed_frame(hdr_u16.astype(np.uint16).data)

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

        # Start async HDR feeder if mpv is active
        mpv_w = self._mpv_widget
        if mpv_w is not None and not self._input_is_hdr:
            self._start_hdr_feeder()
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
                    self._silent_warmup(self._processor, proc_w, proc_h)
                    self.status_message.emit(
                        f"Ready — {self._precision_key} @ {proc_w}×{proc_h}")

            pending_up = self._pending_upscale_mode
            if pending_up is not None:
                self._pending_upscale_mode = None
                self._upscale_mode = pending_up

            # Seek gate
            seek_to = self._seek_frame
            if seek_to is not None:
                self._seek_frame = None
                source.seek(seek_to)
                frame_idx = max(0, seek_to - 1)
                force_position_emit = True
                seek_frame_ready_pending = True
                next_frame_t = time.perf_counter()
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
            if now < next_frame_t:
                time.sleep(next_frame_t - now)
            elif now - next_frame_t > (frame_interval_s * 2.0):
                # After long stalls (seek/pause/compile hiccup), resync.
                next_frame_t = now

            ret, frame = source.read()
            if not ret:
                break

            frame_idx += 1

            # FPS limiter via frame skipping (keeps wall-clock speed).
            if (not seek_frame_ready_pending) and frame_stride > 1 and (frame_idx % frame_stride) != 0:
                next_frame_t += frame_interval_s
                continue

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
                    mpv_w.feed_frame(rgb16.data)
                elif self._sdr_visible:
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
                    try:
                        self._hdr_queue.put_nowait(t_raw.clone())
                    except _queue.Full:
                        # Keep newest frame to avoid persistent HDR lag buildup.
                        try:
                            self._hdr_queue.get_nowait()
                        except _queue.Empty:
                            pass
                        try:
                            self._hdr_queue.put_nowait(t_raw.clone())
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
                            interp = UPSCALE_MODE_TO_CV2_INTERP.get(
                                self._upscale_mode, cv2.INTER_CUBIC
                            )
                        else:
                            interp = cv2.INTER_AREA
                        output = cv2.resize(output, (out_w, out_h), interpolation=interp)
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
                # mpv handles HDR; SDR QLabel still visible
                self.frame_ready.emit(display_frame, display_frame)
            # else: HDR Only + mpv — nothing to emit

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
                    model_mb = os.path.getsize(
                        PRECISIONS[self._precision_key]["model"]) / (1024 * 1024)
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
                })

        source.release()
        self._source = None
        self._stop_hdr_feeder()
        self._stop_app_vram_poll()
        self.playback_finished.emit()
        self.status_message.emit("Playback finished.")


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

class _PrecompileDialog(QDialog):
    """Modal dialog that launches ``compile_kernels.py`` in a separate
    process (zero GPU interference) and streams stdout into a log view."""

    def __init__(self, resolutions: list[str], precision: str = "fp16",
                 model_path: str | None = None, clear_cache: bool = False,
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

_TRITON_CACHE = _pathlib.Path.home() / ".triton" / "cache"


def _compiled_marker_path() -> _pathlib.Path:
    return _TRITON_CACHE / "hdrtvnet_compiled.txt"


def _is_compiled(w: int, h: int, precision: str) -> bool:
    """Check if clean-compiled kernels exist for this resolution+precision."""
    mp = _compiled_marker_path()
    if mp.is_file():
        key = f"{w}x{h}_{precision}"
        return key in mp.read_text(encoding="utf-8").splitlines()
    return False


def _mark_compiled(w: int, h: int, precision: str):
    """Record that kernels for this resolution+precision were compiled cleanly."""
    mp = _compiled_marker_path()
    mp.parent.mkdir(parents=True, exist_ok=True)
    key = f"{w}x{h}_{precision}"
    existing = set()
    if mp.is_file():
        existing = set(mp.read_text(encoding="utf-8").splitlines())
    existing.add(key)
    mp.write_text("\n".join(sorted(existing)) + "\n", encoding="utf-8")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Main Window                                                  ║
# ╚═══════════════════════════════════════════════════════════════╝

class MainWindow(QMainWindow):
    def __init__(self, initial_video=None, initial_resolution=None,
                 initial_precision=None, initial_upscale=None, initial_view=None,
                 initial_autoplay=False, initial_start_frame=None):
        super().__init__()
        self.setWindowTitle("HDRTVNet++ — Real-Time SDR → HDR Pipeline")
        self.setMinimumSize(1024, 600)
        self.resize(1600, 900)
        self.setAcceptDrops(True)

        self._worker = PipelineWorker()
        self._video_path = None
        self._playing = False
        self._compile_dlg = None
        self._last_res = None          # (pw, ph) of last played video
        self._active_precision = None
        self._active_resolution = None
        self._active_upscale = None
        self._active_use_mpv = False
        self._source_hdr_info = {"is_hdr": False, "reason": "unknown"}
        self._last_seek_frame = 0
        self._post_seek_resync_frames = 0
        self._pending_seek_on_resume: int | None = None
        self._audio_last_hard_sync_t = 0.0
        self._resume_audio_after_seek = False
        self._seek_resume_target = 0
        self._seek_resume_started_t = 0.0
        self._audio_player = None
        self._audio_output = None
        self._audio_available = _HAS_QT_AUDIO
        self._volume_percent = 100
        self._auto_muted_low_fps = False
        self._scrub_muted = False
        self._scrub_unmute_seq = 0
        self._low_fps_count = 0
        self._high_fps_count = 0
        self._audio_fade_timer: QTimer | None = None
        self._audio_fade_steps = 12
        self._audio_fade_step_idx = 0
        self._startup_sync_pending = False
        self._last_sdr_frame: np.ndarray | None = None
        self._source_proc_dims: tuple[int, int] | None = None
        self._borderless_full_window = False
        self._saved_window_geometry: QRect | None = None
        self._saved_window_state = Qt.WindowState.WindowNoState
        self._act_borderless_full_window = None
        self._root_layout: QVBoxLayout | None = None
        self._immersive_saved_margins: tuple[int, int, int, int] | None = None
        self._immersive_saved_spacing: int | None = None
        self._immersive_saved_split_handle_width: int | None = None
        self._immersive_saved_view_mode: str | None = None
        self._immersive_saved_vis: dict[str, bool] = {}
        try:
            self._startup_seek_frame = (
                int(initial_start_frame) if initial_start_frame is not None else None
            )
        except (TypeError, ValueError):
            self._startup_seek_frame = None

        self._build_ui()
        self._connect_signals()
        self._init_audio_backend()
        self._load_user_settings(initial_resolution, initial_precision, initial_upscale, initial_view)

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
                legacy_upscale_map = {
                    "Film Bicubic": "Bicubic (GPU)",
                    "Lanczos": "Lanczos (GPU)",
                    "Spline36": "Spline36 (GPU)",
                }
                mapped_upscale = legacy_upscale_map.get(initial_upscale, initial_upscale)
                if mapped_upscale in UPSCALE_MODES:
                    self._cmb_upscale.setCurrentText(mapped_upscale)
                if initial_view in ("Side by Side", "HDR Only", "SDR Only"):
                    self._cmb_view.setCurrentText(initial_view)
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
        self._cmb_prec.addItems(PRECISIONS.keys())
        self._cmb_prec.setFixedWidth(170)
        row1.addWidget(self._cmb_prec)

        row1.addWidget(QLabel("Resolution:"))
        self._cmb_res = QComboBox()
        self._cmb_res.addItems(RESOLUTION_SCALES.keys())
        self._cmb_res.setFixedWidth(100)
        row1.addWidget(self._cmb_res)

        self._lbl_upscale = QLabel("GPU Scale:")
        row1.addWidget(self._lbl_upscale)
        self._cmb_upscale = QComboBox()
        self._cmb_upscale.addItems(UPSCALE_MODES)
        self._cmb_upscale.setFixedWidth(180)
        row1.addWidget(self._cmb_upscale)

        self._btn_apply_settings = QPushButton("Apply")
        self._btn_apply_settings.setFixedWidth(90)
        self._btn_apply_settings.setEnabled(False)
        row1.addWidget(self._btn_apply_settings)

        row1.addWidget(QLabel("View:"))
        self._cmb_view = QComboBox()
        self._cmb_view.addItems(["Side by Side", "HDR Only", "SDR Only"])
        self._cmb_view.setCurrentText("HDR Only")
        self._cmb_view.setFixedWidth(130)
        row1.addWidget(self._cmb_view)

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
        self._lbl_volume = QLabel("Volume:")
        self._sld_volume = QSlider(Qt.Orientation.Horizontal)
        self._sld_volume.setRange(0, 100)
        self._sld_volume.setValue(100)
        self._sld_volume.setFixedWidth(140)
        self._sld_volume.setToolTip("Master volume")
        self._lbl_volume_val = QLabel("100%")
        self._lbl_volume_val.setFixedWidth(42)

        row2.addWidget(self._btn_play)
        row2.addWidget(self._btn_pause)
        row2.addWidget(self._btn_stop)
        row2.addStretch()
        row2.addWidget(self._lbl_volume)
        row2.addWidget(self._sld_volume)
        row2.addWidget(self._lbl_volume_val)
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
        self._split = QSplitter(Qt.Orientation.Horizontal)
        self._disp_sdr = VideoDisplay("SDR Input")
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
        self._split.addWidget(self._disp_sdr)
        self._split.addWidget(self._disp_hdr)
        if _HAS_MPV and self._disp_hdr_mpv is not None:
            self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_mpv)
        self._split.setStretchFactor(0, 1)
        self._split.setStretchFactor(1, 1)
        root.addWidget(self._split, 1)

        # Apply the view mode immediately
        self._on_view(self._cmb_view.currentText())

        # ---- Metrics panel ----
        self._grp_metrics = QGroupBox("Metrics")
        ml = QHBoxLayout(self._grp_metrics)
        ml.setContentsMargins(12, 4, 12, 4)

        self._m = {}
        mono = QFont("Consolas", 9)
        for key in ("fps", "latency", "frame", "res", "gpu", "cpu", "model", "prec"):
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


    # ── Signal wiring ────────────────────────────────────────

    def _connect_signals(self):
        self._btn_file.clicked.connect(self._open_file)
        self._btn_play.clicked.connect(self._play)
        self._btn_pause.clicked.connect(self._toggle_pause)
        self._btn_stop.clicked.connect(self._stop)
        self._btn_apply_settings.clicked.connect(self._apply_runtime_settings)
        self._chk_metrics.toggled.connect(
            lambda on: self._grp_metrics.setVisible(on))
        self._chk_metrics.toggled.connect(lambda _on: self._save_user_settings())
        self._sld_volume.valueChanged.connect(self._on_volume_changed)
        self._cmb_prec.currentTextChanged.connect(self._on_precision)
        self._cmb_prec.currentTextChanged.connect(lambda _v: self._save_user_settings())
        self._cmb_res.currentTextChanged.connect(self._on_resolution)
        self._cmb_res.currentTextChanged.connect(lambda _v: self._save_user_settings())
        self._cmb_upscale.currentTextChanged.connect(self._on_upscale)
        self._cmb_upscale.currentTextChanged.connect(lambda _v: self._save_user_settings())
        self._cmb_view.currentTextChanged.connect(self._on_view)
        self._cmb_view.currentTextChanged.connect(lambda _v: self._save_user_settings())

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
        """Disable scaler selection when full-resolution path has no upscale stage."""
        res_key = self._cmb_res.currentText()
        upscale_enabled = (res_key != "1080p")
        self._lbl_upscale.setEnabled(upscale_enabled)
        self._cmb_upscale.setEnabled(upscale_enabled)
        if upscale_enabled:
            self._cmb_upscale.setToolTip("Select mpv GPU scaling kernel.")
        else:
            self._cmb_upscale.setToolTip("Disabled at 1080p (no upscale step).")

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

        if enabled:
            self._immersive_saved_vis = {k: w.isVisible() for k, w in targets.items()}
            m = self._root_layout.contentsMargins()
            self._immersive_saved_margins = (m.left(), m.top(), m.right(), m.bottom())
            self._immersive_saved_spacing = self._root_layout.spacing()
            self._immersive_saved_split_handle_width = self._split.handleWidth()
            self._immersive_saved_view_mode = self._cmb_view.currentText()

            for w in targets.values():
                w.setVisible(False)
            self._root_layout.setContentsMargins(0, 0, 0, 0)
            self._root_layout.setSpacing(0)
            self._split.setHandleWidth(0)
            self._set_view_mode_silently("HDR Only")
            return

        for key, w in targets.items():
            w.setVisible(self._immersive_saved_vis.get(key, True))

        if self._immersive_saved_margins is not None:
            l, t, r, b = self._immersive_saved_margins
            self._root_layout.setContentsMargins(l, t, r, b)
        if self._immersive_saved_spacing is not None:
            self._root_layout.setSpacing(self._immersive_saved_spacing)
        if self._immersive_saved_split_handle_width is not None:
            self._split.setHandleWidth(self._immersive_saved_split_handle_width)
        if self._immersive_saved_view_mode:
            self._set_view_mode_silently(self._immersive_saved_view_mode)

    def _set_pause_button_labels(self, paused: bool):
        if paused:
            self._btn_pause.setText("▶  Resume")
        else:
            self._btn_pause.setText("⏸  Pause")

    def _toggle_borderless_full_window(self):
        if self._borderless_full_window:
            self._exit_borderless_full_window()
        else:
            self._enter_borderless_full_window()

    def _enter_borderless_full_window(self):
        if self._borderless_full_window:
            return
        self._saved_window_geometry = self.geometry()
        self._saved_window_state = self._without_fullscreen(self.windowState())

        self.menuBar().setVisible(False)
        self.statusBar().setVisible(False)
        self._set_immersive_video_ui(True)

        self.setWindowState(self._without_fullscreen(self.windowState()))
        self.setWindowState(Qt.WindowState.WindowNoState)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.show()
        self.setGeometry(self._safe_borderless_geometry())

        self._borderless_full_window = True
        if self._act_borderless_full_window is not None:
            self._act_borderless_full_window.setChecked(True)
        QTimer.singleShot(0, self._refresh_mpv_after_window_state_change)

    def _exit_borderless_full_window(self):
        if not self._borderless_full_window:
            return
        restore_state = self._without_fullscreen(self._saved_window_state)
        restore_geom = self._saved_window_geometry

        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, False)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
        self.show()
        self.menuBar().setVisible(True)
        self.statusBar().setVisible(True)
        self._set_immersive_video_ui(False)

        self._borderless_full_window = False
        if self._act_borderless_full_window is not None:
            self._act_borderless_full_window.setChecked(False)

        self.setWindowState(self._without_fullscreen(self.windowState()))
        self.setWindowState(Qt.WindowState.WindowNoState)
        if restore_geom is not None:
            self.setGeometry(restore_geom)
        if bool(restore_state & Qt.WindowState.WindowMaximized):
            self.showMaximized()
        else:
            self.showNormal()

        QTimer.singleShot(0, self._refresh_mpv_after_window_state_change)

    def _should_use_mpv_pipeline(self) -> bool:
        return self._disp_hdr_mpv is not None

    def _load_user_settings(self, initial_resolution, initial_precision, initial_upscale, initial_view):
        """Load persisted GUI preferences unless explicitly overridden by CLI."""
        if not os.path.isfile(_PREFS_PATH):
            return
        try:
            with open(_PREFS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        if initial_precision is None:
            p = data.get("precision")
            if p in PRECISIONS:
                self._cmb_prec.setCurrentText(p)
        if initial_resolution is None:
            r = data.get("resolution")
            if r == "Native":
                r = "1080p"
            if r in RESOLUTION_SCALES or r == "Source":
                self._cmb_res.setCurrentText(r)
        if initial_upscale is None:
            u = data.get("upscale")
            if u in UPSCALE_MODES:
                self._cmb_upscale.setCurrentText(u)
        if initial_view is None:
            v = data.get("view")
            if v in ("Side by Side", "HDR Only", "SDR Only"):
                self._cmb_view.setCurrentText(v)

        m = data.get("show_metrics")
        if isinstance(m, bool):
            self._chk_metrics.setChecked(m)
            self._grp_metrics.setVisible(m)

        vol = data.get("volume_percent")
        if isinstance(vol, int):
            self._sld_volume.setValue(max(0, min(100, vol)))

    def _save_user_settings(self):
        data = {
            "precision": self._cmb_prec.currentText(),
            "resolution": self._cmb_res.currentText(),
            "upscale": self._cmb_upscale.currentText(),
            "view": self._cmb_view.currentText(),
            "show_metrics": self._chk_metrics.isChecked(),
            "volume_percent": int(self._volume_percent),
        }
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

    def _start_audio_restore_fade(self, duration_ms: int = 320):
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
            self._apply_volume_to_backends()
            self.statusBar().showMessage("Audio auto-muted (FPS below 20).")
        else:
            self._start_audio_restore_fade()
            self.statusBar().showMessage("Audio restored.")

    def _arm_mute_until_fps_recovery(self):
        """Force mute now; unmute only via measured FPS recovery logic."""
        self._low_fps_count = 0
        self._high_fps_count = 0
        if not self._auto_muted_low_fps:
            self._set_low_fps_mute(True)

    def _update_auto_mute_from_fps(self, fps_value: float):
        """Hysteresis/debounce to avoid sticky mute after transient skips."""
        fps = float(fps_value)
        # Hysteresis band:
        # - Mute only if fps stays low enough.
        # - Unmute only if fps recovers well above threshold.
        low_trip = 18.0
        high_trip = 23.0

        if fps < low_trip:
            self._low_fps_count += 1
            self._high_fps_count = 0
        elif fps > high_trip:
            self._high_fps_count += 1
            self._low_fps_count = 0
        else:
            # Inside deadband: decay counters to avoid stale state.
            self._low_fps_count = max(0, self._low_fps_count - 1)
            self._high_fps_count = max(0, self._high_fps_count - 1)

        # Require consecutive updates to toggle.
        if not self._auto_muted_low_fps and self._low_fps_count >= 3:
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

    def _start_audio_playback(self, path: str):
        if not self._audio_available or self._audio_player is None:
            return
        try:
            self._audio_player.stop()
            self._audio_player.setSource(QUrl.fromLocalFile(path))
            self._audio_player.setPosition(0)
            self._audio_player.setPlaybackRate(1.0)
            self._audio_player.play()
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
        self._startup_sync_pending = False
        if self._audio_available:
            self._seek_audio_seconds(0.0)
            self._set_audio_paused(False)
        self._worker.resume()
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.set_paused(False)

    def _has_pending_setting_changes(self) -> bool:
        if not self._playing:
            return False
        return (
            self._cmb_prec.currentText() != self._active_precision
            or self._cmb_res.currentText() != self._active_resolution
            or self._cmb_upscale.currentText() != self._active_upscale
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
            self._disp_sdr.update_frame(preview)
            if self._disp_hdr_cpu is not None:
                self._disp_hdr_cpu.update_frame(preview)
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_cpu)
        else:
            self._disp_sdr.clear_display()
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
            pw, ph, fps, scale_kernel, audio_path, force_hdr_metadata = pending
            self._disp_hdr_mpv.start_playback(
                pw, ph, fps=fps, scale_kernel=scale_kernel, audio_path=audio_path,
                force_hdr_metadata=force_hdr_metadata,
            )
            self._apply_volume_to_backends()
            if self._startup_sync_pending:
                self._disp_hdr_mpv.set_paused(True)
            self._worker.set_mpv_widget(self._disp_hdr_mpv)
            self._pending_mpv_start = None
            if self._startup_sync_pending:
                QTimer.singleShot(250, self._release_startup_sync)

    def _precompile_kernels(self):
        """Open the pre-compile dialog — runs compile_kernels.py as a
        completely separate process with zero GPU interference."""

        # Ask for resolutions
        text, ok = QInputDialog.getText(
            self, "Pre-compile Kernels",
            "Enter resolutions to compile (e.g. 1920x1080 1440x1080):",
            text="1920x1080",
        )
        if not ok or not text.strip():
            return

        resolutions = text.strip().split()
        # Validate
        for r in resolutions:
            try:
                w, h = r.lower().split("x")
                int(w); int(h)
            except (ValueError, AttributeError):
                QMessageBox.warning(
                    self, "Invalid Resolution",
                    f"'{r}' is not a valid resolution.\n"
                    "Use WxH format (e.g. 1920x1080).",
                )
                return

        # Map GUI precision name → compile_kernels.py precision arg
        prec_map = {
            "FP16": "fp16", "FP32": "fp32",
            "INT8 Mixed (PTQ)": "int8-mixed",
            "INT8 Mixed (QAT)": "int8-mixed",
            "INT8 Full (W8A8)": "int8-full",
        }
        gui_prec = self._cmb_prec.currentText()
        prec_arg = prec_map.get(gui_prec, "fp16")

        # For QAT / PTQ / INT8-full, pass the specific model path
        cfg = PRECISIONS.get(gui_prec, {})
        model_path = cfg.get("model")

        dlg = _PrecompileDialog(
            resolutions, precision=prec_arg, model_path=model_path,
            clear_cache=False, parent=self,
        )
        dlg.exec()          # modal — blocks until user closes
        if dlg.succeeded:
            for r in resolutions:
                w, h = r.lower().split("x")
                _mark_compiled(int(w), int(h), prec_arg)

    def _clear_kernel_cache(self):
        """Delete cached Triton / TorchInductor kernels."""
        import shutil, pathlib, getpass, tempfile

        dirs = []
        triton_dir = pathlib.Path.home() / ".triton" / "cache"
        if triton_dir.exists():
            dirs.append(triton_dir)

        inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if not inductor_dir:
            inductor_dir = os.path.join(
                tempfile.gettempdir(),
                f"torchinductor_{getpass.getuser()}",
            )
        inductor_path = pathlib.Path(inductor_dir)
        if inductor_path.exists():
            dirs.append(inductor_path)

        if not dirs:
            QMessageBox.information(
                self, "Kernel Cache",
                "No Triton / Inductor kernel cache found.",
            )
            return

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
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
            QMessageBox.information(
                self, "Kernel Cache",
                "Cache cleared.  Kernels will recompile on next play.",
            )

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
                upscale=self._cmb_upscale.currentText(),
                view=self._cmb_view.currentText(),
                autoplay=auto_play,
            )
            return

        self._video_path = path
        self._refresh_resolution_options_for_video(path)
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

    def _restart_with_video(self, path, resolution=None, precision=None, upscale=None, view=None,
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
        if upscale in UPSCALE_MODES:
            args += ["--upscale", upscale]
        if view in ("Side by Side", "HDR Only", "SDR Only"):
            args += ["--view", view]
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

        # Set up seek slider
        self._vid_fps = vfps if vfps > 0 else 30.0
        display_fps = _limited_playback_fps(self._vid_fps)
        self._seek_slider.setRange(0, max(0, total_frames - 1))
        self._seek_slider.setValue(0)
        self._seek_slider.setEnabled(True)
        self._seek_slider.setToolTip("Seek while paused is queued and applied on Resume.")
        self._lbl_time.setText("0:00")
        dur_secs = total_frames / self._vid_fps if self._vid_fps > 0 else 0
        self._lbl_duration.setText(self._fmt_time(dur_secs))

        # Map GUI precision to compile arg
        gui_prec = self._cmb_prec.currentText()
        prec_map = {
            "FP16": "fp16", "FP32": "fp32",
            "INT8 Mixed (PTQ)": "int8-mixed",
            "INT8 Mixed (QAT)": "int8-mixed",
            "INT8 Full (W8A8)": "int8-full",
        }
        prec_arg = prec_map.get(gui_prec, "fp16")

        source_is_hdr = bool(self._source_hdr_info.get("is_hdr", False))
        if source_is_hdr:
            self.statusBar().showMessage(
                "HDR input detected. Using model path (OpenCV decode is 8-bit)."
            )

        # Always compile via a clean subprocess — this ensures autotune
        # benchmarks have zero GPU interference from Qt / D3D11 / mpv.
        # If the Triton + Inductor cache is already warm from a previous
        # compile, the subprocess finishes in seconds and auto-closes.
        cfg = PRECISIONS.get(gui_prec, {})
        model_path = cfg.get("model")
        dlg = _PrecompileDialog(
            [f"{pw}x{ph}"], precision=prec_arg,
            model_path=model_path, parent=self,
        )
        dlg.exec()                        # modal — blocks until done
        if dlg.succeeded:
            _mark_compiled(pw, ph, prec_arg)
        else:
            # Compile failed or user closed early — don't start playback
            return

        # ── Start playback ──
        self._last_res = (pw, ph)
        self._playing = True
        self._active_precision = self._cmb_prec.currentText()
        self._active_resolution = self._cmb_res.currentText()
        self._active_upscale = self._cmb_upscale.currentText()
        # Keep mpv initialized whenever available so view switches are UI-only.
        use_mpv_pipeline = (self._disp_hdr_mpv is not None)
        self._active_use_mpv = use_mpv_pipeline
        self._startup_sync_pending = bool(use_mpv_pipeline)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_stack.setCurrentWidget(
                self._disp_hdr_mpv if use_mpv_pipeline else self._disp_hdr_cpu
            )
        self._update_apply_button_state()
        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_stop.setEnabled(True)
        self._btn_file.setEnabled(False)
        self._cmb_prec.setEnabled(True)
        self._set_pause_button_labels(False)

        # Start mpv HDR display AFTER compile finishes (via signal)
        # so that mpv's D3D11 GPU usage doesn't pollute Triton autotuning.
        # mpv receives frames at processing resolution; GPU scaling happens in mpv.
        self._pending_mpv_start = None
        if use_mpv_pipeline and self._disp_hdr_mpv is not None:
            mpv_audio_path = None if self._audio_available else self._video_path
            mpv_scale_kernel = UPSCALE_MODE_TO_MPV_SCALE.get(
                self._cmb_upscale.currentText(), "bicubic"
            )
            self._pending_mpv_start = (
                pw, ph, float(display_fps), mpv_scale_kernel, mpv_audio_path,
                True,
            )
        else:
            self._worker.set_mpv_widget(None)

        self._worker.configure(
            self._video_path, self._cmb_prec.currentText(),
            proc_w=pw, proc_h=ph,
            output_w=ow, output_h=oh,
            upscale_mode=self._cmb_upscale.currentText(),
            input_is_hdr=False,
        )

        # Show loading dialog (in-process model load + cache warmup is fast
        # since subprocess already compiled the kernels)
        self._compile_dlg = _CompileDialog(self)
        self._compile_dlg.show()

        self._worker.start()
        if pw != ow or ph != oh:
            upscale_backend = "mpv GPU" if use_mpv_pipeline else "CPU fallback"
            self.statusBar().showMessage(
                f"Upscale active: {pw}×{ph} -> {ow}×{oh} via {self._cmb_upscale.currentText()} ({upscale_backend})"
            )
        else:
            self.statusBar().showMessage(
                f"No upscale stage: processing at {ow}×{oh}."
            )
        self._post_seek_resync_frames = 0
        self._pending_seek_on_resume = None
        if self._startup_sync_pending:
            self._worker.pause()
        if self._audio_available:
            self._start_audio_playback(self._video_path)
            if self._startup_sync_pending:
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

    def _toggle_pause(self):
        if not self._playing:
            return
        if self._worker.is_paused:
            queued = self._pending_seek_on_resume
            if queued is not None:
                self._worker.request_seek(int(queued))
                fps = getattr(self, '_vid_fps', 30.0)
                self._seek_audio_seconds(int(queued) / max(fps, 1e-6))
                if self._disp_hdr_mpv is not None and not self._audio_available:
                    self._disp_hdr_mpv.seek_seconds(int(queued) / max(fps, 1e-6))
                self._post_seek_resync_frames = 120
                self._resume_audio_after_seek = bool(self._audio_available)
                self._seek_resume_target = int(queued)
                self._seek_resume_started_t = time.perf_counter()
                self._pending_seek_on_resume = None
            self._worker.resume()
            self._set_pause_button_labels(False)
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_mpv.set_paused(False)
            if not self._resume_audio_after_seek:
                self._set_audio_paused(False)
        else:
            self._worker.pause()
            self._set_pause_button_labels(True)
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_mpv.set_paused(True)
            self._set_audio_paused(True)

    def _stop(self):
        self._worker.stop()
        self._worker.wait(10000)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.stop_playback()
        self._stop_audio_playback()
        self._reset_controls()

    # ── Slots: settings ──────────────────────────────────────

    def _on_precision(self, key):
        if self._playing:
            self._update_apply_button_state()

    def _on_resolution(self, scale_key):
        self._sync_upscale_controls()
        if self._playing:
            self._update_apply_button_state()

    def _on_upscale(self, mode):
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
        new_up = self._cmb_upscale.currentText()
        needs_restart = (new_res != self._active_resolution)
        notices: list[str] = []

        if needs_restart:
            self._restart_with_video(
                self._video_path,
                resolution=new_res,
                precision=new_prec,
                upscale=new_up,
                view=self._cmb_view.currentText(),
                autoplay=True,
                start_frame=int(self._seek_slider.value()),
            )
            return

        if new_prec != self._active_precision:
            self._worker.request_precision_change(new_prec)
            notices.append(f"Applying precision change: {new_prec}")
            self._active_precision = new_prec

        if new_up != self._active_upscale:
            old_up = self._active_upscale or "Unknown"
            mpv_scale_kernel = UPSCALE_MODE_TO_MPV_SCALE.get(new_up, "bicubic")
            _, antiring = MpvHDRWidget._kernel_antiring(mpv_scale_kernel)
            applied_now = False
            if self._disp_hdr_mpv is not None and self._active_use_mpv:
                applied_now = self._disp_hdr_mpv.set_scale_kernel(mpv_scale_kernel)
            if self._worker is not None:
                self._worker.request_upscale_change(new_up)
                applied_now = True
            if applied_now:
                notices.append(
                    f"Applying upscale change: {old_up} -> {new_up} (mpv kernel: {mpv_scale_kernel}, antiring: {antiring:.2f})"
                )
            else:
                notices.append(
                    f"Upscale changed: {old_up} -> {new_up}. Will apply on next playback start."
                )

        self._active_resolution = new_res
        self._active_upscale = new_up
        self._save_user_settings()
        self._update_apply_button_state()
        if notices:
            self.statusBar().showMessage(" | ".join(notices))

    def _on_view(self, mode):
        self._disp_sdr.setVisible(mode != "HDR Only")
        self._disp_hdr.setVisible(mode != "SDR Only")
        if mode == "Side by Side":
            # Always restore equal split when returning to side-by-side mode.
            self._split.setSizes([1, 1])
        elif mode == "SDR Only":
            # Give SDR pane all available width.
            self._split.setSizes([1, 0])
        elif mode == "HDR Only":
            # Give HDR pane all available width.
            self._split.setSizes([0, 1])
        if _HAS_MPV and self._disp_hdr_mpv is not None:
            # Show textual placeholder when idle; switch to mpv only during playback.
            if self._playing and mode != "SDR Only":
                self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_mpv)
            else:
                self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_cpu)
        # Let the worker skip unnecessary copies / postprocess
        if self._playing:
            # Keep SDR path running in all views so switching back from HDR-only
            # is instantaneous and frame-accurate without forcing a seek.
            self._worker.set_sdr_visible(True)
            if mode != "HDR Only" and self._last_sdr_frame is not None:
                self._disp_sdr.update_frame(self._last_sdr_frame)

    def _refresh_mpv_after_window_state_change(self):
        if self._disp_hdr_mpv is None or not self._playing:
            return
        if self._cmb_view.currentText() == "SDR Only":
            return
        mpv_scale_kernel = UPSCALE_MODE_TO_MPV_SCALE.get(
            self._cmb_upscale.currentText(), "bicubic"
        )
        # Smooth path: avoid full VO recreation unless scaler command fails.
        if self._disp_hdr_mpv.set_scale_kernel(mpv_scale_kernel):
            return
        try:
            self._disp_hdr_mpv.refresh_surface()
            self._disp_hdr_mpv.set_scale_kernel(mpv_scale_kernel)
        except Exception:
            pass

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
            drift_ms = abs(have_ms - want_ms)
            now_t = time.perf_counter()

            # Post-seek: prefer soft clock correction; avoid hard jumps that crackle.
            if self._post_seek_resync_frames > 0:
                self._post_seek_resync_frames -= 1
                if drift_ms > 220 and (now_t - self._audio_last_hard_sync_t) > 0.8:
                    self._audio_player.setPosition(max(0, want_ms))
                    self._audio_last_hard_sync_t = now_t
                    self._audio_player.setPlaybackRate(1.0)
                elif drift_ms > 60:
                    if have_ms < want_ms:
                        self._audio_player.setPlaybackRate(1.03)
                    else:
                        self._audio_player.setPlaybackRate(0.97)
                else:
                    self._audio_player.setPlaybackRate(1.0)
            elif (current_frame % 10 == 0) and (not self._worker.is_paused):
                if drift_ms > 250 and (now_t - self._audio_last_hard_sync_t) > 1.0:
                    self._audio_player.setPosition(max(0, want_ms))
                    self._audio_last_hard_sync_t = now_t
                    self._audio_player.setPlaybackRate(1.0)
                elif drift_ms > 80:
                    if have_ms < want_ms:
                        self._audio_player.setPlaybackRate(1.01)
                    else:
                        self._audio_player.setPlaybackRate(0.99)
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
            self._schedule_scrub_unmute(120)
            return

        self._worker.request_seek(target_frame)
        self._seek_audio_seconds(target_frame / max(fps, 1e-6))
        self._post_seek_resync_frames = 120
        self._resume_audio_after_seek = bool(self._audio_available)
        self._seek_resume_target = int(target_frame)
        self._seek_resume_started_t = time.perf_counter()
        if self._disp_hdr_mpv is not None and not self._audio_available:
            self._disp_hdr_mpv.seek_seconds(target_frame / max(fps, 1e-6))
        self._schedule_scrub_unmute(120)

    def _schedule_scrub_unmute(self, delay_ms: int = 120):
        token = self._scrub_unmute_seq = (self._scrub_unmute_seq + 1)

        def _release():
            if token != self._scrub_unmute_seq:
                return
            if self._seek_slider.isSliderDown():
                return
            if self._scrub_muted:
                self._scrub_muted = False
                self._apply_volume_to_backends()

        QTimer.singleShot(max(0, int(delay_ms)), _release)

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
        self._last_sdr_frame = sdr
        if self._disp_sdr.isVisible():
            self._disp_sdr.update_frame(sdr)
        # HDR QLabel fallback (mpv gets fed directly from the worker)
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
        self._update_auto_mute_from_fps(float(m.get("fps", 0.0)))
    def _on_status_message(self, text: str):
        """Forward worker status to status bar *and* compile dialog."""
        self.statusBar().showMessage(text)
        if self._compile_dlg is not None:
            self._compile_dlg.set_status(text)

    def _on_finished(self):
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.stop_playback()
        self._stop_audio_playback()
        self._reset_controls()
        self.statusBar().showMessage("Playback finished.")
        self._disp_sdr.clear_display()
        if self._disp_hdr_cpu is not None:
            self._disp_hdr_cpu.clear_display()

    # ── UI helpers ───────────────────────────────────────────

    def _reset_controls(self):
        self._playing = False
        self._active_use_mpv = False
        self._startup_sync_pending = False
        self._auto_muted_low_fps = False
        self._scrub_muted = False
        self._low_fps_count = 0
        self._high_fps_count = 0
        if self._audio_fade_timer is not None:
            self._audio_fade_timer.stop()
        self._apply_volume_to_backends()
        self._post_seek_resync_frames = 0
        self._pending_seek_on_resume = None
        self._resume_audio_after_seek = False
        self._seek_resume_target = 0
        self._seek_resume_started_t = 0.0
        if self._compile_dlg is not None:
            self._compile_dlg.close()
            self._compile_dlg.deleteLater()
            self._compile_dlg = None
        self._btn_play.setEnabled(bool(self._video_path))
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_file.setEnabled(True)
        self._set_pause_button_labels(False)
        self._btn_apply_settings.setEnabled(False)
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
        self._save_user_settings()
        if self._playing:
            self._worker.stop()
            self._worker.wait(10000)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.stop_playback()
        self._stop_audio_playback()
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
    parser.add_argument("--upscale", default=None,
                        help="Initial GPU scaling kernel preset")
    parser.add_argument("--view", default=None,
                        help="Initial view mode (Side by Side/HDR Only/SDR Only)")
    parser.add_argument("--autoplay", default="0",
                        help="Auto-start playback after loading video (0/1)")
    parser.add_argument("--start-frame", default=None,
                        help="Initial frame index to seek to after startup")
    args, _unknown = parser.parse_known_args()

    os.chdir(_ROOT)
    app = QApplication(sys.argv)
    _apply_dark_theme(app)
    win = MainWindow(
        initial_video=args.video,
        initial_resolution=args.resolution,
        initial_precision=args.precision,
        initial_upscale=args.upscale,
        initial_view=args.view,
        initial_autoplay=(str(args.autoplay).strip() == "1"),
        initial_start_frame=args.start_frame,
    )
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
