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
    QTextEdit, QInputDialog, QSlider, QStyle,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QProcess, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QShortcut, QKeySequence

from models.hdrtvnet_torch import HDRTVNetTorch
from video_source import VideoSource

# ── Paths ────────────────────────────────────────────────────
_ROOT = os.path.dirname(_HERE)


def _weight(name):
    return os.path.join(_HERE, "models", "weights", name)


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

    def start_playback(self, width: int, height: int, fps: int = 30):
        """Create an mpv instance and begin reading frames."""
        self.stop_playback()
        self._shutdown.clear()
        self._queue = _queue.Queue(maxsize=4)

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

        # Limit demuxer buffer to ~2 frames so mpv can't build up a
        # large readahead backlog (which causes burst-then-stall with
        # untimed=yes).  Must be >0 or mpv can't buffer even 1 frame.
        frame_bytes = width * height * 6          # RGB48LE
        max_demux = str(frame_bytes * 2)

        player = mpv_lib.MPV(
            wid=wid,
            vo="gpu",
            gpu_api="d3d11",                # d3d11 required for HDR on Windows
            demuxer="rawvideo",
            demuxer_rawvideo_w=str(width),
            demuxer_rawvideo_h=str(height),
            demuxer_rawvideo_mp_format="rgb48le",
            demuxer_rawvideo_fps=str(fps),
            untimed=True,                   # display immediately
            audio="no",
            audio_file_auto="no",
            osc="no",
            input_default_bindings="no",
            input_vo_keyboard="no",
            # ── Minimal-buffer passthrough ──
            cache="no",
            demuxer_max_bytes=max_demux,    # ~2 frames (not 0!)
            demuxer_readahead_secs=0,
            video_sync="desync",            # don't pace to display refresh
            # ── HDR passthrough ──
            target_colorspace_hint="yes",   # signal display about content
            target_trc="pq",                # force PQ output
            target_prim="bt.2020",          # force BT.2020 primaries
            vf="format=colorlevels=full:primaries=bt.2020:gamma=pq",
            log_handler=print,              # surface mpv errors
        )

        self._player = player
        player.play(pipe_url)

        # Delayed diagnostic — poll until video-out-params is populated.
        # The rawvideo demuxer only provides {w, h} in video-params;
        # colorspace tags are injected by our vf=format filter and only
        # visible in video-out-params (which fills asynchronously once
        # the VO has rendered at least one frame).
        def _print_hdr_diag():
            import time as _t
            vop = None
            # Wait up to 60 s (covers Triton compile on first run)
            for _ in range(120):
                _t.sleep(0.5)
                p = self._player
                if p is None:
                    return
                try:
                    vop = p.video_out_params
                except Exception:
                    pass
                # video-out-params gets keys once the VO is live
                if vop and len(vop) > 2:
                    break
            p = self._player
            if p is None:
                return
            try:
                vp = p.video_params or {}
                vop = p.video_out_params or {}

                def _g(d, *keys):
                    for k in keys:
                        v = d.get(k)
                        if v is not None:
                            return str(v)
                    return '?'

                # Also try reading mpv properties directly
                def _prop(name):
                    try:
                        v = getattr(p, name.replace('-', '_'), None)
                        return str(v) if v is not None else '?'
                    except Exception:
                        return '?'

                out_prim = _g(vop, 'primaries', 'colormatrix-primaries')
                out_trc  = _g(vop, 'gamma', 'transfer')
                out_lvl  = _g(vop, 'levels', 'colorlevels')

                print("\n╔══════════ mpv HDR diagnostic ══════════╗")
                print(f"║  video-params keys : {list(vp.keys())}")
                print(f"║  video-out-params  : {list(vop.keys())}")
                print(f"║  VO output prims   : {out_prim}")
                print(f"║  VO output TRC     : {out_trc}")
                print(f"║  VO output levels  : {out_lvl}")
                print(f"║  target-trc        : {_prop('target_trc')}")
                print(f"║  target-prim       : {_prop('target_prim')}")
                print(f"║  target-peak       : {_prop('target_peak')}")
                print(f"║  current-vo        : {_prop('current_vo')}")
                print(f"║  gpu-api           : d3d11 (forced)")
                print(f"║  colorspace-hint   : yes (forced)")
                print("╠════════════════════════════════════════╣")

                # With target-trc=pq & target-prim=bt.2020 set, mpv
                # instructs the D3D11 swapchain to use HDR10 output.
                # The rawvideo demuxer won't tag colour, but vf=format
                # does, and target_* forces the output gamut/TRC.
                t_trc = _prop('target_trc')
                t_prim = _prop('target_prim')
                if 'pq' in t_trc and 'bt.2020' in t_prim:
                    print("║  ✓ target = BT.2020 + PQ")
                    print("║    mpv sends HDR10 to the D3D11 swapchain.")
                    print("║    If Windows 'Use HDR' is ON → true HDR.")
                else:
                    print(f"║  ✗ target = {t_prim}/{t_trc}")
                    print("║    HDR passthrough may not be active.")

                # Check video-out-params if populated
                if out_prim != '?' or out_trc != '?':
                    if 'bt.2020' in out_prim and 'pq' in out_trc:
                        print("║  ✓ VO confirms BT.2020 + PQ output")
                    else:
                        print(f"║  ⚠ VO reports {out_prim}/{out_trc}")
                else:
                    print("║  ℹ video-out-params not yet populated")
                    print("║    (normal for embedded wid= windows)")

                print("╚════════════════════════════════════════╝\n")

                # Emit HDR info dict for the GUI panel
                hdr_info = {
                    "primaries": out_prim,
                    "transfer": out_trc,
                    "levels": out_lvl,
                    "sig_peak": _g(vop, 'sig-peak', 'sig_peak'),
                    "max_cll": _g(vop, 'max-cll', 'max_cll'),
                    "max_fall": _g(vop, 'max-fall', 'max_fall'),
                    "vo": _prop('current_vo'),
                    "gpu_api": "d3d11",
                    "hdr_active": ('pq' in t_trc and 'bt.2020' in t_prim
                                   and 'bt.2020' in out_prim
                                   and 'pq' in out_trc),
                }
                self.hdr_info_ready.emit(hdr_info)
            except Exception as exc:
                import traceback
                print(f"[mpv diag] error: {exc}")
                traceback.print_exc()

        threading.Thread(target=_print_hdr_diag, daemon=True).start()

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
        self._sdr_visible: bool = True   # toggled by main thread
        self._seek_frame: int | None = None   # pending seek request
        self._user_paused: bool = False          # True while user has paused
        self._source: VideoSource | None = None  # ref for seeking

    # ── public API (called from main thread) ──

    def configure(self, video_path, precision_key, proc_w=MAX_W, proc_h=MAX_H):
        self._video_path = video_path
        self._precision_key = precision_key
        self._proc_w = proc_w
        self._proc_h = proc_h

    def request_precision_change(self, key):
        self._pending_precision = key

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
        self._hdr_queue = _queue.Queue(maxsize=2)
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
        if mpv_w is not None:
            self._start_hdr_feeder()

        source = VideoSource(self._video_path, prefetch=8)
        self._source = source
        total_frames = source.frame_count
        vid_fps = source.fps
        frame_times = deque(maxlen=300)
        frame_idx = 0
        process = psutil.Process(os.getpid())
        use_cuda = torch.cuda.is_available()

        while not self._stop_flag:
            # Hot-swap precision
            pending = self._pending_precision
            if pending and pending != self._precision_key:
                self._pending_precision = None
                if not self._load_model(pending):
                    continue

            # Seek gate
            seek_to = self._seek_frame
            if seek_to is not None:
                self._seek_frame = None
                source.seek(seek_to)
                frame_idx = seek_to

            # Pause gate
            self._pause_event.wait()
            if self._stop_flag:
                break

            ret, frame = source.read()
            if not ret:
                break

            t0 = time.perf_counter()

            h, w = frame.shape[:2]
            inp = frame
            if w > MAX_W or h > MAX_H:
                inp = cv2.resize(frame, (MAX_W, MAX_H),
                                 interpolation=cv2.INTER_AREA)

            # ── Fast path: preprocess → infer → (clone for mpv) ──────────
            # When mpv handles HDR display, skip postprocess entirely:
            # the GPU→CPU D2H transfer + numpy conversion would be wasted
            # since the QLabel HDR fallback is not active.  This is the
            # single largest per-frame overhead saved (~1 ms at 1080p).
            with torch.inference_mode():
                tensor, cond = self._processor.preprocess(inp)
                raw_out = self._processor.infer((tensor, cond))

                if mpv_w is not None and self._hdr_queue is not None:
                    # Fast GPU clone — never blocks the loop
                    t_raw = (raw_out[0] if isinstance(raw_out, (tuple, list))
                             else raw_out)
                    try:
                        self._hdr_queue.put_nowait(t_raw.clone())
                    except _queue.Full:
                        pass                 # drop frame, keep latency low

                # Only run the expensive postprocess (GPU→CPU D2H) when
                # the QLabel HDR fallback is the display path.
                need_hdr_cpu = (mpv_w is None)
                if need_hdr_cpu:
                    output = self._processor.postprocess(raw_out)
                    # postprocess() already calls stream.synchronize()
                elif use_cuda:
                    # Sync for timing only (no D2H to wait for)
                    torch.cuda.synchronize()

            t1 = time.perf_counter()
            frame_idx += 1
            frame_ms = (t1 - t0) * 1000.0
            frame_times.append(frame_ms)

            # Position update (every 5 frames to avoid signal flood)
            if frame_idx % 5 == 0:
                self.position_updated.emit(frame_idx, total_frames)

            # Emit only what the UI actually needs
            if need_hdr_cpu:
                # QLabel fallback — both SDR + HDR frames needed
                self.frame_ready.emit(inp.copy(), output.copy())
            elif self._sdr_visible:
                # mpv handles HDR; SDR QLabel still visible
                self.frame_ready.emit(inp.copy(), inp)  # hdr slot unused
            # else: HDR Only + mpv — nothing to emit

            # Re-pause: if user paused and no further seek pending,
            # block again now that this seek frame has been emitted.
            if self._user_paused and self._seek_frame is None:
                self._pause_event.clear()

            # Metrics every 10 frames
            if frame_idx % 10 == 0 and frame_times:
                avg = sum(frame_times) / len(frame_times)
                fps = 1000.0 / avg if avg > 0 else 0
                cpu_mb = process.memory_info().rss / (1024 * 1024)
                gpu_mb = (torch.cuda.memory_allocated() / (1024 * 1024)
                          if use_cuda else 0)
                model_mb = os.path.getsize(
                    PRECISIONS[self._precision_key]["model"]) / (1024 * 1024)
                self.metrics_updated.emit({
                    "fps": fps,
                    "latency_ms": avg,
                    "frame": frame_idx,
                    "cpu_mb": cpu_mb,
                    "gpu_mb": gpu_mb,
                    "model_mb": model_mb,
                    "precision": self._precision_key,
                })

        source.release()
        self._source = None
        self._stop_hdr_feeder()
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
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w,
                       QImage.Format.Format_RGB888).copy()
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
    def __init__(self, initial_video=None):
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

        self._is_fullscreen = False
        self._build_ui()
        self._connect_signals()

        # Auto-open video passed via --video (used by restart)
        if initial_video and os.path.isfile(initial_video):
            QTimer.singleShot(200, lambda: self._set_video(initial_video))

    # ── UI construction ──────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
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

        row1.addWidget(QLabel("View:"))
        self._cmb_view = QComboBox()
        self._cmb_view.addItems(["Side by Side", "HDR Only", "SDR Only"])
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

        row2.addWidget(self._btn_play)
        row2.addWidget(self._btn_pause)
        row2.addWidget(self._btn_stop)
        row2.addStretch()
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
            self._disp_hdr = MpvHDRWidget()
        else:
            self._disp_hdr = VideoDisplay("HDR Output (mpv unavailable)")
        self._split.addWidget(self._disp_sdr)
        self._split.addWidget(self._disp_hdr)
        self._split.setStretchFactor(0, 1)
        self._split.setStretchFactor(1, 1)
        root.addWidget(self._split, 1)

        # ---- Metrics panel ----
        self._grp_metrics = QGroupBox("Metrics")
        ml = QHBoxLayout(self._grp_metrics)
        ml.setContentsMargins(12, 4, 12, 4)

        self._m = {}
        mono = QFont("Consolas", 9)
        for key in ("fps", "latency", "frame", "gpu", "cpu", "model", "prec"):
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

        # ---- Fullscreen shortcut (F11) ----
        self._fs_shortcut = QShortcut(QKeySequence(Qt.Key.Key_F11), self)
        self._fs_shortcut.activated.connect(self._toggle_fullscreen)

    # ── Fullscreen ───────────────────────────────────────────

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode.  F11 or double-click the video area."""
        if self._is_fullscreen:
            self._exit_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self):
        self._is_fullscreen = True
        self.menuBar().hide()
        self.statusBar().hide()
        self._row1_widget.hide()
        self._row2_widget.hide()
        self._row3_widget.hide()
        self._grp_metrics.hide()
        self._grp_hdr.hide()
        self.centralWidget().layout().setContentsMargins(0, 0, 0, 0)
        self.showFullScreen()

    def _exit_fullscreen(self):
        self._is_fullscreen = False
        self.menuBar().show()
        self.statusBar().show()
        self._row1_widget.show()
        self._row2_widget.show()
        self._row3_widget.show()
        if self._chk_metrics.isChecked():
            self._grp_metrics.show()
        self._grp_hdr.show()
        self.centralWidget().layout().setContentsMargins(8, 8, 8, 4)
        self.showNormal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._is_fullscreen:
            self._exit_fullscreen()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self._toggle_fullscreen()

    # ── Signal wiring ────────────────────────────────────────

    def _connect_signals(self):
        self._btn_file.clicked.connect(self._open_file)
        self._btn_play.clicked.connect(self._play)
        self._btn_pause.clicked.connect(self._toggle_pause)
        self._btn_stop.clicked.connect(self._stop)
        self._chk_metrics.toggled.connect(
            lambda on: self._grp_metrics.setVisible(on))
        self._cmb_prec.currentTextChanged.connect(self._on_precision)
        self._cmb_view.currentTextChanged.connect(self._on_view)

        self._worker.frame_ready.connect(self._on_frame)
        self._worker.metrics_updated.connect(self._on_metrics)
        self._worker.status_message.connect(self._on_status_message)
        self._worker.playback_finished.connect(self._on_finished)
        self._worker.compile_ready.connect(self._on_compile_ready)
        self._worker.position_updated.connect(self._on_position)
        self._seek_slider.sliderMoved.connect(self._on_seek)

        # HDR info from mpv
        if isinstance(self._disp_hdr, MpvHDRWidget):
            self._disp_hdr.hdr_info_ready.connect(self._on_hdr_info)

    # ── Slots: file / tools ───────────────────────────────────

    def _on_compile_ready(self):
        """Called on main thread after Triton compile finishes.
        Safe to start mpv now — GPU is free from autotuning."""
        if self._compile_dlg is not None:
            self._compile_dlg.close()
            self._compile_dlg.deleteLater()
            self._compile_dlg = None

        pending = getattr(self, '_pending_mpv_start', None)
        if pending and isinstance(self._disp_hdr, MpvHDRWidget):
            pw, ph, fps = pending
            self._disp_hdr.start_playback(pw, ph, fps=fps)
            self._worker.set_mpv_widget(self._disp_hdr)
            self._pending_mpv_start = None

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

    def _set_video(self, path):
        # Stop current playback if running
        if self._playing:
            self._stop()

        # If a model was already compiled for a different resolution,
        # restart the entire process so torch.compile loads fresh from
        # the inductor disk cache (much faster than in-process re-trace).
        if self._last_res is not None:
            cap = cv2.VideoCapture(path)
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            pw = min(vw, MAX_W)
            ph = min(vh, MAX_H)
            if (pw, ph) != self._last_res:
                self._restart_with_video(path)
                return

        self._video_path = path
        self._lbl_file.setText(os.path.basename(path))
        self._btn_play.setEnabled(True)
        self.setWindowTitle(
            f"HDRTVNet++ — {os.path.basename(path)}")
        self.statusBar().showMessage(f"Selected: {path}")
        # Auto-compile (if needed) and start playback immediately
        QTimer.singleShot(100, self._play)

    def _restart_with_video(self, path):
        """Restart the GUI process with a new video.

        A fresh process avoids stale torch.compile/dynamo state that
        causes slow in-process re-tracing when the resolution changes.
        """
        self.statusBar().showMessage("Restarting for new resolution …")
        # Clean shutdown
        if self._playing:
            self._worker.stop()
            self._worker.wait(5000)
        if isinstance(self._disp_hdr, MpvHDRWidget):
            self._disp_hdr.stop_playback()

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
        rc = _sp.call(args)
        sys.exit(rc)

    # ── Slots: playback ──────────────────────────────────────

    def _play(self):
        if self._playing or not self._video_path:
            return

        # Determine processing resolution
        cap = cv2.VideoCapture(self._video_path)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vfps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if vw > MAX_W or vh > MAX_H:
            pw, ph = MAX_W, MAX_H
        else:
            pw, ph = vw, vh

        # Set up seek slider
        self._vid_fps = vfps if vfps > 0 else 30.0
        self._seek_slider.setRange(0, max(0, total_frames - 1))
        self._seek_slider.setValue(0)
        self._seek_slider.setEnabled(True)
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
        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_stop.setEnabled(True)
        self._btn_file.setEnabled(False)
        self._cmb_prec.setEnabled(True)

        # Start mpv HDR display AFTER compile finishes (via signal)
        # so that mpv's D3D11 GPU usage doesn't pollute Triton autotuning.
        self._pending_mpv_start = None
        if isinstance(self._disp_hdr, MpvHDRWidget):
            self._pending_mpv_start = (pw, ph, int(vfps) if vfps > 0 else 30)
        else:
            self._worker.set_mpv_widget(None)

        self._worker.configure(
            self._video_path, self._cmb_prec.currentText(),
            proc_w=pw, proc_h=ph)

        # Show loading dialog (in-process model load + cache warmup is fast
        # since subprocess already compiled the kernels)
        self._compile_dlg = _CompileDialog(self)
        self._compile_dlg.show()

        self._worker.start()

    def _toggle_pause(self):
        if not self._playing:
            return
        if self._worker.is_paused:
            self._worker.resume()
            self._btn_pause.setText("⏸  Pause")
        else:
            self._worker.pause()
            self._btn_pause.setText("▶  Resume")

    def _stop(self):
        self._worker.stop()
        self._worker.wait(10000)
        if isinstance(self._disp_hdr, MpvHDRWidget):
            self._disp_hdr.stop_playback()
        self._reset_controls()

    # ── Slots: settings ──────────────────────────────────────

    def _on_precision(self, key):
        if self._playing:
            self._worker.request_precision_change(key)

    def _on_view(self, mode):
        self._disp_sdr.setVisible(mode != "HDR Only")
        self._disp_hdr.setVisible(mode != "SDR Only")
        # Let the worker skip unnecessary copies / postprocess
        if self._playing:
            self._worker.set_sdr_visible(mode != "HDR Only")

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
        if not self._seek_slider.isSliderDown():
            self._seek_slider.setValue(current_frame)
        fps = getattr(self, '_vid_fps', 30.0)
        self._lbl_time.setText(self._fmt_time(current_frame / fps))

    def _on_seek(self, frame_number: int):
        """User is dragging or released the seek slider."""
        if self._playing:
            self._worker.request_seek(frame_number)
        # Update time label immediately during drag
        fps = getattr(self, '_vid_fps', 30.0)
        self._lbl_time.setText(self._fmt_time(frame_number / fps))

    def _on_hdr_info(self, info: dict):
        """Update the HDR Info panel from mpv metadata."""
        active = info.get("hdr_active", False)
        if active:
            self._hdr_labels["status"].setText("HDR: ✓ Active")
            self._hdr_labels["status"].setStyleSheet("color: #00e676;")
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
        if self._disp_sdr.isVisible():
            self._disp_sdr.update_frame(sdr)
        # HDR QLabel fallback (mpv gets fed directly from the worker)
        if isinstance(self._disp_hdr, VideoDisplay) and self._disp_hdr.isVisible():
            self._disp_hdr.update_frame(hdr)

    def _on_metrics(self, m):
        self._m["fps"].setText(f"FPS: {m['fps']:.1f}")
        self._m["latency"].setText(f"Latency: {m['latency_ms']:.1f} ms")
        self._m["frame"].setText(f"Frame: {m['frame']}")
        self._m["gpu"].setText(f"GPU: {m['gpu_mb']:.0f} MB")
        self._m["cpu"].setText(f"CPU: {m['cpu_mb']:.0f} MB")
        self._m["model"].setText(f"Model: {m['model_mb']:.2f} MB")
        self._m["prec"].setText(f"Prec: {m['precision']}")

    def _on_status_message(self, text: str):
        """Forward worker status to status bar *and* compile dialog."""
        self.statusBar().showMessage(text)
        if self._compile_dlg is not None:
            self._compile_dlg.set_status(text)

    def _on_finished(self):
        if isinstance(self._disp_hdr, MpvHDRWidget):
            self._disp_hdr.stop_playback()
        self._reset_controls()
        self.statusBar().showMessage("Playback finished.")
        self._disp_sdr.clear_display()
        if isinstance(self._disp_hdr, VideoDisplay):
            self._disp_hdr.clear_display()

    # ── UI helpers ───────────────────────────────────────────

    def _reset_controls(self):
        self._playing = False
        if self._compile_dlg is not None:
            self._compile_dlg.close()
            self._compile_dlg.deleteLater()
            self._compile_dlg = None
        self._btn_play.setEnabled(bool(self._video_path))
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_file.setEnabled(True)
        self._btn_pause.setText("⏸  Pause")
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
        if self._playing:
            self._worker.stop()
            self._worker.wait(10000)
        if isinstance(self._disp_hdr, MpvHDRWidget):
            self._disp_hdr.stop_playback()
        super().closeEvent(event)


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
                        help="Auto-open and play this video on launch")
    args, _unknown = parser.parse_known_args()

    os.chdir(_ROOT)
    app = QApplication(sys.argv)
    _apply_dark_theme(app)
    win = MainWindow(initial_video=args.video)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
