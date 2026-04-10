from __future__ import annotations

import os
import time
import tempfile
import threading
import queue as _queue
from collections import deque

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from PyQt6.QtCore import QThread, pyqtSignal

from video_source import VideoSource
from window_capture_source import WindowCaptureSource
from gui_config import (
    PRECISIONS,
    MAX_W,
    MAX_H,
)
from gui_mpv_widget import MpvHDRWidget
from gui_pipeline_worker_model import PipelineWorkerModelMixin
from gui_pipeline_worker_vram import PipelineWorkerVramMixin
from gui_pipeline_worker_feeders import PipelineWorkerFeedersMixin
from gui_pipeline_worker_compare import PipelineWorkerCompareMixin
from gui_pipeline_worker_objective import PipelineWorkerObjectiveMixin
from gui_pipeline_worker_runtime_metrics import PipelineWorkerRuntimeMetricsMixin
from gui_pipeline_worker_session import PipelineWorkerSessionMixin
from gui_pipeline_worker_frame_processing import PipelineWorkerFrameProcessingMixin
from gui_scaling import _limited_playback_fps
from timer import sleep_until

# Keep wall-clock playback close to target by dropping decode frames when
# inference falls behind. This favors cadence over full-frame coverage.
_REALTIME_CATCHUP_ENABLED = True
_REALTIME_SKIP_LAG_FRAMES = 1.1
_REALTIME_MAX_CATCHUP_SKIP = 6
_PLAYBACK_SOURCE_PREFETCH = 4
_PLAYHEAD_UPDATE_STRIDE_PLAYING = 10
_METRICS_EMIT_INTERVAL_S = 0.20
_LIVE_PRESENT_INTERVAL_MIN_S = 1.0 / 120.0
_LIVE_PRESENT_INTERVAL_MAX_S = 1.0 / 18.0
_LIVE_PRESENT_INTERVAL_SMOOTHING = 0.26
_LIVE_PROCESS_INTERVAL_MAX_S = 0.250
_LIVE_PRESENT_PROCESS_SMOOTHING = 0.18
_LIVE_PRESENT_PROCESS_HEADROOM = 1.16
_LIVE_PRESENT_TARGET_FPS = (
    120.0,
    100.0,
    90.0,
    72.0,
    60.0,
    50.0,
    48.0,
    45.0,
    40.0,
    36.0,
    30.0,
    25.0,
    24.0,
    20.0,
    18.0,
    15.0,
    12.0,
    10.0,
    8.0,
    6.0,
)
_LIVE_PRESENT_SOURCE_TOLERANCE_FPS = 0.75
_LIVE_PRESENT_UPSHIFT_HEADROOM = 1.10
_LIVE_PRESENT_DOWNSHIFT_HEADROOM = 1.00
_LIVE_PRESENT_BUFFER_FRAMES = 2.10
_LIVE_PRESENT_MIN_BUFFER_S = 0.020
_LIVE_PRESENT_MAX_LEAD_FRAMES = 3.50
_LIVE_PRESENT_JITTER_SMOOTHING = 0.30
_LIVE_PRESENT_JITTER_BUFFER_GAIN = 2.20
_LIVE_PRESENT_LATE_BUDGET_MIN_S = 0.100
_LIVE_PRESENT_LATE_BUDGET_MAX_S = 0.300
_LIVE_PRESENT_LATE_BUDGET_FRAMES = 3.80
_LIVE_PRESENT_LATE_DROP_STREAK_MAX = 1
_HERE = os.path.dirname(os.path.abspath(__file__))
_HG_WEIGHTS_PATH = os.path.join(_HERE, "models", "weights", "HG_weights.pth")


def _normalize_runtime_execution_mode(mode: str | None) -> str:
    text = str(mode or "").strip().lower()
    if text == "eager":
        return "eager"
    return "compile"


def _choose_live_present_target_fps(
    source_fps: float,
    sustainable_fps: float,
    current_target_fps: float,
) -> float:
    source_fps = max(1.0, float(source_fps))
    sustainable_fps = max(1.0, float(sustainable_fps))
    current_target_fps = max(1.0, float(current_target_fps))

    source_limit_fps = source_fps + _LIVE_PRESENT_SOURCE_TOLERANCE_FPS
    candidates = [
        fps for fps in _LIVE_PRESENT_TARGET_FPS
        if fps <= source_limit_fps
    ]
    if not candidates:
        return max(1.0, min(source_fps, sustainable_fps))

    desired_target_fps = next(
        (fps for fps in candidates if fps <= sustainable_fps),
        max(1.0, min(source_fps, sustainable_fps)),
    )

    if current_target_fps not in candidates:
        return float(desired_target_fps)

    if desired_target_fps < current_target_fps:
        if sustainable_fps <= (current_target_fps * _LIVE_PRESENT_DOWNSHIFT_HEADROOM):
            return float(desired_target_fps)
        return float(current_target_fps)

    if desired_target_fps > current_target_fps:
        if sustainable_fps >= (desired_target_fps * _LIVE_PRESENT_UPSHIFT_HEADROOM):
            return float(desired_target_fps)
        return float(current_target_fps)

    return float(current_target_fps)

class PipelineWorker(
    PipelineWorkerModelMixin,
    PipelineWorkerVramMixin,
    PipelineWorkerFeedersMixin,
    PipelineWorkerCompareMixin,
    PipelineWorkerObjectiveMixin,
    PipelineWorkerRuntimeMetricsMixin,
    PipelineWorkerSessionMixin,
    PipelineWorkerFrameProcessingMixin,
    QThread,
):
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
    compare_snapshot_ready = pyqtSignal(dict)
    status_message = pyqtSignal(str)
    playback_finished = pyqtSignal()
    compile_ready = pyqtSignal()          # emitted after warmup_compile finishes
    position_updated = pyqtSignal(int, int)  # (current_frame, total_frames)
    seek_frame_ready = pyqtSignal(int)    # emitted after first rendered frame post-seek

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path = None
        self._capture_target: dict | None = None
        self._precision_key = "FP16"
        self._use_hg = True
        self._processor = None
        self._stop_flag = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused
        self._pending_precision = None
        self._pending_predequantize_mode: str | None = None
        self._pending_runtime_execution_mode: str | None = None
        self._pending_capture_fps: float | None = None
        self._paused_control_wake = False
        self._paused_control_refresh_frame: int | None = None
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
        self._flat_temporal_prev_rgb: torch.Tensor | None = None
        self._flat_temporal_prev_luma: torch.Tensor | None = None
        self._flat_temporal_prev_scene: np.ndarray | None = None
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
        self._predequantize_mode: str = "auto"
        self._runtime_execution_mode: str = "compile"
        self._objective_metrics_enabled: bool = False
        self._hdr_ground_truth_path: str | None = None
        self._pending_metrics_cfg: tuple[bool, str | None] | None = None
        self._pending_compare_snapshot: dict | None = None
        self._compare_cached_state: dict | None = None
        self._live_latency_smoothed_ms: float = 0.0
        self._metrics_emit_interval_s: float = _METRICS_EMIT_INTERVAL_S
        self._last_metrics_emit_t: float = 0.0

    # ── public API (called from main thread) ──

    def configure(self, video_path, precision_key, proc_w=MAX_W, proc_h=MAX_H,
                  output_w=MAX_W, output_h=MAX_H, input_is_hdr=False,
                  use_hg=True,
                  predequantize_mode: str = "auto",
                  runtime_execution_mode: str = "compile",
                  objective_metrics_enabled=False,
                  hdr_ground_truth_path: str | None = None,
                  capture_target: dict | None = None):
        self._video_path = video_path
        self._capture_target = dict(capture_target) if isinstance(capture_target, dict) else None
        self._precision_key = precision_key
        self._proc_w = proc_w
        self._proc_h = proc_h
        self._output_w = output_w
        self._output_h = output_h
        self._reset_enhance_history()
        self._input_is_hdr = bool(input_is_hdr)
        self._use_hg = bool(use_hg)
        mode = str(predequantize_mode).strip().lower()
        if mode not in {"auto", "on", "off"}:
            mode = "auto"
        self._predequantize_mode = mode
        self._pending_predequantize_mode = None
        self._runtime_execution_mode = _normalize_runtime_execution_mode(
            runtime_execution_mode
        )
        self._pending_runtime_execution_mode = None
        self._pending_capture_fps = None
        self._objective_metrics_enabled = bool(objective_metrics_enabled)
        self._hdr_ground_truth_path = (
            str(hdr_ground_truth_path).strip() if hdr_ground_truth_path else None
        )
        if self._hdr_ground_truth_path and not os.path.isfile(self._hdr_ground_truth_path):
            self._hdr_ground_truth_path = None
        self._pending_metrics_cfg = None
        self._pending_compare_snapshot = None
        self._compare_cached_state = None
        self._live_latency_smoothed_ms = 0.0
        self._realtime_drop_frames = 0
        self._last_metrics_emit_t = 0.0
        is_live_capture = bool(self._capture_target)
        # File playback benefits from a short startup settle window.
        # Live browser-window capture should render immediately.
        if is_live_capture:
            self._hdr_drop_until_frame = 0
            self._sdr_drop_until_frame = 0
            self._hold_until_t = 0.0
        else:
            self._hdr_drop_until_frame = 2
            self._sdr_drop_until_frame = 2
            self._hold_until_t = time.perf_counter() + 0.5

    def request_objective_metrics_config(self, enabled: bool, hdr_ground_truth_path: str | None):
        path = str(hdr_ground_truth_path).strip() if hdr_ground_truth_path else None
        if path and not os.path.isfile(path):
            path = None
        self._pending_metrics_cfg = (bool(enabled), path)

    def request_compare_snapshot(
        self,
        frame_number: int | None = None,
        hdr_ground_truth_path: str | None = None,
        precision_key: str | None = None,
        force_immediate: bool = False,
    ):
        if frame_number is None:
            cached = self._compare_cached_state
            if isinstance(cached, dict):
                try:
                    frame_number = int(cached.get("frame_idx", self._frame_idx))
                except Exception:
                    frame_number = self._frame_idx
            else:
                frame_number = self._frame_idx
        gt_path = str(hdr_ground_truth_path).strip() if hdr_ground_truth_path else None
        if gt_path and not os.path.isfile(gt_path):
            gt_path = None
        req_precision = str(precision_key).strip() if precision_key else None
        if req_precision and req_precision not in PRECISIONS:
            req_precision = None
        target_frame = max(0, int(frame_number))
        self._pending_compare_snapshot = {
            "frame": target_frame,
            "hdr_gt_path": gt_path,
            "precision_key": req_precision,
            "force_immediate": bool(force_immediate),
        }
        cached = self._compare_cached_state
        use_cached_frame = (
            self._user_paused
            and self._seek_frame is None
            and isinstance(cached, dict)
            and int(cached.get("frame_idx", -1)) == target_frame
        )
        if use_cached_frame:
            self._paused_control_wake = True
            self._paused_control_refresh_frame = None
        elif self._user_paused and bool(force_immediate):
            self._paused_control_wake = True
            self._paused_control_refresh_frame = None
        else:
            self._seek_frame = target_frame
        # Unblock the worker so compare can be emitted immediately when paused,
        # or after a single explicit seek when an anchored frame is requested.
        self._pause_event.set()

    def request_precision_change(self, key):
        self._pending_precision = key
        self._wake_for_paused_control_change()

    def request_predequantize_mode(self, mode: str):
        m = str(mode).strip().lower()
        if m not in {"auto", "on", "off"}:
            m = "auto"
        self._pending_predequantize_mode = m
        self._wake_for_paused_control_change()

    def request_runtime_execution_mode(self, mode: str):
        self._pending_runtime_execution_mode = _normalize_runtime_execution_mode(mode)
        self._wake_for_paused_control_change()

    def request_resolution_change(self, proc_w: int, proc_h: int):
        """Request a processing-resolution change (thread-safe)."""
        self._pending_resolution = (proc_w, proc_h)
        self._wake_for_paused_control_change()

    def request_capture_fps_change(self, fps: float):
        """Request a live browser-window capture FPS cap change."""
        next_fps = max(1.0, float(fps))
        self._pending_capture_fps = next_fps
        if isinstance(self._capture_target, dict):
            self._capture_target["fps"] = next_fps
        source = getattr(self, "_source", None)
        if isinstance(source, WindowCaptureSource):
            try:
                source.set_fps(next_fps)
            except Exception:
                pass

    def _wake_for_paused_control_change(self):
        """Let paused playback process hot-swap work without fully resuming."""
        if self._user_paused:
            self._paused_control_wake = True
            if self._seek_frame is None:
                self._paused_control_refresh_frame = max(0, int(self._frame_idx))
            self._pause_event.set()

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
        if self._capture_target:
            return
        self._hdr_drop_until_frame = max(self._hdr_drop_until_frame,
                                         self._frame_idx + max(0, int(drop_frames)))
        self._sdr_drop_until_frame = max(self._sdr_drop_until_frame,
                                         self._frame_idx + max(0, int(drop_frames)))

    def _reset_enhance_history(self):
        self._enh_prev_luma = None
        self._enh_temporal_detail = None
        self._flat_temporal_prev_rgb = None
        self._flat_temporal_prev_luma = None
        self._flat_temporal_prev_scene = None

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
        accum = detail

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
        self._paused_control_wake = False
        self._paused_control_refresh_frame = None
        self._pause_event.clear()

    def resume(self):
        self._user_paused = False
        self._paused_control_wake = False
        self._paused_control_refresh_frame = None
        self._pause_event.set()

    def stop(self):
        self._stop_flag = True
        self._user_paused = False
        self._paused_control_wake = False
        self._paused_control_refresh_frame = None
        self._pause_event.set()  # unblock if paused

    @property
    def is_paused(self):
        return self._user_paused

    # Model/VRAM/feeder helpers moved to:
    # - gui_pipeline_worker_model.py
    # - gui_pipeline_worker_vram.py
    # - gui_pipeline_worker_feeders.py

    # Main loop

    def run(self):
        self._stop_flag = False
        self._pending_precision = None
        self._pending_predequantize_mode = None
        self._pending_runtime_execution_mode = None

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

        if not self._video_path and not self._capture_target:
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

        # A small decode-ahead buffer smooths content-dependent decode spikes
        # without making seeks feel sluggish.
        if self._capture_target:
            source = WindowCaptureSource(
                float(self._capture_target.get("fps", 24.0) or 24.0),
                title=str(self._capture_target.get("title", "") or ""),
                hwnd=int(self._capture_target.get("hwnd", 0) or 0),
                pid=int(self._capture_target.get("pid", 0) or 0),
                prefetch=1,
                capture_max_w=int(
                    self._capture_target.get("capture_w", self._proc_w) or self._proc_w
                ),
                capture_max_h=int(
                    self._capture_target.get("capture_h", self._proc_h) or self._proc_h
                ),
            )
        else:
            source = VideoSource(self._video_path, prefetch=_PLAYBACK_SOURCE_PREFETCH)
        self._source = source
        is_live_capture = bool(self._capture_target)
        gt_source: VideoSource | None = None
        total_frames = source.frame_count
        src_fps = source.fps if source.fps and source.fps > 0 else 30.0
        out_fps = src_fps if is_live_capture else _limited_playback_fps(src_fps)
        frame_stride = 1 if is_live_capture else max(1, int(round(src_fps / out_fps)))
        frame_interval_s = 1.0 / src_fps
        next_frame_t = time.perf_counter()
        frame_times = deque(maxlen=30)
        model_times = deque(maxlen=30)
        presented_times = deque(maxlen=30)
        metrics_warmup_frames = 0
        frame_idx = 0
        live_video_latency_ms = 0.0
        force_position_emit = False
        seek_frame_ready_pending = False
        process = psutil.Process(os.getpid())
        use_cuda = torch.cuda.is_available()
        objective_note = "Off"
        objective_warned_gt_eof = False
        live_present_t = 0.0
        live_capture_prev_perf = 0.0
        live_source_interval_s = 1.0 / 48.0
        live_process_interval_s = 1.0 / 48.0
        live_present_target_fps = 48.0
        live_present_interval_s = 1.0 / live_present_target_fps
        live_capture_jitter_s = 0.0
        live_late_drop_streak = 0
        (
            psnr_avg,
            sssim_avg,
            deitp_avg,
            hdr_vdp3_avg,
            hdr_vdp3_note,
        ) = self._reset_objective_averages()

        # Local copies of resolution settings (may change mid-playback)
        proc_w, proc_h = self._proc_w, self._proc_h
        out_w, out_h = self._output_w, self._output_h
        lower_res_processing = (proc_w != out_w or proc_h != out_h)
        gt_source, objective_note, objective_warned_gt_eof = self._open_gt_source(
            source=source,
            gt_source=gt_source,
            cur_frame_idx=0,
        )

        while not self._stop_flag:
            if self._sdr_mpv_widget is not None and self._sdr_queue is None:
                self._start_sdr_feeder()

            pending_metrics_cfg = self._pending_metrics_cfg
            if pending_metrics_cfg is not None:
                self._pending_metrics_cfg = None
                self._objective_metrics_enabled = bool(pending_metrics_cfg[0])
                self._hdr_ground_truth_path = pending_metrics_cfg[1]
                (
                    psnr_avg,
                    sssim_avg,
                    deitp_avg,
                    hdr_vdp3_avg,
                    hdr_vdp3_note,
                ) = self._reset_objective_averages()
                gt_source, objective_note, objective_warned_gt_eof = self._open_gt_source(
                    source=source,
                    gt_source=gt_source,
                    cur_frame_idx=frame_idx,
                )

            pending_capture_fps = self._pending_capture_fps
            if is_live_capture and pending_capture_fps is not None:
                self._pending_capture_fps = None
                try:
                    source.set_fps(pending_capture_fps)
                    self.status_message.emit(
                        f"Live browser capture FPS -> {int(round(float(pending_capture_fps)))}"
                    )
                except Exception:
                    pass

            pending_predeq = self._pending_predequantize_mode
            if pending_predeq is not None and pending_predeq == self._predequantize_mode:
                self._pending_predequantize_mode = None
            if (
                (not self._input_is_hdr)
                and pending_predeq is not None
                and pending_predeq != self._predequantize_mode
            ):
                self._pending_predequantize_mode = None
                self._predequantize_mode = pending_predeq
                self.status_message.emit(
                    f"Pre-dequantization mode -> {self._predequantize_mode}. Reloading model ..."
                )
                if not self._load_model(self._precision_key):
                    continue
                self._reset_enhance_history()

            pending_exec_mode = self._pending_runtime_execution_mode
            if pending_exec_mode is not None and pending_exec_mode == self._runtime_execution_mode:
                self._pending_runtime_execution_mode = None
            if (
                (not self._input_is_hdr)
                and pending_exec_mode is not None
                and pending_exec_mode != self._runtime_execution_mode
            ):
                self._pending_runtime_execution_mode = None
                self._runtime_execution_mode = pending_exec_mode
                self.status_message.emit(
                    f"Runtime execution mode -> {self._runtime_execution_mode}. Reloading model ..."
                )
                if not self._load_model(self._precision_key):
                    continue
                self._reset_enhance_history()

            # Hot-swap precision
            pending = self._pending_precision
            if (not self._input_is_hdr) and pending and pending != self._precision_key:
                self._pending_precision = None
                if not self._load_model(pending):
                    continue
                self._reset_enhance_history()

            # Hot-swap processing resolution
            pending_res = self._pending_resolution
            if (not self._input_is_hdr) and pending_res is not None:
                self._pending_resolution = None
                new_pw, new_ph = pending_res
                if (new_pw, new_ph) != (proc_w, proc_h):
                    self.status_message.emit(
                        f"Switching to {new_pw}x{new_ph} ..."
                    )
                    self._proc_w, self._proc_h = new_pw, new_ph
                    proc_w, proc_h = new_pw, new_ph
                    lower_res_processing = (proc_w != out_w or proc_h != out_h)
                    self._reset_enhance_history()
                    self._silent_warmup(self._processor, proc_w, proc_h)
                    self.status_message.emit(
                        f"Ready - {self._precision_key} @ {proc_w}x{proc_h}"
                    )

                self._reset_enhance_history()

            if (
                self._user_paused
                and self._paused_control_wake
                and self._seek_frame is None
                and self._pending_compare_snapshot is not None
            ):
                if self._try_emit_compare_snapshot_from_cache():
                    self._paused_control_wake = False
                    self._paused_control_refresh_frame = None
                    next_frame_t = time.perf_counter()
                    continue

            if self._user_paused and self._paused_control_wake and self._seek_frame is None:
                refresh_frame = self._paused_control_refresh_frame
                if refresh_frame is not None:
                    # Render the current paused frame once using the updated
                    # model/settings so the user sees the change immediately.
                    self._paused_control_refresh_frame = None
                    self._seek_frame = max(0, int(refresh_frame))
                else:
                    # Paused control-only updates finished with no frame redraw
                    # needed; return to the paused wait state cleanly.
                    self._pause_event.clear()
                    self._paused_control_wake = False
                    next_frame_t = time.perf_counter()
                    continue

            # Seek gate
            seek_to = self._seek_frame
            if seek_to is not None:
                self._seek_frame = None
                source.seek(seek_to)
                if gt_source is not None:
                    try:
                        gt_source.seek(seek_to)
                    except Exception:
                        gt_source = self._close_gt_source(gt_source)
                        objective_note = "HDR ground-truth seek failed"
                frame_idx = max(0, seek_to - 1)
                force_position_emit = True
                seek_frame_ready_pending = True
                next_frame_t = time.perf_counter()
                self._sdr_delay_frame = None
                self._reset_enhance_history()
                # Reset post-flush frame-drop watermarks to the new timeline
                # position.  If we keep an old watermark from a later frame,
                # backward seeks can render correctly but never feed a new HDR
                # frame to mpv because every post-seek frame still looks "too
                # early" and gets dropped.
                self._hdr_drop_until_frame = 0
                self._sdr_drop_until_frame = 0
                if (not self._capture_target) and seek_to <= 1:
                    self._hdr_drop_until_frame = 2
                    self._sdr_drop_until_frame = 2
                    self._hold_until_t = time.perf_counter() + 0.5
                # Discard stale FPS history so metrics/auto-mute re-lock quickly.
                frame_times.clear()
                presented_times.clear()
                metrics_warmup_frames = 4
                live_present_t = 0.0
                live_capture_prev_perf = 0.0
                live_source_interval_s = 1.0 / 48.0
                live_process_interval_s = 1.0 / 48.0
                live_present_target_fps = 48.0
                live_present_interval_s = 1.0 / live_present_target_fps
                live_capture_jitter_s = 0.0
                live_late_drop_streak = 0

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
                live_present_t = 0.0
                live_capture_prev_perf = 0.0
                live_source_interval_s = 1.0 / 48.0
                live_process_interval_s = 1.0 / 48.0
                live_present_target_fps = 48.0
                live_present_interval_s = 1.0 / live_present_target_fps
                live_capture_jitter_s = 0.0
                live_late_drop_streak = 0
            if (
                self._user_paused
                and self._paused_control_wake
                and self._seek_frame is None
                and not seek_frame_ready_pending
            ):
                # A paused hot-swap request woke the loop. Bounce back to the top
                # so pending precision/resolution/pre-dequantize work is handled
                # before we decode another frame.
                continue

            # File playback uses a scheduled presentation clock. Live window
            # capture should process as soon as a fresh frame is available.
            now = time.perf_counter()
            lag_s = 0.0
            if not is_live_capture:
                if now < next_frame_t:
                    sleep_until(next_frame_t)
                    now = time.perf_counter()
                else:
                    lag_s = now - next_frame_t
                if self._hold_until_t and now < self._hold_until_t:
                    hold_deadline = min(
                        self._hold_until_t,
                        time.perf_counter() + frame_interval_s,
                    )
                    sleep_until(hold_deadline)

            (
                ret,
                frame,
                gt_frame,
                gt_source,
                objective_note,
                objective_warned_gt_eof,
            ) = self._read_frame_pair(
                source=source,
                gt_source=gt_source,
                objective_note=objective_note,
                objective_warned_gt_eof=objective_warned_gt_eof,
            )
            if not ret:
                break

            frame_idx += 1
            self._frame_idx = frame_idx
            capture_perf_counter = (
                float(getattr(source, "last_capture_perf_counter", 0.0) or 0.0)
                if is_live_capture
                else 0.0
            )

            # Real-time catch-up: if inference is behind wall clock, drop a few
            # decode frames and process the newest one to preserve cadence.
            if (
                (not is_live_capture)
                and _REALTIME_CATCHUP_ENABLED
                and (not seek_frame_ready_pending)
                and lag_s > (frame_interval_s * _REALTIME_SKIP_LAG_FRAMES)
            ):
                skip_n = min(
                    _REALTIME_MAX_CATCHUP_SKIP,
                    max(0, int(lag_s / frame_interval_s)),
                )
                while skip_n > 0:
                    (
                        ret_skip,
                        frame_skip,
                        gt_frame_skip,
                        gt_source,
                        objective_note,
                        objective_warned_gt_eof,
                    ) = self._read_frame_pair(
                        source=source,
                        gt_source=gt_source,
                        objective_note=objective_note,
                        objective_warned_gt_eof=objective_warned_gt_eof,
                    )
                    if not ret_skip:
                        ret = False
                        break
                    frame = frame_skip
                    gt_frame = gt_frame_skip
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

            if is_live_capture:
                present_t = None
                if capture_perf_counter > 0.0:
                    raw_interval_s = 0.0
                    if live_capture_prev_perf > 0.0:
                        raw_interval_s = max(
                            0.0,
                            float(capture_perf_counter) - float(live_capture_prev_perf),
                        )
                    live_capture_prev_perf = float(capture_perf_counter)
                    if raw_interval_s > 0.0:
                        clamped_interval_s = min(
                            _LIVE_PRESENT_INTERVAL_MAX_S,
                            max(_LIVE_PRESENT_INTERVAL_MIN_S, raw_interval_s),
                        )
                        jitter_sample_s = abs(clamped_interval_s - live_source_interval_s)
                        live_capture_jitter_s = (
                            (1.0 - _LIVE_PRESENT_JITTER_SMOOTHING)
                            * live_capture_jitter_s
                            + (_LIVE_PRESENT_JITTER_SMOOTHING * jitter_sample_s)
                        )
                        live_source_interval_s = (
                            (1.0 - _LIVE_PRESENT_INTERVAL_SMOOTHING)
                            * live_source_interval_s
                            + (_LIVE_PRESENT_INTERVAL_SMOOTHING * clamped_interval_s)
                        )
                    required_interval_s = max(
                        live_source_interval_s,
                        live_process_interval_s * _LIVE_PRESENT_PROCESS_HEADROOM,
                    )
                    source_fps = 1.0 / max(live_source_interval_s, 1e-6)
                    sustainable_fps = 1.0 / max(required_interval_s, 1e-6)
                    live_present_target_fps = _choose_live_present_target_fps(
                        source_fps,
                        sustainable_fps,
                        live_present_target_fps,
                    )
                    live_present_interval_s = 1.0 / max(
                        live_present_target_fps,
                        1e-6,
                    )
                    buffer_s = max(
                        _LIVE_PRESENT_MIN_BUFFER_S,
                        (live_present_interval_s * _LIVE_PRESENT_BUFFER_FRAMES)
                        + (live_capture_jitter_s * _LIVE_PRESENT_JITTER_BUFFER_GAIN),
                    )
                    now_t = time.perf_counter()
                    max_lead_s = max(
                        buffer_s,
                        live_present_interval_s * _LIVE_PRESENT_MAX_LEAD_FRAMES,
                    )
                    if live_present_t <= 0.0:
                        live_present_t = now_t + buffer_s
                    else:
                        live_present_t += live_present_interval_s
                        min_present_t = now_t + _LIVE_PRESENT_MIN_BUFFER_S
                        max_present_t = now_t + max_lead_s
                        if live_present_t < min_present_t:
                            live_present_t = min_present_t
                        elif live_present_t > max_present_t:
                            live_present_t = max_present_t
                    present_t = live_present_t
            else:
                present_t = max(next_frame_t, time.perf_counter())

            if is_live_capture and capture_perf_counter > 0.0 and present_t is not None:
                predicted_latency_s = max(0.0, present_t - capture_perf_counter)
                late_budget_s = min(
                    _LIVE_PRESENT_LATE_BUDGET_MAX_S,
                    max(
                        _LIVE_PRESENT_LATE_BUDGET_MIN_S,
                        (live_present_interval_s * _LIVE_PRESENT_LATE_BUDGET_FRAMES)
                        + (live_capture_jitter_s * 1.2),
                    ),
                )
                if (
                    (not seek_frame_ready_pending)
                    and predicted_latency_s > late_budget_s
                    and live_late_drop_streak < _LIVE_PRESENT_LATE_DROP_STREAK_MAX
                ):
                    live_late_drop_streak += 1
                    self._realtime_drop_frames += 1
                    continue
                live_late_drop_streak = 0

            t0 = time.perf_counter()

            display_frame, output, prepared_out, need_hdr_cpu, model_latency_ms = self._process_frame(
                frame=frame,
                frame_idx=frame_idx,
                present_t=present_t,
                out_w=out_w,
                out_h=out_h,
                proc_w=proc_w,
                proc_h=proc_h,
                lower_res_processing=lower_res_processing,
                mpv_w=mpv_w,
                use_cuda=use_cuda,
            )

            should_cache_compare_state = bool(
                self._user_paused or self._pending_compare_snapshot is not None
            )
            if should_cache_compare_state:
                self._cache_compare_state(
                    frame_idx=frame_idx,
                    frame=frame,
                    gt_frame=gt_frame,
                    display_frame=display_frame,
                    output=output,
                    prepared_out=prepared_out,
                    need_hdr_cpu=need_hdr_cpu,
                    out_w=out_w,
                    out_h=out_h,
                    lower_res_processing=lower_res_processing,
                    proc_w=proc_w,
                    proc_h=proc_h,
                )
            elif self._compare_cached_state is not None:
                self._compare_cached_state = None

            # Keep compare work off the normal playback hot path entirely
            # unless the user has actually requested a compare snapshot.
            if self._pending_compare_snapshot is not None:
                self._maybe_emit_compare_snapshot(
                    frame_idx=frame_idx,
                    frame=frame,
                    gt_frame=gt_frame,
                    display_frame=display_frame,
                    output=output,
                    prepared_out=prepared_out,
                    need_hdr_cpu=need_hdr_cpu,
                    out_w=out_w,
                    out_h=out_h,
                    lower_res_processing=lower_res_processing,
                    proc_w=proc_w,
                    proc_h=proc_h,
                )

            objective_note, hdr_vdp3_note = self._update_objective_metrics(
                frame_idx=frame_idx,
                gt_source=gt_source,
                gt_frame=gt_frame,
                need_hdr_cpu=need_hdr_cpu,
                output=output,
                prepared_out=prepared_out,
                out_w=out_w,
                out_h=out_h,
                psnr_avg=psnr_avg,
                sssim_avg=sssim_avg,
                deitp_avg=deitp_avg,
                hdr_vdp3_avg=hdr_vdp3_avg,
                objective_note=objective_note,
                hdr_vdp3_note=hdr_vdp3_note,
            )

            t1 = time.perf_counter()
            if is_live_capture:
                next_frame_t = time.perf_counter()
            else:
                next_frame_t += frame_interval_s
            frame_ms = (t1 - t0) * 1000.0
            if is_live_capture and frame_ms > 0.0:
                process_interval_s = min(
                    _LIVE_PROCESS_INTERVAL_MAX_S,
                    max(_LIVE_PRESENT_INTERVAL_MIN_S, frame_ms / 1000.0),
                )
                live_process_interval_s = (
                    (1.0 - _LIVE_PRESENT_PROCESS_SMOOTHING)
                    * live_process_interval_s
                    + (_LIVE_PRESENT_PROCESS_SMOOTHING * process_interval_s)
                )
            frame_times.append(frame_ms)
            if model_latency_ms > 0.0:
                model_times.append(float(model_latency_ms))
            presentation_stamp = (
                max(t1, present_t)
                if (mpv_w is not None and present_t is not None)
                else t1
            )
            if capture_perf_counter > 0.0:
                raw_live_latency_ms = max(
                    0.0, (presentation_stamp - capture_perf_counter) * 1000.0
                )
                prev_live_latency_ms = float(
                    getattr(self, "_live_latency_smoothed_ms", 0.0) or 0.0
                )
                if prev_live_latency_ms <= 0.0:
                    live_video_latency_ms = raw_live_latency_ms
                else:
                    live_video_latency_ms = (
                        (prev_live_latency_ms * 0.82)
                        + (raw_live_latency_ms * 0.18)
                    )
                self._live_latency_smoothed_ms = float(live_video_latency_ms)
            presented_times.append(presentation_stamp)
            if metrics_warmup_frames > 0:
                metrics_warmup_frames -= 1

            # Keep seek/pause/compare updates immediate, but throttle routine
            # playhead updates so the UI thread is not nudged every frame.
            should_emit_position = (
                force_position_emit
                or seek_frame_ready_pending
                or self._user_paused
                or (frame_idx <= 1)
                or (
                    _PLAYHEAD_UPDATE_STRIDE_PLAYING > 0
                    and (frame_idx % _PLAYHEAD_UPDATE_STRIDE_PLAYING) == 0
                )
            )
            if should_emit_position:
                self.position_updated.emit(frame_idx, total_frames)
            force_position_emit = False
            if seek_frame_ready_pending:
                self.seek_frame_ready.emit(frame_idx)
                seek_frame_ready_pending = False

            # Emit only what the UI actually needs
            emit_sdr_preview = self._sdr_visible and self._sdr_mpv_widget is None
            if need_hdr_cpu:
                # QLabel fallback — both SDR + HDR frames needed
                self.frame_ready.emit(display_frame, output)
            elif emit_sdr_preview:
                # mpv handles HDR; only update the SDR QLabel when that CPU
                # preview path is actually active.
                self.frame_ready.emit(display_frame, display_frame)
            # else: both outputs are handled directly by mpv panes

            if self._hold_until_t and time.perf_counter() >= self._hold_until_t:
                self._hold_until_t = 0.0

            paused_control_refresh_done = bool(
                self._user_paused
                and self._paused_control_wake
                and self._seek_frame is None
            )

            # Re-pause: if user paused and no further seek pending,
            # block again now that this seek frame has been emitted.
            if self._user_paused and self._seek_frame is None:
                self._pause_event.clear()

            self._emit_runtime_metrics_if_ready(
                frame_idx=frame_idx,
                frame_times=frame_times,
                model_times=model_times,
                presented_times=presented_times,
                metrics_warmup_frames=metrics_warmup_frames,
                force=paused_control_refresh_done,
                process=process,
                use_cuda=use_cuda,
                proc_w=proc_w,
                proc_h=proc_h,
                objective_note=objective_note,
                hdr_vdp3_note=hdr_vdp3_note,
                psnr_avg=psnr_avg,
                sssim_avg=sssim_avg,
                deitp_avg=deitp_avg,
                hdr_vdp3_avg=hdr_vdp3_avg,
                hg_weights_path=_HG_WEIGHTS_PATH,
                live_video_latency_ms=live_video_latency_ms,
                is_live_capture=is_live_capture,
            )
            if paused_control_refresh_done:
                self._paused_control_wake = False
                self._paused_control_refresh_frame = None

        source.release()
        self._source = None
        gt_source = self._close_gt_source(gt_source)
        self._stop_hdr_feeder()
        self._stop_sdr_feeder()
        self._stop_app_vram_poll()
        self.playback_finished.emit()
        self.status_message.emit("Playback finished.")

