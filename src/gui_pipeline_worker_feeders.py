from __future__ import annotations

import os
import queue as _queue
import threading
import time

import numpy as np
import torch

from gui_config import LIVE_CAPTURE_DISPLAY_FPS
from gui_mpv_widget import MpvHDRWidget
from timer import sleep_until

_LIVE_SMOOTH_MAX_QUEUE_WAIT_S = 0.042  # ~1/24 second for better 24fps cadence
_LIVE_SMOOTH_MAX_CATCHUP_FRAMES = 3  # Allow more catchup for smoother playback
_FEEDER_GPU_RGB48 = str(
    os.environ.get("HDRTVNET_FEEDER_GPU_RGB48", "1")
).strip().lower() in {"1", "true", "yes", "on"}


def _live_present_interval_s() -> float:
    return 1.0 / max(1.0, float(LIVE_CAPTURE_DISPLAY_FPS or 30.0))


def _next_live_present_deadline(
    deadline_s: float,
    now_s: float,
    interval_s: float,
) -> float:
    """Advance cadence while allowing a tiny bounded refill after stalls."""
    next_s = float(deadline_s) + float(interval_s)
    floor_s = float(now_s) - (
        float(interval_s) * float(_LIVE_SMOOTH_MAX_CATCHUP_FRAMES)
    )
    if next_s < floor_s:
        return floor_s
    return next_s


def _pinned_u16_host_buffer(state: dict, shape: tuple[int, int, int]):
    if state.get("shape") == shape and state.get("tensor") is not None:
        return state["tensor"], state["numpy"]
    tensor = None
    try:
        if torch.cuda.is_available():
            tensor = torch.empty(shape, dtype=torch.uint16, pin_memory=True)
    except Exception:
        tensor = None
    if tensor is None:
        tensor = torch.empty(shape, dtype=torch.uint16)
    arr = tensor.numpy()
    state["shape"] = shape
    state["tensor"] = tensor
    state["numpy"] = arr
    return tensor, arr


def _tensor_to_rgb48_bytes(tensor, host_state: dict) -> bytes:
    with torch.inference_mode():
        prepared = tensor[0] if isinstance(tensor, (tuple, list)) else tensor
        rgb = (
            prepared.squeeze(0)
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .contiguous()
        )

    if (
        _FEEDER_GPU_RGB48
        and getattr(rgb, "device", None) is not None
        and rgb.device.type == "cuda"
    ):
        rgb_u16 = (
            rgb.to(dtype=torch.float32)
            .mul_(65535.0)
            .add_(0.5)
            .to(dtype=torch.uint16)
            .contiguous()
        )
        host_tensor, host_np = _pinned_u16_host_buffer(
            host_state,
            tuple(int(v) for v in rgb_u16.shape),
        )
        non_blocking = False
        try:
            non_blocking = bool(host_tensor.is_pinned())
        except Exception:
            non_blocking = False
        host_tensor.copy_(rgb_u16, non_blocking=non_blocking)
        torch.cuda.current_stream().synchronize()
        return host_np.tobytes()

    rgb_cpu = rgb.cpu().numpy()
    rgb_f32 = rgb_cpu.astype(np.float32, copy=False)
    shape = tuple(int(v) for v in rgb_f32.shape)
    arr = host_state.get("numpy")
    if host_state.get("shape") != shape or arr is None:
        arr = np.empty(shape, dtype=np.uint16)
        host_state["shape"] = shape
        host_state["numpy"] = arr
    np.multiply(rgb_f32, 65535.0, out=rgb_f32)
    np.add(rgb_f32, 0.5, out=rgb_f32)
    np.clip(rgb_f32, 0.0, 65535.0, out=rgb_f32)
    arr[:] = rgb_f32.astype(np.uint16)
    return arr.tobytes()


def _bgr_to_rgb48_bytes(frame: np.ndarray, host_state: dict) -> bytes:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected HxWx3 frame.")
    shape = (int(frame.shape[0]), int(frame.shape[1]), 3)
    arr = host_state.get("numpy")
    if host_state.get("shape") != shape or arr is None:
        arr = np.empty(shape, dtype=np.uint16)
        host_state["shape"] = shape
        host_state["numpy"] = arr
    if frame.dtype == np.uint8:
        np.multiply(frame[:, :, ::-1], np.uint16(257), out=arr, casting="unsafe")
    elif frame.dtype == np.uint16:
        np.copyto(arr, frame[:, :, ::-1], casting="unsafe")
    else:
        src = frame.astype(np.float32, copy=False)
        if src.max(initial=0.0) <= 1.0:
            src = src * 65535.0
        arr[:] = np.clip(src[:, :, ::-1], 0.0, 65535.0).astype(np.uint16)
    return arr.tobytes()


class PipelineWorkerFeedersMixin:
    """HDR/SDR mpv feeder thread helpers for PipelineWorker."""

    def _hdr_feeder_fn(
        self,
        hdr_q: _queue.Queue,
        mpv_widget: MpvHDRWidget,
        live_smooth_cadence: bool,
    ):
        """Drain HDR queue, convert tensors to RGB48LE, and feed mpv."""
        if live_smooth_cadence:
            no_item = object()
            next_present_t = 0.0
            latest_rgb48_bytes: bytes | None = None
            present_interval_s = _live_present_interval_s()
            rgb48_host_state: dict = {}
            while True:
                now = time.perf_counter()
                wait_timeout = 0.2
                if latest_rgb48_bytes is not None and next_present_t > 0.0:
                    wait_timeout = max(
                        0.0,
                        min(_LIVE_SMOOTH_MAX_QUEUE_WAIT_S, next_present_t - now),
                    )

                item = no_item
                try:
                    item = hdr_q.get(timeout=wait_timeout)
                except _queue.Empty:
                    item = no_item

                if item is None:
                    break

                if item is not no_item:
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

                    ready_event = None
                    if isinstance(item, tuple) and len(item) == 3:
                        _source_t, tensor, ready_event = item
                    elif isinstance(item, tuple) and len(item) == 2:
                        _source_t, tensor = item
                    else:
                        tensor = item

                    if ready_event is not None:
                        try:
                            ready_event.synchronize()
                        except Exception:
                            pass

                    latest_rgb48_bytes = _tensor_to_rgb48_bytes(
                        tensor,
                        rgb48_host_state,
                    )
                    if next_present_t <= 0.0:
                        next_present_t = time.perf_counter()

                if latest_rgb48_bytes is None:
                    continue

                now = time.perf_counter()
                if next_present_t <= 0.0:
                    next_present_t = now
                if now < next_present_t:
                    continue

                mpv_widget.feed_frame(latest_rgb48_bytes)
                next_present_t = _next_live_present_deadline(
                    next_present_t,
                    now,
                    present_interval_s,
                )
            return

        rgb48_host_state: dict = {}
        while True:
            try:
                item = hdr_q.get(timeout=0.2)
            except _queue.Empty:
                continue
            if item is None:
                break
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
            ready_event = None
            if isinstance(item, tuple) and len(item) == 3:
                present_t, tensor, ready_event = item
            elif isinstance(item, tuple) and len(item) == 2:
                present_t, tensor = item
            else:
                present_t, tensor = None, item
            if ready_event is not None:
                try:
                    ready_event.synchronize()
                except Exception:
                    pass
            rgb48_bytes = _tensor_to_rgb48_bytes(tensor, rgb48_host_state)
            if present_t is not None:
                now = time.perf_counter()
                if now < present_t:
                    sleep_until(present_t)
            mpv_widget.feed_frame(rgb48_bytes)

    def _sdr_feeder_fn(
        self,
        sdr_q: _queue.Queue,
        mpv_widget: MpvHDRWidget,
        live_smooth_cadence: bool,
    ):
        """Drain SDR queue, convert BGR8 to RGB48LE, and feed mpv."""
        if live_smooth_cadence:
            no_item = object()
            next_present_t = 0.0
            latest_rgb48_bytes: bytes | None = None
            present_interval_s = _live_present_interval_s()
            rgb48_host_state: dict = {}
            while True:
                now = time.perf_counter()
                wait_timeout = 0.2
                if latest_rgb48_bytes is not None and next_present_t > 0.0:
                    wait_timeout = max(
                        0.0,
                        min(_LIVE_SMOOTH_MAX_QUEUE_WAIT_S, next_present_t - now),
                    )

                item = no_item
                try:
                    item = sdr_q.get(timeout=wait_timeout)
                except _queue.Empty:
                    item = no_item

                if item is None:
                    break

                if item is not no_item:
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
                        _source_t, frame = item
                    else:
                        frame = item

                    try:
                        latest_rgb48_bytes = _bgr_to_rgb48_bytes(
                            frame,
                            rgb48_host_state,
                        )
                    except Exception:
                        latest_rgb48_bytes = None

                    if next_present_t <= 0.0:
                        next_present_t = time.perf_counter()

                if latest_rgb48_bytes is None:
                    continue

                now = time.perf_counter()
                if next_present_t <= 0.0:
                    next_present_t = now
                if now < next_present_t:
                    continue

                mpv_widget.feed_frame(latest_rgb48_bytes)
                next_present_t = _next_live_present_deadline(
                    next_present_t,
                    now,
                    present_interval_s,
                )
            return

        rgb48_host_state: dict = {}
        while True:
            try:
                item = sdr_q.get(timeout=0.2)
            except _queue.Empty:
                continue
            if item is None:
                break
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
                rgb48_bytes = _bgr_to_rgb48_bytes(
                    frame,
                    rgb48_host_state,
                )
                if present_t is not None:
                    now = time.perf_counter()
                    if now < present_t:
                        sleep_until(present_t)
                mpv_widget.feed_frame(rgb48_bytes)
            except Exception:
                pass

    def _start_hdr_feeder(self):
        live_smooth_cadence = bool(getattr(self, "_capture_target", None))
        self._hdr_queue = _queue.Queue(maxsize=1)
        self._hdr_thread = threading.Thread(
            target=self._hdr_feeder_fn,
            args=(self._hdr_queue, self._mpv_widget, live_smooth_cadence),
            daemon=True,
        )
        self._hdr_thread.start()

    def _stop_hdr_feeder(self):
        q = self._hdr_queue
        if q is not None:
            try:
                q.put_nowait(None)
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
        if self._sdr_mpv_widget is None or self._sdr_queue is not None:
            return
        live_smooth_cadence = bool(getattr(self, "_capture_target", None))
        self._sdr_queue = _queue.Queue(maxsize=1)
        self._sdr_thread = threading.Thread(
            target=self._sdr_feeder_fn,
            args=(self._sdr_queue, self._sdr_mpv_widget, live_smooth_cadence),
            daemon=True,
        )
        self._sdr_thread.start()

    def _stop_sdr_feeder(self):
        q = self._sdr_queue
        if q is not None:
            try:
                q.put_nowait(None)
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
