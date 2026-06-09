from __future__ import annotations

import os
import queue as _queue
import threading
import time

import numpy as np
import torch

from gui_config import VIDEO_PLAYBACK_BUFFER_FRAMES, VIDEO_PLAYBACK_PRESERVE_ORDER
from gui_mpv_widget import MpvHDRWidget
from timer import prepare_playback_timing_thread, sleep_until

_LIVE_SMOOTH_MAX_QUEUE_WAIT_S = 0.042  # ~1/24 second for better 24fps cadence
_LIVE_SMOOTH_MAX_CATCHUP_FRAMES = 3  # Allow more catchup for smoother playback
_LIVE_SMOOTH_QUEUE_POLL_SLICE_S = 0.0040
_DEFAULT_FEEDER_GPU_RGB48 = "0" if bool(getattr(torch.version, "hip", None)) else "1"
_FEEDER_GPU_RGB48 = str(
    os.environ.get("HDRTVNET_FEEDER_GPU_RGB48", _DEFAULT_FEEDER_GPU_RGB48)
).strip().lower() in {"1", "true", "yes", "on"}


def _gpu_rgb48_ring_frames() -> int:
    try:
        value = int(
            str(os.environ.get("HDRTVNET_FEEDER_GPU_RGB48_RING_FRAMES", "3")).strip()
            or "3"
        )
    except Exception:
        value = 3
    return max(2, min(8, value))


_GPU_RGB48_RING_FRAMES = _gpu_rgb48_ring_frames()


class _PinnedMpvFrame:
    """A CUDA-ready pinned host frame whose slot is released after mpv writes it."""

    def __init__(self, slot: dict, ready_event) -> None:
        self._slot = slot
        self._ready_event = ready_event
        self._ready_waited = False
        self._released = False

    def wait_ready(self) -> None:
        if self._ready_waited:
            return
        event = self._ready_event
        if event is not None:
            event.synchronize()
        self._ready_waited = True

    def buffer_view(self):
        self.wait_ready()
        return memoryview(self._slot["numpy"]).cast("B")

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            self.wait_ready()
        except Exception:
            pass
        try:
            self._slot["free"].set()
        except Exception:
            pass


def _live_present_interval_s(display_fps: float) -> float:
    return 1.0 / max(1.0, float(display_fps or 24.0))


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


def _queue_get_until(q: _queue.Queue, deadline_s: float, no_item):
    while True:
        try:
            return q.get_nowait()
        except _queue.Empty:
            now = time.perf_counter()
            if now >= deadline_s:
                return no_item
            sleep_until(
                min(float(deadline_s), now + _LIVE_SMOOTH_QUEUE_POLL_SLICE_S),
                coarse_margin_s=0.00035,
                spin_margin_s=0.00015,
            )


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


def _pinned_u16_host_ring(state: dict, shape: tuple[int, int, int]):
    key = (shape, int(_GPU_RGB48_RING_FRAMES))
    if state.get("ring_key") == key and state.get("ring_slots"):
        return state["ring_slots"]
    slots = []
    for _ in range(int(_GPU_RGB48_RING_FRAMES)):
        tensor = None
        try:
            if torch.cuda.is_available():
                tensor = torch.empty(shape, dtype=torch.uint16, pin_memory=True)
        except Exception:
            tensor = None
        if tensor is None:
            tensor = torch.empty(shape, dtype=torch.uint16)
        free = threading.Event()
        free.set()
        slots.append({
            "tensor": tensor,
            "numpy": tensor.numpy(),
            "free": free,
        })
    state["ring_key"] = key
    state["ring_slots"] = slots
    state["ring_index"] = 0
    return slots


def _acquire_pinned_u16_slot(state: dict, shape: tuple[int, int, int]):
    slots = _pinned_u16_host_ring(state, shape)
    if not slots:
        return None
    start = int(state.get("ring_index", 0) or 0) % len(slots)
    for offset in range(len(slots)):
        idx = (start + offset) % len(slots)
        slot = slots[idx]
        if slot["free"].is_set():
            slot["free"].clear()
            state["ring_index"] = (idx + 1) % len(slots)
            return slot
    idx = start
    slot = slots[idx]
    if not slot["free"].wait(timeout=0.25):
        return None
    slot["free"].clear()
    state["ring_index"] = (idx + 1) % len(slots)
    return slot


def _cuda_rgb48_state(state: dict, shape: tuple[int, int, int], device):
    key = (shape, str(device))
    if (
        state.get("cuda_key") == key
        and state.get("rgb_f32") is not None
        and state.get("rgb_u16") is not None
        and state.get("stream") is not None
    ):
        return state["stream"], state["rgb_f32"], state["rgb_u16"]
    with torch.cuda.device(device):
        stream = torch.cuda.Stream()
        rgb_f32 = torch.empty(shape, dtype=torch.float32, device=device)
        rgb_u16 = torch.empty(shape, dtype=torch.uint16, device=device)
    state["cuda_key"] = key
    state["stream"] = stream
    state["rgb_f32"] = rgb_f32
    state["rgb_u16"] = rgb_u16
    return stream, rgb_f32, rgb_u16


def _tensor_to_rgb48_bytes(tensor, host_state: dict):
    with torch.inference_mode():
        prepared = tensor[0] if isinstance(tensor, (tuple, list)) else tensor
        rgb = prepared.squeeze(0).permute(1, 2, 0)

    if (
        _FEEDER_GPU_RGB48
        and getattr(rgb, "device", None) is not None
        and rgb.device.type == "cuda"
    ):
        shape = tuple(int(v) for v in rgb.shape)
        stream, rgb_f32, rgb_u16 = _cuda_rgb48_state(
            host_state,
            shape,
            rgb.device,
        )
        slot = _acquire_pinned_u16_slot(host_state, shape)
        if slot is None:
            host_tensor, host_np = _pinned_u16_host_buffer(host_state, shape)
        else:
            host_tensor = slot["tensor"]
            host_np = slot["numpy"]
        non_blocking = False
        try:
            non_blocking = bool(host_tensor.is_pinned())
        except Exception:
            non_blocking = False
        # Keep the 4K RGB48 conversion off the inference stream and reuse the
        # large staging tensors. Reallocating these every frame causes periodic
        # GUI-only latency spikes even when the TensorRT engine itself is flat.
        with torch.cuda.device(rgb.device), torch.cuda.stream(stream):
            rgb_f32.copy_(rgb, non_blocking=True)
            rgb_f32.clamp_(0.0, 1.0).mul_(65535.0).add_(0.5)
            rgb_u16.copy_(rgb_f32, non_blocking=True)
            host_tensor.copy_(rgb_u16, non_blocking=non_blocking)
            ready_event = torch.cuda.Event(enable_timing=False)
            ready_event.record(stream)
        if slot is not None and non_blocking:
            return _PinnedMpvFrame(slot, ready_event)
        stream.synchronize()
        if slot is not None:
            slot["free"].set()
        return host_np.tobytes()

    rgb_cpu = rgb.clamp(0.0, 1.0).contiguous().cpu().numpy()
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


def _repeatable_mpv_payload(payload):
    if hasattr(payload, "buffer_view") and hasattr(payload, "release"):
        try:
            return bytes(payload.buffer_view())
        finally:
            try:
                payload.release()
            except Exception:
                pass
    return payload


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


def _bgr_to_sdr_mpv_bytes(frame: np.ndarray, host_state: dict, raw_format: str):
    fmt = str(raw_format or "rgb48le").strip().lower()
    if fmt == "rgb48le":
        return _bgr_to_rgb48_bytes(frame, host_state)

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected HxWx3 frame.")

    if frame.dtype == np.uint8:
        bgr8 = frame
    elif frame.dtype == np.uint16:
        bgr8 = np.right_shift(frame, 8).astype(np.uint8, copy=False)
    else:
        src = frame.astype(np.float32, copy=False)
        if src.max(initial=0.0) <= 1.0:
            src = src * 255.0
        bgr8 = np.clip(src, 0.0, 255.0).astype(np.uint8)

    if fmt == "bgr24":
        return memoryview(np.ascontiguousarray(bgr8))
    if fmt == "rgb24":
        return memoryview(np.ascontiguousarray(bgr8[:, :, ::-1]))
    return _bgr_to_rgb48_bytes(frame, host_state)


class PipelineWorkerFeedersMixin:
    """HDR/SDR mpv feeder thread helpers for PipelineWorker."""

    def _hdr_feeder_fn(
        self,
        hdr_q: _queue.Queue,
        mpv_widget: MpvHDRWidget,
        live_smooth_cadence: bool,
        display_fps: float,
        preserve_order: bool = False,
    ):
        """Drain HDR queue, convert tensors to RGB48LE, and feed mpv."""
        prepare_playback_timing_thread()
        if live_smooth_cadence:
            no_item = object()
            next_present_t = 0.0
            latest_rgb48_bytes: bytes | None = None
            present_interval_s = _live_present_interval_s(display_fps)
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
                    if latest_rgb48_bytes is not None and next_present_t > 0.0:
                        item = _queue_get_until(
                            hdr_q,
                            now + wait_timeout,
                            no_item,
                        )
                    else:
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

                    source_t = None
                    ready_event = None
                    if isinstance(item, tuple) and len(item) == 3:
                        source_t, tensor, ready_event = item
                    elif isinstance(item, tuple) and len(item) == 2:
                        source_t, tensor = item
                    else:
                        tensor = item

                    if ready_event is not None:
                        try:
                            ready_event.synchronize()
                        except Exception:
                            pass

                    latest_rgb48_bytes = _repeatable_mpv_payload(
                        _tensor_to_rgb48_bytes(
                            tensor,
                            rgb48_host_state,
                        )
                    )
                    try:
                        item_present_t = float(source_t) if source_t is not None else 0.0
                    except Exception:
                        item_present_t = 0.0
                    if item_present_t > 0.0:
                        next_present_t = item_present_t
                    elif next_present_t <= 0.0:
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
            if not preserve_order:
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
        display_fps: float,
        preserve_order: bool = False,
    ):
        """Drain SDR queue, convert BGR8 to RGB48LE, and feed mpv."""
        prepare_playback_timing_thread()
        raw_format = str(getattr(mpv_widget, "raw_video_format", "rgb48le") or "rgb48le")
        if live_smooth_cadence:
            no_item = object()
            next_present_t = 0.0
            latest_sdr_bytes = None
            present_interval_s = _live_present_interval_s(display_fps)
            rgb48_host_state: dict = {}
            while True:
                now = time.perf_counter()
                wait_timeout = 0.2
                if latest_sdr_bytes is not None and next_present_t > 0.0:
                    wait_timeout = max(
                        0.0,
                        min(_LIVE_SMOOTH_MAX_QUEUE_WAIT_S, next_present_t - now),
                    )

                item = no_item
                try:
                    if latest_sdr_bytes is not None and next_present_t > 0.0:
                        item = _queue_get_until(
                            sdr_q,
                            now + wait_timeout,
                            no_item,
                        )
                    else:
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

                    source_t = None
                    if isinstance(item, tuple) and len(item) == 2:
                        source_t, frame = item
                    else:
                        frame = item

                    try:
                        latest_sdr_bytes = _bgr_to_sdr_mpv_bytes(
                            frame,
                            rgb48_host_state,
                            raw_format,
                        )
                    except Exception:
                        latest_sdr_bytes = None

                    try:
                        item_present_t = float(source_t) if source_t is not None else 0.0
                    except Exception:
                        item_present_t = 0.0
                    if item_present_t > 0.0:
                        next_present_t = item_present_t
                    elif next_present_t <= 0.0:
                        next_present_t = time.perf_counter()

                if latest_sdr_bytes is None:
                    continue

                now = time.perf_counter()
                if next_present_t <= 0.0:
                    next_present_t = now
                if now < next_present_t:
                    continue

                mpv_widget.feed_frame(latest_sdr_bytes)
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
            if not preserve_order:
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
                sdr_bytes = _bgr_to_sdr_mpv_bytes(
                    frame,
                    rgb48_host_state,
                    raw_format,
                )
                if present_t is not None:
                    now = time.perf_counter()
                    if now < present_t:
                        sleep_until(present_t)
                mpv_widget.feed_frame(sdr_bytes)
            except Exception:
                pass

    def _start_hdr_feeder(self):
        live_smooth_cadence = bool(getattr(self, "_capture_target", None))
        buffer_frames = (
            1
            if live_smooth_cadence
            else min(3, max(1, int(getattr(self, "_video_playback_buffer_frames", VIDEO_PLAYBACK_BUFFER_FRAMES))))
        )
        preserve_order = bool(
            (not live_smooth_cadence)
            and buffer_frames > 1
            and VIDEO_PLAYBACK_PRESERVE_ORDER
        )
        self._hdr_queue = _queue.Queue(maxsize=buffer_frames)
        display_fps = float(getattr(self, "_live_display_fps", 24.0) or 24.0)
        self._hdr_thread = threading.Thread(
            target=self._hdr_feeder_fn,
            args=(
                self._hdr_queue,
                self._mpv_widget,
                live_smooth_cadence,
                display_fps,
                preserve_order,
            ),
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
        if (
            self._sdr_mpv_widget is None
            or self._sdr_queue is not None
        ):
            return
        live_smooth_cadence = bool(getattr(self, "_capture_target", None))
        buffer_frames = (
            1
            if live_smooth_cadence
            else min(3, max(1, int(getattr(self, "_video_playback_buffer_frames", VIDEO_PLAYBACK_BUFFER_FRAMES))))
        )
        preserve_order = bool(
            (not live_smooth_cadence)
            and buffer_frames > 1
            and VIDEO_PLAYBACK_PRESERVE_ORDER
        )
        self._sdr_queue = _queue.Queue(maxsize=buffer_frames)
        display_fps = float(getattr(self, "_live_display_fps", 24.0) or 24.0)
        self._sdr_thread = threading.Thread(
            target=self._sdr_feeder_fn,
            args=(
                self._sdr_queue,
                self._sdr_mpv_widget,
                live_smooth_cadence,
                display_fps,
                preserve_order,
            ),
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
