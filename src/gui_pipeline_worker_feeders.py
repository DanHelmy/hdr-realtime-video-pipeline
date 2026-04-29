from __future__ import annotations

import queue as _queue
import threading
import time

import numpy as np
import torch

from gui_config import LIVE_CAPTURE_PROCESS_FPS
from gui_mpv_widget import MpvHDRWidget
from timer import sleep_until

_LIVE_SMOOTH_MAX_QUEUE_WAIT_S = 0.050
_LIVE_SMOOTH_MAX_CATCHUP_FRAMES = 2


def _live_present_interval_s() -> float:
    return 1.0 / max(1.0, float(LIVE_CAPTURE_PROCESS_FPS or 24.0))


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

                    with torch.inference_mode():
                        raw_cpu = (
                            tensor.squeeze(0)
                            .clamp_(0.0, 1.0)
                            .permute(1, 2, 0)
                            .contiguous()
                            .cpu()
                            .numpy()
                        )
                    hdr_u16 = (raw_cpu.astype(np.float32).__imul__(65535)).__add__(0.5)
                    np.clip(hdr_u16, 0, 65535, out=hdr_u16)
                    latest_rgb48_bytes = hdr_u16.astype(np.uint16).tobytes()
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
                next_present_t += present_interval_s
                if next_present_t < (
                    now - (present_interval_s * _LIVE_SMOOTH_MAX_CATCHUP_FRAMES)
                ):
                    next_present_t = (
                        now - (present_interval_s * _LIVE_SMOOTH_MAX_CATCHUP_FRAMES)
                    )
            return

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
            with torch.inference_mode():
                raw_cpu = (tensor.squeeze(0)
                           .clamp_(0.0, 1.0)
                           .permute(1, 2, 0)
                           .contiguous()
                           .cpu()
                           .numpy())
            hdr_u16 = (raw_cpu.astype(np.float32).__imul__(65535)).__add__(0.5)
            np.clip(hdr_u16, 0, 65535, out=hdr_u16)
            if present_t is not None:
                now = time.perf_counter()
                if now < present_t:
                    sleep_until(present_t)
            mpv_widget.feed_frame(hdr_u16.astype(np.uint16).data)

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
                        latest_rgb48_bytes = np.ascontiguousarray(
                            frame[:, :, ::-1].astype(np.uint16) * 257
                        ).tobytes()
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
                next_present_t += present_interval_s
                if next_present_t < (
                    now - (present_interval_s * _LIVE_SMOOTH_MAX_CATCHUP_FRAMES)
                ):
                    next_present_t = (
                        now - (present_interval_s * _LIVE_SMOOTH_MAX_CATCHUP_FRAMES)
                    )
            return

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
                rgb16 = np.ascontiguousarray(
                    frame[:, :, ::-1].astype(np.uint16) * 257
                )
                if present_t is not None:
                    now = time.perf_counter()
                    if now < present_t:
                        sleep_until(present_t)
                mpv_widget.feed_frame(rgb16.data)
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
