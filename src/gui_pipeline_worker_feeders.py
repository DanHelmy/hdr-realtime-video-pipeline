from __future__ import annotations

import queue as _queue
import threading
import time

import numpy as np
import torch

from gui_mpv_widget import MpvHDRWidget


class PipelineWorkerFeedersMixin:
    """HDR/SDR mpv feeder thread helpers for PipelineWorker."""

    def _hdr_feeder_fn(self, hdr_q: _queue.Queue, mpv_widget: MpvHDRWidget):
        """Drain HDR queue, convert tensors to RGB48LE, and feed mpv."""
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
            if isinstance(item, tuple) and len(item) == 2:
                present_t, tensor = item
            else:
                present_t, tensor = None, item
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
                    time.sleep(present_t - now)
            mpv_widget.feed_frame(hdr_u16.astype(np.uint16).data)

    def _sdr_feeder_fn(self, sdr_q: _queue.Queue, mpv_widget: MpvHDRWidget):
        """Drain SDR queue, convert BGR8 to RGB48LE, and feed mpv."""
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
                        time.sleep(present_t - now)
                mpv_widget.feed_frame(rgb16.data)
            except Exception:
                pass

    def _start_hdr_feeder(self):
        self._hdr_queue = _queue.Queue(maxsize=1)
        self._hdr_thread = threading.Thread(
            target=self._hdr_feeder_fn,
            args=(self._hdr_queue, self._mpv_widget),
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
        self._sdr_queue = _queue.Queue(maxsize=1)
        self._sdr_thread = threading.Thread(
            target=self._sdr_feeder_fn,
            args=(self._sdr_queue, self._sdr_mpv_widget),
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
