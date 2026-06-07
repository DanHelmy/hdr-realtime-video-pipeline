from __future__ import annotations

import cv2
import numpy as np
import os
import queue as _queue
import time
import torch
import torch.nn.functional as F

from gui_scaling import BEST_CV2_INTERP, _letterbox_bgr, _apply_upscale_sharpen
from timer import sleep_until


class PipelineWorkerFrameProcessingMixin:
    """Per-frame HDR bypass / SDR infer processing for PipelineWorker."""

    @staticmethod
    def _box_blur(x: torch.Tensor, k: int = 3) -> torch.Tensor:
        p = k // 2
        if k >= 9:
            # Large 2D box blurs are hot in frame processing; separable passes keep
            # the same box-filter behavior but avoid an expensive k x k kernel.
            x = F.avg_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, p))
            return F.avg_pool2d(x, kernel_size=(k, 1), stride=1, padding=(p, 0))
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=p)

    def _reset_enhance_history(self):
        self._enh_prev_luma = None
        self._enh_temporal_detail = None

    @staticmethod
    def _queue_latest(q: _queue.Queue | None, item) -> None:
        if q is None:
            return
        try:
            q.put_nowait(item)
            return
        except _queue.Full:
            pass

    def _preserve_display_queue_order(self) -> bool:
        if getattr(self, "_capture_target", None):
            return False
        try:
            return int(getattr(self, "_video_playback_buffer_frames", 1)) > 1
        except Exception:
            return False

    def _queue_display_item(self, q: _queue.Queue | None, item) -> None:
        if q is None:
            return
        if not self._preserve_display_queue_order():
            self._queue_latest(q, item)
            return
        while not bool(getattr(self, "_stop_flag", False)):
            if getattr(self, "_seek_frame", None) is not None:
                return
            try:
                q.put(item, timeout=0.05)
                return
            except _queue.Full:
                continue

        try:
            q.get_nowait()
        except _queue.Empty:
            pass

        try:
            q.put_nowait(item)
        except _queue.Full:
            pass

    def _capture_enhance_history(self):
        return (
            self._enh_prev_luma,
            self._enh_temporal_detail,
        )

    def _restore_enhance_history(self, state) -> None:
        (
            self._enh_prev_luma,
            self._enh_temporal_detail,
        ) = state

    def _prepare_hdr_output_tensor(
        self,
        raw_out,
        lower_res_processing: bool,
    ):
        del lower_res_processing
        return raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

    def _render_hdr_output(
        self,
        prepared_out,
        out_w: int,
        out_h: int,
        *,
        copy_input: bool = False,
    ) -> np.ndarray:
        post_input = prepared_out.clone() if copy_input else prepared_out
        output = self._processor.postprocess(post_input)
        if (output.shape[1], output.shape[0]) != (out_w, out_h):
            if out_w > output.shape[1] or out_h > output.shape[0]:
                output = cv2.resize(output, (out_w, out_h), interpolation=BEST_CV2_INTERP)
                output = _apply_upscale_sharpen(output)
            else:
                output = cv2.resize(output, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return output

    def _stage_hdr_display_tensor(self, prepared_out, use_cuda: bool):
        if not torch.is_tensor(prepared_out):
            try:
                return prepared_out.clone()
            except Exception:
                return prepared_out
        if (
            not use_cuda
            or getattr(prepared_out, "device", None) is None
            or prepared_out.device.type != "cuda"
        ):
            return prepared_out.clone()

        try:
            queue_frames = int(getattr(self, "_video_playback_buffer_frames", 1))
        except Exception:
            queue_frames = 1
        pool_size = max(2, min(16, queue_frames + 2))
        key = (
            tuple(int(v) for v in prepared_out.shape),
            str(prepared_out.device),
            str(prepared_out.dtype),
            int(pool_size),
        )
        pool = getattr(self, "_hdr_display_tensor_pool", None)
        if getattr(self, "_hdr_display_tensor_pool_key", None) != key or not pool:
            pool = [
                torch.empty_like(prepared_out, memory_format=torch.contiguous_format)
                for _ in range(pool_size)
            ]
            self._hdr_display_tensor_pool = pool
            self._hdr_display_tensor_pool_key = key
            self._hdr_display_tensor_pool_idx = 0

        idx = int(getattr(self, "_hdr_display_tensor_pool_idx", 0) or 0) % len(pool)
        staged = pool[idx]
        self._hdr_display_tensor_pool_idx = (idx + 1) % len(pool)
        staged.copy_(prepared_out, non_blocking=True)
        return staged

    def _cuda_timing_events(self):
        events = getattr(self, "_infer_timing_events", None)
        if events is not None:
            return events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        events = (start_event, end_event)
        self._infer_timing_events = events
        return events

    def _process_frame(
        self,
        *,
        frame: np.ndarray,
        frame_idx: int,
        present_t: float | None,
        out_w: int,
        out_h: int,
        proc_w: int,
        proc_h: int,
        lower_res_processing: bool,
        mpv_w,
        use_cuda: bool,
        need_compare_frame: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray, object, bool, float]:
        # File playback may carry an absolute presentation time. Live capture
        # uses the feeder's steady low-FPS clock and mpv's display sync.
        queue_present_t = present_t
        if (
            queue_present_t is None
            and getattr(self, "_capture_target", None)
            and mpv_w is not None
            and self._sdr_visible
            and self._sdr_mpv_widget is not None
        ):
            try:
                capture_fps = float(self._capture_target.get("fps", 24.0) or 24.0)
            except Exception:
                capture_fps = 24.0
            try:
                delay_frames = float(
                    os.environ.get("HDRTVNET_LIVE_PAIR_PRESENT_DELAY_FRAMES", "0.75")
                )
            except Exception:
                delay_frames = 0.75
            interval_s = 1.0 / max(1.0, capture_fps)
            queue_present_t = time.perf_counter() + (
                interval_s * max(0.0, min(2.0, delay_frames))
            )
        sdr_mpv_active = bool(
            self._sdr_visible
            and self._sdr_mpv_widget is not None
            and self._sdr_queue is not None
        )
        need_display_frame = bool(
            self._input_is_hdr
            or mpv_w is None
            or need_compare_frame
            or (self._sdr_visible and not sdr_mpv_active)
        )
        display_frame = _letterbox_bgr(frame, out_w, out_h) if need_display_frame else None
        model_latency_ms = 0.0

        if lower_res_processing:
            # Resize from the decoded source directly so we do not scale
            # already-letterboxed black bars a second time.
            model_inp = _letterbox_bgr(frame, proc_w, proc_h)
        elif display_frame is not None:
            model_inp = display_frame
        elif (frame.shape[1], frame.shape[0]) == (out_w, out_h):
            model_inp = frame
        else:
            model_inp = _letterbox_bgr(frame, out_w, out_h)
        sdr_mpv_frame = model_inp if sdr_mpv_active else display_frame

        if self._input_is_hdr:
            need_hdr_cpu = False
            prepared_out = None
            output = display_frame if display_frame is not None else model_inp
            if (output.shape[1], output.shape[0]) != (out_w, out_h):
                output = cv2.resize(output, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if mpv_w is not None:
                rgb16 = np.ascontiguousarray(output[:, :, ::-1].astype(np.uint16) * 257)
                if queue_present_t is not None:
                    now_t = time.perf_counter()
                    if now_t < queue_present_t:
                        sleep_until(queue_present_t)
                mpv_w.feed_frame(rgb16.data)
            if (
                sdr_mpv_frame is not None
                and self._sdr_mpv_widget is not None
                and self._sdr_queue is not None
                and self._sdr_visible
            ):
                if frame_idx < self._sdr_drop_until_frame:
                    pass
                else:
                    self._sdr_drop_until_frame = 0
                if self._sdr_drop_until_frame == 0:
                    self._queue_display_item(self._sdr_queue, (queue_present_t, sdr_mpv_frame))
            if self._sdr_visible:
                need_hdr_cpu = True
            return display_frame, output, prepared_out, need_hdr_cpu, model_latency_ms

        infer_t0 = 0.0
        cuda_timing = False
        infer_start_event = None
        infer_end_event = None
        if use_cuda:
            try:
                infer_start_event, infer_end_event = self._cuda_timing_events()
                infer_start_event.record(torch.cuda.current_stream())
                cuda_timing = True
            except Exception:
                infer_start_event = None
                infer_end_event = None
                cuda_timing = False
        if not cuda_timing:
            infer_t0 = time.perf_counter()
        with torch.inference_mode():
            tensor, cond = self._processor.preprocess(model_inp)
            raw_out = self._processor.infer((tensor, cond))
        if cuda_timing and infer_end_event is not None:
            try:
                infer_end_event.record(torch.cuda.current_stream())
                infer_end_event.synchronize()
                model_latency_ms = max(
                    0.0,
                    float(infer_start_event.elapsed_time(infer_end_event)),
                )
            except Exception:
                model_latency_ms = 0.0
        else:
            model_latency_ms = max(0.0, (time.perf_counter() - infer_t0) * 1000.0)

        prepared_out = self._prepare_hdr_output_tensor(
            raw_out,
            lower_res_processing,
        )

        if mpv_w is not None and self._hdr_queue is not None:
            ready_event = None
            if frame_idx < self._hdr_drop_until_frame:
                pass
            else:
                self._hdr_drop_until_frame = 0
            if self._hdr_drop_until_frame == 0:
                queued_tensor = self._stage_hdr_display_tensor(prepared_out, use_cuda)
                if use_cuda:
                    ready_event = torch.cuda.Event(enable_timing=False)
                    ready_event.record(torch.cuda.current_stream())
                self._queue_display_item(
                    self._hdr_queue,
                    (queue_present_t, queued_tensor, ready_event),
                )

        if (
            sdr_mpv_frame is not None
            and self._sdr_mpv_widget is not None
            and self._sdr_queue is not None
            and self._sdr_visible
        ):
            if frame_idx < self._sdr_drop_until_frame:
                pass
            else:
                self._sdr_drop_until_frame = 0
            if self._sdr_drop_until_frame == 0:
                self._queue_display_item(self._sdr_queue, (queue_present_t, sdr_mpv_frame))

        need_hdr_cpu = (mpv_w is None)
        if need_hdr_cpu:
            output = self._render_hdr_output(prepared_out, out_w, out_h)
        else:
            output = display_frame if display_frame is not None else frame

        return display_frame, output, prepared_out, need_hdr_cpu, model_latency_ms
