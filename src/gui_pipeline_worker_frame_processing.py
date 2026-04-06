from __future__ import annotations

import cv2
import numpy as np
import queue as _queue
import time
import torch
import torch.nn.functional as F

from gui_scaling import BEST_CV2_INTERP, _letterbox_bgr, _apply_upscale_sharpen
from timer import sleep_until

_FLAT_TEMPORAL_ENABLED = True
_FLAT_TEMPORAL_BRIGHTNESS = 0.80
_FLAT_TEMPORAL_NEUTRAL = 0.12
_FLAT_TEMPORAL_DETAIL = 0.028
_FLAT_TEMPORAL_MOTION = 0.030
_FLAT_TEMPORAL_BLEND = 0.78
_SCENE_CUT_DOWNSCALE = (48, 27)
_SCENE_CUT_THRESHOLD = 18.0


class PipelineWorkerFrameProcessingMixin:
    """Per-frame HDR bypass / SDR infer processing for PipelineWorker."""

    @staticmethod
    def _queue_latest(q: _queue.Queue | None, item) -> None:
        if q is None:
            return
        try:
            q.put_nowait(item)
            return
        except _queue.Full:
            pass

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
            self._flat_temporal_prev_rgb,
            self._flat_temporal_prev_luma,
            self._flat_temporal_prev_scene,
        )

    def _restore_enhance_history(self, state) -> None:
        (
            self._enh_prev_luma,
            self._enh_temporal_detail,
            self._flat_temporal_prev_rgb,
            self._flat_temporal_prev_luma,
            self._flat_temporal_prev_scene,
        ) = state

    def _detect_scene_cut(self, frame_bgr: np.ndarray) -> bool:
        """Cheap scene-cut detection from downscaled SDR luma."""
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            self._flat_temporal_prev_scene = None
            return True

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, _SCENE_CUT_DOWNSCALE, interpolation=cv2.INTER_AREA)
        prev = getattr(self, "_flat_temporal_prev_scene", None)
        self._flat_temporal_prev_scene = small
        if prev is None or prev.shape != small.shape:
            return True
        diff = float(np.mean(np.abs(small.astype(np.float32) - prev.astype(np.float32))))
        return diff > _SCENE_CUT_THRESHOLD

    def _temporally_stabilize_flat_highlights(self, prepared_out, scene_cut: bool):
        """Reduce boiling in bright flat neutral areas with masked temporal blending."""
        if not _FLAT_TEMPORAL_ENABLED or prepared_out is None:
            return prepared_out
        if not torch.is_tensor(prepared_out) or prepared_out.ndim != 4:
            return prepared_out

        peak = prepared_out.amax(dim=1, keepdim=True)
        floor = prepared_out.amin(dim=1, keepdim=True)
        chroma = peak - floor
        luma = (
            0.2126 * prepared_out[:, 0:1, :, :]
            + 0.7152 * prepared_out[:, 1:2, :, :]
            + 0.0722 * prepared_out[:, 2:3, :, :]
        )
        local_mean = F.avg_pool2d(luma, kernel_size=3, stride=1, padding=1)
        detail = (luma - local_mean).abs()

        stabilized = prepared_out
        prev_rgb = getattr(self, "_flat_temporal_prev_rgb", None)
        prev_luma = getattr(self, "_flat_temporal_prev_luma", None)

        if (
            not scene_cut
            and prev_rgb is not None
            and prev_luma is not None
            and prev_rgb.shape == prepared_out.shape
            and prev_luma.shape == luma.shape
        ):
            bright_weight = (
                (peak - _FLAT_TEMPORAL_BRIGHTNESS)
                / max(1.0 - _FLAT_TEMPORAL_BRIGHTNESS, 1e-6)
            ).clamp_(0.0, 1.0)
            neutral_weight = (
                (_FLAT_TEMPORAL_NEUTRAL - chroma) / _FLAT_TEMPORAL_NEUTRAL
            ).clamp_(0.0, 1.0)
            flat_weight = (
                (_FLAT_TEMPORAL_DETAIL - detail) / _FLAT_TEMPORAL_DETAIL
            ).clamp_(0.0, 1.0)
            motion = (luma - prev_luma).abs()
            stable_weight = (
                (_FLAT_TEMPORAL_MOTION - motion) / _FLAT_TEMPORAL_MOTION
            ).clamp_(0.0, 1.0)
            blend = (
                bright_weight * neutral_weight * flat_weight * stable_weight
            ) * _FLAT_TEMPORAL_BLEND
            stabilized = prepared_out * (1.0 - blend) + prev_rgb * blend

        self._flat_temporal_prev_rgb = stabilized.detach()
        self._flat_temporal_prev_luma = luma.detach()
        return stabilized

    def _prepare_hdr_output_tensor(self, raw_out, lower_res_processing: bool, scene_cut: bool):
        prepared = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out
        if lower_res_processing:
            prepared = self._enhance_best_gpu(prepared)
        prepared = self._temporally_stabilize_flat_highlights(prepared, scene_cut)
        return prepared

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

    def _process_frame(
        self,
        *,
        frame: np.ndarray,
        frame_idx: int,
        present_t: float,
        out_w: int,
        out_h: int,
        proc_w: int,
        proc_h: int,
        lower_res_processing: bool,
        mpv_w,
        use_cuda: bool,
    ) -> tuple[np.ndarray | None, np.ndarray, object, bool]:
        need_display_frame = self._input_is_hdr or self._sdr_visible or mpv_w is None
        display_frame = _letterbox_bgr(frame, out_w, out_h) if need_display_frame else None

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
        scene_cut = self._detect_scene_cut(model_inp)

        if self._input_is_hdr:
            need_hdr_cpu = False
            prepared_out = None
            output = display_frame
            if (output.shape[1], output.shape[0]) != (out_w, out_h):
                output = cv2.resize(output, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if mpv_w is not None:
                rgb16 = np.ascontiguousarray(output[:, :, ::-1].astype(np.uint16) * 257)
                now_t = time.perf_counter()
                if now_t < present_t:
                    sleep_until(present_t)
                mpv_w.feed_frame(rgb16.data)
            if (
                display_frame is not None
                and self._sdr_mpv_widget is not None
                and self._sdr_queue is not None
                and self._sdr_visible
            ):
                if frame_idx < self._sdr_drop_until_frame:
                    pass
                else:
                    self._sdr_drop_until_frame = 0
                if self._sdr_drop_until_frame == 0:
                    self._queue_latest(self._sdr_queue, (present_t, display_frame))
            if self._sdr_visible:
                need_hdr_cpu = True
            return display_frame, output, prepared_out, need_hdr_cpu

        with torch.inference_mode():
            tensor, cond = self._processor.preprocess(model_inp)
            raw_out = self._processor.infer((tensor, cond))

        prepared_out = self._prepare_hdr_output_tensor(
            raw_out, lower_res_processing, scene_cut
        )

        if mpv_w is not None and self._hdr_queue is not None:
            ready_event = None
            if frame_idx < self._hdr_drop_until_frame:
                pass
            else:
                self._hdr_drop_until_frame = 0
            if self._hdr_drop_until_frame == 0:
                queued_tensor = prepared_out.clone()
                if use_cuda:
                    ready_event = torch.cuda.Event(enable_timing=False)
                    ready_event.record(torch.cuda.current_stream())
                self._queue_latest(
                    self._hdr_queue,
                    (present_t, queued_tensor, ready_event),
                )
            if use_cuda:
                try:
                    if ready_event is not None:
                        ready_event.synchronize()
                    else:
                        torch.cuda.current_stream().synchronize()
                except Exception:
                    torch.cuda.synchronize()

        if (
            display_frame is not None
            and self._sdr_mpv_widget is not None
            and self._sdr_queue is not None
            and self._sdr_visible
        ):
            if frame_idx < self._sdr_drop_until_frame:
                pass
            else:
                self._sdr_drop_until_frame = 0
            if self._sdr_drop_until_frame == 0:
                self._queue_latest(self._sdr_queue, (present_t, display_frame))

        need_hdr_cpu = (mpv_w is None)
        if need_hdr_cpu:
            output = self._render_hdr_output(prepared_out, out_w, out_h)
        else:
            output = display_frame if display_frame is not None else frame

        return display_frame, output, prepared_out, need_hdr_cpu
