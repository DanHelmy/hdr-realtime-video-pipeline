from __future__ import annotations

import cv2
import numpy as np
import queue as _queue
import time
import torch

from gui_scaling import BEST_CV2_INTERP, _letterbox_bgr, _apply_upscale_sharpen
from timer import sleep_until


class PipelineWorkerFrameProcessingMixin:
    """Per-frame HDR bypass / SDR infer processing for PipelineWorker."""

    @staticmethod
    def _scene_signature_bgr(frame: np.ndarray | None) -> np.ndarray | None:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            return None
        try:
            small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
            b = small[:, :, 0].astype(np.float32)
            g = small[:, :, 1].astype(np.float32)
            r = small[:, :, 2].astype(np.float32)
            return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        except Exception:
            return None

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

    def _apply_mpv_highlight_temporal_stabilizer(
        self,
        prepared_out,
        model_inp: np.ndarray | None,
    ):
        """Viewer-only smoothing for static flat highlights before mpv display."""
        if not isinstance(prepared_out, torch.Tensor):
            return prepared_out
        if prepared_out.ndim not in (3, 4):
            return prepared_out

        with torch.inference_mode():
            squeeze_batch = prepared_out.ndim == 3
            current = prepared_out.unsqueeze(0) if squeeze_batch else prepared_out
            if current.ndim != 4 or current.shape[1] != 3:
                return prepared_out

            scene = self._scene_signature_bgr(model_inp)
            scene_factor = 1.0
            prev_scene = getattr(self, "_flat_temporal_prev_scene", None)
            if scene is not None and isinstance(prev_scene, np.ndarray) and prev_scene.shape == scene.shape:
                scene_delta = float(np.mean(np.abs(scene - prev_scene)))
                if scene_delta > 0.12:
                    self._reset_enhance_history()
                    self._flat_temporal_prev_scene = scene
                    return prepared_out
                scene_factor = float(np.clip((0.075 - scene_delta) / 0.045, 0.0, 1.0))
            if scene is not None:
                self._flat_temporal_prev_scene = scene

            prev_rgb = getattr(self, "_flat_temporal_prev_rgb", None)
            prev_luma = getattr(self, "_flat_temporal_prev_luma", None)
            if (
                not isinstance(prev_rgb, torch.Tensor)
                or not isinstance(prev_luma, torch.Tensor)
                or tuple(prev_rgb.shape) != tuple(current.shape)
                or tuple(prev_luma.shape[-2:]) != tuple(current.shape[-2:])
            ):
                self._flat_temporal_prev_rgb = current.detach().clone()
                cur_luma = (
                    0.2126 * current[:, 0:1].float()
                    + 0.7152 * current[:, 1:2].float()
                    + 0.0722 * current[:, 2:3].float()
                )
                self._flat_temporal_prev_luma = cur_luma.detach().clone()
                return prepared_out

            cur_f = current.float().clamp(0.0, 1.0)
            prev_f = prev_rgb.to(device=current.device, dtype=torch.float32).clamp(0.0, 1.0)
            cur_luma = (
                0.2126 * cur_f[:, 0:1]
                + 0.7152 * cur_f[:, 1:2]
                + 0.0722 * cur_f[:, 2:3]
            )
            prev_luma_f = prev_luma.to(device=current.device, dtype=torch.float32)

            local_mean = self._box_blur(cur_luma, 5)
            local_var = self._box_blur((cur_luma - local_mean) ** 2, 5)
            flat_mask = torch.clamp((0.0025 - local_var) / 0.0025, 0.0, 1.0)
            highlight_mask = torch.clamp((cur_luma - 0.62) / 0.24, 0.0, 1.0)
            temporal_delta = (cur_luma - prev_luma_f).abs()
            stable_mask = torch.clamp((0.045 - temporal_delta) / 0.045, 0.0, 1.0)

            mask = highlight_mask * flat_mask * stable_mask
            if scene_factor <= 0.0:
                self._flat_temporal_prev_rgb = current.detach().clone()
                self._flat_temporal_prev_luma = cur_luma.detach().clone()
                return prepared_out

            # Avoid mask.max().item() here; a CPU readback would sync the GPU every frame.
            strength = (0.22 * scene_factor * mask).expand_as(cur_f)
            stabilized = cur_f * (1.0 - strength) + prev_f * strength
            stabilized = stabilized.clamp(0.0, 1.0).to(dtype=current.dtype)
            stable_luma = (
                0.2126 * stabilized[:, 0:1].float()
                + 0.7152 * stabilized[:, 1:2].float()
                + 0.0722 * stabilized[:, 2:3].float()
            )
            self._flat_temporal_prev_rgb = stabilized.detach().clone()
            self._flat_temporal_prev_luma = stable_luma.detach().clone()
            return stabilized.squeeze(0) if squeeze_batch else stabilized

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
    ) -> tuple[np.ndarray | None, np.ndarray, object, bool, float]:
        # File playback may carry an absolute presentation time. Live capture
        # uses the feeder's steady low-FPS clock and mpv's display sync.
        queue_present_t = present_t
        need_display_frame = self._input_is_hdr or self._sdr_visible or mpv_w is None
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

        if self._input_is_hdr:
            need_hdr_cpu = False
            prepared_out = None
            output = display_frame
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
                    self._queue_latest(self._sdr_queue, (queue_present_t, display_frame))
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
                display_out = self._apply_mpv_highlight_temporal_stabilizer(
                    prepared_out,
                    model_inp,
                )
                queued_tensor = display_out.clone()
                if use_cuda:
                    ready_event = torch.cuda.Event(enable_timing=False)
                    ready_event.record(torch.cuda.current_stream())
                self._queue_latest(
                    self._hdr_queue,
                    (queue_present_t, queued_tensor, ready_event),
                )

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
                self._queue_latest(self._sdr_queue, (queue_present_t, display_frame))

        need_hdr_cpu = (mpv_w is None)
        if need_hdr_cpu:
            output = self._render_hdr_output(prepared_out, out_w, out_h)
        else:
            output = display_frame if display_frame is not None else frame

        return display_frame, output, prepared_out, need_hdr_cpu, model_latency_ms
