from __future__ import annotations

import cv2
import numpy as np
import queue as _queue
import time
import torch

from gui_scaling import BEST_CV2_INTERP, _letterbox_bgr, _apply_upscale_sharpen


class PipelineWorkerFrameProcessingMixin:
    """Per-frame HDR bypass / SDR infer processing for PipelineWorker."""

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
    ) -> tuple[np.ndarray, np.ndarray, object, bool]:
        # Match display/output resolution without stretch (black bars as needed).
        display_frame = _letterbox_bgr(frame, out_w, out_h)

        # Downscale for model if processing at lower resolution.
        if lower_res_processing:
            model_inp = _letterbox_bgr(display_frame, proc_w, proc_h)
        else:
            model_inp = display_frame

        if self._input_is_hdr:
            # Pass-through mode for HDR input: do not run SDR->HDR model.
            need_hdr_cpu = False
            raw_out = None
            output = display_frame
            if (output.shape[1], output.shape[0]) != (out_w, out_h):
                output = cv2.resize(output, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if mpv_w is not None:
                rgb16 = np.ascontiguousarray(
                    output[:, :, ::-1].astype(np.uint16) * 257
                )
                now_t = time.perf_counter()
                if now_t < present_t:
                    time.sleep(present_t - now_t)
                mpv_w.feed_frame(rgb16.data)
            if (
                self._sdr_mpv_widget is not None
                and self._sdr_queue is not None
                and self._sdr_visible
            ):
                if frame_idx < self._sdr_drop_until_frame:
                    pass
                else:
                    self._sdr_drop_until_frame = 0
                if self._sdr_drop_until_frame == 0:
                    try:
                        self._sdr_queue.put_nowait((present_t, display_frame))
                    except _queue.Full:
                        try:
                            self._sdr_queue.get_nowait()
                        except _queue.Empty:
                            pass
                        try:
                            self._sdr_queue.put_nowait((present_t, display_frame))
                        except _queue.Full:
                            pass
            if self._sdr_visible:
                need_hdr_cpu = True
            return display_frame, output, raw_out, need_hdr_cpu

        with torch.inference_mode():
            tensor, cond = self._processor.preprocess(model_inp)
            raw_out = self._processor.infer((tensor, cond))

        if mpv_w is not None and self._hdr_queue is not None:
            t_raw = (raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out)
            if lower_res_processing:
                t_raw = self._enhance_best_gpu(t_raw)
            if frame_idx < self._hdr_drop_until_frame:
                pass
            else:
                self._hdr_drop_until_frame = 0
            if self._hdr_drop_until_frame == 0:
                try:
                    self._hdr_queue.put_nowait((present_t, t_raw.clone()))
                except _queue.Full:
                    try:
                        self._hdr_queue.get_nowait()
                    except _queue.Empty:
                        pass
                    try:
                        self._hdr_queue.put_nowait((present_t, t_raw.clone()))
                    except _queue.Full:
                        pass

        if (
            self._sdr_mpv_widget is not None
            and self._sdr_queue is not None
            and self._sdr_visible
        ):
            if frame_idx < self._sdr_drop_until_frame:
                pass
            else:
                self._sdr_drop_until_frame = 0
            if self._sdr_drop_until_frame == 0:
                try:
                    self._sdr_queue.put_nowait((present_t, display_frame))
                except _queue.Full:
                    try:
                        self._sdr_queue.get_nowait()
                    except _queue.Empty:
                        pass
                    try:
                        self._sdr_queue.put_nowait((present_t, display_frame))
                    except _queue.Full:
                        pass

        # Only run the expensive postprocess (GPU->CPU D2H) when
        # the QLabel HDR fallback is the display path.
        need_hdr_cpu = (mpv_w is None)
        if need_hdr_cpu:
            output = self._processor.postprocess(raw_out)
            # postprocess() already calls stream.synchronize().
            if (output.shape[1], output.shape[0]) != (out_w, out_h):
                if out_w > output.shape[1] or out_h > output.shape[0]:
                    output = cv2.resize(output, (out_w, out_h), interpolation=BEST_CV2_INTERP)
                    output = _apply_upscale_sharpen(output)
                else:
                    output = cv2.resize(output, (out_w, out_h), interpolation=cv2.INTER_AREA)
        elif use_cuda:
            # Sync for timing only (no D2H to wait for).
            torch.cuda.synchronize()
            output = display_frame
        else:
            output = display_frame

        return display_frame, output, raw_out, need_hdr_cpu
