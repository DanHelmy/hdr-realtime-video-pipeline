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
    def _box_blur(x: torch.Tensor, k: int = 3) -> torch.Tensor:
        p = k // 2
        if k >= 9:
            # Large 2D box blurs are hot in HDR sky cleanup; separable passes keep
            # the same box-filter behavior but avoid an expensive k x k kernel.
            x = F.avg_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, p))
            return F.avg_pool2d(x, kernel_size=(k, 1), stride=1, padding=(p, 0))
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=p)

    @staticmethod
    def _env_float_clamped(
        name: str,
        default: float,
        *,
        min_value: float,
        max_value: float,
    ) -> float:
        try:
            value = float(os.environ.get(name, default))
        except Exception:
            value = float(default)
        if not np.isfinite(value):
            value = float(default)
        return float(np.clip(value, min_value, max_value))

    def _hdr_highlight_deband_pattern(self, current: torch.Tensor) -> torch.Tensor:
        h = int(current.shape[-2])
        w = int(current.shape[-1])
        key = (str(current.device), str(current.dtype), h, w)
        cache = getattr(self, "_hdr_highlight_deband_cache", None)
        if isinstance(cache, tuple) and len(cache) == 2 and cache[0] == key:
            pattern = cache[1]
        else:
            rng = np.random.default_rng(41731)
            tile = rng.random((64, 64)).astype(np.float32) - 0.5
            tile = tile - cv2.blur(tile, (5, 5))
            scale = float(np.max(np.abs(tile)))
            if scale > 1e-6:
                tile = tile / scale
            pattern = torch.from_numpy(tile).to(
                device=current.device,
                dtype=current.dtype,
            ).view(1, 1, 64, 64)
            pattern = pattern.repeat(1, 1, (h + 63) // 64, (w + 63) // 64)
            pattern = pattern[:, :, :h, :w].contiguous()
            self._hdr_highlight_deband_cache = (key, pattern)
        if int(current.shape[0]) != 1:
            pattern = pattern.expand(int(current.shape[0]), -1, -1, -1)
        return pattern

    def _apply_hdr_highlight_deband(
        self,
        current: torch.Tensor,
        cur_luma: torch.Tensor,
        *,
        quality_key: str,
    ) -> torch.Tensor:
        high_quality = quality_key in {
            "export",
            "high",
            "hq",
            "full",
            "highlight-high",
            "highlight-hq",
            "highlights-high",
            "deband-high",
            "dither-high",
        }
        default_strength = 0.0022 if high_quality else 0.0018
        strength = self._env_float_clamped(
            "HDRTVNET_HIGHLIGHT_DEBAND_STRENGTH",
            default_strength,
            min_value=0.0,
            max_value=0.01,
        )
        if strength <= 0.0:
            return current
        start = self._env_float_clamped(
            "HDRTVNET_HIGHLIGHT_DEBAND_START",
            0.56,
            min_value=0.0,
            max_value=1.0,
        )
        width = self._env_float_clamped(
            "HDRTVNET_HIGHLIGHT_DEBAND_RANGE",
            0.26,
            min_value=1e-3,
            max_value=1.0,
        )
        mask = torch.clamp((cur_luma - start) / width, 0.0, 1.0)
        mask = mask * mask * (3.0 - 2.0 * mask)
        pattern = self._hdr_highlight_deband_pattern(current)
        return (current + pattern * (strength * mask)).clamp(0.0, 1.0)

    def _apply_hdr_sky_lite_deblock(
        self,
        current: torch.Tensor,
        cur_luma: torch.Tensor,
    ) -> torch.Tensor:
        h = int(current.shape[-2])
        w = int(current.shape[-1])
        if h < 32 or w < 32:
            return current

        div = 6 if max(h, w) >= 1440 else 4
        small_h = max(24, h // div)
        small_w = max(24, w // div)
        if small_h >= h or small_w >= w:
            return current

        def _smooth01(x: torch.Tensor) -> torch.Tensor:
            return x * x * (3.0 - 2.0 * x)

        work = current.float()
        luma = cur_luma.float()
        small_rgb = F.interpolate(
            work,
            size=(small_h, small_w),
            mode="bilinear",
            align_corners=False,
        )
        small_luma = F.interpolate(
            luma,
            size=(small_h, small_w),
            mode="bilinear",
            align_corners=False,
        )
        small_mean = self._box_blur(small_luma, 3)
        small_var = self._box_blur((small_luma - small_mean) ** 2, 3)
        luma_mask = _smooth01(torch.clamp((small_luma - 0.28) / 0.50, 0.0, 1.0))
        flat_mask = _smooth01(torch.clamp((0.030 - small_var) / 0.030, 0.0, 1.0))
        warm_surface = _smooth01(
            torch.clamp(
                ((small_rgb[:, 0:1] - small_rgb[:, 2:3]) - 0.04) / 0.22,
                0.0,
                1.0,
            )
        )
        mask_small = self._box_blur(
            luma_mask * flat_mask * (1.0 - 0.70 * warm_surface),
            3,
        )
        blur_small = self._box_blur(small_rgb, 5)
        mask = F.interpolate(
            mask_small,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        blur = F.interpolate(
            blur_small,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        strength = self._env_float_clamped(
            "HDRTVNET_SKY_LITE_DEBLOCK_STRENGTH",
            0.10,
            min_value=0.0,
            max_value=0.35,
        )
        amount = strength * mask
        return (work * (1.0 - amount) + blur * amount).clamp(0.0, 1.0).to(
            dtype=current.dtype
        )

    def _reset_enhance_history(self):
        self._enh_prev_luma = None
        self._enh_temporal_detail = None
        self._flat_temporal_prev_rgb = None
        self._flat_temporal_prev_luma = None
        self._flat_temporal_prev_scene = None

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

    def _apply_hdr_flat_surface_cleanup(
        self,
        prepared_out,
        model_inp: np.ndarray | None,
        *,
        quality: str = "off",
    ):
        """Smooth flat sky/highlight artifacts for mpv playback and HDR export."""
        if not isinstance(prepared_out, torch.Tensor):
            return prepared_out
        if prepared_out.ndim not in (3, 4):
            return prepared_out

        with torch.inference_mode():
            quality_key = str(quality or "off").strip().lower()
            if quality_key in {"0", "off", "none", "disabled"}:
                return prepared_out
            if quality_key in {"1", "on", "yes", "true"}:
                quality_key = "highlight-high"
            highlight_only = quality_key in {
                "highlight",
                "highlights",
                "highlight-deband",
                "deband",
                "dither",
                "highlight-high",
                "highlight-hq",
                "highlights-high",
                "deband-high",
                "dither-high",
            }
            lite_quality = quality_key in {
                "lite",
                "fast",
                "sky-lite",
                "skylite",
                "playback",
                "balanced",
            }
            export_quality = quality_key in {"export", "high", "hq", "full"}
            if not (highlight_only or lite_quality or export_quality):
                quality_key = "highlight-high"
                highlight_only = True

            highlight_start = 0.34
            highlight_range = 0.46
            flat_var_limit = 0.035 if export_quality else 0.030
            stable_delta_limit = 0.085
            blend_strength = 0.34 if export_quality else 0.22
            mask_feather_kernel = 17 if export_quality else 11
            mask_feather_strength = 0.45 if export_quality else 0.25
            deblock_luma_start = 0.22
            deblock_luma_range = 0.50
            deblock_flat_var_limit = 0.060 if export_quality else 0.045
            deblock_kernel = 21 if export_quality else 11
            deblock_mask_feather_kernel = 31 if export_quality else 17
            deblock_strength = 0.44 if export_quality else 0.30
            edge_guard_limit = 0.018
            motion_guard_start = 0.018
            motion_guard_range = 0.070
            scene_cut_delta = 0.18
            scene_fade_delta = 0.12

            def _smooth01(x: torch.Tensor) -> torch.Tensor:
                return x * x * (3.0 - 2.0 * x)

            squeeze_batch = prepared_out.ndim == 3
            current = prepared_out.unsqueeze(0) if squeeze_batch else prepared_out
            if current.ndim != 4 or current.shape[1] != 3:
                return prepared_out

            if highlight_only:
                cur_f = current.clamp(0.0, 1.0)
                cur_luma = (
                    0.2126 * cur_f[:, 0:1]
                    + 0.7152 * cur_f[:, 1:2]
                    + 0.0722 * cur_f[:, 2:3]
                )
                debanded = self._apply_hdr_highlight_deband(
                    cur_f,
                    cur_luma,
                    quality_key=quality_key,
                )
                debanded = debanded.to(dtype=current.dtype)
                return debanded.squeeze(0) if squeeze_batch else debanded

            if lite_quality:
                cur_f = current.clamp(0.0, 1.0)
                cur_luma = (
                    0.2126 * cur_f[:, 0:1]
                    + 0.7152 * cur_f[:, 1:2]
                    + 0.0722 * cur_f[:, 2:3]
                )
                spatial_f = self._apply_hdr_sky_lite_deblock(cur_f, cur_luma)
                spatial_luma = (
                    0.2126 * spatial_f[:, 0:1]
                    + 0.7152 * spatial_f[:, 1:2]
                    + 0.0722 * spatial_f[:, 2:3]
                )
                debanded = self._apply_hdr_highlight_deband(
                    spatial_f,
                    spatial_luma,
                    quality_key=quality_key,
                )
                debanded = debanded.to(dtype=current.dtype)
                return debanded.squeeze(0) if squeeze_batch else debanded

            scene = self._scene_signature_bgr(model_inp)
            scene_factor = 1.0
            prev_scene = getattr(self, "_flat_temporal_prev_scene", None)
            if scene is not None and isinstance(prev_scene, np.ndarray) and prev_scene.shape == scene.shape:
                scene_delta = float(np.mean(np.abs(scene - prev_scene)))
                if scene_delta > scene_cut_delta:
                    self._reset_enhance_history()
                    self._flat_temporal_prev_scene = scene
                    scene_factor = 0.0
                else:
                    scene_factor = float(np.clip((scene_fade_delta - scene_delta) / scene_fade_delta, 0.0, 1.0))
            if scene is not None:
                self._flat_temporal_prev_scene = scene

            cur_f = current.float().clamp(0.0, 1.0)
            cur_luma = (
                0.2126 * cur_f[:, 0:1]
                + 0.7152 * cur_f[:, 1:2]
                + 0.0722 * cur_f[:, 2:3]
            )
            local_mean = self._box_blur(cur_luma, 5)
            local_var = self._box_blur((cur_luma - local_mean) ** 2, 5)
            local_edge = (cur_luma - local_mean).abs()

            flat_mask = _smooth01(torch.clamp((flat_var_limit - local_var) / flat_var_limit, 0.0, 1.0))
            highlight_mask = _smooth01(torch.clamp((cur_luma - highlight_start) / highlight_range, 0.0, 1.0))
            deblock_luma_mask = _smooth01(
                torch.clamp((cur_luma - deblock_luma_start) / deblock_luma_range, 0.0, 1.0)
            )
            deblock_flat_mask = _smooth01(
                torch.clamp((deblock_flat_var_limit - local_var) / deblock_flat_var_limit, 0.0, 1.0)
            )
            warm_surface = _smooth01(
                torch.clamp(((cur_f[:, 0:1] - cur_f[:, 2:3]) - 0.04) / 0.22, 0.0, 1.0)
            )
            sky_color_guard = 1.0 - 0.75 * warm_surface
            deblock_core = deblock_luma_mask * deblock_flat_mask * sky_color_guard
            deblock_feather = self._box_blur(deblock_core, deblock_mask_feather_kernel)
            deblock_mask = _smooth01(torch.clamp(deblock_core * 0.55 + deblock_feather * 0.70, 0.0, 1.0))

            # Masked blur averages only inside sky-like flat regions, reducing square pop-ins
            # without pulling brick/building colors across hard edges.
            deblock_weight = deblock_mask.expand_as(cur_f)
            deblock_denom = self._box_blur(deblock_mask, deblock_kernel).clamp_min(1e-4)
            deblock_blur = self._box_blur(cur_f * deblock_weight, deblock_kernel) / deblock_denom
            deblock_amount = (deblock_strength * deblock_mask).expand_as(cur_f)
            spatial_f = cur_f * (1.0 - deblock_amount) + deblock_blur * deblock_amount
            spatial_f = spatial_f.clamp(0.0, 1.0)
            spatial_luma = (
                0.2126 * spatial_f[:, 0:1]
                + 0.7152 * spatial_f[:, 1:2]
                + 0.0722 * spatial_f[:, 2:3]
            )
            prev_rgb = getattr(self, "_flat_temporal_prev_rgb", None)
            prev_luma = getattr(self, "_flat_temporal_prev_luma", None)
            if (
                not isinstance(prev_rgb, torch.Tensor)
                or not isinstance(prev_luma, torch.Tensor)
                or tuple(prev_rgb.shape) != tuple(current.shape)
                or tuple(prev_luma.shape[-2:]) != tuple(current.shape[-2:])
            ):
                spatial_out = self._apply_hdr_highlight_deband(
                    spatial_f,
                    spatial_luma,
                    quality_key=quality_key,
                ).to(dtype=current.dtype)
                self._flat_temporal_prev_rgb = spatial_out.detach().clone()
                self._flat_temporal_prev_luma = spatial_luma.detach().clone()
                return spatial_out.squeeze(0) if squeeze_batch else spatial_out

            prev_f = prev_rgb.to(device=current.device, dtype=torch.float32).clamp(0.0, 1.0)
            prev_luma_f = prev_luma.to(device=current.device, dtype=torch.float32)
            temporal_delta = (spatial_luma - prev_luma_f).abs()
            stable_mask = _smooth01(torch.clamp((stable_delta_limit - temporal_delta) / stable_delta_limit, 0.0, 1.0))
            edge_guard = _smooth01(torch.clamp((edge_guard_limit - local_edge) / edge_guard_limit, 0.0, 1.0))
            motion_guard = 1.0 - _smooth01(
                torch.clamp((temporal_delta - motion_guard_start) / motion_guard_range, 0.0, 1.0)
            )
            object_guard = self._box_blur(edge_guard * motion_guard, 5)

            mask = highlight_mask * flat_mask * stable_mask * edge_guard * motion_guard
            feather = self._box_blur(mask, mask_feather_kernel)
            mask = _smooth01(torch.clamp(mask * 0.65 + feather * mask_feather_strength, 0.0, 1.0))
            mask = mask * object_guard
            if scene_factor <= 0.0:
                spatial_out = self._apply_hdr_highlight_deband(
                    spatial_f,
                    spatial_luma,
                    quality_key=quality_key,
                ).to(dtype=current.dtype)
                self._flat_temporal_prev_rgb = spatial_out.detach().clone()
                self._flat_temporal_prev_luma = spatial_luma.detach().clone()
                return spatial_out.squeeze(0) if squeeze_batch else spatial_out

            # Avoid mask.max().item() here; a CPU readback would sync the GPU every frame.
            strength = (blend_strength * scene_factor * mask).expand_as(cur_f)
            stabilized = spatial_f * (1.0 - strength) + prev_f * strength
            stabilized = stabilized.clamp(0.0, 1.0)
            stable_luma = (
                0.2126 * stabilized[:, 0:1].float()
                + 0.7152 * stabilized[:, 1:2].float()
                + 0.0722 * stabilized[:, 2:3].float()
            )
            stabilized = self._apply_hdr_highlight_deband(
                stabilized,
                stable_luma,
                quality_key=quality_key,
            )
            self._flat_temporal_prev_rgb = stabilized.to(dtype=current.dtype).detach().clone()
            self._flat_temporal_prev_luma = stable_luma.detach().clone()
            stabilized = stabilized.to(dtype=current.dtype)
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
                display_out = self._apply_hdr_flat_surface_cleanup(
                    prepared_out,
                    model_inp,
                    quality=os.environ.get("HDRTVNET_PLAYBACK_HDR_CLEANUP", "highlight-high"),
                )
                queued_tensor = display_out if display_out is not prepared_out else display_out.clone()
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
