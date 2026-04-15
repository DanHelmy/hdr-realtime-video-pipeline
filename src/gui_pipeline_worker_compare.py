from __future__ import annotations

import numpy as np
import torch

from gui_hdr_io import read_hdr_video_frame_rgb16, tensor_to_bgr_u16
from gui_scaling import _letterbox_bgr
from gui_objective_metrics import (
    _OBJECTIVE_METRIC_MAX_SIDE,
    _read_video_frame_at,
    _prepare_metric_pair,
    _grade_normalize_pred_to_ref,
    _psnr_bgr,
    _ssim_bgr,
    _delta_e_itp_bgr,
    _hdrvdp3_cli_score,
)


class PipelineWorkerCompareMixin:
    """Compare-snapshot generation helpers for PipelineWorker."""

    def _cache_compare_state(
        self,
        *,
        frame_idx: int,
        frame,
        gt_frame,
        display_frame,
        output,
        prepared_out,
        need_hdr_cpu: bool,
        out_w: int,
        out_h: int,
        lower_res_processing: bool,
        proc_w: int,
        proc_h: int,
    ) -> None:
        self._compare_cached_state = {
            "frame_idx": int(frame_idx),
            "frame": frame,
            "gt_frame": gt_frame,
            "display_frame": display_frame,
            "output": output,
            "prepared_out": prepared_out,
            "need_hdr_cpu": bool(need_hdr_cpu),
            "out_w": int(out_w),
            "out_h": int(out_h),
            "lower_res_processing": bool(lower_res_processing),
            "proc_w": int(proc_w),
            "proc_h": int(proc_h),
        }

    def _try_emit_compare_snapshot_from_cache(self) -> bool:
        compare_request = self._pending_compare_snapshot
        cached = self._compare_cached_state
        if not isinstance(compare_request, dict) or not isinstance(cached, dict):
            return False
        try:
            target_frame = max(0, int(compare_request.get("frame", -1)))
            cached_frame = int(cached.get("frame_idx", -2))
            force_immediate = bool(compare_request.get("force_immediate", False))
        except Exception:
            return False
        if target_frame != cached_frame and not force_immediate:
            return False
        frame = cached.get("frame")
        if not isinstance(frame, np.ndarray):
            return False
        self._maybe_emit_compare_snapshot(
            frame_idx=cached_frame,
            frame=frame,
            gt_frame=cached.get("gt_frame"),
            display_frame=cached.get("display_frame"),
            output=cached.get("output"),
            prepared_out=cached.get("prepared_out"),
            need_hdr_cpu=bool(cached.get("need_hdr_cpu", False)),
            out_w=int(cached.get("out_w", 0) or 0),
            out_h=int(cached.get("out_h", 0) or 0),
            lower_res_processing=bool(cached.get("lower_res_processing", False)),
            proc_w=int(cached.get("proc_w", 0) or 0),
            proc_h=int(cached.get("proc_h", 0) or 0),
        )
        return self._pending_compare_snapshot is None

    def _maybe_emit_compare_snapshot(
        self,
        *,
        frame_idx: int,
        frame: np.ndarray,
        gt_frame: np.ndarray | None,
        display_frame: np.ndarray | None,
        output: np.ndarray | None,
        prepared_out,
        need_hdr_cpu: bool,
        out_w: int,
        out_h: int,
        lower_res_processing: bool,
        proc_w: int,
        proc_h: int,
    ) -> None:
        compare_request = self._pending_compare_snapshot
        compare_target = None
        compare_gt_path = self._hdr_ground_truth_path
        compare_precision_key = None
        force_immediate = False
        if isinstance(compare_request, dict):
            compare_target = max(0, int(compare_request.get("frame", frame_idx)))
            req_gt = compare_request.get("hdr_gt_path")
            if isinstance(req_gt, str) and req_gt.strip():
                compare_gt_path = req_gt.strip()
            req_prec = compare_request.get("precision_key")
            if isinstance(req_prec, str) and req_prec.strip():
                compare_precision_key = req_prec.strip()
            force_immediate = bool(compare_request.get("force_immediate", False))
        elif compare_request is not None:
            compare_target = max(0, int(compare_request))

        if compare_target is None or (
            (not force_immediate) and frame_idx < int(compare_target)
        ):
            return

        cmp_idx = int(compare_target)
        cmp_seek_idx = max(0, int(cmp_idx) - 1)
        cmp_note = ""
        compare_out_w, compare_out_h = int(proc_w), int(proc_h)
        if self._input_is_hdr:
            compare_out_w, compare_out_h = int(out_w), int(out_h)
        compare_proc_w, compare_proc_h = compare_out_w, compare_out_h
        compare_lower_res_processing = False
        runtime_precision_key = str(self._precision_key)
        compare_precision_swapped = False
        if (
            (not self._input_is_hdr)
            and compare_precision_key
            and compare_precision_key != self._precision_key
        ):
            if not self._load_model(
                compare_precision_key,
                announce_ready=False,
                compile_model=False,
            ):
                msg = (
                    f"Requested compare precision {compare_precision_key} "
                    f"unavailable; using {self._precision_key}."
                )
                cmp_note = f"{cmp_note} {msg}".strip()
            else:
                compare_precision_swapped = (
                    str(self._precision_key) != runtime_precision_key
                )
                if compare_precision_swapped:
                    self._reset_enhance_history()

        compare_is_current_frame = (cmp_idx == int(frame_idx))
        cmp_source = frame
        cmp_sdr = None
        cmp_hdr_algo = None
        cmp_algo_precision = "Bypass" if self._input_is_hdr else str(self._precision_key)

        if cmp_sdr is None:
            if not compare_is_current_frame and self._video_path:
                cmp_source = _read_video_frame_at(self._video_path, cmp_seek_idx)
                if cmp_source is None and cmp_seek_idx > 0:
                    cmp_source = _read_video_frame_at(
                        self._video_path, cmp_seek_idx - 1
                    )
                if cmp_source is None:
                    cmp_source = frame
                    cmp_idx = int(frame_idx)
                    cmp_seek_idx = max(0, int(cmp_idx) - 1)
                    compare_is_current_frame = True
                    cmp_note = (
                        "Source frame unavailable at requested position; "
                        "using current decoded frame."
                    )
            cmp_sdr = np.ascontiguousarray(
                _letterbox_bgr(cmp_source, compare_out_w, compare_out_h)
            )

        if cmp_hdr_algo is None and self._input_is_hdr:
            cmp_hdr_algo = np.ascontiguousarray(cmp_sdr.copy())
        elif (
            cmp_hdr_algo is None
            and compare_is_current_frame
            and not compare_precision_swapped
            and prepared_out is not None
        ):
            try:
                # Reuse the already-processed paused frame when possible so
                # compare does not trigger a second compile/infer pass.
                cmp_hdr_algo = np.ascontiguousarray(tensor_to_bgr_u16(prepared_out.clone()))
                if (cmp_hdr_algo.shape[1], cmp_hdr_algo.shape[0]) != (compare_out_w, compare_out_h):
                    cmp_hdr_algo = np.ascontiguousarray(
                        _letterbox_bgr(cmp_hdr_algo, compare_out_w, compare_out_h)
                    )
            except Exception as exc:
                msg = f"HDR Convert snapshot reuse failed ({exc}); rerendering."
                cmp_note = f"{cmp_note} {msg}".strip()
        if cmp_hdr_algo is None and not self._input_is_hdr:
            try:
                # Compare snapshots are rerendered at the active processing
                # resolution so SDR, HDR GT, and HDR Convert all align without
                # forcing a higher display-resolution compile tier.
                cmp_model_inp = _letterbox_bgr(
                    cmp_source, compare_proc_w, compare_proc_h
                )
                with torch.inference_mode():
                    cmp_tensor, cmp_cond = self._processor.preprocess(cmp_model_inp)
                    cmp_raw_out = self._processor.infer((cmp_tensor, cmp_cond))

                saved_enhance_state = self._capture_enhance_history()
                self._reset_enhance_history()
                try:
                    # Compare snapshots should be deterministic and isolated from
                    # live playback temporal/enhancement history.
                    cmp_prepared_out = self._prepare_hdr_output_tensor(
                        cmp_raw_out,
                        compare_lower_res_processing,
                    )
                finally:
                    self._restore_enhance_history(saved_enhance_state)

                cmp_hdr_algo = np.ascontiguousarray(tensor_to_bgr_u16(cmp_prepared_out))
                if (cmp_hdr_algo.shape[1], cmp_hdr_algo.shape[0]) != (compare_out_w, compare_out_h):
                    cmp_hdr_algo = np.ascontiguousarray(
                        _letterbox_bgr(cmp_hdr_algo, compare_out_w, compare_out_h)
                    )
            except Exception as exc:
                cmp_hdr_algo = np.ascontiguousarray(cmp_sdr.copy())
                msg = f"HDR Convert snapshot failed ({exc}); using SDR fallback."
                cmp_note = f"{cmp_note} {msg}".strip()

        cmp_hdr_gt = None
        gt_hdr_mode_note = "HDR fallback (8-bit/OpenCV)"
        if compare_gt_path:
            gt_rgb16 = read_hdr_video_frame_rgb16(compare_gt_path, cmp_seek_idx)
            if gt_rgb16 is None and cmp_seek_idx > 0:
                gt_rgb16 = read_hdr_video_frame_rgb16(compare_gt_path, cmp_seek_idx - 1)
            gt_probe = None
            if gt_rgb16 is not None:
                gt_probe = np.ascontiguousarray(gt_rgb16[:, :, ::-1])
                gt_hdr_mode_note = "true 16-bit HDR decode"
            else:
                gt_probe = _read_video_frame_at(compare_gt_path, cmp_seek_idx)
                if gt_probe is None and cmp_seek_idx > 0:
                    gt_probe = _read_video_frame_at(compare_gt_path, cmp_seek_idx - 1)
            if gt_probe is not None:
                cmp_hdr_gt = np.ascontiguousarray(
                    _letterbox_bgr(gt_probe, compare_out_w, compare_out_h)
                )
            else:
                msg = "HDR GT frame unavailable at this position."
                cmp_note = f"{cmp_note} {msg}".strip()
        elif gt_frame is not None:
            if isinstance(gt_frame, np.ndarray) and gt_frame.dtype == np.uint16:
                gt_hdr_mode_note = "true 16-bit HDR decode"
            cmp_hdr_gt = np.ascontiguousarray(
                _letterbox_bgr(gt_frame, compare_out_w, compare_out_h)
            )
        else:
            msg = "Select HDR GT video to include ground truth in compare view."
            cmp_note = f"{cmp_note} {msg}".strip()

        if (
            not self._input_is_hdr
            and (int(proc_w), int(proc_h)) != (int(out_w), int(out_h))
        ):
            msg = (
                f"Compare aligned at shared {compare_out_w}x{compare_out_h} "
                f"(display canvas is {int(out_w)}x{int(out_h)})."
            )
            cmp_note = f"{cmp_note} {msg}".strip()

        cmp_metrics = {
            "psnr_db": None,
            "sssim": None,
            "delta_e_itp": None,
            "psnr_norm_db": None,
            "sssim_norm": None,
            "delta_e_itp_norm": None,
            "hdr_vdp3": None,
            "obj_note": "",
            "hdr_vdp3_note": "",
        }
        algo_hdr_mode_note = (
            "true 16-bit HDR convert"
            if isinstance(cmp_hdr_algo, np.ndarray) and cmp_hdr_algo.dtype == np.uint16
            else "HDR convert fallback"
        )
        if isinstance(cmp_hdr_algo, np.ndarray) and isinstance(cmp_hdr_gt, np.ndarray):
            try:
                eval_pred, eval_ref = _prepare_metric_pair(
                    cmp_hdr_algo, cmp_hdr_gt, max_side=_OBJECTIVE_METRIC_MAX_SIDE
                )
                cmp_metrics["psnr_db"] = _psnr_bgr(eval_pred, eval_ref)
                cmp_metrics["sssim"] = _ssim_bgr(eval_pred, eval_ref)
                cmp_metrics["delta_e_itp"] = _delta_e_itp_bgr(eval_pred, eval_ref)
                norm_pred, norm_ref = _grade_normalize_pred_to_ref(eval_pred, eval_ref)
                cmp_metrics["psnr_norm_db"] = _psnr_bgr(norm_pred, norm_ref)
                cmp_metrics["sssim_norm"] = _ssim_bgr(norm_pred, norm_ref)
                cmp_metrics["delta_e_itp_norm"] = _delta_e_itp_bgr(norm_pred, norm_ref)
                cmp_metrics["obj_note"] = (
                    f"Computed (raw + normalized, {algo_hdr_mode_note}, {gt_hdr_mode_note})"
                )
            except Exception as exc:
                cmp_metrics["obj_note"] = f"Error ({exc})"
            vdp_score, vdp_note = _hdrvdp3_cli_score(cmp_hdr_algo, cmp_hdr_gt)
            if vdp_score is not None:
                cmp_metrics["hdr_vdp3"] = float(vdp_score)
            elif vdp_note:
                cmp_metrics["hdr_vdp3_note"] = str(vdp_note)
                msg = f"HDR-VDP3 unavailable: {vdp_note}"
                cmp_note = f"{cmp_note} {msg}".strip()
        elif cmp_hdr_gt is None:
            cmp_metrics["obj_note"] = "Need HDR GT"
        else:
            cmp_metrics["obj_note"] = "Unavailable"

        cmp_note = (
            f"{cmp_note} HDR path: {algo_hdr_mode_note}; HDR GT: {gt_hdr_mode_note}."
        ).strip()

        if (
            compare_precision_swapped
            and runtime_precision_key
            and str(self._precision_key) != runtime_precision_key
        ):
            if not self._load_model(runtime_precision_key, announce_ready=False):
                msg = (
                    "Failed to restore player precision after compare; "
                    f"continuing with {self._precision_key}."
                )
                cmp_note = f"{cmp_note} {msg}".strip()
            else:
                self._reset_enhance_history()

        self.compare_snapshot_ready.emit({
            "frame": int(cmp_idx),
            "sdr": cmp_sdr,
            "hdr_algo": cmp_hdr_algo,
            "hdr_gt": cmp_hdr_gt,
            "algo_precision": str(cmp_algo_precision),
            "note": cmp_note,
            "metrics": cmp_metrics,
        })
        self._pending_compare_snapshot = None
        if self._user_paused:
            # Compare requests temporarily wake the paused loop.
            self._pause_event.clear()
