from __future__ import annotations

import numpy as np
import torch

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
        if isinstance(compare_request, dict):
            compare_target = max(0, int(compare_request.get("frame", frame_idx)))
            req_gt = compare_request.get("hdr_gt_path")
            if isinstance(req_gt, str) and req_gt.strip():
                compare_gt_path = req_gt.strip()
            req_prec = compare_request.get("precision_key")
            if isinstance(req_prec, str) and req_prec.strip():
                compare_precision_key = req_prec.strip()
        elif compare_request is not None:
            compare_target = max(0, int(compare_request))

        if compare_target is None or frame_idx < int(compare_target):
            return

        cmp_idx = int(compare_target)
        cmp_note = ""
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

        cmp_source = frame
        cmp_sdr = None
        cmp_hdr_algo = None
        cmp_algo_precision = "Bypass" if self._input_is_hdr else str(self._precision_key)

        if cmp_sdr is None:
            cmp_source = _read_video_frame_at(self._video_path, cmp_idx)
            if cmp_source is None and cmp_idx > 0:
                cmp_source = _read_video_frame_at(self._video_path, cmp_idx - 1)
            if cmp_source is None:
                cmp_source = frame
                cmp_idx = int(frame_idx)
                cmp_note = (
                    "Source frame unavailable at requested position; "
                    "using current decoded frame."
                )
            cmp_sdr = np.ascontiguousarray(_letterbox_bgr(cmp_source, out_w, out_h))

        if cmp_hdr_algo is None and self._input_is_hdr:
            cmp_hdr_algo = np.ascontiguousarray(cmp_sdr.copy())
        elif cmp_hdr_algo is None:
            try:
                if lower_res_processing:
                    cmp_model_inp = _letterbox_bgr(cmp_source, proc_w, proc_h)
                else:
                    cmp_model_inp = cmp_sdr
                with torch.inference_mode():
                    cmp_tensor, cmp_cond = self._processor.preprocess(cmp_model_inp)
                    cmp_raw_out = self._processor.infer((cmp_tensor, cmp_cond))

                saved_enhance_state = self._capture_enhance_history()
                self._reset_enhance_history()
                try:
                    # Compare snapshots should be deterministic and isolated from
                    # live playback temporal/enhancement history.
                    cmp_prepared_out = self._prepare_hdr_output_tensor(
                        cmp_raw_out, lower_res_processing, True
                    )
                finally:
                    self._restore_enhance_history(saved_enhance_state)

                cmp_hdr_algo = np.ascontiguousarray(
                    self._render_hdr_output(cmp_prepared_out, out_w, out_h, copy_input=True)
                )
            except Exception as exc:
                cmp_hdr_algo = np.ascontiguousarray(cmp_sdr.copy())
                msg = f"HDR Convert snapshot failed ({exc}); using SDR fallback."
                cmp_note = f"{cmp_note} {msg}".strip()

        cmp_hdr_gt = None
        if compare_gt_path:
            gt_probe = _read_video_frame_at(compare_gt_path, cmp_idx)
            if gt_probe is None and cmp_idx > 0:
                gt_probe = _read_video_frame_at(compare_gt_path, cmp_idx - 1)
            if gt_probe is not None:
                cmp_hdr_gt = np.ascontiguousarray(_letterbox_bgr(gt_probe, out_w, out_h))
            else:
                msg = "HDR GT frame unavailable at this position."
                cmp_note = f"{cmp_note} {msg}".strip()
        elif gt_frame is not None:
            cmp_hdr_gt = np.ascontiguousarray(_letterbox_bgr(gt_frame, out_w, out_h))
        else:
            msg = "Select HDR GT video to include ground truth in compare view."
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
                cmp_metrics["obj_note"] = "Computed (raw + normalized)"
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
