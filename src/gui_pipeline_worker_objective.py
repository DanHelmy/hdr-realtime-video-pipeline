from __future__ import annotations

import numpy as np

from gui_scaling import _letterbox_bgr
from gui_objective_metrics import (
    _OBJECTIVE_METRIC_SAMPLE_EVERY,
    _OBJECTIVE_HDRVDP3_SAMPLE_EVERY,
    _OBJECTIVE_METRIC_MAX_SIDE,
    _prepare_metric_pair,
    _psnr_bgr,
    _ssim_bgr,
    _delta_e_itp_bgr,
    _hdrvdp3_cli_score,
)


class PipelineWorkerObjectiveMixin:
    """Objective-metrics update helper for PipelineWorker."""

    def _update_objective_metrics(
        self,
        *,
        frame_idx: int,
        gt_source,
        gt_frame: np.ndarray | None,
        need_hdr_cpu: bool,
        output: np.ndarray | None,
        prepared_out,
        out_w: int,
        out_h: int,
        psnr_avg,
        sssim_avg,
        deitp_avg,
        hdr_vdp3_avg,
        objective_note: str,
        hdr_vdp3_note: str,
    ) -> tuple[str, str]:
        if self._objective_metrics_enabled:
            if gt_source is None and not self._hdr_ground_truth_path:
                objective_note = "Need HDR ground-truth video"
            do_metric_sample = (frame_idx % _OBJECTIVE_METRIC_SAMPLE_EVERY == 0)
            if do_metric_sample and gt_frame is not None:
                metric_pred = None
                if need_hdr_cpu or self._input_is_hdr:
                    metric_pred = output
                elif (not self._input_is_hdr) and prepared_out is not None:
                    try:
                        metric_pred = self._render_hdr_output(
                            prepared_out,
                            out_w,
                            out_h,
                            copy_input=True,
                        )
                    except Exception as exc:
                        objective_note = f"Objective metric postprocess failed: {exc}"
                        metric_pred = None

                if metric_pred is not None:
                    metric_ref = _letterbox_bgr(gt_frame, out_w, out_h)
                    try:
                        eval_pred, eval_ref = _prepare_metric_pair(
                            metric_pred, metric_ref, max_side=_OBJECTIVE_METRIC_MAX_SIDE
                        )
                        psnr_avg.update(_psnr_bgr(eval_pred, eval_ref))
                        sssim_avg.update(_ssim_bgr(eval_pred, eval_ref))
                        deitp_avg.update(_delta_e_itp_bgr(eval_pred, eval_ref))
                        objective_note = "Running"
                    except Exception as exc:
                        objective_note = f"Objective metric error: {exc}"

                    if frame_idx % _OBJECTIVE_HDRVDP3_SAMPLE_EVERY == 0:
                        vdp_score, vdp_note = _hdrvdp3_cli_score(metric_pred, metric_ref)
                        if vdp_score is not None:
                            hdr_vdp3_avg.update(vdp_score)
                            hdr_vdp3_note = ""
                        elif vdp_note:
                            hdr_vdp3_note = vdp_note
            elif gt_source is not None and self._hdr_ground_truth_path:
                # Keep objective status text width-stable for UI layout.
                objective_note = "Running"
        else:
            objective_note = "Off"

        return objective_note, hdr_vdp3_note
