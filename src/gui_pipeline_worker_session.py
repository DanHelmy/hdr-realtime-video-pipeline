from __future__ import annotations

import os
import numpy as np

from video_source import VideoSource
from gui_objective_metrics import _RunningAverage


class PipelineWorkerSessionMixin:
    """GT/objective session helpers for PipelineWorker run loop."""

    def _reset_objective_averages(self):
        return (
            _RunningAverage(),
            _RunningAverage(),
            _RunningAverage(),
            _RunningAverage(),
            "",
        )

    def _close_gt_source(self, gt_source: VideoSource | None) -> VideoSource | None:
        if gt_source is not None:
            try:
                gt_source.release()
            except Exception:
                pass
        return None

    def _open_gt_source(
        self,
        *,
        source: VideoSource,
        gt_source: VideoSource | None,
        cur_frame_idx: int = 0,
    ) -> tuple[VideoSource | None, str, bool]:
        gt_source = self._close_gt_source(gt_source)
        objective_warned_gt_eof = False
        if not self._objective_metrics_enabled:
            return gt_source, "Off", objective_warned_gt_eof

        gt_path = self._hdr_ground_truth_path
        if not gt_path:
            self.status_message.emit(
                "Objective metrics need an HDR ground-truth video. Click 'HDR GT ...' to select it."
            )
            return gt_source, "Need HDR ground-truth video", objective_warned_gt_eof
        if not os.path.isfile(gt_path):
            self.status_message.emit(
                f"HDR ground-truth file missing: {gt_path}"
            )
            return gt_source, "HDR ground-truth file not found", objective_warned_gt_eof
        try:
            gt_source = VideoSource(gt_path, prefetch=1)
        except Exception as exc:
            self.status_message.emit(
                f"Failed to open HDR ground-truth video: {exc}"
            )
            return None, "HDR ground-truth open failed", objective_warned_gt_eof

        if cur_frame_idx > 0:
            try:
                gt_source.seek(cur_frame_idx)
            except Exception:
                pass
        if source.frame_count > 0 and gt_source.frame_count > 0:
            diff = abs(int(source.frame_count) - int(gt_source.frame_count))
            if diff > 2:
                self.status_message.emit(
                    "HDR ground-truth frame count differs from input; metrics may drift."
                )
        if source.fps > 0 and gt_source.fps > 0:
            if abs(float(source.fps) - float(gt_source.fps)) > 0.25:
                self.status_message.emit(
                    "HDR ground-truth FPS differs from input; metrics may drift."
                )
        return gt_source, f"HDR GT: {os.path.basename(gt_path)}", objective_warned_gt_eof

    def _read_frame_pair(
        self,
        *,
        source: VideoSource,
        gt_source: VideoSource | None,
        objective_note: str,
        objective_warned_gt_eof: bool,
    ) -> tuple[
        bool,
        np.ndarray | None,
        np.ndarray | None,
        VideoSource | None,
        str,
        bool,
    ]:
        ret_local, frame_local = source.read()
        if not ret_local:
            return False, None, None, gt_source, objective_note, objective_warned_gt_eof
        gt_frame_local = None
        if gt_source is not None:
            ret_gt, gt_frame_local = gt_source.read()
            if not ret_gt:
                gt_source = self._close_gt_source(gt_source)
                objective_note = "HDR ground-truth ended early"
                if not objective_warned_gt_eof:
                    objective_warned_gt_eof = True
                    self.status_message.emit(
                        "HDR ground-truth video ended before input; objective metrics paused."
                    )
                gt_frame_local = None
        return (
            True,
            frame_local,
            gt_frame_local,
            gt_source,
            objective_note,
            objective_warned_gt_eof,
        )
