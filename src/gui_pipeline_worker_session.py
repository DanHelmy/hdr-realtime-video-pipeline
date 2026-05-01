from __future__ import annotations

import os
import numpy as np

from video_source import VideoSource
from gui_objective_metrics import _RunningAverage
from gui_media_probe import _probe_video_sync_info


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
        self._reset_gt_sync_state()
        if gt_source is not None:
            try:
                gt_source.release()
            except Exception:
                pass
        return None

    def _reset_gt_sync_state(self) -> None:
        self._gt_sync_last_index = None
        self._gt_sync_last_frame = None

    @staticmethod
    def _map_frame_index_between_rates(
        frame_idx: int,
        source_fps: float,
        gt_fps: float,
        gt_frame_count: int = 0,
    ) -> int:
        idx = max(0, int(frame_idx))
        src_fps = float(source_fps or 0.0)
        ref_fps = float(gt_fps or 0.0)
        if src_fps > 0.0 and ref_fps > 0.0 and abs(src_fps - ref_fps) > 1e-3:
            idx = int(round((float(idx) / src_fps) * ref_fps))
        if gt_frame_count > 0:
            idx = min(idx, int(gt_frame_count) - 1)
        return max(0, idx)

    def _map_gt_frame_index(
        self,
        source: VideoSource,
        gt_source: VideoSource,
        source_frame_idx: int,
    ) -> int:
        idx = self._map_frame_index_between_rates(
            int(source_frame_idx),
            float(getattr(source, "fps", 0.0) or 0.0),
            float(getattr(gt_source, "fps", 0.0) or 0.0),
            int(getattr(gt_source, "frame_count", 0) or 0),
        )
        idx += int(getattr(self, "_gt_sync_offset_frames", 0) or 0)
        gt_frame_count = int(getattr(gt_source, "frame_count", 0) or 0)
        if gt_frame_count > 0:
            idx = min(idx, gt_frame_count - 1)
        return max(0, int(idx))

    def _read_synced_gt_frame(
        self,
        *,
        source: VideoSource,
        gt_source: VideoSource,
        source_frame_idx: int,
    ) -> tuple[bool, np.ndarray | None]:
        desired_idx = self._map_gt_frame_index(source, gt_source, source_frame_idx)
        last_idx = getattr(self, "_gt_sync_last_index", None)
        last_frame = getattr(self, "_gt_sync_last_frame", None)
        if last_idx is not None and int(last_idx) == int(desired_idx):
            return True, last_frame

        if (
            last_idx is None
            or int(desired_idx) <= int(last_idx)
            or int(desired_idx) > int(last_idx) + 1
        ):
            try:
                gt_source.seek(desired_idx)
            except Exception:
                return False, None

        frame = None
        decoded_idx = -1
        max_reads = 4
        while max_reads > 0:
            ret_gt, frame, decoded_idx = gt_source.read_with_index()
            if not ret_gt:
                return False, None
            if int(decoded_idx) >= int(desired_idx):
                break
            max_reads -= 1

        self._gt_sync_last_index = int(decoded_idx)
        self._gt_sync_last_frame = frame
        return True, frame

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

        self._gt_sync_offset_frames = 0
        source_path = str(getattr(self, "_video_path", "") or "")
        if source_path and os.path.isfile(source_path):
            try:
                sync_info = _probe_video_sync_info(source_path, gt_path, sample_count=3)
                self._gt_sync_offset_frames = int(
                    sync_info.get("offset_frames", 0) or 0
                )
                if self._gt_sync_offset_frames:
                    sync_s = float(sync_info.get("offset_s", 0.0) or 0.0)
                    self.status_message.emit(
                        f"HDR GT sync offset {self._gt_sync_offset_frames:+d} frames ({sync_s:+.3f}s)."
                    )
            except Exception:
                self._gt_sync_offset_frames = 0

        self._reset_gt_sync_state()
        if cur_frame_idx > 0:
            try:
                gt_source.seek(
                    self._map_gt_frame_index(source, gt_source, cur_frame_idx)
                )
            except Exception:
                pass
        if source.frame_count > 0 and gt_source.frame_count > 0:
            diff = abs(int(source.frame_count) - int(gt_source.frame_count))
            if diff > 2:
                src_d = float(getattr(source, "duration", 0.0) or 0.0)
                gt_d = float(getattr(gt_source, "duration", 0.0) or 0.0)
                delta = abs(src_d - gt_d) if src_d > 0.0 and gt_d > 0.0 else 0.0
                suffix = f" by {delta:.2f}s" if delta > 0.0 else ""
                self.status_message.emit(
                    f"HDR ground-truth length differs from input{suffix}; metrics use the overlapping timeline."
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
        ret_local, frame_local, source_idx, _source_pts_s = source.read_with_meta()
        if not ret_local:
            return False, None, None, gt_source, objective_note, objective_warned_gt_eof
        gt_frame_local = None
        if gt_source is not None:
            ret_gt, gt_frame_local = self._read_synced_gt_frame(
                source=source,
                gt_source=gt_source,
                source_frame_idx=source_idx,
            )
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
