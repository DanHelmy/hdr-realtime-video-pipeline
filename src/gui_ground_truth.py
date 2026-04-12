from __future__ import annotations

import os

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from gui_media_probe import (
    _content_similarity_score,
    _probe_hdr_input,
    _probe_video_timing_info,
)


class GroundTruthMixin:
    """HDR ground-truth file selection and compatibility checks for MainWindow."""

    def _update_hdr_ground_truth_label(self):
        if not hasattr(self, "_lbl_hdr_gt") or self._lbl_hdr_gt is None:
            return
        path = self._hdr_ground_truth_path
        if path and os.path.isfile(path):
            name = os.path.basename(path)
            self._lbl_hdr_gt.setText(f"HDR GT: {name}")
            self._lbl_hdr_gt.setToolTip(path)
            self._lbl_hdr_gt.setStyleSheet("color: #c2c6cb;")
        else:
            self._lbl_hdr_gt.setText("HDR GT: none")
            self._lbl_hdr_gt.setToolTip(
                "Select HDR ground-truth video for compare view."
            )
            self._lbl_hdr_gt.setStyleSheet("color: #a2a7ae;")

    def _reset_hdr_ground_truth(self, status_message: str | None = None):
        had_gt = bool(self._hdr_ground_truth_path)
        self._hdr_ground_truth_path = None
        self._objective_metrics_enabled = False
        self._update_hdr_ground_truth_label()
        self._save_user_settings()
        if self._playing:
            self._apply_objective_metrics_to_worker()
        if status_message and had_gt:
            self.statusBar().showMessage(status_message)

    def _validate_hdr_ground_truth(
        self,
        gt_path: str,
        source_path: str | None = None,
    ) -> tuple[bool, str]:
        if not gt_path or not os.path.isfile(gt_path):
            return False, "HDR GT file not found."

        hdr_info = _probe_hdr_input(gt_path)
        if not bool(hdr_info.get("is_hdr", False)):
            reason = str(hdr_info.get("reason", "HDR metadata not detected")).strip()
            return False, f"HDR GT must be an actual HDR video ({reason})."

        src_path = source_path or self._video_path
        if not src_path or not os.path.isfile(src_path):
            return False, "Open the SDR input video first, then select HDR GT."

        src_meta = _probe_video_timing_info(src_path)
        gt_meta = _probe_video_timing_info(gt_path)
        if src_meta is None or gt_meta is None:
            return False, "Unable to read video metadata for compatibility check."

        src_fps = float(src_meta.get("fps", 0.0) or 0.0)
        gt_fps = float(gt_meta.get("fps", 0.0) or 0.0)
        if src_fps > 0.0 and gt_fps > 0.0 and abs(src_fps - gt_fps) > 0.25:
            return False, f"FPS mismatch: source {src_fps:.3f} vs GT {gt_fps:.3f}."

        src_n = int(src_meta.get("frame_count", 0) or 0)
        gt_n = int(gt_meta.get("frame_count", 0) or 0)
        if src_n > 0 and gt_n > 0 and abs(src_n - gt_n) > 2:
            return False, f"Frame-count mismatch: source {src_n} vs GT {gt_n}."

        src_d = float(src_meta.get("duration_s", 0.0) or 0.0)
        gt_d = float(gt_meta.get("duration_s", 0.0) or 0.0)
        if src_d > 0.0 and gt_d > 0.0 and abs(src_d - gt_d) > 0.25:
            return False, f"Duration mismatch: source {src_d:.2f}s vs GT {gt_d:.2f}s."

        src_w = int(src_meta.get("width", 0) or 0)
        src_h = int(src_meta.get("height", 0) or 0)
        gt_w = int(gt_meta.get("width", 0) or 0)
        gt_h = int(gt_meta.get("height", 0) or 0)
        if src_w > 0 and src_h > 0 and gt_w > 0 and gt_h > 0:
            src_ar = float(src_w) / float(src_h)
            gt_ar = float(gt_w) / float(gt_h)
            if abs(src_ar - gt_ar) > 0.01:
                return (
                    False,
                    f"Aspect-ratio mismatch: source {src_w}x{src_h} vs GT {gt_w}x{gt_h}.",
                )

        content_score, sampled = _content_similarity_score(
            src_path, gt_path, sample_count=5
        )
        if content_score is None or sampled < 3:
            return False, "Could not verify content match from sampled frames."
        if content_score < 0.38:
            return (
                False,
                "Content mismatch: GT does not look like the same video "
                f"(similarity {content_score:.2f}).",
            )

        return True, f"Validated (same-content similarity {content_score:.2f})."

    def _apply_objective_metrics_to_worker(self):
        if self._worker is None:
            return
        self._worker.request_objective_metrics_config(
            bool(self._objective_metrics_enabled),
            self._hdr_ground_truth_path,
        )

    def _pick_hdr_ground_truth_file(self):
        if getattr(self, "_source_mode", "video") != "video":
            QMessageBox.information(
                self,
                "HDR Ground Truth",
                "HDR ground-truth compare is only available for file-based video playback.",
            )
            return
        if not self._video_path or not os.path.isfile(self._video_path):
            QMessageBox.information(
                self,
                "HDR Ground Truth",
                "Open the SDR input video first, then select HDR GT.",
            )
            return
        start_dir = (
            self._last_open_dir if os.path.isdir(self._last_open_dir) else os.getcwd()
        )
        if self._hdr_ground_truth_path and os.path.isfile(self._hdr_ground_truth_path):
            start_dir = os.path.dirname(self._hdr_ground_truth_path)
        elif self._video_path and os.path.isfile(self._video_path):
            start_dir = os.path.dirname(self._video_path)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDR Ground-Truth Video",
            start_dir,
            "Video (*.mp4 *.avi *.mkv *.mov *.webm *.flv);;All (*)",
        )
        if not path:
            return
        ok, note = self._validate_hdr_ground_truth(path, source_path=self._video_path)
        if not ok:
            QMessageBox.warning(self, "HDR Ground Truth", note)
            return
        self._hdr_ground_truth_path = path
        self._objective_metrics_enabled = False
        try:
            gt_dir = os.path.dirname(path)
            if gt_dir and os.path.isdir(gt_dir):
                self._last_open_dir = gt_dir
        except Exception:
            pass
        self._update_hdr_ground_truth_label()
        self._save_user_settings()
        self.statusBar().showMessage(
            f"HDR ground-truth set: {os.path.basename(path)} ({note}) — compare mode only."
        )
        if self._playing:
            self._apply_objective_metrics_to_worker()
