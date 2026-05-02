from __future__ import annotations

import os

from PyQt6.QtCore import QObject, QThread, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

from gui_media_probe import (
    _probe_hdr_input,
    _probe_video_active_area_info,
    _probe_video_sync_info,
    _probe_video_timing_info,
    _validate_video_timing_compatibility,
)


class _GroundTruthValidationCanceled(RuntimeError):
    pass


def _ground_truth_cancel_requested(cancel_check) -> bool:
    if not callable(cancel_check):
        return False
    try:
        return bool(cancel_check())
    except Exception:
        return False


def _raise_if_ground_truth_canceled(cancel_check) -> None:
    if _ground_truth_cancel_requested(cancel_check):
        raise _GroundTruthValidationCanceled()


def _validate_hdr_ground_truth_pair(
    gt_path: str,
    source_path: str | None = None,
    *,
    cancel_check=None,
) -> tuple[bool, str]:
    if not gt_path or not os.path.isfile(gt_path):
        return False, "HDR GT file not found."

    _raise_if_ground_truth_canceled(cancel_check)
    hdr_info = _probe_hdr_input(gt_path)
    if not bool(hdr_info.get("is_hdr", False)):
        reason = str(hdr_info.get("reason", "HDR metadata not detected")).strip()
        return False, f"HDR GT must be an actual HDR video ({reason})."

    src_path = source_path
    if not src_path or not os.path.isfile(src_path):
        return False, "Open the SDR input video first, then select HDR GT."

    _raise_if_ground_truth_canceled(cancel_check)
    src_meta = _probe_video_timing_info(src_path)
    gt_meta = _probe_video_timing_info(gt_path)
    ok, timing_error, notes = _validate_video_timing_compatibility(
        src_meta,
        gt_meta,
        source_label="source",
        gt_label="GT",
        metadata_error_message="Unable to read video metadata for compatibility check.",
        enforce_sync_tolerance=False,
    )
    if not ok:
        return False, str(
            timing_error or "Unable to read video metadata for compatibility check."
        )

    src_w = int(src_meta.get("width", 0) or 0)
    src_h = int(src_meta.get("height", 0) or 0)
    gt_w = int(gt_meta.get("width", 0) or 0)
    gt_h = int(gt_meta.get("height", 0) or 0)
    if src_w > 0 and src_h > 0 and gt_w > 0 and gt_h > 0:
        src_ar = float(src_w) / float(src_h)
        gt_ar = float(gt_w) / float(gt_h)
        if abs(src_ar - gt_ar) > 0.01:
            _raise_if_ground_truth_canceled(cancel_check)
            src_active = _probe_video_active_area_info(
                src_path,
                sample_count=5,
                cancel_check=cancel_check,
            )
            _raise_if_ground_truth_canceled(cancel_check)
            gt_active = _probe_video_active_area_info(
                gt_path,
                sample_count=5,
                cancel_check=cancel_check,
            )
            _raise_if_ground_truth_canceled(cancel_check)
            src_active_ar = (
                float(src_active.get("active_aspect", 0.0) or 0.0)
                if isinstance(src_active, dict)
                else 0.0
            )
            gt_active_ar = (
                float(gt_active.get("active_aspect", 0.0) or 0.0)
                if isinstance(gt_active, dict)
                else 0.0
            )
            if (
                src_active_ar > 0.0
                and gt_active_ar > 0.0
                and abs(src_active_ar - gt_active_ar) <= 0.04
            ):
                src_aw = int(src_active.get("active_width", src_w))
                src_ah = int(src_active.get("active_height", src_h))
                gt_aw = int(gt_active.get("active_width", gt_w))
                gt_ah = int(gt_active.get("active_height", gt_h))
                notes.append(
                    "active picture aspect matches after black-bar crop "
                    f"({src_aw}x{src_ah} vs {gt_aw}x{gt_ah})"
                )
            else:
                return (
                    False,
                    f"Aspect-ratio mismatch: source {src_w}x{src_h} vs GT {gt_w}x{gt_h}.",
                )

    _raise_if_ground_truth_canceled(cancel_check)
    sync_info = _probe_video_sync_info(
        src_path,
        gt_path,
        sample_count=3,
        cancel_check=cancel_check,
    )
    if bool(sync_info.get("canceled", False)):
        raise _GroundTruthValidationCanceled()
    _raise_if_ground_truth_canceled(cancel_check)
    content_score = sync_info.get("score")
    sampled = int(sync_info.get("sampled", 0) or 0)
    if content_score is None or sampled < 3:
        return False, "Could not verify content match from sampled frames."
    content_score = float(content_score)
    if content_score < 0.38:
        return (
            False,
            "Content mismatch: GT does not look like the same video "
            f"(similarity {content_score:.2f}).",
        )
    try:
        sync_offset_frames = int(sync_info.get("offset_frames", 0) or 0)
        sync_offset_s = float(sync_info.get("offset_s", 0.0) or 0.0)
    except Exception:
        sync_offset_frames = 0
        sync_offset_s = 0.0
    if sync_offset_frames:
        notes.append(
            f"GT sync offset {sync_offset_frames:+d} frames ({sync_offset_s:+.3f}s)"
        )

    suffix = ""
    if notes:
        suffix = "; " + "; ".join(notes)
    return True, f"Validated (same-content similarity {content_score:.2f}{suffix})."


class _GroundTruthValidationWorker(QObject):
    resolved = pyqtSignal(int, str, bool, str)
    canceled = pyqtSignal(int)
    failed = pyqtSignal(int, str)

    def __init__(self, generation: int, gt_path: str, source_path: str):
        super().__init__()
        self._generation = int(generation)
        self._gt_path = str(gt_path or "")
        self._source_path = str(source_path or "")
        self._cancel_requested = False

    def _is_canceled(self) -> bool:
        return bool(self._cancel_requested)

    @pyqtSlot()
    def cancel(self):
        self._cancel_requested = True

    @pyqtSlot()
    def run(self):
        if self._is_canceled():
            self.canceled.emit(self._generation)
            return
        try:
            ok, note = _validate_hdr_ground_truth_pair(
                self._gt_path,
                self._source_path,
                cancel_check=self._is_canceled,
            )
        except _GroundTruthValidationCanceled:
            self.canceled.emit(self._generation)
            return
        except Exception as exc:
            self.failed.emit(
                self._generation,
                str(exc).strip() or "HDR ground-truth validation failed.",
            )
            return
        if self._is_canceled():
            self.canceled.emit(self._generation)
            return
        self.resolved.emit(self._generation, self._gt_path, bool(ok), str(note or ""))


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
        return _validate_hdr_ground_truth_pair(
            gt_path,
            source_path or self._video_path,
        )

    def _apply_objective_metrics_to_worker(self):
        if self._worker is None:
            return
        self._worker.request_objective_metrics_config(
            bool(self._objective_metrics_enabled),
            self._hdr_ground_truth_path,
        )

    def _create_hdr_ground_truth_progress_dialog(self, gt_path: str | None = None):
        if self._hdr_gt_progress_dlg is not None:
            return self._hdr_gt_progress_dlg
        label = "Loading HDR ground-truth ..."
        if gt_path:
            label = f"Loading HDR ground-truth: {os.path.basename(gt_path)} ..."
        progress = QProgressDialog(label, "Cancel", 0, 0, self)
        progress.setWindowTitle("HDR Ground Truth")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.canceled.connect(self._request_cancel_hdr_ground_truth_validation)
        progress.show()
        self._hdr_gt_progress_dlg = progress
        return progress

    def _close_hdr_ground_truth_progress_dialog(self):
        progress = self._hdr_gt_progress_dlg
        self._hdr_gt_progress_dlg = None
        if progress is None:
            return
        try:
            progress.blockSignals(True)
            progress.close()
            progress.deleteLater()
        except Exception:
            pass

    def _resume_after_hdr_ground_truth_validation_if_needed(self):
        should_resume = bool(
            getattr(self, "_hdr_gt_resume_after_validation", False)
        )
        self._hdr_gt_resume_after_validation = False
        if not should_resume:
            return
        if not self._playing or self._worker is None:
            return

        def _resume():
            if self._playing and self._worker is not None and self._worker.is_paused:
                self._toggle_pause()

        QTimer.singleShot(0, _resume)

    def _cleanup_hdr_ground_truth_validation(self):
        thread = getattr(self, "_hdr_gt_validation_thread", None)
        if thread is not None and thread.isRunning():
            return
        self._hdr_gt_validation_thread = None
        self._hdr_gt_validation_worker = None

    def _request_cancel_hdr_ground_truth_validation(self):
        worker = getattr(self, "_hdr_gt_validation_worker", None)
        if worker is None:
            self._close_hdr_ground_truth_progress_dialog()
            self._resume_after_hdr_ground_truth_validation_if_needed()
            return
        if self._hdr_gt_progress_dlg is not None:
            try:
                self._hdr_gt_progress_dlg.setLabelText(
                    "Canceling HDR ground-truth load ..."
                )
                self._hdr_gt_progress_dlg.setCancelButton(None)
            except Exception:
                pass
        try:
            worker.cancel()
        except Exception:
            pass
        self.statusBar().showMessage("Canceling HDR ground-truth load ...")

    def _cancel_hdr_ground_truth_validation(
        self,
        *,
        wait: bool = False,
        invalidate: bool = False,
    ):
        worker = getattr(self, "_hdr_gt_validation_worker", None)
        thread = getattr(self, "_hdr_gt_validation_thread", None)
        if worker is None and thread is None:
            return
        if invalidate:
            self._hdr_gt_validation_generation = (
                int(getattr(self, "_hdr_gt_validation_generation", 0)) + 1
            )
        self._hdr_gt_resume_after_validation = False
        try:
            if worker is not None:
                worker.cancel()
        except Exception:
            pass
        self._close_hdr_ground_truth_progress_dialog()
        if wait and thread is not None:
            try:
                thread.quit()
            except Exception:
                pass
            try:
                thread.wait(5000)
            except Exception:
                pass
            self._cleanup_hdr_ground_truth_validation()

    def _begin_hdr_ground_truth_validation(self, path: str):
        if getattr(self, "_hdr_gt_validation_thread", None) is not None:
            self.statusBar().showMessage(
                "HDR ground-truth load already in progress ..."
            )
            return

        source_path = self._video_path
        if not source_path or not os.path.isfile(source_path):
            QMessageBox.information(
                self,
                "HDR Ground Truth",
                "Open the SDR input video first, then select HDR GT.",
            )
            return

        should_resume = bool(
            self._playing
            and self._worker is not None
            and not self._worker.is_paused
        )
        if should_resume:
            self._toggle_pause()
        self._hdr_gt_resume_after_validation = should_resume
        self._hdr_gt_validation_generation = (
            int(getattr(self, "_hdr_gt_validation_generation", 0)) + 1
        )
        generation = int(self._hdr_gt_validation_generation)
        self._create_hdr_ground_truth_progress_dialog(path)

        thread = QThread(self)
        worker = _GroundTruthValidationWorker(generation, path, source_path)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.resolved.connect(self._on_hdr_ground_truth_validation_resolved)
        worker.failed.connect(self._on_hdr_ground_truth_validation_failed)
        worker.canceled.connect(self._on_hdr_ground_truth_validation_canceled)
        worker.resolved.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.canceled.connect(thread.quit)
        worker.resolved.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.canceled.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_hdr_ground_truth_validation)

        self._hdr_gt_validation_thread = thread
        self._hdr_gt_validation_worker = worker
        thread.start()
        self.statusBar().showMessage(
            f"Loading HDR ground-truth: {os.path.basename(path)} ..."
        )

    def _on_hdr_ground_truth_validation_resolved(
        self,
        generation: int,
        path: str,
        ok: bool,
        note: str,
    ):
        if int(generation) != int(getattr(self, "_hdr_gt_validation_generation", 0)):
            return
        self._close_hdr_ground_truth_progress_dialog()
        if not ok:
            self._resume_after_hdr_ground_truth_validation_if_needed()
            QMessageBox.warning(
                self,
                "HDR Ground Truth",
                str(note or "HDR ground-truth validation failed."),
            )
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
            f"HDR ground-truth set: {os.path.basename(path)} ({note}) - compare mode only."
        )
        if self._playing:
            self._apply_objective_metrics_to_worker()
        self._resume_after_hdr_ground_truth_validation_if_needed()

    def _on_hdr_ground_truth_validation_canceled(self, generation: int):
        if int(generation) != int(getattr(self, "_hdr_gt_validation_generation", 0)):
            return
        self._close_hdr_ground_truth_progress_dialog()
        self.statusBar().showMessage("HDR ground-truth load canceled.")
        self._resume_after_hdr_ground_truth_validation_if_needed()

    def _on_hdr_ground_truth_validation_failed(
        self,
        generation: int,
        message: str,
    ):
        if int(generation) != int(getattr(self, "_hdr_gt_validation_generation", 0)):
            return
        self._close_hdr_ground_truth_progress_dialog()
        self.statusBar().showMessage("HDR ground-truth load failed.")
        self._resume_after_hdr_ground_truth_validation_if_needed()
        QMessageBox.warning(
            self,
            "HDR Ground Truth",
            str(message or "HDR ground-truth validation failed."),
        )

    def _pick_hdr_ground_truth_file(self):
        if getattr(self, "_hdr_gt_validation_thread", None) is not None:
            self.statusBar().showMessage(
                "HDR ground-truth load already in progress ..."
            )
            return
        if bool(getattr(self, "_export_interaction_locked", False)):
            self.statusBar().showMessage(
                "HDR GT selection is locked while export is running. Finish or cancel the export first."
            )
            return
        if bool(getattr(self, "_benchmark_interaction_locked", False)):
            self.statusBar().showMessage(
                "HDR GT selection is locked while benchmark is open. Close benchmark first."
            )
            return
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
        self._begin_hdr_ground_truth_validation(path)
