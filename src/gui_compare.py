from __future__ import annotations

import glob
import os
import shutil
import webbrowser

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QMessageBox

from gui_mpv_widget import MpvHDRWidget
from gui_scaling import (
    BEST_MPV_SCALE,
    FSR_SHADER_PATH,
    FILMGRAIN_SHADER_PATH,
    SSIM_SUPERRES_SHADER_PATH,
    _ensure_filmgrain_shader,
    _ensure_fsr_shader,
    _ensure_ssim_superres_shader,
    _normalize_shader_paths,
)
from gui_widgets import CompareFrameDialog

try:
    import mpv as mpv_lib

    _HAS_MPV = True
except (OSError, ImportError):
    mpv_lib = None
    _HAS_MPV = False

_MPV_DIAG = os.environ.get("HDRTVNET_MPV_DIAG", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_OCTAVE_DOWNLOAD_URL = "https://octave.org/download#ms-windows"


def _octave_executable() -> str | None:
    for name in ("octave-cli", "octave", "octave-cli.exe", "octave.exe"):
        p = shutil.which(name)
        if p:
            return p
    if os.name == "nt":
        patterns = [
            r"C:\Program Files\GNU Octave\Octave-*\mingw64\bin\octave-cli.exe",
            r"C:\Program Files\GNU Octave\Octave-*\mingw64\bin\octave.exe",
            r"C:\Program Files\Octave\Octave-*\mingw64\bin\octave-cli.exe",
            r"C:\Program Files\Octave\Octave-*\mingw64\bin\octave.exe",
        ]
        for pat in patterns:
            hits = sorted(glob.glob(pat), reverse=True)
            for hit in hits:
                if os.path.isfile(hit):
                    return hit
    return None


class CompareViewMixin:
    """Compare-view related helpers for MainWindow."""

    def _default_compare_frame_anchor(self) -> int | None:
        worker = getattr(self, "_worker", None)
        if worker is None:
            return None
        if worker.is_paused:
            queued = getattr(self, "_pending_seek_on_resume", None)
            if queued is not None:
                return max(0, int(queued))
            if hasattr(self, "_sync_anchor_frame"):
                try:
                    return max(0, int(self._sync_anchor_frame()))
                except Exception:
                    pass
            return max(0, int(getattr(self, "_last_seek_frame", 0)))
        return None

    def _compare_precision_options(self) -> list[str]:
        if not hasattr(self, "_cmb_prec") or self._cmb_prec is None:
            return []
        return [
            self._cmb_prec.itemText(i)
            for i in range(self._cmb_prec.count())
            if str(self._cmb_prec.itemText(i) or "").strip()
        ]

    def _new_mpv_widget(self) -> MpvHDRWidget:
        return MpvHDRWidget(
            mpv_lib=mpv_lib,
            mpv_diag=_MPV_DIAG,
            normalize_shader_paths=_normalize_shader_paths,
            ensure_fsr_shader=_ensure_fsr_shader,
            ensure_ssim_superres_shader=_ensure_ssim_superres_shader,
            ensure_filmgrain_shader=_ensure_filmgrain_shader,
            best_mpv_scale=BEST_MPV_SCALE,
            fsr_shader_path=FSR_SHADER_PATH,
            ssim_superres_shader_path=SSIM_SUPERRES_SHADER_PATH,
            filmgrain_shader_path=FILMGRAIN_SHADER_PATH,
        )

    def _ensure_compare_dialog(self) -> CompareFrameDialog:
        if self._compare_dialog is None:
            self._compare_dialog = CompareFrameDialog(
                mpv_available=_HAS_MPV,
                mpv_widget_factory=self._new_mpv_widget if _HAS_MPV else None,
                best_mpv_scale=BEST_MPV_SCALE,
                parent=self,
            )
            self._compare_dialog.setModal(False)
            self._compare_dialog.compare_requested.connect(
                self._on_compare_requested
            )
        current_prec = str(self._active_precision or self._cmb_prec.currentText() or "").strip()
        self._compare_dialog.set_precision_options(
            self._compare_precision_options(),
            selected=current_prec or None,
        )
        min_frame, max_frame = self._compare_frame_bounds()
        current_frame = int(
            getattr(
                self,
                "_compare_anchor_frame",
                getattr(self, "_last_seek_frame", min_frame),
            )
            or min_frame
        )
        self._compare_dialog.set_frame_bounds(
            min_frame=min_frame,
            max_frame=max_frame,
            current_frame=current_frame,
        )
        return self._compare_dialog

    def _compare_frame_bounds(self) -> tuple[int, int]:
        if getattr(self, "_source_mode", "video") != "video":
            current = max(
                0,
                int(
                    getattr(
                        self,
                        "_compare_anchor_frame",
                        getattr(self, "_last_seek_frame", 0),
                    )
                    or 0
                ),
            )
            return current, current
        max_frame = 0
        if hasattr(self, "_seek_slider") and self._seek_slider is not None:
            try:
                max_frame = max(0, int(self._seek_slider.maximum()) + 1)
            except Exception:
                max_frame = 0
        if max_frame <= 0:
            try:
                max_frame = max(
                    0,
                    int(
                        getattr(
                            self,
                            "_compare_anchor_frame",
                            getattr(self, "_last_seek_frame", 0),
                        )
                    ),
                )
            except Exception:
                max_frame = 0
        min_frame = 1 if max_frame > 0 else 0
        return min_frame, max_frame

    def _request_compare_snapshot(
        self,
        precision_key: str | None = None,
        frame_number: int | None = None,
    ):
        if bool(getattr(self, "_export_interaction_locked", False)):
            self.statusBar().showMessage(
                "Compare is locked while export is running. Finish or cancel the export first."
            )
            return
        if not self._playing:
            self.statusBar().showMessage("Start playback first, then use Compare.")
            return
        if self._compare_snapshot_pending:
            self.statusBar().showMessage("Compare snapshot already in progress ...")
            return

        self._warn_if_octave_missing_for_compare()

        # Pause first to freeze playback context for deterministic compare.
        if not self._worker.is_paused:
            self._toggle_pause()

        anchor_frame = frame_number
        preserve_resume_seek = False
        if anchor_frame is None:
            anchor_frame = self._default_compare_frame_anchor()
            preserve_resume_seek = (
                self._worker.is_paused
                and getattr(self, "_pending_seek_on_resume", None) is not None
            )

        # Compare can inspect the queued paused playhead without consuming it.
        if not preserve_resume_seek:
            self._pending_seek_on_resume = None
        self._compare_snapshot_pending = True

        req_precision = str(precision_key or "").strip()
        if not req_precision:
            req_precision = str(
                self._active_precision or self._cmb_prec.currentText() or ""
            ).strip()
        dlg = self._compare_dialog
        if dlg is not None:
            dlg.set_recompare_busy(True)

        self.statusBar().showMessage(
            "Preparing compare snapshot (SDR, HDR GT, HDR Convert) ..."
        )
        if anchor_frame is None:
            # Let the worker snapshot its actual current decoded frame. The
            # UI slider/labels are not always updated every frame while
            # playing, so using them here can drift by several frames.
            self._note_compare_request_for_logging(
                precision_key=req_precision or None,
                frame_number=None,
            )
            self._worker.request_compare_snapshot(
                hdr_ground_truth_path=self._hdr_ground_truth_path,
                precision_key=req_precision or None,
            )
        else:
            self._note_compare_request_for_logging(
                precision_key=req_precision or None,
                frame_number=anchor_frame,
            )
            self._worker.request_compare_snapshot(
                max(0, int(anchor_frame)),
                hdr_ground_truth_path=self._hdr_ground_truth_path,
                precision_key=req_precision or None,
                force_immediate=True,
            )

    def _compare_current_frame(self):
        if getattr(self, "_source_mode", "video") != "video":
            self.statusBar().showMessage(
                "Compare is only available in Video Player mode."
            )
            return
        self._request_compare_snapshot()

    def _on_compare_requested(self, precision_key: str, frame_number: int):
        key = str(precision_key or "").strip()
        if not key:
            return
        anchor = max(0, int(frame_number))
        self._request_compare_snapshot(precision_key=key, frame_number=anchor)

    def _on_compare_snapshot_ready(self, payload: dict):
        self._compare_snapshot_pending = False
        frame_idx = int(payload.get("frame", self._last_seek_frame))
        self._compare_anchor_frame = int(frame_idx)
        sdr = payload.get("sdr")
        hdr_algo = payload.get("hdr_algo")
        hdr_gt = payload.get("hdr_gt")
        algo_precision = str(payload.get("algo_precision", "") or "").strip()
        note = str(payload.get("note", "") or "")
        metrics = (
            payload.get("metrics")
            if isinstance(payload.get("metrics"), dict)
            else {}
        )

        if isinstance(sdr, np.ndarray):
            self._last_sdr_frame = sdr
        if isinstance(hdr_algo, np.ndarray):
            self._last_hdr_frame = hdr_algo

        self._resolve_compare_request_for_logging(
            frame_number=frame_idx,
            precision_key=algo_precision or None,
            note=note,
            metrics=metrics,
        )

        dlg = self._ensure_compare_dialog()
        dlg.set_recompare_busy(False)
        algo_label = "HDR Convert"
        if algo_precision:
            algo_label = f"HDR Convert ({algo_precision})"
        if algo_precision:
            dlg.set_precision_options(
                self._compare_precision_options(),
                selected=algo_precision,
            )
        min_frame, max_frame = self._compare_frame_bounds()
        dlg.set_frame_bounds(
            min_frame=min_frame,
            max_frame=max_frame,
            current_frame=frame_idx,
        )
        dlg.set_frames(
            frame_idx,
            sdr,
            hdr_gt,
            hdr_algo,
            note=note,
            metrics=metrics,
            hdr_algo_label=algo_label,
        )
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

        self.statusBar().showMessage(f"Compare view ready at frame {frame_idx}.")

    def _warn_if_octave_missing_for_compare(self):
        if getattr(self, "_compare_octave_warning_shown", False):
            return
        self._compare_octave_warning_shown = True
        if getattr(self, "_suppress_octave_compare_warning", False):
            return
        if _octave_executable():
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("GNU Octave Not Detected")
        box.setText(
            "GNU Octave was not detected.\n\n"
            "Compare will still open, but HDR-VDP3 score will be unavailable "
            "until Octave is installed.\n\n"
            f"Download GNU Octave:\n{_OCTAVE_DOWNLOAD_URL}",
        )
        never_warn = QCheckBox("Do not show this warning again")
        box.setCheckBox(never_warn)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        open_btn = box.addButton(
            "Open Octave Download Page",
            QMessageBox.ButtonRole.ActionRole,
        )
        while True:
            box.exec()
            if box.clickedButton() is open_btn:
                try:
                    webbrowser.open(_OCTAVE_DOWNLOAD_URL, new=2)
                except Exception:
                    pass
                continue
            break

        if never_warn.isChecked():
            self._suppress_octave_compare_warning = True
            try:
                self._save_user_settings()
            except Exception:
                pass
