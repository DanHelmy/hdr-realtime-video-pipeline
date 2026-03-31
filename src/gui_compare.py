from __future__ import annotations

import os

import numpy as np

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


class CompareViewMixin:
    """Compare-view related helpers for MainWindow."""

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
            self._compare_dialog.precision_requested.connect(
                self._on_compare_precision_requested
            )
        current_prec = str(self._active_precision or self._cmb_prec.currentText() or "").strip()
        self._compare_dialog.set_precision_options(
            self._compare_precision_options(),
            selected=current_prec or None,
        )
        return self._compare_dialog

    def _request_compare_snapshot(
        self,
        precision_key: str | None = None,
        frame_number: int | None = None,
    ):
        if not self._playing:
            self.statusBar().showMessage("Start playback first, then use Compare.")
            return
        if self._compare_snapshot_pending:
            self.statusBar().showMessage("Compare snapshot already in progress ...")
            return

        # Pause first to freeze playback context for deterministic compare.
        if not self._worker.is_paused:
            self._toggle_pause()

        target_frame = (
            int(self._seek_slider.value())
            if frame_number is None
            else max(0, int(frame_number))
        )
        # Compare seek is immediate; avoid re-applying an old queued seek on resume.
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
        self._worker.request_compare_snapshot(
            target_frame,
            hdr_ground_truth_path=self._hdr_ground_truth_path,
            precision_key=req_precision or None,
        )

    def _compare_current_frame(self):
        self._request_compare_snapshot()

    def _on_compare_precision_requested(self, precision_key: str):
        key = str(precision_key or "").strip()
        if not key:
            return
        anchor = getattr(self, "_compare_anchor_frame", None)
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
