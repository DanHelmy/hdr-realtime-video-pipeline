from __future__ import annotations

import json
import os

import cv2
import numpy as np

from PyQt6.QtCore import QEvent

from gui_config import (
    _available_precision_keys,
    MAX_W,
    MAX_H,
    RESOLUTION_SCALES,
)
from gui_scaling import (
    UPSCALER_CHOICES,
    DEFAULT_UPSCALER,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PREFS_PATH = os.path.join(_ROOT, ".gui_prefs.json")


class SettingsPreviewMixin:
    """Settings persistence and idle preview/timeline helpers for MainWindow."""

    def _sync_upscale_controls(self):
        """Enable/disable upscale selector based on resolution preset."""
        if not hasattr(self, "_cmb_upscale") or self._cmb_upscale is None:
            return
        scale_key = self._cmb_res.currentText()
        allow = scale_key in {"540p", "720p"}
        self._cmb_upscale.blockSignals(True)
        if allow:
            self._cmb_upscale.setEnabled(True)
            if self._cmb_upscale.currentText() not in UPSCALER_CHOICES:
                self._cmb_upscale.setCurrentText(DEFAULT_UPSCALER)
        else:
            self._cmb_upscale.setCurrentText(DEFAULT_UPSCALER)
            self._cmb_upscale.setEnabled(False)
        self._cmb_upscale.blockSignals(False)

    def _refresh_resolution_options_for_video(self, path: str):
        """Show only processing presets that do not exceed source resolution."""
        options = list(RESOLUTION_SCALES.keys())
        source_dims = None
        current = self._cmb_res.currentText()
        self._source_proc_dims = None

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if vw > 0 and vh > 0:
                source_dims = (vw, vh)
                filtered = []
                for key, dims in RESOLUTION_SCALES.items():
                    tw, th = (MAX_W, MAX_H) if dims is None else dims
                    if tw <= vw and th <= vh:
                        filtered.append(key)
                if filtered:
                    options = filtered
                    self._source_proc_dims = None
                else:
                    sw = max(2, vw - (vw % 2))
                    sh = max(2, vh - (vh % 2))
                    self._source_proc_dims = (sw, sh)
                    options = ["Source"]
        else:
            cap.release()

        self._cmb_res.blockSignals(True)
        self._cmb_res.clear()
        self._cmb_res.addItems(options)
        if current in options:
            self._cmb_res.setCurrentText(current)
        else:
            self._cmb_res.setCurrentIndex(0)
        self._cmb_res.blockSignals(False)
        self._sync_upscale_controls()

        if source_dims is not None and source_dims[1] < MAX_H:
            self.statusBar().showMessage(
                f"Source is {source_dims[0]}x{source_dims[1]}: hiding higher processing presets."
            )

    def _load_user_settings(
        self,
        initial_resolution,
        initial_precision,
        initial_view,
        initial_use_hg,
        initial_upscale,
        initial_film_grain,
        initial_hdr_gt,
    ):
        """Load persisted GUI preferences unless explicitly overridden by CLI."""
        data = {}
        if os.path.isfile(_PREFS_PATH):
            try:
                with open(_PREFS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        if initial_precision is None:
            p = data.get("precision")
            if p in _available_precision_keys():
                self._cmb_prec.setCurrentText(p)
        if initial_resolution is None:
            r = data.get("resolution")
            if r in RESOLUTION_SCALES or r == "Source":
                self._cmb_res.setCurrentText(r)
        if hasattr(self, "_cmb_upscale"):
            if isinstance(initial_upscale, str) and initial_upscale in UPSCALER_CHOICES:
                self._cmb_upscale.setCurrentText(initial_upscale)
            else:
                um = data.get("upscale_mode")
                if isinstance(um, str) and um in UPSCALER_CHOICES:
                    self._cmb_upscale.setCurrentText(um)
        if hasattr(self, "_chk_film_grain") and self._chk_film_grain is not None:
            if initial_film_grain is None:
                fg = data.get("film_grain")
                if isinstance(fg, bool):
                    self._chk_film_grain.setChecked(fg)
            else:
                self._chk_film_grain.setChecked(str(initial_film_grain).strip() == "1")
        if initial_view is None:
            v = data.get("view")
            if v == "Tabbed":
                self._cmb_view.setCurrentText("Tabbed")
        # Restore last active tab (defaults to HDR if missing).
        if self._video_tabs is not None:
            saved_tab = data.get("active_tab", "HDR")
            for i in range(self._video_tabs.count()):
                if self._video_tabs.tabText(i) == saved_tab:
                    self._video_tabs.setCurrentIndex(i)
                    break
            else:
                # If saved tab not found, default to HDR.
                for i in range(self._video_tabs.count()):
                    if self._video_tabs.tabText(i) == "HDR":
                        self._video_tabs.setCurrentIndex(i)
                        break
        self._chk_hg.setChecked(bool(data.get("use_hg", True)))
        if initial_use_hg is not None:
            self._chk_hg.setChecked(str(initial_use_hg).strip() == "1")

        m = data.get("show_metrics")
        if isinstance(m, bool):
            self._chk_metrics.setChecked(m)
            self._grp_metrics.setVisible(m)

        vol = data.get("volume_percent")
        if isinstance(vol, int):
            self._sld_volume.setValue(max(0, min(100, vol)))
        aidx = data.get("audio_track")
        if isinstance(aidx, int):
            self._selected_audio_track = max(0, int(aidx))
        hc = data.get("hide_cursor_idle")
        if isinstance(hc, bool):
            self._chk_hide_cursor.setChecked(hc)

        last_dir = data.get("last_open_dir")
        if (
            isinstance(last_dir, str)
            and last_dir.strip()
            and os.path.isdir(last_dir.strip())
        ):
            self._last_open_dir = last_dir.strip()
        else:
            self._last_open_dir = _ROOT

        last_export_dir = data.get("last_export_dir")
        if (
            isinstance(last_export_dir, str)
            and last_export_dir.strip()
            and os.path.isdir(last_export_dir.strip())
        ):
            self._last_export_dir = last_export_dir.strip()
        else:
            self._last_export_dir = self._last_open_dir

        predeq_mode = str(data.get("predequantize_mode", "auto")).strip().lower()
        if predeq_mode not in {"auto", "on", "off"}:
            predeq_mode = "auto"
        self._predequantize_mode = predeq_mode

        self._suppress_hip_sdk_warning = bool(
            data.get("suppress_hip_sdk_warning", False)
        )
        self._suppress_octave_compare_warning = bool(
            data.get("suppress_octave_compare_warning", False)
        )

        gt_path = None
        # Do not persist HDR GT across app sessions; only accept explicit CLI pass-through.
        if isinstance(initial_hdr_gt, str) and initial_hdr_gt.strip():
            gt_path = initial_hdr_gt.strip()
        if gt_path and not os.path.isfile(gt_path):
            gt_path = None
        self._hdr_ground_truth_path = gt_path
        # Keep GT for compare-only path to avoid runtime GT decode overhead.
        self._objective_metrics_enabled = False
        self._update_hdr_ground_truth_label()

        self._sync_upscale_controls()

    def _save_user_settings(self):
        data = {
            "precision": self._cmb_prec.currentText(),
            "resolution": self._cmb_res.currentText(),
            "upscale_mode": self._cmb_upscale.currentText()
            if hasattr(self, "_cmb_upscale")
            else DEFAULT_UPSCALER,
            "view": self._cmb_view.currentText(),
            "show_metrics": self._chk_metrics.isChecked(),
            "use_hg": self._chk_hg.isChecked(),
            "film_grain": bool(
                getattr(self, "_chk_film_grain", None) and self._chk_film_grain.isChecked()
            ),
            "volume_percent": int(self._volume_percent),
            "audio_track": int(self._selected_audio_track),
            "hide_cursor_idle": bool(self._cursor_idle_enabled),
            "predequantize_mode": str(
                getattr(self, "_predequantize_mode", "auto")
            ),
            "suppress_hip_sdk_warning": bool(
                getattr(self, "_suppress_hip_sdk_warning", False)
            ),
            "suppress_octave_compare_warning": bool(
                getattr(self, "_suppress_octave_compare_warning", False)
            ),
            "last_open_dir": self._last_open_dir
            if os.path.isdir(self._last_open_dir)
            else _ROOT,
            "last_export_dir": self._last_export_dir
            if os.path.isdir(self._last_export_dir)
            else (
                self._last_open_dir
                if os.path.isdir(self._last_open_dir)
                else _ROOT
            ),
        }
        if self._video_tabs is not None and self._video_tabs.currentIndex() >= 0:
            data["active_tab"] = self._video_tabs.tabText(self._video_tabs.currentIndex())
        try:
            with open(_PREFS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def eventFilter(self, obj, event):
        et = event.type()
        if et == QEvent.Type.Resize:
            self._position_ui_overlay()
        if et in (
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.Wheel,
            QEvent.Type.KeyPress,
        ):
            self._show_cursor()
            self._arm_cursor_idle_timer()
            if self._ui_hidden:
                self._show_ui_overlay_temporarily()
        return super().eventFilter(obj, event)

    def _has_pending_setting_changes(self) -> bool:
        if not self._playing:
            return False
        upscale_changed = False
        if hasattr(self, "_cmb_upscale") and self._cmb_res.currentText() in {"540p", "720p"}:
            upscale_changed = self._cmb_upscale.currentText() != self._active_upscale_mode
        film_grain_changed = False
        if hasattr(self, "_chk_film_grain") and self._chk_film_grain is not None:
            film_grain_changed = (
                self._chk_film_grain.isChecked() != self._active_film_grain
            )
        return (
            self._cmb_prec.currentText() != self._active_precision
            or self._cmb_res.currentText() != self._active_resolution
            or self._chk_hg.isChecked() != self._active_use_hg
            or upscale_changed
            or film_grain_changed
        )

    def _update_apply_button_state(self):
        self._btn_apply_settings.setEnabled(self._has_pending_setting_changes())

    def _choose_preview_frame(self, path: str) -> np.ndarray | None:
        """Pick a representative frame (non-black, high-detail) for idle preview."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                probe_ids = [
                    int(total * 0.08),
                    int(total * 0.22),
                    int(total * 0.38),
                    int(total * 0.55),
                    int(total * 0.72),
                ]
            else:
                probe_ids = [0]

            best = None
            best_score = -1.0
            for fid in probe_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fid))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = float(gray.std())
                if score > best_score:
                    best = frame
                    best_score = score

            if best is not None:
                return best

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    def _show_idle_preview(self, path: str):
        """Render selected-video preview without starting playback."""
        preview = self._choose_preview_frame(path)
        if preview is not None:
            if self._disp_sdr_cpu is not None:
                self._disp_sdr_cpu.update_frame(preview)
            if self._disp_hdr_cpu is not None:
                self._disp_hdr_cpu.update_frame(preview)
            if self._disp_sdr_mpv is not None:
                self._disp_sdr_stack.setCurrentWidget(self._disp_sdr_cpu)
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_cpu)
        else:
            if self._disp_sdr_cpu is not None:
                self._disp_sdr_cpu.clear_display()
            if self._disp_hdr_cpu is not None:
                self._disp_hdr_cpu.clear_display()

    def _prepare_idle_timeline(self, path: str):
        """Populate duration labels/slider in idle state."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._seek_slider.setEnabled(False)
            self._seek_slider.setValue(0)
            self._lbl_time.setText("0:00")
            self._lbl_duration.setText("0:00")
            return
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()

        self._vid_fps = fps if fps > 0 else 30.0
        self._seek_slider.setRange(0, max(0, total_frames - 1))
        self._seek_slider.setValue(0)
        self._seek_slider.setEnabled(False)
        self._lbl_time.setText("0:00")
        dur_secs = total_frames / self._vid_fps if self._vid_fps > 0 else 0.0
        self._lbl_duration.setText(self._fmt_time(dur_secs))
