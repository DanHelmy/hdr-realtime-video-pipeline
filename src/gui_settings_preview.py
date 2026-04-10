from __future__ import annotations

import json
import os

import cv2
import numpy as np

from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QInputDialog

from gui_config import (
    SOURCE_MODE_VIDEO,
    SOURCE_MODE_WINDOW,
    _available_precision_keys,
    MAX_W,
    MAX_H,
    RESOLUTION_SCALES,
    SOURCE_MODE_LABELS,
    _capture_fps_value_from_label,
    _normalize_capture_fps_label,
    _normalize_source_mode,
    _source_mode_label,
    _max_processing_preset_for_source,
    _processing_preset_dims,
    _processing_preset_options_for_source,
    _source_is_below_processing_preset,
)
from gui_scaling import (
    UPSCALER_CHOICES,
    DEFAULT_UPSCALER,
)


def _normalize_runtime_execution_mode(mode: str | None) -> str:
    text = str(mode or "").strip().lower()
    if text in {"eager", "compile"}:
        return text
    return "compile"

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
        top_key = str(getattr(self, "_source_max_resolution_key", "1080p") or "1080p")
        top_dims = _processing_preset_dims(top_key)
        sel_dims = _processing_preset_dims(scale_key)
        allow = (
            scale_key in {"540p", "720p"}
            and sel_dims[0] < top_dims[0]
            and sel_dims[1] < top_dims[1]
        )
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
        """Limit processing presets to the nearest source-sized preset bucket."""
        options = list(RESOLUTION_SCALES.keys())
        source_dims = None
        current = self._cmb_res.currentText()
        self._source_proc_dims = None
        self._source_video_dims = None
        self._source_max_resolution_key = "1080p"

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if vw > 0 and vh > 0:
                source_dims = (vw, vh)
                self._source_video_dims = source_dims
                self._source_max_resolution_key = _max_processing_preset_for_source(
                    vw, vh
                )
                options = _processing_preset_options_for_source(vw, vh)
        else:
            cap.release()

        if current == "Source":
            current = self._source_max_resolution_key

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
            top_key = self._source_max_resolution_key
            msg = (
                f"Source is {source_dims[0]}x{source_dims[1]}: "
                f"max processing/output preset is {top_key}."
            )
            if _source_is_below_processing_preset(
                source_dims[0], source_dims[1], top_key
            ):
                msg += f" No upscale is applied when {top_key} is selected."
            self.statusBar().showMessage(msg)

    def _load_user_settings(
        self,
        initial_resolution,
        initial_precision,
        initial_view,
        initial_use_hg,
        initial_upscale,
        initial_film_grain,
        initial_hdr_gt,
        initial_source_mode=None,
        initial_capture_fps=None,
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
        self._source_mode_prompt_pending = "source_mode" not in data and initial_source_mode is None
        source_mode = _normalize_source_mode(
            initial_source_mode
            if initial_source_mode is not None
            else data.get("source_mode", SOURCE_MODE_VIDEO)
        )
        self._source_mode = source_mode
        self._cmb_source_mode.setCurrentText(_source_mode_label(source_mode))
        capture_fps_label = _normalize_capture_fps_label(
            initial_capture_fps
            if initial_capture_fps is not None
            else data.get("capture_fps", self._capture_fps_label)
        )
        self._capture_fps_label = capture_fps_label
        self._capture_fps_value = _capture_fps_value_from_label(capture_fps_label)
        self._cmb_capture_fps.setCurrentText(capture_fps_label)
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
        # Restore last active tab when available; otherwise keep the current tab.
        if self._video_tabs is not None:
            saved_tab = str(data.get("active_tab", "") or "").strip()
            if saved_tab:
                for i in range(self._video_tabs.count()):
                    if self._video_tabs.tabText(i) == saved_tab:
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
        self._runtime_execution_mode = _normalize_runtime_execution_mode(
            data.get("runtime_execution_mode", "compile")
        )

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
        self._refresh_source_mode_ui()

    def _save_user_settings(self):
        data = {
            "source_mode": str(getattr(self, "_source_mode", SOURCE_MODE_VIDEO)),
            "capture_fps": str(
                self._cmb_capture_fps.currentText()
                if hasattr(self, "_cmb_capture_fps") and self._cmb_capture_fps is not None
                else getattr(self, "_capture_fps_label", _normalize_capture_fps_label(None))
            ),
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
            "runtime_execution_mode": str(
                _normalize_runtime_execution_mode(
                    getattr(self, "_runtime_execution_mode", "compile")
                )
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

    def _current_source_available(self) -> bool:
        if str(getattr(self, "_source_mode", SOURCE_MODE_VIDEO)) == SOURCE_MODE_WINDOW:
            return getattr(self, "_capture_target", None) is not None
        return bool(getattr(self, "_video_path", None))

    def _refresh_source_mode_ui(self):
        mode = _normalize_source_mode(getattr(self, "_source_mode", SOURCE_MODE_VIDEO))
        self._source_mode = mode
        is_window = mode == SOURCE_MODE_WINDOW
        if hasattr(self, "_cmb_source_mode") and self._cmb_source_mode is not None:
            self._cmb_source_mode.blockSignals(True)
            self._cmb_source_mode.setCurrentText(_source_mode_label(mode))
            self._cmb_source_mode.setEnabled(not self._playing)
            self._cmb_source_mode.blockSignals(False)
        if hasattr(self, "_capture_fps_container") and self._capture_fps_container is not None:
            self._capture_fps_container.setVisible(False)
        if hasattr(self, "_cmb_capture_fps") and self._cmb_capture_fps is not None:
            self._cmb_capture_fps.blockSignals(True)
            self._cmb_capture_fps.setCurrentText(
                _normalize_capture_fps_label(getattr(self, "_capture_fps_label", None))
            )
            self._cmb_capture_fps.setEnabled(False)
            self._cmb_capture_fps.blockSignals(False)
        if hasattr(self, "_btn_file") and self._btn_file is not None:
            self._btn_file.setText("Choose Browser Window ..." if is_window else "Open Video ...")
        if hasattr(self, "_act_open_source") and self._act_open_source is not None:
            self._act_open_source.setText("&Choose Browser Window ..." if is_window else "&Open Video ...")
        if hasattr(self, "_act_export_video") and self._act_export_video is not None:
            self._act_export_video.setEnabled(not is_window)
        if hasattr(self, "_btn_pause") and self._btn_pause is not None:
            self._btn_pause.setVisible(not is_window)
            if is_window:
                self._btn_pause.setEnabled(False)
        if hasattr(self, "_btn_compare") and self._btn_compare is not None:
            self._btn_compare.setVisible(not is_window)
            if is_window:
                self._btn_compare.setEnabled(False)
        if hasattr(self, "_btn_hdr_gt") and self._btn_hdr_gt is not None:
            self._btn_hdr_gt.setVisible(not is_window)
            self._btn_hdr_gt.setEnabled(not is_window)
            if is_window:
                self._btn_hdr_gt.setToolTip(
                    "HDR ground-truth compare is only available for file-based video playback."
                )
            else:
                self._btn_hdr_gt.setToolTip(
                    "Choose an HDR ground-truth video for compare / objective metrics."
                )
        if hasattr(self, "_lbl_hdr_gt") and self._lbl_hdr_gt is not None:
            self._lbl_hdr_gt.setVisible(not is_window)
        if hasattr(self, "_row3_widget") and self._row3_widget is not None:
            self._row3_widget.setVisible(not is_window)
        if hasattr(self, "_cmb_audio_track") and self._cmb_audio_track is not None and is_window:
            self._cmb_audio_track.blockSignals(True)
            self._cmb_audio_track.clear()
            target = getattr(self, "_capture_target", None)
            has_tab_audio_sync = bool(str(getattr(target, "session_id", "") or "").strip())
            if has_tab_audio_sync:
                self._cmb_audio_track.addItem("Chrome extension delayed audio")
                tooltip = (
                    "Browser Window Capture is experimental and only supported with Google Chrome. "
                    "Chrome's 'Use graphics acceleration when available' must be turned off. "
                    "HDRTVNet++ stays silent while the Chrome extension delays and plays the tab audio locally. "
                    "Adjust the delay in the extension popup while playback is running. "
                    "Exact sync depends on your PC, browser, and processing load."
                )
            else:
                self._cmb_audio_track.addItem("Chrome Audio Sync not active")
                tooltip = (
                    "Browser Window Capture is experimental and only supported with Google Chrome. "
                    "Chrome's 'Use graphics acceleration when available' must be turned off. "
                    "Start Chrome Audio Sync in the Chrome extension if you want delayed local browser audio. "
                    "Without it, HDRTVNet++ stays silent and Chrome keeps playing its own audio locally, which can lead the video."
                )
            self._cmb_audio_track.setEnabled(False)
            self._cmb_audio_track.setToolTip(tooltip)
            self._cmb_audio_track.blockSignals(False)
        elif hasattr(self, "_cmb_audio_track") and self._cmb_audio_track is not None:
            path = getattr(self, "_video_path", None)
            if path and os.path.isfile(path):
                try:
                    self._refresh_audio_tracks_for_video(path)
                except Exception:
                    pass
            else:
                self._cmb_audio_track.blockSignals(True)
                self._cmb_audio_track.clear()
                self._cmb_audio_track.addItem("Select a video first")
                self._cmb_audio_track.setEnabled(False)
                self._cmb_audio_track.setToolTip(
                    "Choose a source video to inspect audio tracks."
                )
                self._cmb_audio_track.blockSignals(False)
        if hasattr(self, "_lbl_file") and self._lbl_file is not None:
            if is_window:
                target = getattr(self, "_capture_target", None)
                self._lbl_file.setText(target.label if target is not None else "No browser window selected")
            else:
                path = getattr(self, "_video_path", None)
                self._lbl_file.setText(os.path.basename(path) if path else "No video selected")
        if hasattr(self, "_m") and isinstance(self._m, dict) and "frame" in self._m:
            self._m["frame"].setText("Frame: Live" if is_window else "Frame: -")
        if not self._playing and hasattr(self, "_btn_play") and self._btn_play is not None:
            self._btn_play.setEnabled(self._current_source_available())

    def _maybe_prompt_source_mode_choice(self):
        if not bool(getattr(self, "_source_mode_prompt_pending", False)):
            return
        self._source_mode_prompt_pending = False
        options = [
            SOURCE_MODE_LABELS[SOURCE_MODE_VIDEO],
            SOURCE_MODE_LABELS[SOURCE_MODE_WINDOW],
        ]
        current_idx = 0 if self._source_mode == SOURCE_MODE_VIDEO else 1
        choice, ok = QInputDialog.getItem(
            self,
            "Choose Source Mode",
            "How do you want to use HDRTVNet++ first?",
            options,
            current_idx,
            False,
        )
        if ok and str(choice or "").strip():
            self._source_mode = _normalize_source_mode(str(choice))
        else:
            self._source_mode = SOURCE_MODE_VIDEO
        self._refresh_source_mode_ui()
        self._save_user_settings()

    def _show_idle_preview_frame(self, preview: np.ndarray | None):
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

    def _set_source_resolution_options_for_dims(self, src_w: int, src_h: int):
        options = list(RESOLUTION_SCALES.keys())
        current = self._cmb_res.currentText()
        self._source_proc_dims = None
        self._source_video_dims = (int(src_w), int(src_h))
        self._source_max_resolution_key = "1080p"
        if current == "Source":
            current = self._source_max_resolution_key
        self._cmb_res.blockSignals(True)
        self._cmb_res.clear()
        self._cmb_res.addItems(options)
        if current in options:
            self._cmb_res.setCurrentText(current)
        else:
            self._cmb_res.setCurrentIndex(0)
        self._cmb_res.blockSignals(False)
        self._sync_upscale_controls()
        msg = (
            f"Browser window source is {int(src_w)}x{int(src_h)}: "
            "browser-window capture processing/output remains capped at 1080p."
        )
        self.statusBar().showMessage(msg)

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
        self._show_idle_preview_frame(self._choose_preview_frame(path))

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

    def _prepare_live_timeline(self, fps: float):
        self._vid_fps = float(fps) if fps and fps > 0 else 24.0
        self._seek_slider.setRange(0, 0)
        self._seek_slider.setValue(0)
        self._seek_slider.setEnabled(False)
        self._lbl_time.setText("0:00")
        self._lbl_duration.setText("LIVE")

    def _on_source_mode_changed(self, label: str):
        new_mode = _normalize_source_mode(label)
        if new_mode == getattr(self, "_source_mode", SOURCE_MODE_VIDEO):
            self._refresh_source_mode_ui()
            return
        if self._playing:
            self._cmb_source_mode.blockSignals(True)
            self._cmb_source_mode.setCurrentText(_source_mode_label(self._source_mode))
            self._cmb_source_mode.blockSignals(False)
            self.statusBar().showMessage(
                "Stop playback before switching between Video Player and Browser Window Capture."
            )
            return
        self._source_mode = new_mode
        self._capture_fps_label = _normalize_capture_fps_label(
            getattr(self, "_cmb_capture_fps", None).currentText()
            if hasattr(self, "_cmb_capture_fps") and self._cmb_capture_fps is not None
            else getattr(self, "_capture_fps_label", None)
        )
        self._capture_fps_value = _capture_fps_value_from_label(self._capture_fps_label)
        self._refresh_source_mode_ui()
