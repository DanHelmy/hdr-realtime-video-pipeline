from __future__ import annotations

from PyQt6.QtCore import Qt

from gui_scaling import BEST_MPV_SCALE, DEFAULT_UPSCALER


class LifecycleMixin:
    """UI lifecycle and top-level event handlers for MainWindow."""

    def _reset_controls(self):
        try:
            self.setWindowTitle("HDRTVNet++ — Real-Time SDR → HDR Pipeline")
        except Exception:
            pass
        self._playing = False
        self._compare_snapshot_pending = False
        self._active_use_mpv = False
        self._sdr_mpv_feed_from_worker = False
        self._active_mpv_scale_kernel = BEST_MPV_SCALE
        self._active_mpv_scale_antiring = 0.15
        self._active_mpv_cas = 0.0
        self._active_upscale_mode = DEFAULT_UPSCALER
        self._active_film_grain = False
        self._startup_sync_pending = False
        self._auto_muted_low_fps = False
        self._scrub_muted = False
        self._ui_hidden = False
        self._hdr_mpv_screen_sig = None
        self._sdr_mpv_screen_sig = None
        self._screen_hooked_handles.clear()
        self._last_mpv_notice = ""
        self._last_mpv_notice_t = 0.0
        self._user_pause_override_startup = False
        self._deferred_mpv_refresh = False
        if self._audio_fade_timer is not None:
            self._audio_fade_timer.stop()
        self._apply_volume_to_backends()
        self._post_seek_resync_frames = 0
        self._pending_seek_on_resume = None
        self._resume_audio_after_seek = False
        self._seek_resume_target = 0
        self._seek_resume_started_t = 0.0
        self._audio_seek_guard_until = 0.0
        self._audio_resync_pending = False
        self._audio_fps_recovered = True
        self._auto_mute_rearm_until = 0.0
        self._fps_prev_sample = None
        self._fps_is_stable = False
        self._fps_stable_count = 0
        self._fps_unstable_count = 0
        self._fps_stable_since_t = 0.0
        self._startup_audio_gate_active = False
        self._playhead_relock_token = 0
        self._pending_playhead_relock_on_unmute = False
        self._pending_playhead_relock_pre_delay_ms = -1
        self._pending_playhead_relock_first_ms = 35
        self._pending_playhead_relock_settle_ms = 220
        self._relock_hold_muted = False
        if self._window_refresh_timer is not None:
            self._window_refresh_timer.stop()
        if self._cursor_idle_timer is not None:
            self._cursor_idle_timer.stop()
        if self._periodic_relock_timer is not None:
            self._periodic_relock_timer.stop()
        if self._periodic_relock_timer is not None:
            self._periodic_relock_timer.stop()
        self._show_cursor()
        if self._compile_dlg is not None:
            self._compile_dlg.close()
            self._compile_dlg.deleteLater()
            self._compile_dlg = None
        self._btn_play.setEnabled(bool(self._video_path))
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_compare.setEnabled(False)
        self._btn_file.setEnabled(True)
        if self._btn_toggle_ui is not None:
            self._btn_toggle_ui.setEnabled(False)
            self._btn_toggle_ui.setText("Hide UI")
        if self._ui_overlay_btn is not None:
            self._ui_overlay_btn.hide()
        if self._row3_widget is not None:
            self._row3_widget.setVisible(True)
        self._set_pause_button_labels(False)
        self._btn_apply_settings.setEnabled(False)
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_stack.setCurrentWidget(self._disp_sdr_cpu)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_cpu)
        self._seek_slider.setEnabled(False)
        self._seek_slider.setValue(0)
        self._lbl_time.setText("0:00")
        self._audio_sync_frame_hint = 0
        self._last_sdr_frame = None
        self._last_hdr_frame = None
        if self._compare_dialog is not None and self._compare_dialog.isVisible():
            self._compare_dialog.close()
        # Reset HDR panel
        self._hdr_labels["status"].setText("HDR: waiting\u2026")
        self._hdr_labels["status"].setStyleSheet("")
        self._hdr_labels["primaries"].setText("Primaries: \u2014")
        self._hdr_labels["transfer"].setText("Transfer: \u2014")
        self._hdr_labels["peak"].setText("Peak: \u2014")
        self._hdr_labels["vo"].setText("VO: \u2014")
        if hasattr(self, "_m") and isinstance(getattr(self, "_m", None), dict):
            reset_map = {
                "fps": "FPS: —",
                "latency": "Latency: —",
                "frame": "Frame: —",
                "res": "Res: —",
                "gpu": "VRAM: —",
                "cpu": "CPU: —",
                "model": "Model: —",
                "prec": "Prec: —",
            }
            for key, text in reset_map.items():
                lbl = self._m.get(key)
                if lbl is None:
                    continue
                try:
                    lbl.setText(text)
                except Exception:
                    continue

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        exts = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv")
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(exts):
                self._set_video(path)
                break

    def closeEvent(self, event):
        self._ui_closing = True
        self._save_user_settings()
        if self._export_worker is not None:
            try:
                self._export_worker.cancel()
            except Exception:
                pass
        if self._export_progress_dlg is not None:
            self._export_progress_dlg.close()
        if self._window_refresh_timer is not None:
            self._window_refresh_timer.stop()
        if self._cursor_idle_timer is not None:
            self._cursor_idle_timer.stop()
        self._show_cursor()
        self._dock_video_pane("sdr")
        self._dock_video_pane("hdr")
        if self._playing:
            self._worker.stop()
            self._worker.wait(10000)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.stop_playback()
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.stop_playback()
        self._stop_audio_playback()
        self._set_process_priority(False)
        if self._export_thread is not None:
            self._export_thread.quit()
            self._export_thread.wait(5000)
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F11:
            self._toggle_borderless_full_window()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space and self._playing:
            self._toggle_pause()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self._borderless_full_window:
            self._exit_borderless_full_window()
            event.accept()
            return
        super().keyPressEvent(event)
