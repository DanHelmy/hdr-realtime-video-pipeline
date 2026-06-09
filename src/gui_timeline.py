from __future__ import annotations

import time

from PyQt6.QtCore import QPoint, QTimer
from PyQt6.QtGui import QGuiApplication

from gui_config import SOURCE_MODE_WINDOW, _normalize_source_mode


class TimelineSeekMixin:
    """Timeline synchronization + seek/position slot helpers for MainWindow."""

    def _is_live_window_capture_active(self) -> bool:
        return (
            self._playing
            and _normalize_source_mode(getattr(self, "_source_mode", None))
            == SOURCE_MODE_WINDOW
        )

    def _sync_anchor_frame(self) -> int:
        """Best-effort current frame anchor for A/V relock."""
        if self._seek_slider.isSliderDown():
            return max(0, int(self._seek_slider.value()))
        try:
            hint = int(getattr(self, "_audio_sync_frame_hint", self._last_seek_frame))
        except Exception:
            hint = int(self._last_seek_frame)
        return max(0, hint)

    def _reset_audio_playback_rate(self):
        if not self._audio_available or self._audio_player is None:
            return
        try:
            self._audio_player.setPlaybackRate(1.0)
        except Exception:
            pass

    def _resync_audio_to_current_timeline(self):
        if not self._playing:
            return
        if bool(getattr(self, "_video_prebuffer_pending", False)):
            return
        if self._is_live_window_capture_active():
            return
        fps = getattr(self, "_vid_fps", 30.0)
        sec = float(self._sync_anchor_frame()) / max(fps, 1e-6)
        if self._audio_available:
            self._force_audio_seek(sec)
            audio_release_blocked = (
                self._auto_muted_low_fps
                or self._scrub_muted
                or self._relock_hold_muted
                or self._pending_playhead_relock_on_unmute
            )
            if (
                not self._worker.is_paused
                and not self._startup_sync_pending
                and not audio_release_blocked
            ):
                self._set_audio_paused(False)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.seek_seconds(sec)
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.seek_seconds(sec)

    def _flush_display_queues(self, drop_frames: int = 2):
        if self._worker is None:
            return
        try:
            self._worker.flush_display_queues(drop_frames=drop_frames)
        except AttributeError:
            self._worker.flush_hdr_queue(drop_frames=drop_frames)

    def _seek_mpv_panes(self, sec: float):
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.seek_seconds(sec)
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.seek_seconds(sec)

    def _file_prebuffer_available(self) -> bool:
        return bool(
            self._playing
            and getattr(self, "_active_use_mpv", False)
            and self._worker is not None
            and not self._is_live_window_capture_active()
        )

    def _cancel_video_prebuffer(self):
        self._video_prebuffer_pending = False
        self._video_prebuffer_reason = ""
        try:
            self._worker.cancel_display_prebuffer()
        except Exception:
            pass

    def _begin_video_prebuffer(
        self,
        anchor_frame: int,
        *,
        reason: str = "playback",
        resume_worker: bool = True,
    ) -> bool:
        if not self._file_prebuffer_available():
            return False
        try:
            target = max(1, int(getattr(self, "_video_prebuffer_target_frames", 12)))
        except Exception:
            target = 12
        self._video_prebuffer_pending = True
        self._video_prebuffer_anchor_frame = max(0, int(anchor_frame))
        self._video_prebuffer_reason = str(reason or "playback")
        self._startup_audio_gate_active = False
        self._ui_resync_gate_strict = False
        self._auto_muted_low_fps = False
        self._audio_fps_recovered = True
        self._resume_audio_after_seek = False
        self._pending_playhead_relock_on_unmute = False
        self._relock_hold_muted = False
        self._scrub_muted = True
        self._apply_volume_to_backends()
        if self._audio_available:
            self._set_audio_paused(True)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.set_paused(True)
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.set_paused(True)
        self._flush_display_queues(drop_frames=0)
        try:
            self._worker.request_display_prebuffer(target)
        except Exception:
            self._video_prebuffer_pending = False
            self._scrub_muted = False
            self._apply_volume_to_backends()
            return False
        if resume_worker:
            self._worker.resume()
        self.statusBar().showMessage(f"Buffering {target} frames before playback ...")
        return True

    def _on_display_prebuffer_ready(self, frame_number: int, buffered_frames: int):
        if not bool(getattr(self, "_video_prebuffer_pending", False)):
            return
        if not self._playing or self._worker is None:
            self._cancel_video_prebuffer()
            return
        if self._user_pause_override_startup:
            self._startup_sync_pending = False
            self._startup_frame_relock_pending = False
            self._cancel_video_prebuffer()
            return
        anchor_frame = max(0, int(getattr(self, "_video_prebuffer_anchor_frame", 0)))
        try:
            release_frame = max(anchor_frame, int(frame_number))
        except Exception:
            release_frame = anchor_frame
        fps = getattr(self, "_vid_fps", 30.0)
        sec = float(release_frame) / max(fps, 1e-6)
        self._video_prebuffer_pending = False
        self._video_prebuffer_reason = ""
        self._startup_sync_pending = False
        self._startup_frame_relock_pending = False
        self._startup_frame_relock_token += 1
        self._startup_audio_gate_active = False
        self._ui_resync_gate_strict = False
        self._auto_muted_low_fps = False
        self._scrub_muted = False
        self._relock_hold_muted = False
        self._pending_playhead_relock_on_unmute = False
        self._audio_fps_recovered = True
        self._audio_resync_pending = False
        self._audio_seek_guard_until = time.perf_counter() + 0.35
        self._last_seek_frame = int(release_frame)
        self._audio_sync_frame_hint = int(release_frame)
        try:
            if not self._seek_slider.isSliderDown():
                self._seek_slider.setValue(int(release_frame))
                self._lbl_time.setText(self._fmt_time(sec))
        except Exception:
            pass
        if self._audio_available:
            self._reset_audio_playback_rate()
            self._force_audio_seek(sec)
        elif self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.seek_seconds(sec)
        self._worker.resume()
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.set_paused(False)
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.set_paused(False)
        self._apply_volume_to_backends()
        if self._audio_available and not self._worker.is_paused:
            self._reset_audio_playback_rate()
            self._set_audio_paused(False)
        if self._active_use_mpv:
            self._schedule_playhead_skip_relock(first_delay_ms=30, settle_delay_ms=180)
        if bool(getattr(self, "_scrub_preview_hold_until_prebuffer", False)):
            self._stop_scrub_preview()
        self._scrub_preview_hold_until_prebuffer = False
        try:
            shown = max(1, int(buffered_frames))
        except Exception:
            shown = int(getattr(self, "_video_prebuffer_target_frames", 12))
        self.statusBar().showMessage(f"Playback buffered ({shown} frames).")

    def _relock_timeline(self, delay_ms: int = 0, drop_frames: int = 2):
        """Force a timeline relock after UI actions."""
        if not self._playing:
            return
        if self._is_live_window_capture_active():
            return
        if bool(getattr(self, "_video_prebuffer_pending", False)):
            return

        def _do():
            if not self._playing:
                return
            self._flush_display_queues(drop_frames=drop_frames)
            self._resync_audio_to_current_timeline()

        if delay_ms > 0:
            QTimer.singleShot(int(delay_ms), _do)
        else:
            _do()

    def _clear_deferred_playhead_relock(self):
        self._deferred_playhead_relock_token = (
            int(getattr(self, "_deferred_playhead_relock_token", 0)) + 1
        )
        self._playhead_relock_token = int(getattr(self, "_playhead_relock_token", 0)) + 1
        self._pending_playhead_relock_on_unmute = False
        self._pending_playhead_relock_started_t = 0.0
        self._pending_playhead_relock_pre_delay_ms = -1
        self._relock_hold_muted = False
        try:
            self._apply_volume_to_backends()
        except Exception:
            pass

    def _schedule_state_change_relock(
        self,
        delay_ms: int = 0,
        drop_frames: int = 2,
        settle_delay_ms: int | None = None,
        settle_drop_frames: int = 1,
    ):
        """Bounded relock batch for visual state changes."""
        if not self._playing:
            return
        if self._is_live_window_capture_active():
            return
        if bool(getattr(self, "_video_prebuffer_pending", False)):
            return
        token = int(getattr(self, "_state_relock_token", 0)) + 1
        self._state_relock_token = token
        if not bool(getattr(self, "_resume_audio_after_seek", False)):
            self._clear_deferred_playhead_relock()

        def _do(expected_token: int, frames_to_drop: int):
            if expected_token != int(getattr(self, "_state_relock_token", 0)):
                return
            if not self._playing:
                return
            if self._is_live_window_capture_active():
                return
            if bool(getattr(self, "_video_prebuffer_pending", False)):
                return
            self._flush_display_queues(drop_frames=frames_to_drop)
            self._resync_audio_to_current_timeline()

        QTimer.singleShot(
            max(0, int(delay_ms)),
            lambda expected=token: _do(expected, int(drop_frames)),
        )
        if settle_delay_ms is not None:
            QTimer.singleShot(
                max(0, int(settle_delay_ms)),
                lambda expected=token: _do(expected, int(settle_drop_frames)),
            )

    def _schedule_playhead_skip_relock(
        self,
        first_delay_ms: int = 35,
        settle_delay_ms: int = 220,
    ):
        """Fast relock path for timeline skips: early snap + short settle pass."""
        if not self._playing or not getattr(self, "_active_use_mpv", False):
            return
        token = int(getattr(self, "_playhead_relock_token", 0)) + 1
        self._playhead_relock_token = token

        self._relock_timeline(delay_ms=max(0, int(first_delay_ms)), drop_frames=2)

        def _settle():
            if token != int(getattr(self, "_playhead_relock_token", 0)):
                return
            if not self._playing or self._worker is None or self._worker.is_paused:
                return
            self._relock_timeline(drop_frames=1)

        QTimer.singleShot(max(80, int(settle_delay_ms)), _settle)

    def _request_playhead_skip_relock_after_unmute(
        self, first_delay_ms: int = 35, settle_delay_ms: int = 220
    ):
        """Run skip relock after audio unmutes; fall back to immediate if already unmuted."""
        if not self._playing or not getattr(self, "_active_use_mpv", False):
            return
        if not (self._auto_muted_low_fps or self._scrub_muted or self._relock_hold_muted):
            self._schedule_playhead_skip_relock(
                first_delay_ms=first_delay_ms, settle_delay_ms=settle_delay_ms
            )
            return
        self._pending_playhead_relock_on_unmute = True
        self._pending_playhead_relock_started_t = time.perf_counter()
        self._pending_playhead_relock_pre_delay_ms = -1
        self._pending_playhead_relock_first_ms = int(first_delay_ms)
        self._pending_playhead_relock_settle_ms = int(settle_delay_ms)

    def _scrub_preview_available(self) -> bool:
        return bool(
            self._playing
            and self._video_path
            and getattr(self, "_scrub_preview_popup", None) is not None
            and getattr(self, "_scrub_preview_mpv", None) is not None
            and not self._is_live_window_capture_active()
        )

    def _position_scrub_preview_popup(self, frame_number: int):
        popup = getattr(self, "_scrub_preview_popup", None)
        slider = getattr(self, "_seek_slider", None)
        if popup is None or slider is None:
            return
        try:
            min_value = int(slider.minimum())
            max_value = int(slider.maximum())
            span = max(1, max_value - min_value)
            ratio = (max(min_value, min(max_value, int(frame_number))) - min_value) / span
            thumb_x = int(round(ratio * max(1, slider.width())))
            slider_pos = slider.mapToGlobal(QPoint(thumb_x, 0))
            popup.adjustSize()
            popup_w = max(1, int(popup.width()))
            popup_h = max(1, int(popup.height()))
            margin = 8
            x = int(slider_pos.x() - popup_w / 2)
            try:
                screen = QGuiApplication.screenAt(slider_pos)
                if screen is None:
                    screen = slider.screen() or popup.screen() or self.screen()
                geo = screen.availableGeometry() if screen is not None else self.geometry()
                left = int(geo.left()) + margin
                right = int(geo.right()) - popup_w - margin + 1
                top = int(geo.top()) + margin
            except Exception:
                left = margin
                right = max(margin, int(self.width()) - popup_w - margin)
                top = margin
            x = max(left, min(max(left, right), x))
            y = int(slider_pos.y() - popup_h - 10)
            if y < top:
                y = int(slider_pos.y() + slider.height() + 10)
            popup.move(x, y)
            popup.raise_()
        except Exception:
            pass

    def _start_scrub_preview(self, frame_number: int) -> bool:
        if not self._scrub_preview_available():
            return False
        fps = getattr(self, "_vid_fps", 30.0)
        sec = max(0.0, float(frame_number) / max(fps, 1e-6))
        preview = self._scrub_preview_mpv
        try:
            ok = bool(preview.start_preview(self._video_path, sec))
        except Exception:
            ok = False
        if ok:
            self._scrub_preview_active = True
            self._position_scrub_preview_popup(frame_number)
            self._scrub_preview_popup.show()
            self._scrub_preview_popup.raise_()
        return ok

    def _update_scrub_preview(self, frame_number: int):
        if not self._scrub_preview_available():
            return
        fps = getattr(self, "_vid_fps", 30.0)
        sec = max(0.0, float(frame_number) / max(fps, 1e-6))
        if not bool(getattr(self, "_scrub_preview_active", False)):
            self._start_scrub_preview(frame_number)
            return
        try:
            self._scrub_preview_mpv.seek_seconds(sec)
            self._position_scrub_preview_popup(frame_number)
            self._scrub_preview_popup.show()
            self._scrub_preview_popup.raise_()
        except Exception:
            pass

    def _stop_scrub_preview(self):
        if getattr(self, "_scrub_preview_mpv", None) is not None:
            try:
                self._scrub_preview_mpv.stop_preview()
            except Exception:
                pass
        if getattr(self, "_scrub_preview_popup", None) is not None:
            try:
                self._scrub_preview_popup.hide()
            except Exception:
                pass
        self._scrub_preview_active = False
        self._scrub_preview_hold_until_prebuffer = False

    def _maybe_dispatch_startup_frame_relock(self):
        if not bool(getattr(self, "_startup_frame_relock_pending", False)):
            return
        if not self._playing or self._is_live_window_capture_active():
            self._startup_frame_relock_pending = False
            return
        if not getattr(self, "_active_use_mpv", False):
            self._startup_frame_relock_pending = False
            return
        if self._startup_sync_pending:
            return

        self._startup_frame_relock_pending = False
        token = int(getattr(self, "_startup_frame_relock_token", 0)) + 1
        self._startup_frame_relock_token = token

        def _relock_if_current(delay_ms: int, drop_frames: int):
            def _do():
                if token != int(getattr(self, "_startup_frame_relock_token", 0)):
                    return
                if not self._playing or self._worker is None or self._worker.is_paused:
                    return
                self._relock_timeline(drop_frames=drop_frames)

            QTimer.singleShot(max(0, int(delay_ms)), _do)

        _relock_if_current(40, 3)
        _relock_if_current(220, 2)
        _relock_if_current(620, 1)

    def _force_audio_seek(self, sec: float):
        """Aggressive audio resync: double-seek to reduce drift after UI changes."""
        if not self._audio_available:
            return
        self._seek_audio_seconds(sec)

        def _second_seek():
            self._seek_audio_seconds(sec)

        QTimer.singleShot(20, _second_seek)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Format seconds as M:SS or H:MM:SS."""
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def _on_position(self, current_frame: int, total_frames: int):
        """Update seek slider + time labels from worker."""
        self._last_seek_frame = int(current_frame)
        self._audio_sync_frame_hint = int(current_frame)
        self._maybe_dispatch_startup_frame_relock()
        if (
            total_frames > 0
            and not self._seek_slider.isSliderDown()
            and int(self._seek_slider.value()) != int(current_frame)
        ):
            self._seek_slider.setValue(current_frame)
        fps = getattr(self, "_vid_fps", 30.0)
        self._lbl_time.setText(self._fmt_time(current_frame / fps))

        # On startup or when returning to the beginning, re-anchor mpv to 0
        # to eliminate initial HDR lag.
        if (
            (not self._is_live_window_capture_active())
            and self._disp_hdr_mpv is not None
            and current_frame <= 1
        ):
            now_t = time.perf_counter()
            if (now_t - self._mpv_start_resync_t) > 0.8:
                self._mpv_start_resync_t = now_t
                self._seek_mpv_panes(0.0)

        if self._resume_audio_after_seek and not self._worker.is_paused:
            settled = abs(int(current_frame) - int(self._seek_resume_target)) <= 1
            timed_out = (time.perf_counter() - self._seek_resume_started_t) > 0.45
            if settled or timed_out:
                audio_release_blocked = (
                    self._auto_muted_low_fps
                    or self._scrub_muted
                    or self._relock_hold_muted
                    or self._pending_playhead_relock_on_unmute
                )
                if not audio_release_blocked:
                    self._reset_audio_playback_rate()
                    self._set_audio_paused(False)
                    self._resume_audio_after_seek = False
                    self._audio_fps_recovered = True
                    self._audio_resync_pending = False
                    if self._scrub_muted and (not self._seek_slider.isSliderDown()):
                        self._scrub_muted = False
                        self._apply_volume_to_backends()
                        if self._pending_playhead_relock_on_unmute:
                            self._dispatch_pending_playhead_relock_after_unmute()

        if (
            self._playing
            and self._audio_available
            and self._audio_player is not None
            and (not self._is_live_window_capture_active())
            and not self._seek_slider.isSliderDown()
        ):
            want_ms = int((current_frame / max(fps, 1e-6)) * 1000.0)
            have_ms = int(self._audio_player.position())
            drift_signed_ms = float(have_ms - want_ms)  # + => audio ahead, - => audio behind
            drift_ms = abs(drift_signed_ms)
            now_t = time.perf_counter()
            in_seek_guard = now_t < self._audio_seek_guard_until

            # Post-seek stabilization window: hold neutral rate, avoid correction thrash.
            if self._post_seek_resync_frames > 0:
                self._post_seek_resync_frames -= 1
                self._audio_player.setPlaybackRate(1.0)
            elif (current_frame % self._audio_drift_check_stride == 0) and (
                not self._worker.is_paused
            ):
                # One-shot sync only after FPS recovery. Do not keep correcting
                # until another low-FPS mute/seek event re-arms this gate.
                if (
                    self._audio_resync_pending
                    and self._audio_fps_recovered
                    and (not in_seek_guard)
                    and drift_ms > 220
                ):
                    self._audio_player.setPosition(max(0, want_ms))
                    self._audio_last_hard_sync_t = now_t
                    self._audio_resync_pending = False
                elif (not in_seek_guard) and drift_ms > 2200 and (
                    now_t - self._audio_last_hard_sync_t
                ) > 10.0:
                    # Emergency recovery only.
                    self._audio_player.setPosition(max(0, want_ms))
                    self._audio_last_hard_sync_t = now_t
                    self._audio_player.setPlaybackRate(1.0)
                elif drift_signed_ms > 320:
                    self._audio_player.setPlaybackRate(0.9997)
                elif drift_signed_ms < -320:
                    self._audio_player.setPlaybackRate(1.0003)
                else:
                    self._audio_player.setPlaybackRate(1.0)

    def _on_seek_pressed(self):
        if bool(getattr(self, "_export_interaction_locked", False)):
            self.statusBar().showMessage(
                "Seeking is locked while export is running. Finish or cancel the export first."
            )
            return
        if not self._playing:
            return
        self._pending_seek_on_resume = None
        self._scrub_unmute_seq += 1
        self._start_scrub_preview(int(self._seek_slider.value()))

    def _on_seek(self, frame_number: int):
        if bool(getattr(self, "_export_interaction_locked", False)):
            return
        if not self._playing:
            return
        self._pending_seek_on_resume = None
        self._last_seek_frame = int(frame_number)
        fps = getattr(self, "_vid_fps", 30.0)
        self._lbl_time.setText(self._fmt_time(frame_number / fps))
        self._update_scrub_preview(int(frame_number))

    def _on_seek_value_changed(self, frame_number: int):
        if not self._playing:
            return
        try:
            slider_down = bool(self._seek_slider.isSliderDown())
        except Exception:
            slider_down = False
        if not slider_down:
            return
        self._on_seek(int(frame_number))

    def _on_seek_released(self):
        if bool(getattr(self, "_export_interaction_locked", False)):
            self.statusBar().showMessage(
                "Seeking is locked while export is running. Finish or cancel the export first."
            )
            return
        if not self._playing:
            return
        self._pending_seek_on_resume = None
        if self._audio_available:
            self._reset_audio_playback_rate()
            self._set_audio_paused(True)
        target_frame = int(self._seek_slider.value())
        self._last_seek_frame = target_frame
        self._audio_sync_frame_hint = target_frame
        fps = getattr(self, "_vid_fps", 30.0)
        target_sec = target_frame / max(fps, 1e-6)
        was_paused = bool(self._worker.is_paused)

        prebuffering = False
        if not was_paused:
            prebuffering = self._begin_video_prebuffer(
                target_frame,
                reason="seek",
                resume_worker=False,
            )
            self._scrub_preview_hold_until_prebuffer = bool(prebuffering)
        if not prebuffering:
            self._flush_display_queues(drop_frames=4)
        self._worker.request_seek(target_frame)
        if not prebuffering:
            self._force_audio_seek(target_sec)
        now_t = time.perf_counter()
        self._audio_seek_guard_until = now_t + 1.0
        self._audio_resync_pending = True
        self._audio_fps_recovered = False
        if self._audio_available:
            QTimer.singleShot(420, self._ensure_selected_audio_track_qt)
        self._post_seek_resync_frames = 120
        self._resume_audio_after_seek = bool(self._audio_available and not prebuffering)
        self._seek_resume_target = int(target_frame)
        self._seek_resume_started_t = time.perf_counter()
        self._seek_mpv_panes(target_sec)
        if prebuffering:
            self._worker.resume()
            self._set_pause_button_labels(False)
            return
        if was_paused:
            self._lbl_time.setText(self._fmt_time(target_sec))
            self.statusBar().showMessage(
                f"Seeked to {self._fmt_time(target_sec)}."
            )
            QTimer.singleShot(350, self._stop_scrub_preview)
        else:
            QTimer.singleShot(120, self._stop_scrub_preview)
        self._request_playhead_skip_relock_after_unmute()

    def _on_seek_frame_ready(self, frame_number: int):
        """Worker callback: first rendered frame after seek is now visible."""
        if not self._playing:
            return
        if bool(getattr(self, "_video_prebuffer_pending", False)):
            return
        try:
            frame_idx = max(0, int(frame_number))
        except Exception:
            return

        self._last_seek_frame = frame_idx
        self._audio_sync_frame_hint = frame_idx
        fps = getattr(self, "_vid_fps", 30.0)
        sec = float(frame_idx) / max(fps, 1e-6)

        # Re-anchor audio to the exact first visible post-seek frame.
        if self._audio_available:
            self._seek_audio_seconds(sec)
            if (
                not self._worker.is_paused
                and not self._startup_sync_pending
                and not (
                    self._auto_muted_low_fps
                    or self._scrub_muted
                    or self._relock_hold_muted
                    or self._pending_playhead_relock_on_unmute
                )
            ):
                self._set_audio_paused(False)

        # Deterministic relock on first rendered post-seek frame.
        if self._active_use_mpv:
            self._request_playhead_skip_relock_after_unmute(
                first_delay_ms=35, settle_delay_ms=220
            )
            if self._pending_playhead_relock_on_unmute:
                self._dispatch_pending_playhead_relock_after_unmute()
        else:
            self._relock_timeline(drop_frames=1)
