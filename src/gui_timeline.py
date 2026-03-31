from __future__ import annotations

import time

from PyQt6.QtCore import QTimer


class TimelineSeekMixin:
    """Timeline synchronization + seek/position slot helpers for MainWindow."""

    def _sync_anchor_frame(self) -> int:
        """Best-effort current frame anchor for A/V relock."""
        if self._seek_slider.isSliderDown():
            return max(0, int(self._seek_slider.value()))
        try:
            hint = int(getattr(self, "_audio_sync_frame_hint", self._last_seek_frame))
        except Exception:
            hint = int(self._last_seek_frame)
        return max(0, hint)

    def _resync_audio_to_current_timeline(self):
        if not self._playing:
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

    def _relock_timeline(self, delay_ms: int = 0, drop_frames: int = 2):
        """Force a timeline relock (audio + HDR queue) after UI actions."""
        if not self._playing:
            return

        def _do():
            if not self._playing:
                return
            if self._worker is not None:
                self._worker.flush_hdr_queue(drop_frames=drop_frames)
            self._resync_audio_to_current_timeline()

        if delay_ms > 0:
            QTimer.singleShot(int(delay_ms), _do)
        else:
            _do()

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
        self._pending_playhead_relock_pre_delay_ms = -1
        self._pending_playhead_relock_first_ms = int(first_delay_ms)
        self._pending_playhead_relock_settle_ms = int(settle_delay_ms)

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
        if not self._seek_slider.isSliderDown():
            self._seek_slider.setValue(current_frame)
        fps = getattr(self, "_vid_fps", 30.0)
        self._lbl_time.setText(self._fmt_time(current_frame / fps))

        # On startup or when returning to the beginning, re-anchor mpv to 0
        # to eliminate initial HDR lag.
        if self._disp_hdr_mpv is not None and current_frame <= 1:
            now_t = time.perf_counter()
            if (now_t - self._mpv_start_resync_t) > 0.8:
                self._mpv_start_resync_t = now_t
                self._disp_hdr_mpv.seek_seconds(0.0)

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
                    self._set_audio_paused(False)
                    self._resume_audio_after_seek = False
                    if self._scrub_muted and (not self._seek_slider.isSliderDown()):
                        self._scrub_muted = False
                        self._apply_volume_to_backends()
                        if self._pending_playhead_relock_on_unmute:
                            self._dispatch_pending_playhead_relock_after_unmute()

        if (
            self._playing
            and self._audio_available
            and self._audio_player is not None
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
        if not self._playing:
            return
        self._scrub_unmute_seq += 1
        if self._audio_available:
            self._set_audio_paused(True)
        self._arm_mute_until_fps_recovery()
        if not self._scrub_muted:
            self._scrub_muted = True
            self._apply_volume_to_backends()

    def _on_seek(self, frame_number: int):
        if not self._playing:
            return
        # Fast click-seek can race sliderPressed; enforce mute/pause here too.
        if self._audio_available:
            self._set_audio_paused(True)
        self._arm_mute_until_fps_recovery()
        if not self._scrub_muted:
            self._scrub_muted = True
            self._apply_volume_to_backends()
        self._last_seek_frame = int(frame_number)
        fps = getattr(self, "_vid_fps", 30.0)
        self._lbl_time.setText(self._fmt_time(frame_number / fps))

    def _on_seek_released(self):
        if not self._playing:
            return
        # Defensive gate for very fast release where press handler may lag.
        if self._audio_available:
            self._set_audio_paused(True)
        self._arm_mute_until_fps_recovery()
        if not self._scrub_muted:
            self._scrub_muted = True
            self._apply_volume_to_backends()
        target_frame = int(self._seek_slider.value())
        self._last_seek_frame = target_frame
        self._audio_sync_frame_hint = target_frame
        fps = getattr(self, "_vid_fps", 30.0)

        if self._worker.is_paused:
            self._pending_seek_on_resume = target_frame
            self._lbl_time.setText(self._fmt_time(target_frame / max(fps, 1e-6)))
            self.statusBar().showMessage(
                f"Seek queued to {self._fmt_time(target_frame / max(fps, 1e-6))}. Press Resume to apply."
            )
            return

        self._worker.request_seek(target_frame)
        self._seek_audio_seconds(target_frame / max(fps, 1e-6))
        now_t = time.perf_counter()
        self._audio_seek_guard_until = now_t + 1.0
        self._audio_resync_pending = True
        self._audio_fps_recovered = False
        if self._audio_available:
            QTimer.singleShot(420, self._ensure_selected_audio_track_qt)
        self._post_seek_resync_frames = 120
        self._resume_audio_after_seek = bool(self._audio_available)
        self._seek_resume_target = int(target_frame)
        self._seek_resume_started_t = time.perf_counter()
        if self._disp_hdr_mpv is not None and not self._audio_available:
            self._disp_hdr_mpv.seek_seconds(target_frame / max(fps, 1e-6))
        self._request_playhead_skip_relock_after_unmute()

    def _on_seek_frame_ready(self, frame_number: int):
        """Worker callback: first rendered frame after seek is now visible."""
        if not self._playing:
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
