from __future__ import annotations

import math
import time

from PyQt6.QtCore import QTimer


class AutoMuteMixin:
    """Low-FPS auto-mute policy helpers for MainWindow."""

    def _is_stability_hold_satisfied(self, strict: bool = False) -> bool:
        """Require continuous stable FPS for a minimum hold duration."""
        if not bool(getattr(self, "_fps_is_stable", False)):
            return False
        need = int(
            getattr(
                self,
                "_audio_stable_need_strict" if strict else "_audio_stable_need",
                8 if strict else 6,
            )
        )
        if int(getattr(self, "_fps_stable_count", 0)) < max(1, need):
            return False
        hold_s = float(
            getattr(
                self,
                "_audio_stable_hold_strict_s" if strict else "_audio_stable_hold_s",
                0.65 if strict else 0.45,
            )
        )
        stable_since = float(getattr(self, "_fps_stable_since_t", 0.0))
        if stable_since <= 0.0:
            return False
        return (time.perf_counter() - stable_since) >= max(0.0, hold_s)

    def _update_fps_stability_state(self, fps_value: float) -> bool:
        """Track whether FPS is stable enough to safely restore audio."""
        try:
            fps = float(fps_value)
        except (TypeError, ValueError):
            fps = float("nan")

        prev = getattr(self, "_fps_prev_sample", None)
        min_fps = float(getattr(self, "_audio_stability_min_fps", 6.0))
        delta_abs = float(getattr(self, "_audio_stability_delta_abs", 1.2))
        delta_rel = float(getattr(self, "_audio_stability_delta_rel", 0.10))

        if not math.isfinite(fps):
            self._fps_prev_sample = None
            self._fps_is_stable = False
            self._fps_stable_count = 0
            self._fps_stable_since_t = 0.0
            self._fps_unstable_count += 1
            return False

        self._fps_prev_sample = fps
        if prev is None or not math.isfinite(float(prev)):
            self._fps_is_stable = False
            self._fps_stable_count = 0
            self._fps_unstable_count = 0
            self._fps_stable_since_t = 0.0
            return False

        prev_fps = float(prev)
        allowed_delta = max(delta_abs, delta_rel * max(abs(prev_fps), abs(fps), 1.0))
        stable_now = (
            fps >= min_fps
            and prev_fps >= min_fps
            and abs(fps - prev_fps) <= allowed_delta
        )
        was_stable = bool(getattr(self, "_fps_is_stable", False))
        if stable_now:
            self._fps_stable_count += 1
            self._fps_unstable_count = max(0, self._fps_unstable_count - 1)
            if (not was_stable) or float(getattr(self, "_fps_stable_since_t", 0.0)) <= 0.0:
                self._fps_stable_since_t = time.perf_counter()
        else:
            self._fps_unstable_count += 1
            self._fps_stable_count = max(0, self._fps_stable_count - 1)
            self._fps_stable_since_t = 0.0
        self._fps_is_stable = stable_now
        return stable_now

    def _dispatch_pending_playhead_relock_after_unmute(self) -> bool:
        """Dispatch deferred seek relock once mute guards have actually cleared."""
        if not self._pending_playhead_relock_on_unmute:
            return False
        if not self._playing or not getattr(self, "_active_use_mpv", False):
            self._pending_playhead_relock_on_unmute = False
            self._pending_playhead_relock_pre_delay_ms = -1
            return False
        if (
            self._worker is None
            or self._worker.is_paused
            or self._startup_sync_pending
            or self._startup_audio_gate_active
        ):
            return False

        if self._seek_slider.isSliderDown():
            return False
        if self._scrub_muted:
            self._scrub_muted = False
            self._apply_volume_to_backends()

        if self._auto_muted_low_fps or self._scrub_muted or self._relock_hold_muted:
            return False

        self._pending_playhead_relock_on_unmute = False
        pending_pre_ms = int(getattr(self, "_pending_playhead_relock_pre_delay_ms", -1))
        self._pending_playhead_relock_pre_delay_ms = -1
        first_ms = int(getattr(self, "_pending_playhead_relock_first_ms", 35))
        settle_ms = int(getattr(self, "_pending_playhead_relock_settle_ms", 220))
        pre_delay_ms = (
            pending_pre_ms
            if pending_pre_ms >= 0
            else 0
        )
        post_fade_ms = int(getattr(self, "_post_relock_fade_ms", 80))

        # Hold mute until relock settles.
        self._relock_hold_muted = True
        self._apply_volume_to_backends()
        if self._audio_available:
            self._set_audio_paused(True)

        def _run_relock():
            if not self._playing:
                self._relock_hold_muted = False
                return
            if self._auto_muted_low_fps or self._scrub_muted:
                self._pending_playhead_relock_on_unmute = True
                self._pending_playhead_relock_pre_delay_ms = int(pre_delay_ms)
                self._pending_playhead_relock_first_ms = first_ms
                self._pending_playhead_relock_settle_ms = settle_ms
                self._relock_hold_muted = False
                self._apply_volume_to_backends()
                return
            self._schedule_playhead_skip_relock(
                first_delay_ms=first_ms,
                settle_delay_ms=settle_ms,
            )

            # Release hold after relock settle and apply a short fade-in.
            def _release_after_relock():
                if not self._playing:
                    self._relock_hold_muted = False
                    return
                if self._auto_muted_low_fps or self._scrub_muted:
                    self._relock_hold_muted = False
                    self._apply_volume_to_backends()
                    return
                self._relock_hold_muted = False
                self._apply_volume_to_backends()
                if (
                    self._audio_available
                    and not self._worker.is_paused
                    and not self._startup_sync_pending
                ):
                    # Final anchor right before unmute to avoid tiny audio-ahead drift.
                    fps = getattr(self, "_vid_fps", 30.0)
                    anchor_frame = (
                        int(self._sync_anchor_frame())
                        if hasattr(self, "_sync_anchor_frame")
                        else int(getattr(self, "_last_seek_frame", 0))
                    )
                    sec = float(anchor_frame) / max(fps, 1e-6)
                    self._seek_audio_seconds(sec)
                    QTimer.singleShot(20, lambda: self._seek_audio_seconds(sec))
                    self._set_audio_paused(False)
                now_t = time.perf_counter()
                grace_s = float(getattr(self, "_post_recovery_mute_grace_s", 1.6))
                self._auto_mute_rearm_until = max(
                    float(getattr(self, "_auto_mute_rearm_until", 0.0)),
                    now_t + max(0.0, grace_s),
                )
                self._start_audio_restore_fade(duration_ms=post_fade_ms)
                self.statusBar().showMessage("Audio restored (playback stable).")

            QTimer.singleShot(
                max(50, int(first_ms) + int(settle_ms)),
                _release_after_relock,
            )

        QTimer.singleShot(max(0, pre_delay_ms), _run_relock)
        return True

    def _set_low_fps_mute(self, enabled: bool):
        enabled = bool(enabled)
        if enabled and not bool(getattr(self, "_enable_low_fps_audio_mute", True)):
            return
        if self._auto_muted_low_fps == enabled:
            return
        self._auto_muted_low_fps = enabled
        if enabled:
            # Arm one-shot A/V re-sync for when playback recovers.
            self._audio_resync_pending = True
            self._audio_fps_recovered = False
            self._relock_hold_muted = False
            # Keep audio clock from running ahead while muted.
            if self._audio_available:
                self._set_audio_paused(True)
            self._apply_volume_to_backends()
            self.statusBar().showMessage("Audio auto-muted (playback unstable).")
        else:
            self._audio_fps_recovered = True
            # Re-anchor audio to current video position before unmuting.
            if self._audio_available and self._playing:
                fps = getattr(self, "_vid_fps", 30.0)
                anchor_frame = (
                    int(self._sync_anchor_frame())
                    if hasattr(self, "_sync_anchor_frame")
                    else int(getattr(self, "_last_seek_frame", 0))
                )
                self._seek_audio_seconds(float(anchor_frame) / max(fps, 1e-6))
                if (
                    self._active_use_mpv
                    and not self._pending_playhead_relock_on_unmute
                ):
                    self._pending_playhead_relock_on_unmute = True
                    self._pending_playhead_relock_pre_delay_ms = int(
                        getattr(self, "_stability_relock_pre_delay_ms", 0)
                    )
                    self._pending_playhead_relock_first_ms = int(
                        getattr(self, "_stability_relock_first_ms", 60)
                    )
                    self._pending_playhead_relock_settle_ms = int(
                        getattr(self, "_stability_relock_settle_ms", 300)
                    )
                if (
                    not self._pending_playhead_relock_on_unmute
                    and not self._worker.is_paused
                    and not self._startup_sync_pending
                    and not self._startup_audio_gate_active
                ):
                    self._set_audio_paused(False)
                elif self._pending_playhead_relock_on_unmute or self._startup_audio_gate_active:
                    self._set_audio_paused(True)
            if self._pending_playhead_relock_on_unmute and not self._startup_audio_gate_active:
                self._dispatch_pending_playhead_relock_after_unmute()
            else:
                if not self._startup_audio_gate_active:
                    if self._scrub_muted and (not self._seek_slider.isSliderDown()):
                        self._scrub_muted = False
                        self._apply_volume_to_backends()
                    now_t = time.perf_counter()
                    grace_s = float(getattr(self, "_post_recovery_mute_grace_s", 1.6))
                    self._auto_mute_rearm_until = max(
                        float(getattr(self, "_auto_mute_rearm_until", 0.0)),
                        now_t + max(0.0, grace_s),
                    )
                    self._start_audio_restore_fade()
            if (
                not self._startup_audio_gate_active
                and not self._pending_playhead_relock_on_unmute
            ):
                self.statusBar().showMessage("Audio restored (playback stable).")

    def _arm_mute_until_fps_recovery(self):
        """Force mute now; unmute only via measured FPS recovery logic."""
        self._fps_prev_sample = None
        self._fps_is_stable = False
        self._fps_stable_count = 0
        self._fps_unstable_count = 0
        self._fps_stable_since_t = 0.0
        self._auto_mute_rearm_until = 0.0
        self._audio_resync_pending = True
        if not bool(getattr(self, "_enable_low_fps_audio_mute", True)):
            self._audio_fps_recovered = True
            return
        self._audio_fps_recovered = False
        if not self._auto_muted_low_fps:
            self._set_low_fps_mute(True)

    def _update_auto_mute_from_fps(self, fps_value: float):
        """Auto-mute policy based on sustained instability, not FPS target hit."""
        if not bool(getattr(self, "_enable_low_fps_audio_mute", True)):
            self._fps_prev_sample = None
            self._fps_is_stable = False
            self._fps_stable_count = 0
            self._fps_unstable_count = 0
            self._fps_stable_since_t = 0.0
            if self._auto_muted_low_fps:
                self._set_low_fps_mute(False)
            return

        self._update_fps_stability_state(fps_value)
        now_t = time.perf_counter()
        rearm_until = float(getattr(self, "_auto_mute_rearm_until", 0.0))
        unstable_need = max(1, int(getattr(self, "_audio_unstable_need", 4)))
        stable_ready = self._is_stability_hold_satisfied(strict=False)
        muting_armed = now_t >= rearm_until
        if (not self._auto_muted_low_fps) and (not muting_armed):
            self._fps_unstable_count = 0

        if (
            (not self._auto_muted_low_fps)
            and muting_armed
            and self._fps_unstable_count >= unstable_need
        ):
            self._set_low_fps_mute(True)
            self._fps_unstable_count = 0
        elif self._auto_muted_low_fps and stable_ready:
            self._set_low_fps_mute(False)
            self._fps_stable_count = 0
