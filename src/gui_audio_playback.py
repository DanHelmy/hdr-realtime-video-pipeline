from __future__ import annotations

from PyQt6.QtCore import QTimer, QUrl

try:
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

    _HAS_QT_AUDIO = True
except ImportError:
    QMediaPlayer = None
    QAudioOutput = None
    _HAS_QT_AUDIO = False

from gui_media_probe import _probe_audio_streams


class AudioPlaybackMixin:
    """Audio backend, track selection, and volume control helpers for MainWindow."""

    def _init_audio_backend(self):
        """Initialize Qt audio backend (used for seekable timeline audio)."""
        if not _HAS_QT_AUDIO:
            self._audio_available = False
            return
        try:
            self._audio_player = QMediaPlayer(self)
            self._audio_output = QAudioOutput(self)
            self._audio_output.setVolume(self._volume_percent / 100.0)
            self._audio_player.setAudioOutput(self._audio_output)
            self._audio_available = True
        except Exception:
            self._audio_player = None
            self._audio_output = None
            self._audio_available = False

    @staticmethod
    def _format_audio_track_label(track: dict, fallback_idx: int) -> str:
        n = int(track.get("ordinal", fallback_idx)) + 1
        lang = str(track.get("language") or "und").upper()
        codec = str(track.get("codec") or "audio").upper()
        ch = int(track.get("channels") or 0)
        ch_sfx = f" {ch}ch" if ch > 0 else ""
        title = str(track.get("title") or "").strip()
        default_sfx = " (Default)" if bool(track.get("is_default", False)) else ""
        base = f"{n}. {lang} {codec}{ch_sfx}"
        if title:
            return f"{base} - {title}{default_sfx}"
        return f"{base}{default_sfx}"

    def _refresh_audio_tracks_for_video(self, path: str):
        tracks = _probe_audio_streams(path)
        self._audio_tracks = tracks
        self._cmb_audio_track.blockSignals(True)
        self._cmb_audio_track.clear()
        if not tracks:
            self._cmb_audio_track.addItem("No audio tracks detected")
            self._cmb_audio_track.setEnabled(False)
            self._selected_audio_track = 0
            self._cmb_audio_track.blockSignals(False)
            return

        default_idx = 0
        for i, t in enumerate(tracks):
            self._cmb_audio_track.addItem(self._format_audio_track_label(t, i), i)
            if bool(t.get("is_default", False)):
                default_idx = i

        preferred = self._selected_audio_track
        if preferred < 0 or preferred >= len(tracks):
            preferred = default_idx
        self._selected_audio_track = preferred
        self._cmb_audio_track.setCurrentIndex(preferred)
        self._cmb_audio_track.setEnabled(len(tracks) > 1)
        if len(tracks) > 1:
            self._cmb_audio_track.setToolTip("Choose audio stream from source file.")
        else:
            self._cmb_audio_track.setToolTip("Single audio stream detected.")
        self._cmb_audio_track.blockSignals(False)

    def _apply_selected_audio_track_qt_async(self):
        if not self._audio_available or self._audio_player is None:
            return
        if not self._audio_tracks:
            return
        target = max(0, min(int(self._selected_audio_track), len(self._audio_tracks) - 1))
        self._audio_apply_token += 1
        token = self._audio_apply_token

        def _try_apply(attempt: int = 0):
            if token != self._audio_apply_token:
                return
            p = self._audio_player
            if p is None:
                return
            try:
                qtracks = p.audioTracks()
            except Exception:
                qtracks = []
            if qtracks:
                idx = max(0, min(target, len(qtracks) - 1))
                try:
                    p.setActiveAudioTrack(idx)
                    return
                except Exception:
                    pass
            if attempt < 20:
                QTimer.singleShot(120, lambda: _try_apply(attempt + 1))

        _try_apply(0)

    def _ensure_selected_audio_track_qt(self):
        """Re-assert selected Qt audio track if backend switched tracks after seek/rebuffer."""
        if not self._audio_available or self._audio_player is None or not self._audio_tracks:
            return
        p = self._audio_player
        try:
            qtracks = p.audioTracks()
        except Exception:
            qtracks = []
        if not qtracks:
            return
        target = max(0, min(int(self._selected_audio_track), len(qtracks) - 1))
        try:
            active = int(p.activeAudioTrack())
        except Exception:
            active = None
        if active != target:
            try:
                p.setActiveAudioTrack(target)
            except Exception:
                pass

    def _apply_selected_audio_track_mpv_async(self):
        if self._audio_available:
            return
        if self._disp_hdr_mpv is None or not self._audio_tracks:
            return
        target = max(0, min(int(self._selected_audio_track), len(self._audio_tracks) - 1))
        self._audio_apply_token += 1
        token = self._audio_apply_token

        def _try_apply(attempt: int = 0):
            if token != self._audio_apply_token:
                return
            if self._disp_hdr_mpv is None:
                return
            if self._disp_hdr_mpv.set_audio_track_ordinal(target):
                return
            if attempt < 20:
                QTimer.singleShot(150, lambda: _try_apply(attempt + 1))

        _try_apply(0)

    def _apply_volume_to_backends(self):
        """Apply current volume/mute policy to Qt audio and mpv fallback audio."""
        muted = self._auto_muted_low_fps or self._scrub_muted or self._relock_hold_muted
        if muted and self._audio_fade_timer is not None:
            self._audio_fade_timer.stop()
        if self._audio_output is not None:
            self._audio_output.setVolume(
                0.0 if muted else (self._volume_percent / 100.0)
            )
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.set_muted(muted)
            if not muted:
                self._disp_hdr_mpv.set_volume_percent(self._volume_percent)

    def _start_audio_restore_fade(self, duration_ms: int | None = None):
        """Smoothly restore audio level after auto-mute release."""
        if self._auto_muted_low_fps:
            return
        if duration_ms is None:
            duration_ms = int(getattr(self, "_audio_restore_fade_ms", 140))
        if self._audio_fade_timer is None:
            self._audio_fade_timer = QTimer(self)
            self._audio_fade_timer.timeout.connect(self._on_audio_fade_tick)
        self._audio_fade_step_idx = 0
        if self._audio_output is not None:
            self._audio_output.setVolume(0.0)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.set_muted(False)
            self._disp_hdr_mpv.set_volume_percent(0)
        step_ms = max(10, int(duration_ms / max(1, self._audio_fade_steps)))
        self._audio_fade_timer.start(step_ms)

    def _on_audio_fade_tick(self):
        if self._auto_muted_low_fps:
            if self._audio_fade_timer is not None:
                self._audio_fade_timer.stop()
            return
        self._audio_fade_step_idx += 1
        ratio = min(1.0, self._audio_fade_step_idx / max(1, self._audio_fade_steps))
        target = max(0.0, min(1.0, self._volume_percent / 100.0))
        if self._audio_output is not None:
            self._audio_output.setVolume(target * ratio)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.set_volume_percent(int(round(self._volume_percent * ratio)))
        if ratio >= 1.0 and self._audio_fade_timer is not None:
            self._audio_fade_timer.stop()
            self._apply_volume_to_backends()

    def _on_volume_changed(self, value: int):
        self._volume_percent = int(value)
        self._lbl_volume_val.setText(f"{self._volume_percent}%")
        if self._audio_fade_timer is not None and self._audio_fade_timer.isActive():
            # Fade tick uses current _volume_percent as target.
            self._save_user_settings()
            return
        self._apply_volume_to_backends()
        self._save_user_settings()

    def _on_audio_track_changed(self, index: int):
        if index < 0:
            return
        self._selected_audio_track = int(index)
        self._save_user_settings()
        if not self._playing:
            return
        if self._audio_available:
            self._apply_selected_audio_track_qt_async()
        else:
            self._apply_selected_audio_track_mpv_async()

    def _start_audio_playback(self, path: str):
        if not self._audio_available or self._audio_player is None:
            return
        try:
            self._audio_player.stop()
            self._audio_player.setSource(QUrl.fromLocalFile(path))
            self._audio_player.setPosition(0)
            self._audio_player.setPlaybackRate(1.0)
            self._audio_player.play()
            self._apply_selected_audio_track_qt_async()
            self._apply_volume_to_backends()
        except Exception as exc:
            self.statusBar().showMessage(f"Qt audio unavailable: {exc}")
            self._audio_available = False

    def _stop_audio_playback(self):
        if self._audio_player is not None:
            try:
                self._audio_player.stop()
            except Exception:
                pass

    def _set_audio_paused(self, paused: bool):
        if not self._audio_available or self._audio_player is None:
            return
        try:
            if paused:
                self._audio_player.pause()
            else:
                self._audio_player.setPlaybackRate(1.0)
                self._audio_player.play()
        except Exception:
            pass

    def _seek_audio_seconds(self, sec: float):
        if not self._audio_available or self._audio_player is None:
            return
        try:
            self._audio_player.setPosition(int(max(0.0, sec) * 1000.0))
        except Exception:
            pass

    def _release_startup_sync(self):
        """Unpause worker/mpv/audio together after startup warm sync."""
        if not self._playing or not self._startup_sync_pending:
            return
        if self._user_pause_override_startup:
            # User explicitly paused during startup; do not auto-resume.
            self._startup_sync_pending = False
            return
        self._startup_sync_pending = False
        if self._audio_available:
            self._seek_audio_seconds(0.0)
            # Keep audio paused until startup FPS gate opens.
            self._set_audio_paused(True)
        self._worker.resume()
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.set_paused(False)
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.set_paused(False)
        if self._startup_audio_gate_active:
            self._arm_mute_until_fps_recovery()
        # Force an initial timeline relock so audio isn't ahead if user never seeks.
        self._relock_timeline(delay_ms=160, drop_frames=3)
        # Follow-up relock after startup UI interactions to keep HDR aligned.
        QTimer.singleShot(520, lambda: self._relock_timeline(drop_frames=2))
