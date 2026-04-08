from __future__ import annotations

import time
import numpy as np
from PyQt6.QtCore import QTimer

from gui_config import SOURCE_MODE_WINDOW, _normalize_source_mode


class WorkerSlotsMixin:
    """Worker/mpv signal handlers and live metrics UI updates."""

    def _on_mpv_notice(self, text: str):
        note = str(text or "").strip()
        if not note:
            return
        now_t = time.perf_counter()
        last_note = str(getattr(self, "_last_mpv_notice", "") or "")
        last_t = float(getattr(self, "_last_mpv_notice_t", 0.0) or 0.0)
        # Prevent duplicate notices when both HDR and SDR mpv panes report.
        if note == last_note and (now_t - last_t) < 2.0:
            return
        self._last_mpv_notice = note
        self._last_mpv_notice_t = now_t
        self.statusBar().showMessage(note, 5500)

    def _on_hdr_info(self, info: dict):
        """Update the HDR Info panel from mpv metadata."""
        vo_confirmed = bool(info.get("hdr_vo_confirmed", False))
        metadata_forced = bool(info.get("hdr_metadata_forced", False))
        if vo_confirmed:
            self._hdr_labels["status"].setText("HDR: Active (VO confirmed)")
            self._hdr_labels["status"].setStyleSheet("color: #00e676;")
        elif metadata_forced:
            self._hdr_labels["status"].setText("HDR: Metadata tagged (VO unconfirmed)")
            self._hdr_labels["status"].setStyleSheet("color: #ffca28;")
        else:
            self._hdr_labels["status"].setText("HDR: Inactive")
            self._hdr_labels["status"].setStyleSheet("color: #ff5252;")
        self._hdr_labels["primaries"].setText(f"Primaries: {info.get('primaries', '?')}")
        self._hdr_labels["transfer"].setText(f"Transfer: {info.get('transfer', '?')}")
        peak_raw = info.get("sig_peak", "?")
        try:
            nits = float(peak_raw) * 203  # sig-peak is relative to ref white
            peak_str = f"{nits:.0f} nits (sig-peak {float(peak_raw):.1f})"
        except (ValueError, TypeError):
            peak_str = str(peak_raw)
        self._hdr_labels["peak"].setText(f"Peak: {peak_str}")
        self._hdr_labels["vo"].setText(f"VO: {info.get('vo', '?')}/{info.get('gpu_api', '?')}")

    def _on_frame(self, sdr, hdr):
        sdr_show = sdr
        self._last_sdr_frame = sdr_show
        self._last_hdr_frame = hdr
        if (
            self._disp_sdr_mpv is not None
            and self._playing
            and self._active_use_mpv
            and not self._sdr_mpv_feed_from_worker
        ):
            try:
                rgb16 = np.ascontiguousarray(sdr_show[:, :, ::-1].astype(np.uint16) * 257)
                self._disp_sdr_mpv.feed_frame(rgb16.data)
            except Exception:
                pass
        if self._disp_sdr_cpu is not None and self._disp_sdr_cpu.isVisible():
            self._disp_sdr_cpu.update_frame(sdr_show)
        # QLabel fallbacks (mpv panes are fed directly from the worker)
        if self._disp_hdr_cpu is not None and self._disp_hdr_cpu.isVisible():
            self._disp_hdr_cpu.update_frame(hdr)

    def _on_metrics(self, m):
        self._m["fps"].setText(f"FPS: {m['fps']:.1f}")
        is_window_source = (
            _normalize_source_mode(getattr(self, "_source_mode", None))
            == SOURCE_MODE_WINDOW
        )
        shown_latency_ms = float(m.get("latency_ms", 0.0) or 0.0)
        self._m["latency"].setText(f"Latency: {shown_latency_ms:.1f} ms")
        self._m["frame"].setText(
            "Frame: Live" if is_window_source else f"Frame: {m['frame']}"
        )
        self._m["res"].setText(f"Res: {m['proc_res']}")
        self._m["gpu"].setText(f"VRAM: {m['gpu_mb']:.0f} MB")
        self._m["cpu"].setText(f"CPU: {m['cpu_mb']:.0f} MB")
        self._m["model"].setText(f"Model: {m['model_mb']:.2f} MB")
        self._m["prec"].setText(f"Prec: {m['precision']}")

        try:
            self._audio_sync_frame_hint = max(
                0, int(m.get("frame", getattr(self, "_last_seek_frame", 0)))
            )
        except Exception:
            self._audio_sync_frame_hint = max(0, int(getattr(self, "_last_seek_frame", 0)))

        fps_now = float(m.get("fps", 0.0))
        if is_window_source:
            self._fps_prev_sample = None
            self._fps_is_stable = False
            self._fps_stable_count = 0
            self._fps_unstable_count = 0
            self._fps_stable_since_t = 0.0
            if self._auto_muted_low_fps:
                self._set_low_fps_mute(False)
        else:
            self._update_auto_mute_from_fps(fps_now)
        if self._startup_audio_gate_active:
            bypass_gate = not bool(
                getattr(self, "_enable_low_fps_audio_mute", True)
            )
            gate_ready = (
                True
                if bypass_gate
                else self._is_stability_hold_satisfied(
                    strict=bool(self._ui_resync_gate_strict)
                )
            )
            if gate_ready:
                self._startup_audio_gate_active = False
                self._ui_resync_gate_strict = False
                if self._scrub_muted:
                    self._scrub_muted = False
                    self._apply_volume_to_backends()
                if (
                    self._audio_available
                    and not self._worker.is_paused
                    and not self._startup_sync_pending
                    and not self._auto_muted_low_fps
                    and not self._scrub_muted
                    and not self._relock_hold_muted
                ):
                    fps = getattr(self, "_vid_fps", 30.0)
                    cur_frame = int(m.get("frame", self._last_seek_frame))
                    sec = float(cur_frame) / max(fps, 1e-6)
                    self._force_audio_seek(sec)
                    if self._active_use_mpv:
                        if not self._pending_playhead_relock_on_unmute:
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
                        self._dispatch_pending_playhead_relock_after_unmute()
                    else:
                        self._set_audio_paused(False)
                        now_t = time.perf_counter()
                        grace_s = float(getattr(self, "_post_recovery_mute_grace_s", 1.6))
                        self._auto_mute_rearm_until = max(
                            float(getattr(self, "_auto_mute_rearm_until", 0.0)),
                            now_t + max(0.0, grace_s),
                        )
        # Startup fast-seek edge case: deferred relock can remain pending
        # until another mute/unmute cycle unless we explicitly re-check here.
        if self._pending_playhead_relock_on_unmute:
            self._dispatch_pending_playhead_relock_after_unmute()

    def _on_status_message(self, text: str):
        """Forward worker status to status bar and compile dialog."""
        self.statusBar().showMessage(text)
        if self._compile_dlg is not None:
            self._compile_dlg.set_status(text)
        if self._precision_swap_pending is not None and text.startswith("Ready"):
            ready_key = str(text)
            for prefix in ("Ready — ", "Ready - ", "Ready â€” "):
                if ready_key.startswith(prefix):
                    ready_key = ready_key.split(prefix, 1)[-1].strip()
                    break
            ready_prec = ready_key.split("@", 1)[0].strip()
            pending_prec = str(self._precision_swap_pending).split("[", 1)[0].strip()
            if ready_prec == pending_prec:
                self._resume_after_precision_swap()
        if self._precision_swap_pending is not None and text.startswith("ERROR:"):
            self._resume_after_precision_swap(force=True)

    def _on_finished(self):
        restart_on_eof = bool(
            self._video_path
            and (not self._suppress_eof_restart_once)
            and (not self._ui_closing)
        )
        # Consume one-shot suppression (manual stop/restart paths).
        self._suppress_eof_restart_once = False
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.stop_playback()
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.stop_playback()
        self._stop_audio_playback()
        self._set_process_priority(False)
        self._reset_controls()
        if restart_on_eof:
            self.statusBar().showMessage("Playback finished. Restarting app ...")
            QTimer.singleShot(0, self._restart_app_clean)
        else:
            self.statusBar().showMessage("Playback finished.")
        if self._disp_sdr_cpu is not None:
            self._disp_sdr_cpu.clear_display()
        if self._disp_hdr_cpu is not None:
            self._disp_hdr_cpu.clear_display()
