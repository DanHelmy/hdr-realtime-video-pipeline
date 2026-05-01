from __future__ import annotations

import os

from PyQt6.QtCore import QTimer, Qt

from gui_config import (
    DEFAULT_USE_HG,
    PRECISIONS,
    RESOLUTION_SCALES,
    SOURCE_MODE_VIDEO,
    _normalize_source_mode,
)
from window_capture_source import target_from_hwnd
from gui_scaling import UPSCALER_CHOICES


class StateInitMixin:
    """MainWindow state initialization split from gui.py."""

    def _init_runtime_state(self, initial_start_frame, root_dir: str, has_qt_audio: bool):
        self._video_path = None
        self._source_mode = SOURCE_MODE_VIDEO
        self._source_mode_prompt_pending = False
        self._capture_target = None
        self._playing = False
        self._compile_dlg = None
        self._pending_mpv_start = None
        self._pending_sdr_mpv_start = None
        self._last_res = None
        self._active_precision = None
        self._active_resolution = None
        self._active_use_mpv = False
        self._sdr_mpv_feed_from_worker = False
        self._active_use_hg = bool(DEFAULT_USE_HG)
        self._active_film_grain = False
        self._source_hdr_info = {"is_hdr": False, "reason": "unknown"}
        self._last_seek_frame = 0
        self._audio_sync_frame_hint = 0
        self._post_seek_resync_frames = 0
        self._pending_seek_on_resume = None
        self._audio_last_hard_sync_t = 0.0
        self._audio_seek_guard_until = 0.0
        self._audio_resync_pending = False
        self._audio_fps_recovered = True
        self._auto_mute_rearm_until = 0.0
        self._post_recovery_mute_grace_s = max(
            0.0, float(os.environ.get("HDRTVNET_POST_RECOVERY_MUTE_GRACE_S", "1.6"))
        )
        self._precision_swap_mute_grace_s = max(
            0.0, float(os.environ.get("HDRTVNET_PRECISION_SWAP_MUTE_GRACE_S", "2.5"))
        )
        self._audio_drift_check_stride = 10
        self._startup_audio_gate_active = False
        self._user_pause_override_startup = False
        self._ui_resync_gate_strict = False
        self._fps_prev_sample = None
        self._fps_is_stable = False
        self._fps_stable_count = 0
        self._fps_unstable_count = 0
        self._fps_stable_since_t = 0.0
        self._audio_stable_hold_s = max(
            0.0, float(os.environ.get("HDRTVNET_AUDIO_STABLE_HOLD_S", "0.55"))
        )
        self._audio_stable_hold_strict_s = max(
            self._audio_stable_hold_s,
            float(os.environ.get("HDRTVNET_AUDIO_STABLE_HOLD_STRICT_S", "0.80")),
        )
        self._audio_stability_min_fps = max(
            1.0, float(os.environ.get("HDRTVNET_AUDIO_STABILITY_MIN_FPS", "6.0"))
        )
        self._audio_stability_delta_abs = max(
            0.05, float(os.environ.get("HDRTVNET_AUDIO_STABILITY_DELTA_ABS", "1.2"))
        )
        self._audio_stability_delta_rel = max(
            0.01, float(os.environ.get("HDRTVNET_AUDIO_STABILITY_DELTA_REL", "0.10"))
        )
        self._audio_stable_need = max(
            1, int(os.environ.get("HDRTVNET_AUDIO_STABILITY_SAMPLES", "8"))
        )
        self._audio_stable_need_strict = max(
            self._audio_stable_need + 1,
            int(os.environ.get("HDRTVNET_AUDIO_STABILITY_SAMPLES_STRICT", "12")),
        )
        self._audio_unstable_need = max(
            1, int(os.environ.get("HDRTVNET_AUDIO_UNSTABLE_SAMPLES", "4"))
        )
        self._enable_low_fps_audio_mute = (
            os.environ.get("HDRTVNET_ENABLE_LOW_FPS_AUDIO_MUTE", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        # Low/high FPS mute thresholds are scaled from this 24fps baseline.
        self._low_fps_trip = 19.5
        self._high_fps_trip = 22.5
        self._resume_audio_after_seek = False
        self._seek_resume_target = 0
        self._seek_resume_started_t = 0.0
        self._precision_swap_pending = None
        self._precision_pause_armed = False
        self._precision_swap_timer = None
        self._last_user_pause_t = 0.0
        self._audio_player = None
        self._audio_output = None
        self._audio_available = bool(has_qt_audio)
        self._live_audio_controller = None
        self._live_audio_active = False
        self._pending_live_audio_start_target = None
        self._pending_live_audio_retry_at = 0.0
        self._live_audio_compile_ready = False
        self._live_audio_has_latency_hint = False
        self._live_audio_default_delay_ms = max(
            50.0, float(os.environ.get("HDRTVNET_LIVE_AUDIO_DEFAULT_DELAY_MS", "95"))
        )
        self._live_audio_delay_margin_ms = max(
            0.0, float(os.environ.get("HDRTVNET_LIVE_AUDIO_DELAY_MARGIN_MS", "10"))
        )
        self._live_audio_min_delay_ms = max(
            40.0, float(os.environ.get("HDRTVNET_LIVE_AUDIO_MIN_DELAY_MS", "70"))
        )
        self._live_audio_max_delay_ms = max(
            self._live_audio_min_delay_ms,
            float(os.environ.get("HDRTVNET_LIVE_AUDIO_MAX_DELAY_MS", "650")),
        )
        self._live_tab_audio_floor_delay_ms = max(
            self._live_audio_min_delay_ms,
            float(os.environ.get("HDRTVNET_LIVE_TAB_AUDIO_FLOOR_DELAY_MS", "80")),
        )
        self._live_audio_target_delay_ms = self._live_audio_default_delay_ms
        self._audio_tracks = []
        self._selected_audio_track = 0
        self._audio_apply_token = 0
        self._volume_percent = 100
        self._auto_muted_low_fps = False
        self._scrub_muted = False
        self._scrub_unmute_seq = 0
        self._audio_fade_timer = None
        self._audio_fade_steps = 8
        self._audio_fade_step_idx = 0
        self._audio_restore_fade_ms = max(
            80, int(os.environ.get("HDRTVNET_AUDIO_RESTORE_FADE_MS", "140"))
        )
        self._post_relock_fade_ms = max(
            40, int(os.environ.get("HDRTVNET_POST_RELOCK_FADE_MS", "80"))
        )
        self._stability_relock_first_ms = max(
            10, int(os.environ.get("HDRTVNET_STABILITY_RELOCK_FIRST_MS", "60"))
        )
        self._stability_relock_settle_ms = max(
            80, int(os.environ.get("HDRTVNET_STABILITY_RELOCK_SETTLE_MS", "300"))
        )
        self._stability_relock_pre_delay_ms = max(
            0, int(os.environ.get("HDRTVNET_STABILITY_RELOCK_PRE_DELAY_MS", "0"))
        )
        self._relock_hold_muted = False
        self._proc_priority_saved = None
        self._app_active = True
        self._deferred_mpv_refresh = False
        self._cursor_idle_timer = None
        self._cursor_idle_enabled = True
        self._cursor_hidden = False
        self._cursor_idle_ms = 1500
        self._startup_sync_pending = False
        self._startup_frame_relock_pending = False
        self._startup_frame_relock_token = 0
        self._last_sdr_frame = None
        self._last_hdr_frame = None
        self._compare_dialog = None
        self._compare_snapshot_pending = False
        self._compare_progress_dlg = None
        self._compare_resume_after_cancel = False
        self._compare_anchor_frame = None
        self._source_proc_dims = None
        self._source_video_dims = None
        self._source_max_resolution_key = "1080p"
        self._last_open_dir = root_dir
        self._last_export_dir = root_dir
        self._predequantize_mode = "auto"
        self._runtime_execution_mode = "compile"
        self._suppress_hip_sdk_warning = False
        self._startup_hip_sdk_warning_shown = False
        self._suppress_octave_compare_warning = False
        self._compare_octave_warning_shown = False
        self._autotune_warning_needed = False
        self._hdr_ground_truth_path = None
        self._objective_metrics_enabled = False
        self._playback_log_enabled = False
        self._playback_log_session_started = False
        self._playback_log_source_label = ""
        self._playback_log_dir = os.path.join(root_dir, "logs", "playback_sessions")
        self._playback_log_compare_events = []
        self._playback_log_pending_compare_index = None
        self._playback_log_runtime_samples = []
        self._playback_log_started_wall_time = ""
        self._playback_log_started_perf = 0.0
        self._borderless_full_window = False
        self._mpv_start_resync_t = 0.0
        self._ui_hidden = False
        self._ui_overlay_btn = None
        self._ui_overlay_timer = None
        self._ui_overlay_hide_ms = 1200
        self._ui_anim_effects = {}
        self._ui_anim_running = {}
        self._ui_anim_duration_ms = 0
        self._layout_freeze_depth = 0
        self._saved_window_geometry = None
        self._saved_window_state = Qt.WindowState.WindowNoState
        self._window_toggle_last_t = 0.0
        self._window_toggle_cooldown_s = 0.25
        self._window_refresh_timer = None
        self._window_refresh_soft_only = False
        self._screen_hooked_handles = set()
        self._hdr_mpv_screen_sig = None
        self._sdr_mpv_screen_sig = None
        self._suppress_eof_restart_once = False
        self._last_mpv_notice = ""
        self._last_mpv_notice_t = 0.0
        self._overlay_reposition_timer = None
        self._ui_pause_timer = None
        self._ui_pause_duration_ms = 180
        self._periodic_relock_timer = None
        self._periodic_relock_ms = max(
            400, int(os.environ.get("HDRTVNET_PERIODIC_RELOCK_MS", "1200"))
        )
        self._periodic_relock_first_ms = max(
            120, int(os.environ.get("HDRTVNET_PERIODIC_RELOCK_FIRST_MS", "450"))
        )
        self._periodic_relock_drift_s = max(
            0.0, float(os.environ.get("HDRTVNET_PERIODIC_RELOCK_DRIFT_S", "0.045"))
        )
        self._playhead_relock_token = 0
        self._pending_playhead_relock_on_unmute = False
        self._pending_playhead_relock_pre_delay_ms = -1
        self._pending_playhead_relock_first_ms = 35
        self._pending_playhead_relock_settle_ms = 220
        self._act_borderless_full_window = None
        self._root_layout = None
        self._immersive_saved_margins = None
        self._immersive_saved_spacing = None
        self._immersive_saved_view_mode = None
        self._immersive_saved_vis = {}
        self._video_tabs = None
        self._sdr_tab_host = None
        self._hdr_tab_host = None
        self._side_tab_host = None
        self._side_sdr_host = None
        self._side_hdr_host = None
        self._sdr_float_window = None
        self._hdr_float_window = None
        self._ui_closing = False
        self._export_thread = None
        self._export_worker = None
        self._export_progress_dlg = None
        self._export_compile_dlg = None
        self._export_interaction_locked = False
        self._export_saved_enabled_states = {}
        self._benchmark_interaction_locked = False
        self._benchmark_saved_enabled_states = {}
        try:
            self._startup_seek_frame = (
                int(initial_start_frame) if initial_start_frame is not None else None
            )
        except (TypeError, ValueError):
            self._startup_seek_frame = None

    def _queue_initial_video_open(
        self,
        initial_video,
        initial_source_mode,
        initial_capture_hwnd,
        initial_capture_session_id,
        initial_capture_title,
        initial_capture_browser_name,
        initial_capture_process_name,
        initial_capture_url,
        initial_resolution,
        initial_precision,
        initial_view,
        initial_autoplay,
        initial_upscale,
    ):
        def _boot_open():
            source_mode = _normalize_source_mode(initial_source_mode or self._source_mode)
            self._source_mode = source_mode
            self._refresh_source_mode_ui()
            if initial_precision in PRECISIONS:
                self._cmb_prec.setCurrentText(initial_precision)
            legacy_resolution_map = {
                "Native": "1080p",
            }
            mapped_resolution = legacy_resolution_map.get(initial_resolution, initial_resolution)
            if mapped_resolution in RESOLUTION_SCALES or mapped_resolution == "Source":
                self._cmb_res.setCurrentText(mapped_resolution)
            if isinstance(initial_upscale, str) and initial_upscale in UPSCALER_CHOICES:
                if hasattr(self, "_cmb_upscale"):
                    self._cmb_upscale.setCurrentText(initial_upscale)
            if initial_view == "Tabbed":
                self._cmb_view.setCurrentText("Tabbed")
            if source_mode != SOURCE_MODE_VIDEO and initial_capture_hwnd:
                target = target_from_hwnd(
                    initial_capture_hwnd,
                    title=initial_capture_title or "",
                    browser_name=initial_capture_browser_name or "",
                    process_name=initial_capture_process_name or "",
                    source_url=initial_capture_url or "",
                    session_id=initial_capture_session_id or "",
                )
                if target is not None:
                    self._set_window_capture_source(target, auto_play=bool(initial_autoplay))
                    return
            if initial_video and os.path.isfile(initial_video):
                self._set_video(initial_video, auto_play=bool(initial_autoplay))

        QTimer.singleShot(200, _boot_open)
