from __future__ import annotations

from PyQt6.QtWidgets import QApplication


class SignalWiringMixin:
    """Centralize Qt signal/slot wiring for MainWindow."""

    def _connect_signals(self):
        app = QApplication.instance()
        if app is not None:
            app.applicationStateChanged.connect(self._on_app_state_changed)

        self._btn_file.clicked.connect(self._open_file)
        self._btn_play.clicked.connect(self._play)
        self._btn_pause.clicked.connect(self._toggle_pause)
        self._btn_stop.clicked.connect(self._stop_and_restart)
        self._btn_compare.clicked.connect(self._compare_current_frame)
        self._btn_apply_settings.clicked.connect(self._apply_runtime_settings)
        self._btn_hdr_gt.clicked.connect(self._pick_hdr_ground_truth_file)
        self._chk_metrics.toggled.connect(
            lambda on: self._grp_metrics.setVisible(on))
        self._chk_metrics.toggled.connect(lambda _on: self._save_user_settings())
        self._chk_hide_cursor.toggled.connect(self._on_hide_cursor_toggled)
        self._chk_hide_cursor.toggled.connect(lambda _on: self._save_user_settings())
        self._sld_volume.valueChanged.connect(self._on_volume_changed)
        self._cmb_audio_track.currentIndexChanged.connect(self._on_audio_track_changed)
        self._cmb_prec.currentTextChanged.connect(self._on_precision)
        self._chk_hg.stateChanged.connect(self._on_hg_toggle)
        self._cmb_prec.currentTextChanged.connect(lambda _v: self._save_user_settings())
        self._cmb_res.currentTextChanged.connect(self._on_resolution)
        self._cmb_res.currentTextChanged.connect(lambda _v: self._save_user_settings())
        self._cmb_upscale.currentTextChanged.connect(self._on_upscale_changed)
        self._cmb_upscale.currentTextChanged.connect(lambda _v: self._save_user_settings())
        if hasattr(self, "_chk_film_grain") and self._chk_film_grain is not None:
            self._chk_film_grain.stateChanged.connect(self._on_film_grain_changed)
        self._cmb_view.currentTextChanged.connect(self._on_view)
        self._cmb_view.currentTextChanged.connect(lambda _v: self._save_user_settings())
        self._btn_pop_sdr.clicked.connect(self._toggle_sdr_popout)
        self._btn_pop_hdr.clicked.connect(self._toggle_hdr_popout)
        self._btn_toggle_ui.clicked.connect(self._toggle_ui_visibility)
        if self._video_tabs is not None:
            self._video_tabs.currentChanged.connect(self._on_video_tab_changed)

        self._worker.frame_ready.connect(self._on_frame)
        self._worker.metrics_updated.connect(self._on_metrics)
        self._worker.compare_snapshot_ready.connect(self._on_compare_snapshot_ready)
        self._worker.status_message.connect(self._on_status_message)
        self._worker.playback_finished.connect(self._on_finished)
        self._worker.compile_ready.connect(self._on_compile_ready)
        self._worker.position_updated.connect(self._on_position)
        self._worker.seek_frame_ready.connect(self._on_seek_frame_ready)
        self._seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self._seek_slider.sliderMoved.connect(self._on_seek)
        self._seek_slider.sliderReleased.connect(self._on_seek_released)

        # HDR info from mpv
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.hdr_info_ready.connect(self._on_hdr_info)
            self._disp_hdr_mpv.runtime_notice.connect(self._on_mpv_notice)
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.runtime_notice.connect(self._on_mpv_notice)
