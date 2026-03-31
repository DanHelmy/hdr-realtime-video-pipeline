from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui_config import _available_precision_keys, RESOLUTION_SCALES
from gui_compare import _HAS_MPV
from gui_scaling import UPSCALER_CHOICES, DEFAULT_UPSCALER
from gui_widgets import VideoDisplay


class UiBuilderMixin:
    """Main-window UI construction split into smaller builders."""

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        self._root_layout = root
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(6)

        self._build_menu_bar()
        self._build_top_controls_row(root)
        self._build_playback_controls_row(root)
        self._build_timeline_row(root)
        self._build_video_displays(root)
        self._build_metrics_panel(root)
        self._build_hdr_panel(root)
        self._build_status_and_overlay()

    def _build_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction("📂  &Open Video …", self._open_file)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close)

        tools_menu = menu_bar.addMenu("&Tools")
        tools_menu.addAction("INT8 &Pre-dequantization ...", self._choose_predequantize_mode)
        tools_menu.addAction("⚙  &Pre-compile Kernels …", self._precompile_kernels)
        tools_menu.addSeparator()
        tools_menu.addAction("🗑  Clear &Kernel Cache …", self._clear_kernel_cache)

        view_menu = menu_bar.addMenu("&View")
        self._act_borderless_full_window = view_menu.addAction(
            "Borderless Full Window\tF11",
            self._toggle_borderless_full_window,
        )
        self._act_borderless_full_window.setCheckable(True)

    def _build_top_controls_row(self, root: QVBoxLayout):
        self._row1_widget = QWidget()
        row1 = QHBoxLayout(self._row1_widget)
        row1.setContentsMargins(0, 0, 0, 0)

        self._btn_file = QPushButton("📂  Open Video ...")
        self._btn_file.setFixedHeight(32)
        self._lbl_file = QLabel("No video selected")
        self._lbl_file.setStyleSheet("color: #999; padding-left: 8px;")
        row1.addWidget(self._btn_file)
        row1.addWidget(self._lbl_file, 1)

        row1.addWidget(QLabel("Precision:"))
        self._cmb_prec = QComboBox()
        self._cmb_prec.addItems(_available_precision_keys())
        self._cmb_prec.setFixedWidth(170)
        row1.addWidget(self._cmb_prec)
        self._chk_hg = QCheckBox("Use HG")
        self._chk_hg.setChecked(True)
        self._chk_hg.setToolTip("Enable highlight refinement (HG).")
        row1.addWidget(self._chk_hg)

        row1.addWidget(QLabel("Resolution:"))
        self._cmb_res = QComboBox()
        self._cmb_res.addItems(RESOLUTION_SCALES.keys())
        self._cmb_res.setFixedWidth(100)
        row1.addWidget(self._cmb_res)

        row1.addWidget(QLabel("Upscale:"))
        self._cmb_upscale = QComboBox()
        self._cmb_upscale.addItems(UPSCALER_CHOICES)
        self._cmb_upscale.setFixedWidth(130)
        self._cmb_upscale.setToolTip(
            "Upscale kernel for 540p/720p. 1080p stays native (no upscale)."
        )
        row1.addWidget(self._cmb_upscale)

        self._chk_film_grain = QCheckBox("Film Grain")
        self._chk_film_grain.setToolTip("Restore film grain using mpv shader.")
        if not _HAS_MPV:
            self._chk_film_grain.setEnabled(False)
            self._chk_film_grain.setToolTip("Requires mpv (libmpv-2.dll).")
        row1.addWidget(self._chk_film_grain)

        self._btn_apply_settings = QPushButton("Apply")
        self._btn_apply_settings.setFixedWidth(90)
        self._btn_apply_settings.setEnabled(False)
        row1.addWidget(self._btn_apply_settings)

        self._cmb_view = QComboBox()
        self._cmb_view.addItems(["Tabbed"])
        self._cmb_view.setCurrentText("Tabbed")
        self._btn_pop_sdr = QPushButton("Pop SDR")
        self._btn_pop_sdr.setFixedWidth(88)
        self._btn_pop_hdr = QPushButton("Pop HDR")
        self._btn_pop_hdr.setFixedWidth(88)
        row1.addWidget(self._btn_pop_sdr)
        row1.addWidget(self._btn_pop_hdr)
        self._btn_toggle_ui = QPushButton("Hide UI")
        self._btn_toggle_ui.setFixedWidth(90)
        self._btn_toggle_ui.setEnabled(False)
        row1.addWidget(self._btn_toggle_ui)

        root.addWidget(self._row1_widget)

    def _build_playback_controls_row(self, root: QVBoxLayout):
        self._row2_widget = QWidget()
        row2 = QHBoxLayout(self._row2_widget)
        row2.setContentsMargins(0, 0, 0, 0)

        self._btn_play = QPushButton("▶  Play")
        self._btn_pause = QPushButton("⏸  Pause")
        self._btn_stop = QPushButton("⏹  Stop")
        self._btn_compare = QPushButton("Compare")
        for b in (self._btn_play, self._btn_pause, self._btn_stop):
            b.setFixedSize(100, 30)
        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_compare.setFixedSize(110, 30)
        self._btn_compare.setEnabled(False)
        self._btn_compare.setToolTip(
            "Pause and open a 3-way frame compare (SDR, HDR GT, HDR Convert)."
        )

        self._chk_metrics = QCheckBox("Show Metrics")
        self._chk_metrics.setChecked(True)
        self._btn_hdr_gt = QPushButton("HDR GT ...")
        self._btn_hdr_gt.setFixedWidth(100)
        self._btn_hdr_gt.setToolTip(
            "Select HDR ground-truth video (same content/timing as the input)."
        )
        self._lbl_hdr_gt = QLabel("HDR GT: none")
        self._lbl_hdr_gt.setStyleSheet("color: #999;")
        self._lbl_hdr_gt.setMinimumWidth(220)
        self._chk_hide_cursor = QCheckBox("Hide Cursor")
        self._chk_hide_cursor.setChecked(True)
        self._lbl_volume = QLabel("Volume:")
        self._sld_volume = QSlider(Qt.Orientation.Horizontal)
        self._sld_volume.setRange(0, 100)
        self._sld_volume.setValue(100)
        self._sld_volume.setFixedWidth(140)
        self._sld_volume.setToolTip("Master volume")
        self._lbl_volume_val = QLabel("100%")
        self._lbl_volume_val.setFixedWidth(42)
        self._lbl_audio_track = QLabel("Audio:")
        self._cmb_audio_track = QComboBox()
        self._cmb_audio_track.setFixedWidth(260)
        self._cmb_audio_track.setEnabled(False)
        self._cmb_audio_track.setToolTip("Load a video with multiple audio tracks.")

        row2.addWidget(self._btn_play)
        row2.addWidget(self._btn_pause)
        row2.addWidget(self._btn_stop)
        row2.addWidget(self._btn_compare)
        row2.addWidget(self._btn_hdr_gt)
        row2.addWidget(self._lbl_hdr_gt, 1)
        row2.addStretch()
        row2.addWidget(self._lbl_volume)
        row2.addWidget(self._sld_volume)
        row2.addWidget(self._lbl_volume_val)
        row2.addWidget(self._lbl_audio_track)
        row2.addWidget(self._cmb_audio_track)
        row2.addWidget(self._chk_hide_cursor)
        row2.addWidget(self._chk_metrics)
        root.addWidget(self._row2_widget)

    def _build_timeline_row(self, root: QVBoxLayout):
        self._row3_widget = QWidget()
        row3 = QHBoxLayout(self._row3_widget)
        row3.setContentsMargins(0, 0, 0, 0)

        self._lbl_time = QLabel("0:00")
        self._lbl_time.setFixedWidth(50)
        self._lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._lbl_time.setFont(QFont("Consolas", 9))

        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 0)
        self._seek_slider.setEnabled(False)
        self._seek_slider.setTracking(True)

        self._lbl_duration = QLabel("0:00")
        self._lbl_duration.setFixedWidth(50)
        self._lbl_duration.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._lbl_duration.setFont(QFont("Consolas", 9))

        row3.addWidget(self._lbl_time)
        row3.addWidget(self._seek_slider, 1)
        row3.addWidget(self._lbl_duration)
        root.addWidget(self._row3_widget)

    def _build_video_displays(self, root: QVBoxLayout):
        if _HAS_MPV:
            self._disp_sdr_mpv = self._new_mpv_widget()
            self._disp_sdr_cpu = VideoDisplay("SDR Input")
            self._disp_sdr_stack = QStackedWidget()
            self._disp_sdr_stack.addWidget(self._disp_sdr_mpv)
            self._disp_sdr_stack.addWidget(self._disp_sdr_cpu)
            self._disp_sdr = self._disp_sdr_stack
        else:
            self._disp_sdr_mpv = None
            self._disp_sdr_cpu = VideoDisplay("SDR Input")
            self._disp_sdr = self._disp_sdr_cpu

        if _HAS_MPV:
            self._disp_hdr_mpv = self._new_mpv_widget()
            self._disp_hdr_cpu = VideoDisplay("HDR Output")
            self._disp_hdr_stack = QStackedWidget()
            self._disp_hdr_stack.addWidget(self._disp_hdr_mpv)
            self._disp_hdr_stack.addWidget(self._disp_hdr_cpu)
            self._disp_hdr = self._disp_hdr_stack
        else:
            self._disp_hdr_mpv = None
            self._disp_hdr_cpu = VideoDisplay("HDR Output")
            self._disp_hdr = self._disp_hdr_cpu

        self._video_tabs = QTabWidget()
        self._video_tabs.setDocumentMode(True)
        self._sdr_tab_host = QWidget()
        sdr_tab_layout = QVBoxLayout(self._sdr_tab_host)
        sdr_tab_layout.setContentsMargins(0, 0, 0, 0)
        sdr_tab_layout.setSpacing(0)
        sdr_tab_layout.addWidget(self._disp_sdr)

        self._hdr_tab_host = QWidget()
        hdr_tab_layout = QVBoxLayout(self._hdr_tab_host)
        hdr_tab_layout.setContentsMargins(0, 0, 0, 0)
        hdr_tab_layout.setSpacing(0)
        hdr_tab_layout.addWidget(self._disp_hdr)

        self._side_tab_host = QWidget()
        side_tab_layout = QVBoxLayout(self._side_tab_host)
        side_tab_layout.setContentsMargins(0, 0, 0, 0)
        side_tab_layout.setSpacing(0)
        side_split = QSplitter(Qt.Orientation.Horizontal)
        self._side_sdr_host = QWidget()
        side_sdr_layout = QVBoxLayout(self._side_sdr_host)
        side_sdr_layout.setContentsMargins(0, 0, 0, 0)
        side_sdr_layout.setSpacing(0)
        self._side_hdr_host = QWidget()
        side_hdr_layout = QVBoxLayout(self._side_hdr_host)
        side_hdr_layout.setContentsMargins(0, 0, 0, 0)
        side_hdr_layout.setSpacing(0)
        side_split.addWidget(self._side_sdr_host)
        side_split.addWidget(self._side_hdr_host)
        side_split.setStretchFactor(0, 1)
        side_split.setStretchFactor(1, 1)
        side_tab_layout.addWidget(side_split, 1)

        self._video_tabs.addTab(self._sdr_tab_host, "SDR")
        self._video_tabs.addTab(self._hdr_tab_host, "HDR")
        self._video_tabs.addTab(self._side_tab_host, "Side by Side")
        self._video_tabs.setCurrentIndex(1)
        if _HAS_MPV and self._disp_sdr_mpv is not None:
            self._disp_sdr_stack.setCurrentWidget(self._disp_sdr_mpv)
        if _HAS_MPV and self._disp_hdr_mpv is not None:
            self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_mpv)
        root.addWidget(self._video_tabs, 1)

        # Apply the view mode immediately.
        self._on_view(self._cmb_view.currentText())

    def _build_metrics_panel(self, root: QVBoxLayout):
        self._grp_metrics = QWidget()
        mroot = QHBoxLayout(self._grp_metrics)
        mroot.setContentsMargins(0, 0, 0, 0)
        mroot.setSpacing(8)

        self._grp_perf_metrics = QGroupBox("Performance Metrics")
        pl = QGridLayout(self._grp_perf_metrics)
        pl.setContentsMargins(10, 4, 10, 6)
        pl.setHorizontalSpacing(16)
        pl.setVerticalSpacing(2)

        self._m = {}
        mono = QFont("Consolas", 9)
        perf_keys = ("fps", "latency", "frame", "res", "gpu", "cpu", "model", "prec", "upscale")
        for idx, key in enumerate(perf_keys):
            lbl = QLabel(f"{key}: —")
            lbl.setFont(mono)
            pl.addWidget(lbl, idx // 5, idx % 5)
            self._m[key] = lbl

        mroot.addWidget(self._grp_perf_metrics, 1)
        root.addWidget(self._grp_metrics)

    def _build_hdr_panel(self, root: QVBoxLayout):
        self._grp_hdr = QGroupBox("HDR Output")
        hl = QHBoxLayout(self._grp_hdr)
        hl.setContentsMargins(12, 4, 12, 4)
        self._hdr_labels = {}
        mono = QFont("Consolas", 9)
        for key, default in [
            ("status", "HDR: waiting…"),
            ("primaries", "Primaries: —"),
            ("transfer", "Transfer: —"),
            ("peak", "Peak: —"),
            ("vo", "VO: —"),
        ]:
            lbl = QLabel(default)
            lbl.setFont(mono)
            hl.addWidget(lbl)
            self._hdr_labels[key] = lbl
        root.addWidget(self._grp_hdr)

    def _build_status_and_overlay(self):
        self.statusBar().showMessage(
            "Ready — open a video file to begin.  "
            "You can also drag-and-drop a video onto this window."
        )
        self._sync_upscale_controls()

        self._ui_overlay_btn = QPushButton("Show UI", self)
        self._ui_overlay_btn.setFixedSize(110, 30)
        self._ui_overlay_btn.setStyleSheet(
            "QPushButton { background: rgba(30,30,30,210); color: #eee; "
            "border: 1px solid #666; border-radius: 6px; }"
            "QPushButton:hover { background: rgba(50,50,50,230); }"
        )
        self._ui_overlay_btn.clicked.connect(self._toggle_ui_visibility)
        self._ui_overlay_btn.hide()
