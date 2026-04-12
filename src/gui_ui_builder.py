from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui_config import (
    _available_precision_keys,
    RESOLUTION_SCALES,
    SOURCE_MODE_LABELS,
    _capture_fps_label_options,
)
from gui_compare import _HAS_MPV
from gui_scaling import UPSCALER_CHOICES, DEFAULT_UPSCALER
from gui_widgets import VideoDisplay


class UiBuilderMixin:
    """Main-window UI construction split into smaller builders."""

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("AppSurface")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        self._root_layout = root
        root.setContentsMargins(14, 12, 14, 8)
        root.setSpacing(10)

        self._build_menu_bar()
        self._build_source_mode_row(root)
        self._build_workspace_area(root)
        self._build_status_and_overlay()

    def _build_workspace_area(self, root: QVBoxLayout):
        self._workspace_split = QSplitter(Qt.Orientation.Horizontal)
        self._workspace_split.setObjectName("WorkspaceSplit")
        self._workspace_split.setChildrenCollapsible(False)
        self._workspace_split.setHandleWidth(10)

        self._viewer_workspace = QWidget()
        self._viewer_workspace.setObjectName("ViewerWorkspace")
        viewer_layout = QVBoxLayout(self._viewer_workspace)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(10)

        self._inspector_workspace = QScrollArea()
        self._inspector_workspace.setObjectName("InspectorWorkspace")
        self._inspector_workspace.setWidgetResizable(True)
        self._inspector_workspace.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._inspector_workspace.setMinimumWidth(340)
        self._inspector_workspace.setMaximumWidth(430)

        self._inspector_content = QWidget()
        self._inspector_content.setObjectName("InspectorContent")
        inspector_layout = QVBoxLayout(self._inspector_content)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(10)
        inspector_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)

        self._build_top_controls_row(inspector_layout)
        self._build_video_displays(viewer_layout)
        self._build_timeline_row(viewer_layout)
        self._build_playback_controls_row(viewer_layout)
        self._build_hdr_panel(inspector_layout)
        self._build_metrics_panel(inspector_layout)
        inspector_layout.addStretch(1)
        self._inspector_workspace.setWidget(self._inspector_content)

        self._workspace_split.addWidget(self._viewer_workspace)
        self._workspace_split.addWidget(self._inspector_workspace)
        self._workspace_split.setStretchFactor(0, 1)
        self._workspace_split.setStretchFactor(1, 0)
        self._workspace_split.setSizes([1240, 380])
        root.addWidget(self._workspace_split, 1)

    def _build_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        self._act_open_source = file_menu.addAction("&Open Video ...", self._open_file)
        self._act_export_video = file_menu.addAction("&Export Video ...", self._open_export_dialog)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close)

        tools_menu = menu_bar.addMenu("&Tools")
        tools_menu.addAction("Runtime Execution Mode ...", self._choose_runtime_execution_mode)
        tools_menu.addAction("INT8 &Pre-dequantization ...", self._choose_predequantize_mode)
        tools_menu.addAction("&Pre-compile Kernels ...", self._precompile_kernels)
        tools_menu.addSeparator()
        tools_menu.addAction("Clear &Kernel Cache ...", self._clear_kernel_cache)

        view_menu = menu_bar.addMenu("&View")
        self._act_borderless_full_window = view_menu.addAction(
            "Borderless Full Window\tF11",
            self._toggle_borderless_full_window,
        )
        self._act_borderless_full_window.setCheckable(True)

    def _build_source_mode_row(self, root: QVBoxLayout):
        self._row0_widget = QWidget()
        self._row0_widget.setObjectName("ModeToolbar")
        row0 = QHBoxLayout(self._row0_widget)
        row0.setContentsMargins(14, 10, 14, 10)
        row0.setSpacing(8)

        self._lbl_source_mode = QLabel("Mode:")
        self._lbl_source_mode.setProperty("muted", True)
        row0.addWidget(self._lbl_source_mode)

        self._cmb_source_mode = QComboBox()
        self._cmb_source_mode.addItems(list(SOURCE_MODE_LABELS.values()))
        self._cmb_source_mode.setFixedWidth(255)
        self._cmb_source_mode.setToolTip(
            "Choose whether HDRTVNet++ opens video files or uses the experimental Google Chrome browser-window path. Chrome's 'Use graphics acceleration when available' must be off for Browser Window Capture."
        )
        row0.addWidget(self._cmb_source_mode)

        self._capture_fps_container = QWidget()
        capture_row = QHBoxLayout(self._capture_fps_container)
        capture_row.setContentsMargins(0, 0, 0, 0)
        capture_row.setSpacing(6)
        self._lbl_capture_fps = QLabel("Capture FPS:")
        self._lbl_capture_fps.setProperty("muted", True)
        capture_row.addWidget(self._lbl_capture_fps)
        self._cmb_capture_fps = QComboBox()
        self._cmb_capture_fps.addItems(_capture_fps_label_options())
        self._cmb_capture_fps.setFixedWidth(90)
        self._cmb_capture_fps.setToolTip(
            "Applies only to Browser Window Capture mode."
        )
        capture_row.addWidget(self._cmb_capture_fps)
        row0.addWidget(self._capture_fps_container)

        row0.addStretch(1)
        root.addWidget(self._row0_widget)

    def _build_top_controls_row(self, root: QVBoxLayout):
        self._row1_widget = QGroupBox("Render Settings")
        self._row1_widget.setObjectName("InspectorPanel")
        row1 = QGridLayout(self._row1_widget)
        row1.setContentsMargins(12, 16, 12, 12)
        row1.setHorizontalSpacing(10)
        row1.setVerticalSpacing(10)
        row1.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        self._btn_file = QPushButton("Open Video ...")
        self._btn_file.setMinimumHeight(32)
        self._btn_file.setProperty("role", "primary")
        self._lbl_file = QLabel("No video selected")
        self._lbl_file.setProperty("pill", True)
        self._lbl_file.setProperty("muted", True)
        self._lbl_file.setWordWrap(True)
        row1.addWidget(self._btn_file, 0, 0, 1, 2)
        row1.addWidget(self._lbl_file, 1, 0, 1, 2)

        lbl_precision = QLabel("Precision:")
        lbl_precision.setProperty("muted", True)
        row1.addWidget(lbl_precision, 2, 0)
        self._cmb_prec = QComboBox()
        self._cmb_prec.addItems(_available_precision_keys())
        self._cmb_prec.setMinimumWidth(170)
        row1.addWidget(self._cmb_prec, 2, 1)
        self._chk_hg = QCheckBox("Use HG")
        self._chk_hg.setChecked(True)
        self._chk_hg.setToolTip("Enable highlight refinement (HG) (very heavy).")

        lbl_resolution = QLabel("Resolution:")
        lbl_resolution.setProperty("muted", True)
        row1.addWidget(lbl_resolution, 3, 0)
        self._cmb_res = QComboBox()
        self._cmb_res.addItems(RESOLUTION_SCALES.keys())
        self._cmb_res.setMinimumWidth(120)
        row1.addWidget(self._cmb_res, 3, 1)

        lbl_upscale = QLabel("Upscale:")
        lbl_upscale.setProperty("muted", True)
        row1.addWidget(lbl_upscale, 4, 0)
        self._cmb_upscale = QComboBox()
        self._cmb_upscale.addItems(UPSCALER_CHOICES)
        self._cmb_upscale.setMinimumWidth(150)
        self._cmb_upscale.setToolTip(
            "Upscale kernel for 540p/720p. 1080p stays native (no upscale)."
        )
        row1.addWidget(self._cmb_upscale, 4, 1)

        self._chk_film_grain = QCheckBox("Film Grain")
        self._chk_film_grain.setToolTip("Restore film grain using mpv shader.")
        if not _HAS_MPV:
            self._chk_film_grain.setEnabled(False)
            self._chk_film_grain.setToolTip("Requires mpv (libmpv-2.dll).")

        toggles_row = QWidget()
        toggles_layout = QHBoxLayout(toggles_row)
        toggles_layout.setContentsMargins(0, 0, 0, 0)
        toggles_layout.setSpacing(12)
        toggles_layout.addWidget(self._chk_hg)
        toggles_layout.addWidget(self._chk_film_grain)
        toggles_layout.addStretch(1)
        row1.addWidget(toggles_row, 5, 0, 1, 2)

        self._btn_apply_settings = QPushButton("Apply")
        self._btn_apply_settings.setEnabled(False)
        self._btn_apply_settings.setProperty("role", "primary")
        self._btn_apply_settings.setMinimumHeight(32)
        row1.addWidget(self._btn_apply_settings, 6, 0, 1, 2)

        self._cmb_view = QComboBox()
        self._cmb_view.addItems(["Tabbed"])
        self._cmb_view.setCurrentText("Tabbed")
        self._cmb_view.hide()
        self._btn_pop_sdr = QPushButton("Pop SDR")
        self._btn_pop_sdr.setProperty("toolbar", "compact")
        self._btn_pop_hdr = QPushButton("Pop HDR")
        self._btn_pop_hdr.setProperty("toolbar", "compact")
        self._btn_toggle_ui = QPushButton("Hide UI")
        self._btn_toggle_ui.setEnabled(False)
        self._btn_toggle_ui.setProperty("toolbar", "compact")

        viewer_tools = QWidget()
        viewer_tools_layout = QGridLayout(viewer_tools)
        viewer_tools_layout.setContentsMargins(0, 0, 0, 0)
        viewer_tools_layout.setHorizontalSpacing(8)
        viewer_tools_layout.setVerticalSpacing(8)
        viewer_tools_layout.addWidget(self._btn_pop_sdr, 0, 0)
        viewer_tools_layout.addWidget(self._btn_pop_hdr, 0, 1)
        viewer_tools_layout.addWidget(self._btn_toggle_ui, 1, 0, 1, 2)
        row1.addWidget(viewer_tools, 7, 0, 1, 2)

        root.addWidget(self._row1_widget)

    def _build_playback_controls_row(self, root: QVBoxLayout):
        self._row2_widget = QWidget()
        self._row2_widget.setObjectName("PlaybackToolbar")
        row2_root = QVBoxLayout(self._row2_widget)
        row2_root.setContentsMargins(14, 10, 14, 10)
        row2_root.setSpacing(8)

        self._btn_play = QPushButton("Play")
        self._btn_pause = QPushButton("Pause")
        self._btn_stop = QPushButton("Stop")
        self._btn_compare = QPushButton("Compare")
        for b in (self._btn_play, self._btn_pause, self._btn_stop):
            b.setFixedSize(100, 30)
            b.setProperty("toolbar", "compact")
        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_play.setProperty("role", "success")
        self._btn_stop.setProperty("role", "warning")
        self._btn_compare.setFixedSize(110, 30)
        self._btn_compare.setEnabled(False)
        self._btn_compare.setProperty("role", "primary")
        self._btn_compare.setProperty("toolbar", "compact")
        self._btn_compare.setToolTip(
            "Pause and open a 3-way frame compare (SDR, HDR GT, HDR Convert)."
        )

        transport_row = QHBoxLayout()
        transport_row.setContentsMargins(0, 0, 0, 0)
        transport_row.setSpacing(8)
        transport_row.addWidget(self._btn_play)
        transport_row.addWidget(self._btn_pause)
        transport_row.addWidget(self._btn_stop)
        transport_row.addWidget(self._btn_compare)
        transport_row.addStretch(1)
        row2_root.addLayout(transport_row)

        self._chk_metrics = QCheckBox("Show Metrics")
        self._chk_metrics.setChecked(True)
        self._btn_hdr_gt = QPushButton("HDR GT ...")
        self._btn_hdr_gt.setFixedWidth(100)
        self._btn_hdr_gt.setProperty("toolbar", "compact")
        self._btn_hdr_gt.setToolTip(
            "Select HDR ground-truth video (same content/timing as the input)."
        )
        self._lbl_hdr_gt = QLabel("HDR GT: none")
        self._lbl_hdr_gt.setProperty("pill", True)
        self._lbl_hdr_gt.setProperty("muted", True)
        self._lbl_hdr_gt.setMinimumWidth(220)
        self._chk_hide_cursor = QCheckBox("Hide Cursor")
        self._chk_hide_cursor.setChecked(True)
        self._lbl_volume = QLabel("Volume:")
        self._lbl_volume.setProperty("muted", True)
        self._sld_volume = QSlider(Qt.Orientation.Horizontal)
        self._sld_volume.setRange(0, 100)
        self._sld_volume.setValue(100)
        self._sld_volume.setFixedWidth(140)
        self._sld_volume.setToolTip("Master volume")
        self._lbl_volume_val = QLabel("100%")
        self._lbl_volume_val.setFixedWidth(42)
        self._lbl_volume_val.setProperty("metricChip", True)
        self._lbl_audio_track = QLabel("Audio:")
        self._lbl_audio_track.setProperty("muted", True)
        self._cmb_audio_track = QComboBox()
        self._cmb_audio_track.setFixedWidth(260)
        self._cmb_audio_track.setEnabled(False)
        self._cmb_audio_track.setToolTip("Load a video with multiple audio tracks.")

        session_row = QHBoxLayout()
        session_row.setContentsMargins(0, 0, 0, 0)
        session_row.setSpacing(8)
        session_row.addWidget(self._btn_hdr_gt)
        session_row.addWidget(self._lbl_hdr_gt, 1)
        session_row.addSpacing(10)
        session_row.addWidget(self._lbl_volume)
        session_row.addWidget(self._sld_volume)
        session_row.addWidget(self._lbl_volume_val)
        session_row.addSpacing(10)
        session_row.addWidget(self._lbl_audio_track)
        session_row.addWidget(self._cmb_audio_track)
        session_row.addSpacing(10)
        session_row.addWidget(self._chk_hide_cursor)
        session_row.addWidget(self._chk_metrics)
        row2_root.addLayout(session_row)

        root.addWidget(self._row2_widget)

    def _build_timeline_row(self, root: QVBoxLayout):
        self._row3_widget = QWidget()
        self._row3_widget.setObjectName("TimelineToolbar")
        row3 = QHBoxLayout(self._row3_widget)
        row3.setContentsMargins(14, 8, 14, 8)
        row3.setSpacing(10)

        self._lbl_time = QLabel("0:00")
        self._lbl_time.setFixedWidth(50)
        self._lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._lbl_time.setFont(QFont("Consolas", 9))
        self._lbl_time.setProperty("metricChip", True)

        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 0)
        self._seek_slider.setEnabled(False)
        self._seek_slider.setTracking(True)

        self._lbl_duration = QLabel("0:00")
        self._lbl_duration.setFixedWidth(50)
        self._lbl_duration.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._lbl_duration.setFont(QFont("Consolas", 9))
        self._lbl_duration.setProperty("metricChip", True)

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
        self._video_tabs.setObjectName("VideoTabs")
        self._video_tabs.setDocumentMode(True)
        self._video_tabs.tabBar().setDrawBase(False)
        self._video_tabs.tabBar().setObjectName("ViewerTabBar")
        self._sdr_tab_host = QWidget()
        sdr_tab_layout = QVBoxLayout(self._sdr_tab_host)
        sdr_tab_layout.setContentsMargins(6, 6, 6, 6)
        sdr_tab_layout.setSpacing(0)
        sdr_tab_layout.addWidget(self._disp_sdr)

        self._hdr_tab_host = QWidget()
        hdr_tab_layout = QVBoxLayout(self._hdr_tab_host)
        hdr_tab_layout.setContentsMargins(6, 6, 6, 6)
        hdr_tab_layout.setSpacing(0)
        hdr_tab_layout.addWidget(self._disp_hdr)

        self._side_tab_host = QWidget()
        side_tab_layout = QVBoxLayout(self._side_tab_host)
        side_tab_layout.setContentsMargins(6, 6, 6, 6)
        side_tab_layout.setSpacing(0)
        side_split = QSplitter(Qt.Orientation.Horizontal)
        side_split.setHandleWidth(12)
        self._side_sdr_host = QWidget()
        side_sdr_layout = QVBoxLayout(self._side_sdr_host)
        side_sdr_layout.setContentsMargins(0, 0, 3, 0)
        side_sdr_layout.setSpacing(0)
        self._side_hdr_host = QWidget()
        side_hdr_layout = QVBoxLayout(self._side_hdr_host)
        side_hdr_layout.setContentsMargins(3, 0, 0, 0)
        side_hdr_layout.setSpacing(0)
        side_split.addWidget(self._side_sdr_host)
        side_split.addWidget(self._side_hdr_host)
        side_split.setStretchFactor(0, 1)
        side_split.setStretchFactor(1, 1)
        side_tab_layout.addWidget(side_split, 1)

        self._video_tabs.addTab(self._sdr_tab_host, "SDR")
        self._video_tabs.addTab(self._hdr_tab_host, "HDR")
        self._video_tabs.addTab(self._side_tab_host, "Side by Side")
        if _HAS_MPV and self._disp_sdr_mpv is not None:
            self._disp_sdr_stack.setCurrentWidget(self._disp_sdr_mpv)
        if _HAS_MPV and self._disp_hdr_mpv is not None:
            self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_mpv)
        root.addWidget(self._video_tabs, 1)

        # Apply the view mode immediately.
        self._on_view(self._cmb_view.currentText())

    def _build_metrics_panel(self, root: QVBoxLayout):
        self._grp_metrics = QGroupBox("Performance Metrics")
        self._grp_metrics.setObjectName("MetricsCard")
        pl = QGridLayout(self._grp_metrics)
        pl.setContentsMargins(12, 16, 12, 12)
        pl.setHorizontalSpacing(14)
        pl.setVerticalSpacing(8)
        pl.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        self._m = {}
        mono = QFont("Consolas", 9)
        perf_keys = (
            "fps",
            "latency",
            "frame",
            "res",
            "gpu",
            "cpu",
            "model",
            "prec",
        )
        for idx, key in enumerate(perf_keys):
            lbl = QLabel(f"{key}: —")
            lbl.setFont(mono)
            lbl.setProperty("metricChip", True)
            pl.addWidget(lbl, idx // 2, idx % 2)
            self._m[key] = lbl

        root.addWidget(self._grp_metrics)

    def _build_hdr_panel(self, root: QVBoxLayout):
        self._grp_hdr = QGroupBox("HDR Output")
        self._grp_hdr.setObjectName("HdrCard")
        hl = QGridLayout(self._grp_hdr)
        hl.setContentsMargins(12, 16, 12, 12)
        hl.setHorizontalSpacing(12)
        hl.setVerticalSpacing(8)
        hl.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
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
            lbl.setProperty("metricChip", True)
            index = len(self._hdr_labels)
            if key == "status":
                hl.addWidget(lbl, index, 0, 1, 2)
            else:
                hl.addWidget(lbl, index, 0, 1, 2)
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
        self._ui_overlay_btn.setProperty("role", "ghost")
        self._ui_overlay_btn.clicked.connect(self._toggle_ui_visibility)
        self._ui_overlay_btn.hide()

