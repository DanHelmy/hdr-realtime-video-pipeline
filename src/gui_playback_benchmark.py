from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time

from PyQt6.QtGui import QTextCursor
from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui_config import (
    DEFAULT_RESOLUTION_KEY,
    RESOLUTION_SCALES,
    _available_precision_keys,
    _processing_preset_dims,
)
from gui_window_utils import configure_independent_window

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SORT_ROLE = int(Qt.ItemDataRole.UserRole.value) + 1
_SESSION_ROLE = int(Qt.ItemDataRole.UserRole.value) + 2


_PRECISION_TO_RUN = {
    "FP32": "fp32",
    "FP16": "fp16",
    "INT8 Mixed (PTQ)": "int8-mixed-ptq",
    "INT8 Mixed (QAT)": "int8-mixed-qat",
    "INT8 Mixed (QAT) (Film)": "int8-mixed-qat-film",
    "INT8 Full (PTQ)": "int8-full-ptq",
    "INT8 Full (QAT)": "int8-full-qat",
    "INT8 Full (QAT) (Film)": "int8-full-qat-film",
    "FP8 Mixed (PTQ)": "fp8-mixed-ptq",
    "FP8 Mixed (QAT)": "fp8-mixed-qat",
    "FP8 Mixed (QAT) (Film)": "fp8-mixed-qat-film",
    "FP8 Full (PTQ)": "fp8-full-ptq",
    "FP8 Full (QAT)": "fp8-full-qat",
    "FP8 Full (QAT) (Film)": "fp8-full-qat-film",
}


class _ResultTableItem(QTableWidgetItem):
    def __lt__(self, other):
        if isinstance(other, QTableWidgetItem):
            left = self.data(_SORT_ROLE)
            right = other.data(_SORT_ROLE)
            if left is not None and right is not None:
                try:
                    return float(left) < float(right)
                except Exception:
                    pass
        return super().__lt__(other)


def _fmt_float(value, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    if not (number == number):
        return "-"
    return f"{number:.{digits}f}{suffix}"


def _metric_last(payload: dict, key: str):
    samples = payload.get("runtime_metrics")
    if isinstance(samples, list):
        for sample in reversed(samples):
            if isinstance(sample, dict) and sample.get(key) is not None:
                return sample.get(key)
    summary = payload.get("runtime_metric_summary")
    if isinstance(summary, dict):
        stat = summary.get(key)
        if isinstance(stat, dict):
            return stat.get("last", stat.get("avg"))
    return None


def _metric_max(payload: dict, key: str):
    summary = payload.get("runtime_metric_summary")
    if isinstance(summary, dict):
        stat = summary.get(key)
        if isinstance(stat, dict):
            return stat.get("max", stat.get("last", stat.get("avg")))
    return _metric_last(payload, key)


_PRECISION_DISPLAY_LABELS = {
    "fp32": "FP32",
    "fp16": "FP16",
    "int8_mixed_ptq_trt_native": "INT8 Mixed PTQ",
    "int8-mixed-ptq": "INT8 Mixed PTQ",
    "int8_mixed_qat_trt_native": "INT8 Mixed QAT",
    "int8-mixed-qat": "INT8 Mixed QAT",
    "int8_mixed_qat_film_trt_native": "INT8 Mixed QAT Film",
    "int8-mixed-qat-film": "INT8 Mixed QAT Film",
    "int8_full_ptq_trt_native": "INT8 Full PTQ",
    "int8-full-ptq": "INT8 Full PTQ",
    "int8_full_qat_trt_native": "INT8 Full QAT",
    "int8-full-qat": "INT8 Full QAT",
    "int8_full_qat_film_trt_native": "INT8 Full QAT Film",
    "int8-full-qat-film": "INT8 Full QAT Film",
    "fp8_mixed_ptq_trt_native": "FP8 Mixed PTQ",
    "fp8-mixed-ptq": "FP8 Mixed PTQ",
    "fp8_mixed_qat_trt_native": "FP8 Mixed QAT",
    "fp8-mixed-qat": "FP8 Mixed QAT",
    "fp8_mixed_qat_film_trt_native": "FP8 Mixed QAT Film",
    "fp8-mixed-qat-film": "FP8 Mixed QAT Film",
    "fp8_full_ptq_trt_native": "FP8 Full PTQ",
    "fp8-full-ptq": "FP8 Full PTQ",
    "fp8_full_qat_trt_native": "FP8 Full QAT",
    "fp8-full-qat": "FP8 Full QAT",
    "fp8_full_qat_film_trt_native": "FP8 Full QAT Film",
    "fp8-full-qat-film": "FP8 Full QAT Film",
}


def _display_precision_label(value) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "-"
    key = raw.lower().replace("-", "_")
    if key in _PRECISION_DISPLAY_LABELS:
        return _PRECISION_DISPLAY_LABELS[key]
    if raw in _PRECISION_DISPLAY_LABELS:
        return _PRECISION_DISPLAY_LABELS[raw]
    label = raw.replace("_trt_native", "").replace("_", " ").replace("-", " ")
    label = re.sub(r"\bint8\b", "INT8", label, flags=re.IGNORECASE)
    label = re.sub(r"\bfp8\b", "FP8", label, flags=re.IGNORECASE)
    label = re.sub(r"\bfp16\b", "FP16", label, flags=re.IGNORECASE)
    label = re.sub(r"\bfp32\b", "FP32", label, flags=re.IGNORECASE)
    label = re.sub(r"\bqat\b", "QAT", label, flags=re.IGNORECASE)
    label = re.sub(r"\bptq\b", "PTQ", label, flags=re.IGNORECASE)
    return " ".join(part for part in label.split() if part)


def _session_sort_key(path: pathlib.Path) -> tuple[str, str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        settings = payload.get("settings") if isinstance(payload, dict) else {}
        return (
            str(settings.get("resolution") or ""),
            str(settings.get("precision") or ""),
            str(path.parent.name),
        )
    except Exception:
        return ("", "", str(path.parent.name))


class PlaybackPerformanceBenchmarkDialog(QDialog):
    """GUI wrapper for the mpv-backed CLI playback benchmark."""

    def __init__(
        self,
        *,
        initial_video_path: str | None,
        suggested_dir: str,
        initial_precision_key: str | None,
        initial_use_hg: bool,
        logs_root: str,
        parent=None,
    ):
        super().__init__(None)
        self._owner_widget = parent
        configure_independent_window(self)
        self.setSizeGripEnabled(True)
        self.setWindowTitle("Playback Performance Benchmark")
        self.setModal(True)
        self.resize(1120, 760)

        self._suggested_dir = suggested_dir if os.path.isdir(suggested_dir) else str(_ROOT)
        self._last_source_dir = self._suggested_dir
        self._logs_root = logs_root
        self._process: QProcess | None = None
        self._stdout_text = ""
        self._stderr_text = ""
        self._batch_dir: pathlib.Path | None = None
        self._run_started_t = 0.0
        self._run_started_wall_t = 0.0
        self._result_session_dirs: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        intro = QLabel(
            "Runs a wall-clock realtime playback benchmark through the same HDR mpv raw-video display backend used by playback, without the selected upscale shader. "
            "It uses source-FPS pacing and catch-up frame skipping like the GUI player. "
            "The video player stays locked while this window is open so the GPU path is not shared with normal playback."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        source_group = QGroupBox("Source")
        source_form = QFormLayout(source_group)
        self._txt_video = QLineEdit()
        if initial_video_path:
            self._txt_video.setText(initial_video_path)
        self._btn_browse_video = QPushButton("Browse ...")
        source_row = QWidget()
        source_row_l = QHBoxLayout(source_row)
        source_row_l.setContentsMargins(0, 0, 0, 0)
        source_row_l.setSpacing(6)
        source_row_l.addWidget(self._txt_video, 1)
        source_row_l.addWidget(self._btn_browse_video)
        source_form.addRow("Video:", source_row)
        root.addWidget(source_group)

        option_grid = QGridLayout()
        precision_group = QGroupBox("Precisions")
        precision_layout = QVBoxLayout(precision_group)
        self._lst_precisions = QListWidget()
        self._lst_precisions.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._populate_precision_list(initial_precision_key)
        precision_layout.addWidget(self._lst_precisions)

        precision_btns = QHBoxLayout()
        self._btn_all_precisions = QPushButton("Check All ...")
        self._btn_clear_precisions = QPushButton("Clear")
        precision_btns.addWidget(self._btn_all_precisions)
        precision_btns.addWidget(self._btn_clear_precisions)
        precision_btns.addStretch(1)
        precision_layout.addLayout(precision_btns)
        option_grid.addWidget(precision_group, 0, 0, 1, 1)

        resolution_group = QGroupBox("Resolutions")
        resolution_layout = QVBoxLayout(resolution_group)
        self._lst_resolutions = QListWidget()
        self._lst_resolutions.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._populate_resolution_list()
        resolution_layout.addWidget(self._lst_resolutions)

        resolution_btns = QHBoxLayout()
        self._btn_all_resolutions = QPushButton("Check All")
        self._btn_clear_resolutions = QPushButton("Clear")
        resolution_btns.addWidget(self._btn_all_resolutions)
        resolution_btns.addWidget(self._btn_clear_resolutions)
        resolution_btns.addStretch(1)
        resolution_layout.addLayout(resolution_btns)
        option_grid.addWidget(resolution_group, 0, 1, 1, 1)

        runtime_group = QGroupBox("Runtime")
        runtime_form = QFormLayout(runtime_group)
        self._spn_duration = QSpinBox()
        self._spn_duration.setRange(5, 600)
        self._spn_duration.setValue(30)
        self._spn_duration.setSuffix(" s")
        self._spn_warmup = QSpinBox()
        self._spn_warmup.setRange(0, 600)
        self._spn_warmup.setValue(60)
        self._spn_sample = QSpinBox()
        self._spn_sample.setRange(1, 600)
        self._spn_sample.setValue(60)
        self._chk_hg = QCheckBox("Use HG highlight refinement")
        self._chk_hg.setChecked(bool(initial_use_hg))
        self._chk_loop = QCheckBox("Loop source if the clip ends")
        self._chk_loop.setChecked(True)
        self._txt_logs_root = QLineEdit(self._logs_root)
        self._btn_logs_root = QPushButton("Browse ...")
        logs_row = QWidget()
        logs_row_l = QHBoxLayout(logs_row)
        logs_row_l.setContentsMargins(0, 0, 0, 0)
        logs_row_l.setSpacing(6)
        logs_row_l.addWidget(self._txt_logs_root, 1)
        logs_row_l.addWidget(self._btn_logs_root)
        runtime_form.addRow("Duration:", self._spn_duration)
        runtime_form.addRow("Warmup frames:", self._spn_warmup)
        runtime_form.addRow("Log every N frames:", self._spn_sample)
        runtime_form.addRow("", self._chk_hg)
        runtime_form.addRow("", self._chk_loop)
        runtime_form.addRow("Logs root:", logs_row)
        option_grid.addWidget(runtime_group, 0, 2, 1, 1)
        option_grid.setColumnStretch(0, 1)
        option_grid.setColumnStretch(1, 1)
        option_grid.setColumnStretch(2, 1)
        root.addLayout(option_grid)

        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        results_body = QHBoxLayout()
        results_left = QWidget()
        results_left_layout = QVBoxLayout(results_left)
        results_left_layout.setContentsMargins(0, 0, 0, 0)
        results_left_layout.setSpacing(6)
        self._tbl_results = QTableWidget(0, 13)
        self._tbl_results.setHorizontalHeaderLabels(
            [
                "Precision",
                "Res",
                "HG",
                "FPS",
                "1% Low",
                "Latency",
                "Model",
                "Render",
                "VRAM",
                "CPU",
                "Engine",
                "Frames",
                "Dropped",
            ]
        )
        self._tbl_results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tbl_results.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tbl_results.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tbl_results.setAlternatingRowColors(True)
        self._tbl_results.setSortingEnabled(True)
        self._tbl_results.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        header = self._tbl_results.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        self._apply_result_column_defaults()
        results_left_layout.addWidget(self._tbl_results, 1)

        self._txt_output = QPlainTextEdit()
        self._txt_output.setReadOnly(True)
        self._txt_output.setMaximumHeight(140)
        results_left_layout.addWidget(self._txt_output)

        preview_group = QGroupBox("Live mpv Preview")
        preview_layout = QVBoxLayout(preview_group)
        self._preview_surface = QFrame()
        self._preview_surface.setObjectName("playbackBenchmarkPreviewSurface")
        self._preview_surface.setMinimumSize(320, 180)
        self._preview_surface.setMaximumWidth(380)
        self._preview_surface.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self._preview_surface.setAutoFillBackground(True)
        self._preview_surface.setProperty("videoSurface", True)
        self._preview_surface.setStyleSheet(
            "#playbackBenchmarkPreviewSurface { background: #000; border: 1px solid #303844; }"
        )
        preview_layout.addWidget(self._preview_surface)
        preview_note = QLabel(
            "Embedded mpv output from the active benchmark run. "
            "This is the same HDR raw-video display path used for playback, with no FSR/SSimSuperRes upscale shader."
        )
        preview_note.setWordWrap(True)
        preview_layout.addWidget(preview_note)
        preview_layout.addStretch(1)

        results_body.addWidget(results_left, 1)
        results_body.addWidget(preview_group, 0)
        results_layout.addLayout(results_body, 1)
        root.addWidget(results_group, 1)

        footer = QHBoxLayout()
        self._lbl_status = QLabel("Ready")
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._btn_run = QPushButton("Run Benchmark")
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_open_logs = QPushButton("Open Result Folder")
        self._btn_open_logs.setEnabled(False)
        self._btn_close = QPushButton("Close")
        footer.addWidget(self._lbl_status, 1)
        footer.addWidget(self._progress, 0)
        footer.addWidget(self._btn_run)
        footer.addWidget(self._btn_cancel)
        footer.addWidget(self._btn_open_logs)
        footer.addWidget(self._btn_close)
        root.addLayout(footer)

        self._btn_browse_video.clicked.connect(self._browse_video)
        self._btn_logs_root.clicked.connect(self._browse_logs_root)
        self._btn_all_precisions.clicked.connect(self._choose_precision_check_scope)
        self._btn_clear_precisions.clicked.connect(lambda: self._check_all(self._lst_precisions, False))
        self._btn_all_resolutions.clicked.connect(lambda: self._check_all(self._lst_resolutions, True))
        self._btn_clear_resolutions.clicked.connect(lambda: self._check_all(self._lst_resolutions, False))
        self._btn_run.clicked.connect(self._run_benchmark)
        self._btn_cancel.clicked.connect(self._cancel_benchmark)
        self._btn_open_logs.clicked.connect(self._open_result_folder)
        self._btn_close.clicked.connect(self.close)

    def last_source_dir(self) -> str:
        return self._last_source_dir

    def last_session_dir(self) -> str | None:
        if self._batch_dir is not None and self._batch_dir.is_dir():
            return str(self._batch_dir)
        return None

    def _apply_result_column_defaults(self):
        widths = (150, 88, 42, 72, 76, 86, 82, 78, 78, 76, 82, 70, 76)
        header = self._tbl_results.horizontalHeader()
        header.setMinimumSectionSize(38)
        for col, width in enumerate(widths):
            self._tbl_results.setColumnWidth(col, width)

    def _populate_precision_list(self, initial_precision_key: str | None):
        available = [
            key for key in _available_precision_keys()
            if key in _PRECISION_TO_RUN
        ]
        if not available:
            available = list(_PRECISION_TO_RUN.keys())
        for key in available:
            item = QListWidgetItem(key)
            item.setData(Qt.ItemDataRole.UserRole, _PRECISION_TO_RUN[key])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = key == initial_precision_key
            if not initial_precision_key and key == "INT8 Mixed (QAT)":
                checked = True
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self._lst_precisions.addItem(item)

    def _populate_resolution_list(self):
        for key in RESOLUTION_SCALES.keys():
            width, height = _processing_preset_dims(key)
            item = QListWidgetItem(f"{key} ({width}x{height})")
            item.setData(Qt.ItemDataRole.UserRole, f"{width}x{height}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if key == DEFAULT_RESOLUTION_KEY
                else Qt.CheckState.Unchecked
            )
            self._lst_resolutions.addItem(item)

    @staticmethod
    def _check_all(list_widget: QListWidget, checked: bool):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(list_widget.count()):
            list_widget.item(i).setCheckState(state)

    def _available_precision_labels_in_list(self) -> list[str]:
        labels: list[str] = []
        for i in range(self._lst_precisions.count()):
            labels.append(str(self._lst_precisions.item(i).text()))
        return labels

    def _main_precision_labels(self) -> list[str]:
        available = self._available_precision_labels_in_list()

        def _first_available(candidates: tuple[str, ...]) -> str | None:
            for candidate in candidates:
                if candidate in available:
                    return candidate
            return None

        selected = {
            key
            for key in (
                _first_available(("FP16",)),
                _first_available(("FP32",)),
                _first_available(
                    (
                        "INT8 Mixed (QAT)",
                        "INT8 Mixed (PTQ)",
                        "INT8 Mixed (QAT) (Film)",
                    )
                ),
                _first_available(
                    (
                        "INT8 Full (QAT)",
                        "INT8 Full (PTQ)",
                        "INT8 Full (QAT) (Film)",
                    )
                ),
            )
            if key
        }
        return [key for key in available if key in selected]

    def _set_precision_label_checks(self, labels: set[str]):
        for i in range(self._lst_precisions.count()):
            item = self._lst_precisions.item(i)
            item.setCheckState(
                Qt.CheckState.Checked
                if str(item.text()) in labels
                else Qt.CheckState.Unchecked
            )

    def _choose_precision_check_scope(self):
        box = QMessageBox(self)
        box.setWindowTitle("Check Precisions")
        box.setText("Select which precision set?")
        box.setInformativeText(
            "Main Ones selects one representative run for each family: "
            "FP16, FP32, INT8 Mixed, and INT8 Full. "
            "Literally All selects every available precision preset, including PTQ and Film variants."
        )
        main_btn = box.addButton("Main Ones", QMessageBox.ButtonRole.AcceptRole)
        all_btn = box.addButton("Literally All", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(main_btn)
        box.exec()

        clicked = box.clickedButton()
        if clicked is main_btn:
            labels = set(self._main_precision_labels())
            if not labels:
                QMessageBox.warning(
                    self,
                    "Playback Benchmark",
                    "No main precision presets are currently available.",
                )
                return
            self._set_precision_label_checks(labels)
        elif clicked is all_btn:
            self._check_all(self._lst_precisions, True)

    @staticmethod
    def _checked_data(list_widget: QListWidget) -> list[str]:
        values: list[str] = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                values.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return values

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Benchmark Video",
            self._last_source_dir,
            "Video Files (*.mp4 *.mkv *.mov *.avi *.m4v *.webm);;All Files (*)",
        )
        if path:
            self._txt_video.setText(path)
            folder = os.path.dirname(path)
            if folder and os.path.isdir(folder):
                self._last_source_dir = folder

    def _browse_logs_root(self):
        start = self._txt_logs_root.text().strip() or self._logs_root
        path = QFileDialog.getExistingDirectory(self, "Select Benchmark Logs Root", start)
        if path:
            self._txt_logs_root.setText(path)

    def _selected_rows_session_dir(self) -> str | None:
        row = self._tbl_results.currentRow()
        if row < 0:
            return None
        item = self._tbl_results.item(row, 0)
        if item is None:
            return None
        session_dir = item.data(_SESSION_ROLE)
        if session_dir and os.path.isdir(str(session_dir)):
            return str(session_dir)
        return None

    def _open_result_folder(self):
        target = self._selected_rows_session_dir()
        if target is None and self._batch_dir is not None:
            target = str(self._batch_dir)
        if not target:
            return
        try:
            os.startfile(target)
        except Exception:
            QMessageBox.information(self, "Open Result Folder", target)

    def _set_running(self, running: bool):
        for widget in (
            self._txt_video,
            self._btn_browse_video,
            self._lst_precisions,
            self._lst_resolutions,
            self._spn_duration,
            self._spn_warmup,
            self._spn_sample,
            self._chk_hg,
            self._chk_loop,
            self._txt_logs_root,
            self._btn_logs_root,
            self._btn_all_precisions,
            self._btn_clear_precisions,
            self._btn_all_resolutions,
            self._btn_clear_resolutions,
        ):
            widget.setEnabled(not running)
        self._btn_run.setEnabled(not running)
        self._btn_cancel.setEnabled(running)
        self._btn_close.setEnabled(not running)
        self._progress.setRange(0, 0 if running else 1)
        if not running:
            self._progress.setValue(0)

    def _run_benchmark(self):
        video = self._txt_video.text().strip().strip('"')
        if not os.path.isfile(video):
            QMessageBox.warning(self, "Playback Benchmark", "Choose a valid video file first.")
            return
        runs = self._checked_data(self._lst_precisions)
        if not runs:
            QMessageBox.warning(self, "Playback Benchmark", "Choose at least one precision.")
            return
        resolutions = self._checked_data(self._lst_resolutions)
        if not resolutions:
            QMessageBox.warning(self, "Playback Benchmark", "Choose at least one resolution.")
            return
        logs_root = self._txt_logs_root.text().strip() or self._logs_root
        os.makedirs(logs_root, exist_ok=True)

        self._stdout_text = ""
        self._stderr_text = ""
        self._batch_dir = None
        self._result_session_dirs = []
        self._tbl_results.setRowCount(0)
        self._txt_output.clear()
        self._btn_open_logs.setEnabled(False)
        self._set_running(True)
        self._lbl_status.setText("Realtime playback benchmark running through mpv ...")
        self._run_started_t = time.perf_counter()
        self._run_started_wall_t = time.time()

        script = str(_HERE / "cli_playback_benchmark.py")
        args = [
            script,
            "--video",
            video,
            "--resolutions",
            *resolutions,
            "--runs",
            *runs,
            "--duration-s",
            str(int(self._spn_duration.value())),
            "--warmup-frames",
            str(int(self._spn_warmup.value())),
            "--sample-interval",
            str(int(self._spn_sample.value())),
            "--use-hg",
            "1" if self._chk_hg.isChecked() else "0",
            "--display",
            "--display-backend",
            "mpv",
            "--display-wid",
            str(int(self._preview_surface.winId())),
            "--wall-clock",
            "--playback-mode",
            "realtime",
            "--out-root",
            logs_root,
        ]
        if self._chk_loop.isChecked():
            args.append("--loop-source")

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(args)
        process.setWorkingDirectory(str(_ROOT))
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._benchmark_finished)
        self._process = process
        process.start()
        if not process.waitForStarted(3000):
            self._set_running(False)
            self._lbl_status.setText("Benchmark failed to start.")
            QMessageBox.warning(
                self,
                "Playback Benchmark",
                "Could not start the benchmark subprocess.",
            )

    def _append_output(self, text: str):
        if not text:
            return
        self._txt_output.moveCursor(QTextCursor.MoveOperation.End)
        self._txt_output.insertPlainText(text)
        self._txt_output.moveCursor(QTextCursor.MoveOperation.End)

    def _read_stdout(self):
        process = self._process
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._stdout_text += text
        self._append_output(text)
        for line in text.splitlines():
            match = re.search(r"^\[bench\]\s+Output:\s+(.+)$", line.strip())
            if match:
                self._batch_dir = pathlib.Path(match.group(1).strip())

    def _read_stderr(self):
        process = self._process
        if process is None:
            return
        text = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        self._stderr_text += text
        self._append_output(text)

    def _discover_batch_dir(self) -> pathlib.Path | None:
        if self._batch_dir is not None and self._batch_dir.is_dir():
            return self._batch_dir
        root = pathlib.Path(self._txt_logs_root.text().strip() or self._logs_root)
        if not root.is_dir():
            return None
        candidates = [
            p for p in root.glob("*_cli_batch")
            if p.is_dir()
            and p.stat().st_mtime >= max(0.0, self._run_started_wall_t - 2.0)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _benchmark_finished(self, exit_code: int, _status):
        self._set_running(False)
        self._process = None
        batch_dir = self._discover_batch_dir()
        self._batch_dir = batch_dir
        if exit_code != 0:
            self._lbl_status.setText(f"Benchmark failed with exit code {exit_code}.")
            QMessageBox.warning(
                self,
                "Playback Benchmark Failed",
                "Benchmark did not complete. Review the output log in this window.",
            )
            return
        if batch_dir is None:
            self._lbl_status.setText("Benchmark completed, but no result folder was found.")
            return
        rows = self._load_result_rows(batch_dir)
        self._populate_result_table(rows)
        self._btn_open_logs.setEnabled(bool(rows))
        self._lbl_status.setText(
            f"Benchmark completed: {len(rows)} run(s). Logs: {batch_dir}"
        )

    def _load_result_rows(self, batch_dir: pathlib.Path) -> list[dict]:
        rows: list[dict] = []
        session_files = sorted(batch_dir.rglob("session.json"), key=_session_sort_key)
        for session_json in session_files:
            try:
                payload = json.loads(session_json.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            settings = payload.get("settings")
            worker_summary = payload.get("worker_summary")
            if not isinstance(settings, dict) or not isinstance(worker_summary, dict):
                continue
            precision_raw = str(settings.get("precision") or "-")
            row = {
                "precision": _display_precision_label(precision_raw),
                "precision_raw": precision_raw,
                "resolution": str(settings.get("resolution") or "-"),
                "hg": "on" if bool(settings.get("use_hg")) else "off",
                "fps": _metric_last(payload, "fps"),
                "fps_1p_low": _metric_last(payload, "fps_1p_low"),
                "latency_ms": _metric_last(payload, "latency_ms"),
                "model_ms": (
                    worker_summary.get("avg_model_latency_ms")
                    if worker_summary.get("avg_model_latency_ms") is not None
                    else _metric_last(payload, "model_latency_ms")
                ),
                "render_ms": _metric_last(payload, "render_ms"),
                "gpu_mb": _metric_max(payload, "gpu_mb"),
                "cpu_mb": _metric_max(payload, "cpu_mb"),
                "model_mb": _metric_last(payload, "model_mb"),
                "frames": worker_summary.get("timed_frames"),
                "dropped": worker_summary.get("dropped_frames", _metric_last(payload, "dropped_frames")),
                "session_dir": str(session_json.parent),
            }
            rows.append(row)
        return rows

    def _populate_result_table(self, rows: list[dict]):
        self._tbl_results.setSortingEnabled(False)
        self._tbl_results.setRowCount(0)
        self._result_session_dirs = []
        for row_data in rows:
            row = self._tbl_results.rowCount()
            self._tbl_results.insertRow(row)
            values = (
                (row_data["precision"], None),
                (row_data["resolution"], None),
                (row_data["hg"], 1 if row_data["hg"] == "on" else 0),
                (_fmt_float(row_data["fps"], 2), row_data["fps"]),
                (_fmt_float(row_data["fps_1p_low"], 2), row_data["fps_1p_low"]),
                (_fmt_float(row_data["latency_ms"], 2, " ms"), row_data["latency_ms"]),
                (_fmt_float(row_data["model_ms"], 2, " ms"), row_data["model_ms"]),
                (_fmt_float(row_data["render_ms"], 2, " ms"), row_data["render_ms"]),
                (_fmt_float(row_data["gpu_mb"], 0, " MB"), row_data["gpu_mb"]),
                (_fmt_float(row_data["cpu_mb"], 0, " MB"), row_data["cpu_mb"]),
                (_fmt_float(row_data["model_mb"], 2, " MB"), row_data["model_mb"]),
                (str(row_data["frames"] or "-"), row_data["frames"]),
                (str(row_data["dropped"] or 0), row_data["dropped"]),
            )
            tooltip = (
                f"Run id: {row_data.get('precision_raw') or row_data['precision']}\n"
                f"Resolution: {row_data['resolution']} | HG: {row_data['hg']}\n"
                f"Session: {row_data['session_dir']}"
            )
            for col, (value, sort_value) in enumerate(values):
                item = _ResultTableItem(value)
                if col == 0:
                    item.setData(_SESSION_ROLE, row_data["session_dir"])
                if sort_value is not None:
                    item.setData(_SORT_ROLE, sort_value)
                item.setToolTip(tooltip)
                if col >= 3:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._tbl_results.setItem(row, col, item)
            self._result_session_dirs.append(row_data["session_dir"])
        self._apply_result_column_defaults()
        self._tbl_results.setSortingEnabled(True)
        if rows:
            self._tbl_results.selectRow(0)

    def _cancel_benchmark(self):
        process = self._process
        if process is None:
            return
        self._lbl_status.setText("Canceling benchmark ...")
        process.terminate()
        if not process.waitForFinished(1500):
            process.kill()

    def closeEvent(self, event):
        if self._process is not None:
            answer = QMessageBox.question(
                self,
                "Benchmark Running",
                "Cancel the running playback benchmark?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._cancel_benchmark()
        super().closeEvent(event)
