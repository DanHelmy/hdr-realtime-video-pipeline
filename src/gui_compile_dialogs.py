"""Compile/precompile dialog classes extracted from gui.py."""

from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt, QProcess, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MAX_W = 1920
_DEFAULT_MAX_H = 1080


class _CompileDialog(QDialog):
    """Non-modal dialog shown while Triton kernels are being compiled
    in-process (loading screen during playback start)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compiling Kernels")
        self.setFixedSize(460, 160)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        title = QLabel("\u2699  Compiling optimized GPU kernels \u2026")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(title)

        detail = QLabel(
            "First run at this resolution may take 2\u20135 minutes.\n"
            "Subsequent runs load from cache in seconds."
        )
        detail.setStyleSheet("color: #aaa;")
        layout.addWidget(detail)

        bar = QProgressBar()
        bar.setRange(0, 0)                 # indeterminate / busy animation
        bar.setFixedHeight(6)
        bar.setTextVisible(False)
        layout.addWidget(bar)

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color: #6af;")
        layout.addWidget(self._lbl_status)

    def set_status(self, text: str):
        self._lbl_status.setText(text)


class _PrecompileOptionsDialog(QDialog):
    """Collect precision + resolution choices before launching compile."""

    def __init__(
        self,
        initial_precision: str,
        initial_resolution: str,
        precision_keys: list[str],
        max_w: int = _DEFAULT_MAX_W,
        max_h: int = _DEFAULT_MAX_H,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Pre-compile Kernels")
        self.setMinimumSize(420, 260)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        self._precision_combo = QComboBox()
        self._precision_combo.addItems(precision_keys)
        if initial_precision in precision_keys:
            self._precision_combo.setCurrentText(initial_precision)

        self._res_combo = QComboBox()
        res_options = [
            (f"1080p ({max_w}x{max_h})", f"{max_w}x{max_h}", "1080p"),
            ("720p (1280x720)", "1280x720", "720p"),
            ("540p (960x540)", "960x540", "540p"),
        ]
        for label, res_value, _res_key in res_options:
            self._res_combo.addItem(label, res_value)
        if initial_resolution:
            for i, (_label, _value, res_key) in enumerate(res_options):
                if initial_resolution == res_key:
                    self._res_combo.setCurrentIndex(i)
                    break

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Precision:"))
        layout.addWidget(self._precision_combo)

        layout.addWidget(QLabel("Resolution to compile:"))
        layout.addWidget(self._res_combo)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_ok = QPushButton("Start")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

    def selected_resolutions(self) -> list[str]:
        return [str(self._res_combo.currentData())]

    def selected_precision(self) -> str:
        return self._precision_combo.currentText()


class _PrecompileDialog(QDialog):
    """Modal dialog that launches ``compile_kernels.py`` in a separate
    process (zero GPU interference) and streams stdout into a log view."""

    def __init__(self, resolutions: list[str], precision: str = "fp16",
                 model_path: str | None = None, use_hg: bool = True,
                 hg_weights: str | None = None, clear_cache: bool = False,
                 predequantize_mode: str = "auto",
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pre-compile Kernels")
        self.setMinimumSize(540, 340)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        self._resolutions = resolutions
        self._precision = precision
        self._model_path = model_path
        self._use_hg = bool(use_hg)
        self._hg_weights = hg_weights
        self._clear_cache = clear_cache
        self._predequantize_mode = str(predequantize_mode).strip().lower()
        if self._predequantize_mode not in {"auto", "on", "off"}:
            self._predequantize_mode = "auto"
        self._process: QProcess | None = None
        self._finished_ok = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        title = QLabel("\u2699  Compiling optimised GPU kernels \u2026")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(title)

        detail = QLabel(
            f"Resolutions: {', '.join(resolutions)}  |  Precision: {precision}\n"
            "This runs in a clean process for best kernel quality.\n"
            "First compilation may take 2\u20135 minutes per resolution."
        )
        detail.setStyleSheet("color: #aaa;")
        layout.addWidget(detail)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)
        self._bar.setFixedHeight(6)
        self._bar.setTextVisible(False)
        layout.addWidget(self._bar)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setStyleSheet(
            "QTextEdit { background: #111; color: #ccc; border: 1px solid #333; }"
        )
        layout.addWidget(self._log, 1)

        btn_row = QHBoxLayout()
        self._btn_close = QPushButton("Close")
        self._btn_close.setFixedSize(90, 28)
        self._btn_close.setEnabled(False)
        self._btn_close.clicked.connect(self.accept)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_close)
        layout.addLayout(btn_row)

        # Start compilation immediately
        self._start()

    def _start(self):
        script = os.path.join(_HERE, "compile_kernels.py")
        args = [script] + self._resolutions
        args += ["--precision", self._precision]
        if self._model_path:
            args += ["--model", self._model_path]
        args += ["--use-hg", "1" if self._use_hg else "0"]
        args += ["--predequantize", self._predequantize_mode]
        if self._hg_weights:
            args += ["--hg-weights", self._hg_weights]
        if self._clear_cache:
            args += ["--clear-cache"]

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        # Force UTF-8 so Unicode chars (→, ×, etc.) in model print() don't
        # crash on Windows cp1252 console encoding.
        env = self._process.processEnvironment()
        if env.isEmpty():
            from PyQt6.QtCore import QProcessEnvironment
            env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        if os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
            env.insert("TORCHINDUCTOR_CACHE_DIR", os.environ["TORCHINDUCTOR_CACHE_DIR"])
        if os.environ.get("TRITON_CACHE_DIR"):
            env.insert("TRITON_CACHE_DIR", os.environ["TRITON_CACHE_DIR"])
        self._process.setProcessEnvironment(env)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_finished)

        self._log.append(f"$ python {' '.join(os.path.basename(a) for a in args)}\n")
        self._process.start(sys.executable, ["-u"] + args)

    def _on_stdout(self):
        data = self._process.readAllStandardOutput()
        text = bytes(data).decode("utf-8", errors="replace").rstrip()
        if text:
            self._log.append(text)
            # Auto-scroll
            sb = self._log.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _on_finished(self, exit_code, _status):
        self._bar.setRange(0, 1)
        self._bar.setValue(1)
        self._btn_close.setEnabled(True)

        if exit_code == 0:
            self._finished_ok = True
            self._log.append("\n\u2705  Done — kernels cached to disk.")
            self._log.append("Starting playback ...")
            # Auto-close after a brief pause so the user sees the result
            QTimer.singleShot(800, self.accept)
        else:
            self._log.append(f"\n\u274c  Process exited with code {exit_code}.")
            self._log.append("Check the log above for errors.")

        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
            self._process.waitForFinished(3000)
        super().closeEvent(event)

    @property
    def succeeded(self) -> bool:
        return self._finished_ok
