"""Compile/precompile dialog classes extracted from gui.py."""

from __future__ import annotations

import os
import subprocess
import sys

from PyQt6.QtCore import QProcess, QProcessEnvironment, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from gui_window_utils import configure_independent_window

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_DEFAULT_MAX_W = 1920
_DEFAULT_MAX_H = 1080


def _default_quality_trials() -> int:
    # Keep the normal GUI path fast. Users can opt into repeated clean trials
    # from the dialog or with HDRTVNET_GUI_AUTOTUNE_QUALITY_TRIALS.
    raw = os.environ.get(
        "HDRTVNET_GUI_AUTOTUNE_QUALITY_TRIALS",
        os.environ.get("HDRTVNET_AUTOTUNE_QUALITY_TRIALS", "1"),
    )
    try:
        value = int(raw)
    except Exception:
        value = 1
    return min(10, max(1, value))


def _python_executable_for_clean_subprocess() -> str:
    """Return the Python that should run helper scripts from the GUI.

    After GUI restarts, ``sys.executable`` can point at the base interpreter on
    some Windows venv layouts. Prefer the project venv when it exists so tools
    use the same installed torch/PyQt environment as the launcher.
    """
    candidates = []
    if os.name == "nt":
        candidates.append(os.path.join(_ROOT, "venv", "Scripts", "python.exe"))
    else:
        candidates.append(os.path.join(_ROOT, "venv", "bin", "python"))
    candidates.append(sys.executable)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(candidate)
        key = os.path.normcase(path)
        if key in seen:
            continue
        seen.add(key)
        if os.path.isfile(path):
            return path
    return sys.executable


def _kill_process_tree(process: QProcess | None) -> None:
    if process is None:
        return
    if process.state() == QProcess.ProcessState.NotRunning:
        return
    if os.name == "nt":
        try:
            pid = int(process.processId())
        except Exception:
            pid = 0
        if pid > 0:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
                return
            except Exception:
                pass
    process.kill()


class _CompileDialog(QDialog):
    """Non-modal dialog shown while Triton kernels are being compiled
    in-process (loading screen during playback start)."""

    def __init__(self, parent=None):
        super().__init__(None)
        self._owner_widget = parent
        self.setWindowTitle("Compiling Kernels")
        self.setMinimumSize(520, 180)
        self.resize(520, 180)
        configure_independent_window(self, minimize=False, maximize=False, close=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        self._lbl_title = QLabel("Compiling optimized GPU kernels...")
        self._lbl_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self._lbl_title.setWordWrap(True)
        layout.addWidget(self._lbl_title)

        self._lbl_detail = QLabel(
            "First run at this resolution may take 2\u20135 minutes.\n"
            "Subsequent runs load from cache in seconds."
        )
        self._lbl_detail.setWordWrap(True)
        self._lbl_detail.setProperty("muted", True)
        layout.addWidget(self._lbl_detail)

        bar = QProgressBar()
        bar.setRange(0, 0)                 # indeterminate / busy animation
        bar.setFixedHeight(6)
        bar.setTextVisible(False)
        layout.addWidget(bar)

        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setProperty("accentText", True)
        layout.addWidget(self._lbl_status)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setVisible(False)
        layout.addWidget(self._log, 1)

    def set_message(self, *, window_title: str, title: str, detail: str):
        self.setWindowTitle(window_title)
        self._lbl_title.setText(title)
        self._lbl_detail.setText(detail)

    def configure_for_tensorrt(
        self,
        *,
        cache_miss: bool,
        width: int,
        height: int,
        precision: str,
    ):
        resolution = f"{int(width)}x{int(height)}"
        if cache_miss:
            self.set_message(
                window_title="Building TensorRT Engine",
                title="Building optimized TensorRT engine...",
                detail=(
                    f"First run for {precision} at {resolution} can take a few minutes.\n"
                    "The finished engine is cached and reused on later runs."
                ),
            )
            self.set_status("Exporting ONNX and optimizing TensorRT tactics ...")
        else:
            self.set_message(
                window_title="Loading TensorRT Engine",
                title="Loading cached TensorRT engine...",
                detail=(
                    f"TensorRT engine cache found for {precision} at {resolution}.\n"
                    "Playback will start as soon as the runtime is ready."
                ),
            )
            self.set_status("Loading cached TensorRT engine ...")

    def set_status(self, text: str):
        self._lbl_status.setText(text)

    def append_log(self, text: str):
        line = str(text or "").strip()
        if not line:
            return
        if not self._log.isVisible():
            self._log.setVisible(True)
            self.resize(max(self.width(), 680), max(self.height(), 380))
        self._log.append(line)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())


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
        super().__init__(None)
        self._owner_widget = parent
        self.setWindowTitle("Pre-compile Kernels")
        self.setMinimumSize(420, 260)
        configure_independent_window(self, maximize=False)

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

        layout.addWidget(QLabel("Autotune quality:"))
        self._quality_combo = QComboBox()
        quality_options = [
            ("Fast single trial (recommended)", 1),
            ("Best-of-3 clean trials", 3),
            ("Best-of-5 clean trials", 5),
        ]
        for label, value in quality_options:
            self._quality_combo.addItem(label, value)
        default_trials = _default_quality_trials()
        closest = min(
            range(len(quality_options)),
            key=lambda idx: abs(quality_options[idx][1] - default_trials),
        )
        self._quality_combo.setCurrentIndex(closest)
        layout.addWidget(self._quality_combo)

        self._clean_compile = QCheckBox("Clean recompile first")
        self._clean_compile.setChecked(True)
        self._clean_compile.setToolTip(
            "Deletes the current project kernel cache before compiling so stale "
            "or unlucky autotune picks cannot be reused."
        )
        layout.addWidget(self._clean_compile)

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

    def selected_quality_trials(self) -> int:
        try:
            return int(self._quality_combo.currentData())
        except Exception:
            return _default_quality_trials()

    def selected_clean_compile(self) -> bool:
        return bool(self._clean_compile.isChecked())


class _PrecompileDialog(QDialog):
    """Modal dialog that launches ``compile_kernels.py`` in a separate
    process (zero GPU interference) and streams stdout into a log view."""

    def __init__(self, resolutions: list[str], precision: str = "fp16",
                 model_path: str | None = None, use_hg: bool = True,
                 hg_weights: str | None = None, clear_cache: bool = False,
                 predequantize_mode: str = "auto",
                 quality_trials: int | None = None,
                 parent=None):
        super().__init__(None)
        self._owner_widget = parent
        self.setWindowTitle("Pre-compile Kernels")
        self.setMinimumSize(540, 340)
        configure_independent_window(self)

        self._resolutions = resolutions
        self._precision = precision
        self._model_path = model_path
        self._use_hg = bool(use_hg)
        self._hg_weights = hg_weights
        self._clear_cache = clear_cache
        self._quality_trials = _default_quality_trials() if quality_trials is None else int(quality_trials)
        self._quality_trials = min(10, max(1, self._quality_trials))
        self._predequantize_mode = str(predequantize_mode).strip().lower()
        if self._predequantize_mode not in {"auto", "on", "off"}:
            self._predequantize_mode = "auto"
        self._process: QProcess | None = None
        self._finished_ok = False
        self._finish_handled = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        title = QLabel("Compiling optimized GPU kernels...")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(title)

        detail = QLabel(
            f"Resolutions: {', '.join(resolutions)}  |  Precision: {precision}\n"
            + (
                f"Best-of-{self._quality_trials} clean autotune trials; fastest cache wins.\n"
                f"Expect roughly {self._quality_trials}x first-compile time.\n"
                if self._quality_trials > 1
                else "This runs in a clean process for best kernel quality.\n"
            )
            + "A short GPU clock pre-warm runs before autotune timing.\n"
            + "First compilation may take 2\u20135 minutes per resolution.\n"
            "If autotune quality looks unusually bad, restarting the PC before "
            "recompiling can help."
        )
        detail.setProperty("muted", True)
        layout.addWidget(detail)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)
        self._bar.setFixedHeight(6)
        self._bar.setTextVisible(False)
        layout.addWidget(self._bar)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        layout.addWidget(self._log, 1)

        btn_row = QHBoxLayout()
        self._btn_close = QPushButton("Cancel")
        self._btn_close.setFixedSize(90, 28)
        self._btn_close.clicked.connect(self._on_close_clicked)
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
        if self._quality_trials > 1:
            args += ["--quality-trials", str(self._quality_trials)]

        self._process = QProcess(self)
        self._process.setWorkingDirectory(_ROOT)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        # Force UTF-8 so Unicode chars (→, ×, etc.) in model print() don't
        # crash on Windows cp1252 console encoding.
        env = self._process.processEnvironment()
        if env.isEmpty():
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
        self._process.errorOccurred.connect(self._on_process_error)

        python_exe = _python_executable_for_clean_subprocess()
        self._log.append(
            f"$ {os.path.basename(python_exe)} "
            f"{' '.join(os.path.basename(a) for a in args)}\n"
        )
        self._process.start(python_exe, ["-u"] + args)
        QTimer.singleShot(5000, self._check_started)

    def _on_stdout(self):
        if self._process is None:
            return
        data = self._process.readAllStandardOutput()
        text = bytes(data).decode("utf-8", errors="replace").rstrip()
        if text:
            self._log.append(text)
            # Auto-scroll
            sb = self._log.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _check_started(self):
        if self._finish_handled or self._process is None:
            return
        if self._process.state() != QProcess.ProcessState.NotRunning:
            return
        self._finish_with_error(
            "Compiler process did not stay running. Check the log above for errors."
        )

    def _on_process_error(self, error):
        if self._finish_handled:
            return
        detail = ""
        try:
            detail = self._process.errorString() if self._process is not None else ""
        except Exception:
            detail = ""
        self._finish_with_error(f"Compiler process error: {error}. {detail}".strip())

    def _finish_with_error(self, message: str):
        if self._finish_handled:
            return
        self._finish_handled = True
        self._bar.setRange(0, 1)
        self._bar.setValue(1)
        self._btn_close.setText("Close")
        self._btn_close.setEnabled(True)
        self._log.append(f"\n{message}")
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, exit_code, _status):
        if self._finish_handled:
            return
        self._finish_handled = True
        self._bar.setRange(0, 1)
        self._bar.setValue(1)
        self._btn_close.setText("Close")
        self._btn_close.setEnabled(True)

        if exit_code == 0:
            self._finished_ok = True
            self._log.append("\nDone - kernels cached to disk.")
            self._log.append("Pre-compile finished.")
            # Auto-close after a brief pause so the user sees the result
            QTimer.singleShot(800, self.accept)
        else:
            self._log.append(f"\nProcess exited with code {exit_code}.")
            self._log.append("Check the log above for errors.")

        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_close_clicked(self):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._log.append("\nCanceling compiler process ...")
            _kill_process_tree(self._process)
            self._process.waitForFinished(3000)
            self._finished_ok = False
            self.reject()
            return
        self.accept()

    def closeEvent(self, event):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            _kill_process_tree(self._process)
            self._process.waitForFinished(3000)
        super().closeEvent(event)

    @property
    def succeeded(self) -> bool:
        return self._finished_ok
