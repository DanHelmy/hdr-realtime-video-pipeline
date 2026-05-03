from __future__ import annotations

import gc
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui_config import PRECISIONS, _available_precision_keys, _select_model_path
from gui_media_probe import _probe_hdr_input, _probe_video_timing_info
from gui_pipeline_worker_frame_processing import PipelineWorkerFrameProcessingMixin
from gui_pipeline_worker_model import PipelineWorkerModelMixin, _resolve_predequantize_arg
from gui_scaling import _letterbox_bgr
from models.hdrtvnet_torch import (
    HDRTVNetTensorRT,
    HDRTVNetTorch,
    _HAS_COMPILE,
    _HAS_HIP_SDK,
    _HAS_TRITON,
    _IS_NVIDIA,
    _IS_ROCM,
)
from video_source import VideoSource

EXPORT_HDR_TARGET_PEAK_NITS = 1001.0


def _ensure_even(value: int) -> int:
    value = max(2, int(value))
    return value if (value % 2) == 0 else value + 1


def _sanitize_filename_part(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text or "").strip())
    safe = re.sub(r"_+", "_", safe).strip("._-")
    return safe or "export"


def _fmt_fps_tag(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


@dataclass
class ExportJobConfig:
    source_path: str
    output_path: str
    precision_key: str
    use_hg: bool
    width: int
    height: int
    fps: float
    use_max_autotune: bool
    predequantize_mode: str


def _export_autotune_unavailable_reason() -> str:
    if not torch.cuda.is_available():
        return "Requires a CUDA/ROCm GPU device."
    if not _HAS_COMPILE:
        return "This PyTorch build does not support torch.compile."
    if not _HAS_TRITON:
        return "Triton is not installed."
    if os.name == "nt" and _IS_ROCM and not _HAS_HIP_SDK:
        return "ROCm on Windows needs the AMD HIP SDK installed."
    return ""


class ExportOptionsDialog(QDialog):
    """Independent export configuration dialog."""

    def __init__(
        self,
        *,
        initial_source_path: str | None,
        suggested_dir: str,
        initial_precision_key: str | None = None,
        initial_use_hg: bool | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export Video")
        self.setModal(True)
        self.resize(760, 520)

        self._suggested_dir = suggested_dir
        self._source_info: dict | None = None
        self._source_aspect: float = 16.0 / 9.0
        self._path_user_edited = False
        self._syncing_aspect = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        intro = QLabel(
            "Export uses its own settings. Playback tab precision, HG, and "
            "resolution choices are not reused here."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        output_layout.setContentsMargins(0, 8, 0, 0)
        output_layout.setSpacing(10)
        tabs.addTab(output_tab, "Output")

        source_group = QGroupBox("Source")
        source_form = QFormLayout(source_group)
        self._txt_source = QLineEdit()
        self._txt_source.setReadOnly(True)
        self._btn_source = QPushButton("Browse...")
        src_row = QHBoxLayout()
        src_row.setContentsMargins(0, 0, 0, 0)
        src_row.addWidget(self._txt_source, 1)
        src_row.addWidget(self._btn_source)
        src_widget = QWidget()
        src_widget.setLayout(src_row)
        source_form.addRow("Video:", src_widget)
        self._lbl_source_info = QLabel("Native: -")
        self._lbl_source_info.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        source_form.addRow("Native:", self._lbl_source_info)
        output_layout.addWidget(source_group)

        render_group = QGroupBox("Render")
        render_grid = QGridLayout(render_group)
        render_grid.setHorizontalSpacing(10)
        render_grid.setVerticalSpacing(8)

        self._spn_width = QSpinBox()
        self._spn_width.setRange(2, 16384)
        self._spn_width.setSingleStep(2)
        self._spn_width.setAccelerated(True)
        self._spn_width.setKeyboardTracking(False)
        self._spn_height = QSpinBox()
        self._spn_height.setRange(2, 16384)
        self._spn_height.setSingleStep(2)
        self._spn_height.setAccelerated(True)
        self._spn_height.setKeyboardTracking(False)
        self._chk_keep_aspect = QCheckBox("Keep source aspect")
        self._chk_keep_aspect.setChecked(True)
        self._spn_fps = QDoubleSpinBox()
        self._spn_fps.setRange(1.0, 240.0)
        self._spn_fps.setDecimals(3)
        self._spn_fps.setSingleStep(1.0)
        self._spn_fps.setKeyboardTracking(False)
        self._btn_reset_native = QPushButton("Reset to Native")

        render_grid.addWidget(QLabel("Width:"), 0, 0)
        render_grid.addWidget(self._spn_width, 0, 1)
        render_grid.addWidget(QLabel("Height:"), 0, 2)
        render_grid.addWidget(self._spn_height, 0, 3)
        render_grid.addWidget(self._chk_keep_aspect, 1, 0, 1, 2)
        render_grid.addWidget(QLabel("FPS:"), 1, 2)
        render_grid.addWidget(self._spn_fps, 1, 3)
        render_grid.addWidget(self._btn_reset_native, 2, 0, 1, 2)

        self._lbl_scale_note = QLabel(
            "If the output aspect ratio does not match the source, export uses "
            "Resolve-style \"Scale entire image to fit\" behavior: the image is "
            "resized to fit inside the chosen canvas and padded with black bars."
        )
        self._lbl_scale_note.setWordWrap(True)
        render_grid.addWidget(self._lbl_scale_note, 3, 0, 1, 4)
        output_layout.addWidget(render_group)

        file_group = QGroupBox("File")
        file_form = QFormLayout(file_group)
        self._txt_output = QLineEdit()
        self._btn_output = QPushButton("Browse...")
        out_row = QHBoxLayout()
        out_row.setContentsMargins(0, 0, 0, 0)
        out_row.addWidget(self._txt_output, 1)
        out_row.addWidget(self._btn_output)
        out_widget = QWidget()
        out_widget.setLayout(out_row)
        file_form.addRow("Output:", out_widget)
        self._lbl_codec = QLabel(
            "Container / Codec: MOV / ProRes 422 HQ (video) + PCM audio if present"
        )
        self._lbl_codec.setWordWrap(True)
        file_form.addRow("Format:", self._lbl_codec)
        output_layout.addWidget(file_group)

        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        model_layout.setContentsMargins(0, 8, 0, 0)
        model_layout.setSpacing(10)
        tabs.addTab(model_tab, "Model")

        model_group = QGroupBox("Model Preset")
        model_form = QFormLayout(model_group)
        self._cmb_precision = QComboBox()
        self._cmb_precision.addItems(_available_precision_keys())
        self._chk_use_hg = QCheckBox("Use HG highlight refinement")
        self._chk_use_hg.setChecked(True)
        if (
            initial_precision_key
            and initial_precision_key in _available_precision_keys()
        ):
            self._cmb_precision.setCurrentText(initial_precision_key)
        if initial_use_hg is not None:
            self._chk_use_hg.setChecked(bool(initial_use_hg))
        model_form.addRow("Preset:", self._cmb_precision)
        model_form.addRow("", self._chk_use_hg)
        model_layout.addWidget(model_group)

        disclaimer = QLabel(
            "Export is intentionally limited to ProRes 422 HQ. Most source files are "
            "already compressed with H.265/HEVC or similar delivery codecs; exporting "
            "back to H.265 would add another lossy generation. ProRes 422 HQ is used "
            "as a high-quality mezzanine format instead.\n\n"
            "Note: ProRes 422 HQ is visually high quality and edit-friendly, but it is "
            "not mathematically lossless."
        )
        disclaimer.setWordWrap(True)
        model_layout.addWidget(disclaimer)
        model_layout.addStretch(1)

        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        advanced_layout.setContentsMargins(0, 8, 0, 0)
        advanced_layout.setSpacing(10)
        tabs.addTab(advanced_tab, "Advanced")

        perf_group = QGroupBox("Performance")
        perf_form = QFormLayout(perf_group)
        self._chk_use_max_autotune = QCheckBox(
            "Use experimental max-autotune compile"
        )
        self._chk_use_max_autotune.setChecked(False)
        autotune_reason = _export_autotune_unavailable_reason()
        if autotune_reason:
            self._chk_use_max_autotune.setEnabled(False)
            self._chk_use_max_autotune.setToolTip(autotune_reason)
        perf_form.addRow("", self._chk_use_max_autotune)
        self._lbl_autotune_note = QLabel(
            "Off by default. This keeps export in eager mode for the most stable "
            "behavior.\n\n"
            "Turn it on only for longer fixed-resolution exports when you want to "
            "trade a slower startup for potentially better throughput."
        )
        self._lbl_autotune_note.setWordWrap(True)
        perf_form.addRow("Max autotune:", self._lbl_autotune_note)
        if autotune_reason:
            self._lbl_autotune_availability = QLabel(
                f"Unavailable on this setup: {autotune_reason}"
            )
        else:
            self._lbl_autotune_availability = QLabel(
                "Available on this setup. Export will compile in-process without "
                "using the precompile cache/restart flow."
            )
        self._lbl_autotune_availability.setWordWrap(True)
        perf_form.addRow("Availability:", self._lbl_autotune_availability)
        advanced_layout.addWidget(perf_group)

        int8_group = QGroupBox("INT8")
        int8_form = QFormLayout(int8_group)
        self._cmb_predequantize = QComboBox()
        self._cmb_predequantize.addItem("Auto", "auto")
        self._cmb_predequantize.addItem("Force On", "on")
        self._cmb_predequantize.addItem("Force Off", "off")
        int8_form.addRow("Pre-dequantize:", self._cmb_predequantize)
        self._lbl_predequant_note = QLabel("")
        self._lbl_predequant_note.setWordWrap(True)
        int8_form.addRow("Note:", self._lbl_predequant_note)
        advanced_layout.addWidget(int8_group)
        advanced_layout.addStretch(1)
        if _IS_NVIDIA:
            perf_group.hide()
            int8_group.hide()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        ok_button.setText("Start Export")
        ok_button.setAutoDefault(False)
        ok_button.setDefault(False)
        cancel_button.setAutoDefault(False)
        cancel_button.setDefault(False)
        root.addWidget(buttons)

        self._btn_source.clicked.connect(self._browse_source)
        self._btn_output.clicked.connect(self._browse_output)
        self._btn_reset_native.clicked.connect(self._reset_to_native)
        self._txt_output.textEdited.connect(self._mark_output_path_dirty)
        self._spn_width.valueChanged.connect(self._on_width_changed)
        self._spn_height.valueChanged.connect(self._on_height_changed)
        self._spn_fps.valueChanged.connect(self._update_output_path_if_auto)
        self._cmb_precision.currentTextChanged.connect(self._update_output_path_if_auto)
        self._chk_use_hg.toggled.connect(self._update_output_path_if_auto)
        self._cmb_precision.currentTextChanged.connect(self._update_advanced_state)
        self._chk_keep_aspect.toggled.connect(self._on_keep_aspect_toggled)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        for widget in (
            self._txt_output,
            self._spn_width,
            self._spn_height,
            self._spn_fps,
        ):
            widget.installEventFilter(self)
        for editor in (
            self._spn_width.lineEdit(),
            self._spn_height.lineEdit(),
            self._spn_fps.lineEdit(),
        ):
            if editor is not None:
                editor.installEventFilter(self)

        if initial_source_path and os.path.isfile(initial_source_path):
            if not self._set_source_path(initial_source_path):
                self._reset_defaults()
        else:
            self._reset_defaults()
        self._update_advanced_state()

    def _reset_defaults(self):
        self._source_info = None
        self._source_aspect = 16.0 / 9.0
        self._lbl_source_info.setText("Native: -")
        self._syncing_aspect = True
        try:
            self._spn_width.setValue(1920)
            self._spn_height.setValue(1080)
            self._spn_fps.setValue(30.0)
        finally:
            self._syncing_aspect = False
        self._update_output_path_if_auto()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and event.key() in (
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
        ):
            if obj in {
                self._txt_output,
                self._spn_width,
                self._spn_height,
                self._spn_fps,
                self._spn_width.lineEdit(),
                self._spn_height.lineEdit(),
                self._spn_fps.lineEdit(),
            }:
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def _browse_source(self):
        start_dir = self._suggested_dir
        current = self._txt_source.text().strip()
        if current and os.path.isfile(current):
            start_dir = os.path.dirname(current)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Source Video",
            start_dir,
            "Video (*.mp4 *.avi *.mkv *.mov *.webm *.flv);;All (*)",
        )
        if path:
            self._set_source_path(path)

    def _browse_output(self):
        start_dir = self._suggested_dir
        current = self._txt_output.text().strip()
        if current:
            cand_dir = os.path.dirname(current)
            if cand_dir and os.path.isdir(cand_dir):
                start_dir = cand_dir
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export ProRes 422 HQ",
            self._txt_output.text().strip() or start_dir,
            "QuickTime MOV (*.mov)",
        )
        if path:
            if not path.lower().endswith(".mov"):
                path += ".mov"
            self._path_user_edited = True
            self._txt_output.setText(path)

    def _set_source_path(self, path: str) -> bool:
        hdr_info = _probe_hdr_input(path)
        if bool(hdr_info.get("is_hdr", False)):
            QMessageBox.warning(
                self,
                "Unsupported Input",
                "HDR input videos are not supported for conversion.\n\n"
                f"Selected file appears HDR ({hdr_info.get('reason', 'HDR metadata detected')}).\n"
                "Please choose an SDR source video.",
            )
            return False

        self._txt_source.setText(path)
        self._suggested_dir = os.path.dirname(path) or self._suggested_dir
        info = _probe_video_timing_info(path)
        self._source_info = info
        if info:
            self._source_aspect = float(info["width"]) / max(1.0, float(info["height"]))
            self._lbl_source_info.setText(
                f"{info['width']}x{info['height']} @ {float(info['fps'] or 0.0):.3f} fps"
            )
            self._syncing_aspect = True
            try:
                self._spn_width.setValue(_ensure_even(int(info["width"] or 1920)))
                self._spn_height.setValue(_ensure_even(int(info["height"] or 1080)))
                self._spn_fps.setValue(max(1.0, float(info["fps"] or 30.0)))
            finally:
                self._syncing_aspect = False
        else:
            self._reset_defaults()
        self._path_user_edited = False
        self._update_output_path_if_auto()
        return True

    def _mark_output_path_dirty(self, _text: str):
        self._path_user_edited = True

    def _reset_to_native(self):
        info = self._source_info or {}
        self._syncing_aspect = True
        try:
            self._spn_width.setValue(_ensure_even(int(info.get("width") or 1920)))
            self._spn_height.setValue(_ensure_even(int(info.get("height") or 1080)))
            self._spn_fps.setValue(max(1.0, float(info.get("fps") or 30.0)))
        finally:
            self._syncing_aspect = False
        self._update_output_path_if_auto()

    def _on_keep_aspect_toggled(self, checked: bool):
        if checked:
            self._sync_height_from_width()
        self._update_output_path_if_auto()

    def _on_width_changed(self, value: int):
        if self._syncing_aspect:
            return
        if self._chk_keep_aspect.isChecked():
            self._sync_height_from_width(width=value)
        self._update_output_path_if_auto()

    def _on_height_changed(self, value: int):
        if self._syncing_aspect:
            return
        if self._chk_keep_aspect.isChecked():
            self._sync_width_from_height(height=value)
        self._update_output_path_if_auto()

    def _sync_height_from_width(self, *, width: int | None = None):
        if width is None:
            width = int(self._spn_width.value())
        width = _ensure_even(width)
        height = _ensure_even(int(round(width / max(self._source_aspect, 1e-6))))
        self._syncing_aspect = True
        try:
            self._spn_width.setValue(width)
            self._spn_height.setValue(height)
        finally:
            self._syncing_aspect = False

    def _sync_width_from_height(self, *, height: int | None = None):
        if height is None:
            height = int(self._spn_height.value())
        height = _ensure_even(height)
        width = _ensure_even(int(round(height * self._source_aspect)))
        self._syncing_aspect = True
        try:
            self._spn_height.setValue(height)
            self._spn_width.setValue(width)
        finally:
            self._syncing_aspect = False

    def _default_output_path(self) -> str:
        source_path = self._txt_source.text().strip()
        if source_path and os.path.isfile(source_path):
            out_dir = os.path.dirname(source_path)
            stem = os.path.splitext(os.path.basename(source_path))[0]
        else:
            out_dir = self._suggested_dir
            stem = "hdrtvnet_export"
        prec = _sanitize_filename_part(self._cmb_precision.currentText())
        hg_tag = "hg" if self._chk_use_hg.isChecked() else "nohg"
        wh = f"{int(self._spn_width.value())}x{int(self._spn_height.value())}"
        fps_tag = _fmt_fps_tag(float(self._spn_fps.value()))
        name = f"{stem}_hdrtvnet_{prec}_{hg_tag}_{wh}_{fps_tag}fps_prores.mov"
        return os.path.join(out_dir, name)

    def _update_output_path_if_auto(self, *_args):
        if self._path_user_edited:
            return
        self._txt_output.setText(self._default_output_path())

    def _selected_predequantize_mode(self) -> str:
        value = self._cmb_predequantize.currentData()
        if value in {"auto", "on", "off"}:
            return str(value)
        return "auto"

    def _update_advanced_state(self, *_args):
        is_int8 = str(self._cmb_precision.currentText()).lower().startswith("int8")
        self._cmb_predequantize.setEnabled(is_int8)
        if is_int8:
            self._lbl_predequant_note.setText(
                "Only affects INT8 presets. `Auto` is the safest default. "
                "`Force On` can help GPUs that do not benefit from INT8 tensor-core "
                "style execution. `Force Off` keeps the quantized layers live at runtime."
            )
        else:
            self._lbl_predequant_note.setText(
                "Only used for INT8 export presets. FP16 and FP32 ignore this setting."
            )

    def selected_config(self) -> ExportJobConfig | None:
        source_path = self._txt_source.text().strip()
        if not source_path or not os.path.isfile(source_path):
            QMessageBox.warning(self, "Export", "Choose a source video first.")
            return None

        hdr_info = _probe_hdr_input(source_path)
        if bool(hdr_info.get("is_hdr", False)):
            QMessageBox.warning(
                self,
                "Unsupported Input",
                "HDR input videos are not supported for conversion.\n\n"
                f"Selected file appears HDR ({hdr_info.get('reason', 'HDR metadata detected')}).\n"
                "Please choose an SDR source video.",
            )
            return None

        output_path = self._txt_output.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Export", "Choose an output .mov file.")
            return None
        if not output_path.lower().endswith(".mov"):
            output_path += ".mov"
        try:
            if os.path.normcase(os.path.abspath(output_path)) == os.path.normcase(
                os.path.abspath(source_path)
            ):
                QMessageBox.warning(
                    self,
                    "Export",
                    "Output path must be different from the source video.",
                )
                return None
        except Exception:
            pass
        out_dir = os.path.dirname(output_path) or "."
        if not os.path.isdir(out_dir):
            QMessageBox.warning(self, "Export", "Output folder does not exist.")
            return None

        precision_key = self._cmb_precision.currentText()
        model_path = _select_model_path(precision_key, self._chk_use_hg.isChecked())
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(
                self,
                "Export",
                f"Selected model weights were not found:\n{model_path}",
            )
            return None

        return ExportJobConfig(
            source_path=source_path,
            output_path=output_path,
            precision_key=precision_key,
            use_hg=bool(self._chk_use_hg.isChecked()),
            width=_ensure_even(int(self._spn_width.value())),
            height=_ensure_even(int(self._spn_height.value())),
            fps=max(1.0, float(self._spn_fps.value())),
            use_max_autotune=bool(
                self._chk_use_max_autotune.isChecked()
                and self._chk_use_max_autotune.isEnabled()
            ),
            predequantize_mode=self._selected_predequantize_mode(),
        )

    def accept(self):
        config = self.selected_config()
        if config is None:
            return
        self._accepted_config = config
        super().accept()

    def export_config(self) -> ExportJobConfig | None:
        return getattr(self, "_accepted_config", None)


class VideoExportWorker(QObject, PipelineWorkerFrameProcessingMixin):
    compile_ready = pyqtSignal()
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)
    canceled = pyqtSignal(str)

    def __init__(self, config: ExportJobConfig):
        super().__init__()
        self._config = config
        self._cancel_requested = threading.Event()
        self._runtime_lock = threading.Lock()
        self._source = None
        self._ffmpeg_proc = None
        self._processor = None
        self._rgb48_host_tensor = None
        self._rgb48_host_np = None
        self._rgb48_host_shape = None
        self._reset_enhance_history()

    def cancel(self):
        self._cancel_requested.set()
        with self._runtime_lock:
            ffmpeg_proc = self._ffmpeg_proc
        if ffmpeg_proc is not None:
            try:
                if ffmpeg_proc.stdin is not None:
                    ffmpeg_proc.stdin.close()
            except Exception:
                pass
            try:
                ffmpeg_proc.terminate()
            except Exception:
                pass

    def _fail(self, message: str):
        self.failed.emit(message)

    def _canceled(self) -> bool:
        return bool(self._cancel_requested.is_set())

    def _cleanup_runtime(self, source, ffmpeg_proc, processor):
        with self._runtime_lock:
            self._source = None
            self._ffmpeg_proc = None
            self._processor = None
            self._rgb48_host_tensor = None
            self._rgb48_host_np = None
            self._rgb48_host_shape = None

        if source is not None:
            try:
                source.release()
            except Exception:
                pass

        if ffmpeg_proc is not None:
            try:
                if ffmpeg_proc.stdin is not None and not ffmpeg_proc.stdin.closed:
                    ffmpeg_proc.stdin.close()
            except Exception:
                pass
            try:
                ffmpeg_proc.terminate()
            except Exception:
                pass
            try:
                ffmpeg_proc.wait(timeout=3)
            except Exception:
                try:
                    ffmpeg_proc.kill()
                except Exception:
                    pass

        if processor is not None:
            try:
                if hasattr(processor, "model"):
                    del processor.model
            except Exception:
                pass
            try:
                del processor
            except Exception:
                pass

        try:
            torch._dynamo.reset()
        except Exception:
            pass

        try:
            gc.collect()
        except Exception:
            pass

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception:
                pass

    def _ensure_rgb48_host_buffer(self, height: int, width: int):
        shape = (int(height), int(width), 3)
        if self._rgb48_host_tensor is not None and self._rgb48_host_shape == shape:
            return
        tensor = None
        try:
            if torch.cuda.is_available():
                tensor = torch.empty(shape, dtype=torch.uint16, pin_memory=True)
        except Exception:
            tensor = None
        if tensor is None:
            tensor = torch.empty(shape, dtype=torch.uint16)
        self._rgb48_host_tensor = tensor
        self._rgb48_host_np = tensor.numpy()
        self._rgb48_host_shape = shape

    def _tensor_to_rgb48_bytes(self, tensor) -> bytes:
        with torch.inference_mode():
            prepared = tensor[0] if isinstance(tensor, (tuple, list)) else tensor
            rgb_u16 = (
                prepared.squeeze(0)
                .clamp(0.0, 1.0)
                .mul(65535.0)
                .add_(0.5)
                .to(dtype=torch.uint16)
                .permute(1, 2, 0)
                .contiguous()
            )
        if rgb_u16.device.type == "cuda":
            self._ensure_rgb48_host_buffer(rgb_u16.shape[0], rgb_u16.shape[1])
            self._rgb48_host_tensor.copy_(rgb_u16, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            return self._rgb48_host_np.tobytes()
        return rgb_u16.cpu().numpy().tobytes()

    def run(self):
        source = None
        ffmpeg_proc = None
        processor = None
        finished_output_path = None
        writer_thread = None
        writer_stop = threading.Event()
        frame_queue = queue.Queue(maxsize=3)
        writer_error: list[Exception] = []
        writer_sentinel = object()
        try:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                self._fail("ffmpeg was not found on PATH. Install FFmpeg to export.")
                return

            cfg = PRECISIONS.get(self._config.precision_key, {})
            model_path = _select_model_path(
                self._config.precision_key,
                self._config.use_hg,
            )
            if not model_path or not os.path.isfile(model_path):
                self._fail(f"Model weights not found:\n{model_path}")
                return

            src_info = _probe_video_timing_info(self._config.source_path)
            if not src_info:
                self._fail("Could not read source video metadata.")
                return
            src_fps = max(1.0, float(src_info.get("fps") or 30.0))
            frame_count = max(0, int(src_info.get("frame_count") or 0))

            if self._config.use_max_autotune:
                self.progress.emit(
                    0,
                    "Loading export model (max-autotune enabled; reusing cached playback kernels when available) ...",
                )
            else:
                self.progress.emit(0, "Loading export model ...")
            if _IS_NVIDIA:
                processor = HDRTVNetTensorRT(
                    model_path,
                    device="auto",
                    precision=str(cfg.get("precision") or "fp16"),
                    engine_width=int(self._config.width),
                    engine_height=int(self._config.height),
                    mode_name=f"{self._config.precision_key}_{'hg' if self._config.use_hg else 'nohg'}",
                    use_hg=self._config.use_hg,
                )
            else:
                processor = HDRTVNetTorch(
                    model_path,
                    device="auto",
                    precision=str(cfg.get("precision") or "fp16"),
                    compile_model=bool(self._config.use_max_autotune),
                    compile_mode="max-autotune",
                    predequantize=_resolve_predequantize_arg(
                        str(self._config.predequantize_mode)
                    ),
                    use_hg=self._config.use_hg,
                )
            with self._runtime_lock:
                self._processor = processor

            if self._config.use_max_autotune and not _IS_NVIDIA:
                self.progress.emit(
                    0,
                    f"Warming up kernels for {int(self._config.width)}x{int(self._config.height)} ({self._config.precision_key}) ...",
                )
                PipelineWorkerModelMixin._silent_warmup(
                    processor,
                    int(self._config.width),
                    int(self._config.height),
                )
            self.compile_ready.emit()
            self.progress.emit(0, "Preparing export pipeline ...")

            vf_filters: list[str] = []
            vf_filters.append("deband")
            vf_filters.append(
                "zscale="
                "matrixin=gbr:"
                "transferin=smpte2084:"
                "primariesin=bt2020:"
                "rangein=full:"
                "matrix=bt2020nc:"
                "transfer=smpte2084:"
                "primaries=bt2020:"
                "range=limited:"
                f"dither=error_diffusion:npl={EXPORT_HDR_TARGET_PEAK_NITS:.0f}"
            )
            vf_filters.append("format=yuv422p10le")
            if abs(float(self._config.fps) - src_fps) > 1e-3:
                vf_filters.append(f"fps={float(self._config.fps):.6f}")

            ffmpeg_cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb48le",
                "-s:v",
                f"{int(self._config.width)}x{int(self._config.height)}",
                "-r",
                f"{src_fps:.6f}",
                "-color_range",
                "pc",
                "-colorspace",
                "bt2020nc",
                "-color_trc",
                "smpte2084",
                "-color_primaries",
                "bt2020",
                "-i",
                "-",
                "-i",
                self._config.source_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
            ]
            if vf_filters:
                ffmpeg_cmd += ["-vf", ",".join(vf_filters)]
            ffmpeg_cmd += [
                "-c:v",
                "prores_ks",
                "-profile:v",
                "3",
                "-pix_fmt",
                "yuv422p10le",
                "-bsf:v",
                "prores_metadata=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc",
                "-color_range",
                "tv",
                "-colorspace",
                "bt2020nc",
                "-color_trc",
                "smpte2084",
                "-color_primaries",
                "bt2020",
                "-vendor",
                "apl0",
                "-c:a",
                "pcm_s16le",
                "-movflags",
                "+faststart+write_colr",
                self._config.output_path,
            ]

            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            with self._runtime_lock:
                self._ffmpeg_proc = ffmpeg_proc

            source = VideoSource(self._config.source_path, prefetch=2)
            with self._runtime_lock:
                self._source = source

            def _writer_loop():
                try:
                    while True:
                        if writer_stop.is_set() and frame_queue.empty():
                            break
                        try:
                            item = frame_queue.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if item is writer_sentinel:
                            break
                        if ffmpeg_proc.stdin is None:
                            raise RuntimeError("FFmpeg stdin was not available.")
                        ffmpeg_proc.stdin.write(item)
                except Exception as exc:
                    if not self._canceled():
                        writer_error.append(exc)
                finally:
                    try:
                        if ffmpeg_proc.stdin is not None and not ffmpeg_proc.stdin.closed:
                            ffmpeg_proc.stdin.close()
                    except Exception:
                        pass

            writer_thread = threading.Thread(
                target=_writer_loop,
                name="hdrtvnet-export-writer",
                daemon=True,
            )
            writer_thread.start()

            frames_done = 0
            last_emit_t = 0.0

            while not self._canceled():
                if writer_error:
                    raise RuntimeError(str(writer_error[0]))
                ret, frame = source.read()
                if not ret:
                    break
                if self._canceled():
                    raise InterruptedError("Export canceled by user.")
                model_inp = _letterbox_bgr(
                    frame,
                    int(self._config.width),
                    int(self._config.height),
                )
                with torch.inference_mode():
                    tensor, cond = processor.preprocess(model_inp)
                    raw_out = processor.infer((tensor, cond))
                    prepared_out = self._prepare_hdr_output_tensor(
                        raw_out,
                        lower_res_processing=False,
                    )
                    export_out = self._apply_hdr_flat_surface_cleanup(
                        prepared_out,
                        model_inp,
                        quality=os.environ.get("HDRTVNET_EXPORT_HDR_CLEANUP", "highlight-high"),
                    )
                output_rgb48 = self._tensor_to_rgb48_bytes(export_out)
                if self._canceled():
                    raise InterruptedError("Export canceled by user.")
                while not self._canceled():
                    if writer_error:
                        raise RuntimeError(str(writer_error[0]))
                    try:
                        frame_queue.put(output_rgb48, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                frames_done += 1

                now_t = time.perf_counter()
                if (
                    frames_done == 1
                    or frames_done == frame_count
                    or (now_t - last_emit_t) >= 0.2
                ):
                    last_emit_t = now_t
                    pct = 0
                    if frame_count > 0:
                        pct = max(0, min(99, int((frames_done * 100) / frame_count)))
                    self.progress.emit(
                        pct,
                        f"Encoding frame {frames_done}"
                        + (f" / {frame_count}" if frame_count > 0 else ""),
                    )

            if self._canceled():
                raise InterruptedError("Export canceled by user.")

            self.progress.emit(99, "Finalizing ProRes export ...")
            writer_stop.set()
            while True:
                if writer_error:
                    raise RuntimeError(str(writer_error[0]))
                try:
                    frame_queue.put(writer_sentinel, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if writer_thread is not None:
                writer_thread.join()
            if writer_error:
                raise RuntimeError(str(writer_error[0]))
            stderr_text = ""
            if ffmpeg_proc.stderr is not None:
                stderr_text = ffmpeg_proc.stderr.read().decode("utf-8", errors="replace")
            return_code = ffmpeg_proc.wait()
            if return_code != 0:
                raise RuntimeError(stderr_text.strip() or "FFmpeg export failed.")

            finished_output_path = self._config.output_path
        except InterruptedError as exc:
            try:
                if os.path.isfile(self._config.output_path):
                    os.remove(self._config.output_path)
            except Exception:
                pass
            self.canceled.emit(str(exc))
        except Exception as exc:
            try:
                if os.path.isfile(self._config.output_path):
                    os.remove(self._config.output_path)
            except Exception:
                pass
            self._fail(str(exc))
        finally:
            writer_stop.set()
            if writer_thread is not None and writer_thread.is_alive():
                try:
                    frame_queue.put_nowait(writer_sentinel)
                except Exception:
                    pass
                writer_thread.join(timeout=2.0)
            self._cleanup_runtime(source, ffmpeg_proc, processor)
            if finished_output_path:
                self.progress.emit(100, "Export complete.")
                self.finished.emit(finished_output_path)
