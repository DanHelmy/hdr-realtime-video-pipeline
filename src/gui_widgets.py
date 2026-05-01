"""Reusable GUI widget classes extracted from gui.py."""

from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt, QObject, QTimer, QEvent, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui_hdr_io import frame_to_rgb48_bytes


class _KernelCacheClearWorker(QObject):
    finished = pyqtSignal(bool)

    def __init__(self, dirs):
        super().__init__()
        self._dirs = dirs

    def run(self):
        import shutil
        ok = True
        for d in self._dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                ok = False
        self.finished.emit(ok)


class VideoDisplay(QLabel):
    """QLabel that efficiently renders a BGR numpy frame."""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self._title = title
        self.setObjectName("VideoDisplay")
        self.setProperty("videoSurface", True)
        self.setMinimumSize(240, 135)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setFont(QFont("Segoe UI", 10))
        self.setText(title)

    def update_frame(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        # Use BGR888 directly to avoid per-frame color conversion cost.
        qimg = QImage(
            bgr.data, w, h, 3 * w, QImage.Format.Format_BGR888
        ).copy()
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.setPixmap(scaled)

    def clear_display(self):
        self.clear()
        self.setText(self._title)


class _CompareVideoPane(QWidget):
    """Single compare pane with mpv color-managed path + CPU fallback."""

    expand_requested = pyqtSignal()

    def __init__(
        self,
        title: str,
        force_hdr_metadata: bool,
        mpv_available: bool,
        mpv_widget_factory,
        best_mpv_scale: str,
        preview_scale_kernel: str | None = None,
        preview_fps: float = 1.0,
        preview_scale_antiring: float | None = None,
        preview_cas_strength: float | None = None,
        preview_film_grain: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("CompareVideoPane")
        self._title = str(title)
        self._force_hdr_metadata = bool(force_hdr_metadata)
        self._last_size: tuple[int, int] | None = None
        self._last_frame: np.ndarray | None = None
        self._last_unavailable_text = self._title
        self._last_mpv_error: str | None = None
        self._mpv = None
        self._stack: QStackedWidget | None = None
        self._mpv_widget_factory = mpv_widget_factory
        self._best_mpv_scale = str(best_mpv_scale)
        self._preview_scale_kernel = str(preview_scale_kernel or best_mpv_scale)
        self._preview_fps = float(preview_fps) if preview_fps and preview_fps > 0 else 1.0
        self._preview_scale_antiring = (
            None if preview_scale_antiring is None else float(preview_scale_antiring)
        )
        self._preview_cas_strength = (
            None if preview_cas_strength is None else float(preview_cas_strength)
        )
        self._preview_film_grain = bool(preview_film_grain)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self._lbl_title = QLabel(self._title)
        self._lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_title.setProperty("eyebrow", True)
        root.addWidget(self._lbl_title)

        self._cpu = VideoDisplay(self._title)
        if mpv_available and self._mpv_widget_factory is not None:
            self._mpv = self._mpv_widget_factory()
            self._stack = QStackedWidget()
            self._stack.addWidget(self._mpv)
            self._stack.addWidget(self._cpu)
            self._stack.setCurrentWidget(self._cpu)
            root.addWidget(self._stack, 1)
        else:
            root.addWidget(self._cpu, 1)

        for widget in (self, self._lbl_title, self._cpu, self._stack, self._mpv):
            if widget is not None:
                widget.installEventFilter(self)

    def set_title(self, title: str):
        self._title = str(title or "")
        self._lbl_title.setText(self._title)

    def _ensure_mpv(self, w: int, h: int):
        if self._mpv is None:
            return
        size = (int(w), int(h))
        if self._last_size == size:
            return
        started = self._mpv.start_playback(
            width=int(w),
            height=int(h),
            fps=float(self._preview_fps),
            scale_kernel=self._preview_scale_kernel,
            scale_antiring=self._preview_scale_antiring,
            cas_strength=self._preview_cas_strength,
            force_hdr_metadata=self._force_hdr_metadata,
            film_grain=self._preview_film_grain,
        )
        if not started:
            raise RuntimeError(
                getattr(self._mpv, "_last_scale_error", None)
                or "mpv preview startup failed."
            )
        self._last_size = size

    def set_frame(self, bgr: np.ndarray | None, unavailable_text: str):
        self._last_unavailable_text = str(unavailable_text or self._title)
        self._last_frame = (
            np.ascontiguousarray(bgr) if isinstance(bgr, np.ndarray) else None
        )
        self._last_mpv_error = None
        if not isinstance(bgr, np.ndarray):
            if self._mpv is not None:
                try:
                    self._mpv.stop_playback()
                except Exception:
                    pass
                self._last_size = None
            self._cpu.clear_display()
            self._cpu.setText(self._last_unavailable_text)
            if self._stack is not None:
                self._stack.setCurrentWidget(self._cpu)
            return

        frame = self._last_frame
        h, w = frame.shape[:2]
        if self._mpv is not None and self._stack is not None:
            try:
                self._ensure_mpv(w, h)
                self._mpv.feed_frame(frame_to_rgb48_bytes(frame))
                self._stack.setCurrentWidget(self._mpv)
                return
            except Exception as exc:
                # Graceful fallback to CPU preview when mpv path fails.
                self._last_mpv_error = str(exc)
                pass

        cpu_frame = np.ascontiguousarray(frame)
        if cpu_frame.dtype == np.uint16:
            cpu_frame = ((cpu_frame.astype(np.float32) / 65535.0) * 255.0).astype(np.uint8)
        self._cpu.update_frame(cpu_frame)
        if self._stack is not None:
            self._stack.setCurrentWidget(self._cpu)

    def refresh_surface(self) -> bool:
        frame = self._last_frame
        if self._mpv is None or self._stack is None or not isinstance(frame, np.ndarray):
            return False
        try:
            h, w = frame.shape[:2]
            # Recreate the rawvideo surface on the current monitor so HDR
            # metadata is re-evaluated after show/move/screen changes.
            self._last_size = None
            self._ensure_mpv(w, h)
            self._mpv.feed_frame(frame_to_rgb48_bytes(frame))
            self._stack.setCurrentWidget(self._mpv)
            self._last_mpv_error = None
            return True
        except Exception as exc:
            self._last_mpv_error = str(exc)
            cpu_frame = np.ascontiguousarray(frame)
            if cpu_frame.dtype == np.uint16:
                cpu_frame = (
                    (cpu_frame.astype(np.float32) / 65535.0) * 255.0
                ).astype(np.uint8)
            self._cpu.update_frame(cpu_frame)
            self._stack.setCurrentWidget(self._cpu)
            return False

    def stop(self):
        if self._mpv is not None:
            try:
                self._mpv.stop_playback()
            except Exception:
                pass
        self._last_size = None

    def eventFilter(self, watched, event):
        if event is not None and event.type() == QEvent.Type.MouseButtonDblClick:
            self.expand_requested.emit()
            return True
        return super().eventFilter(watched, event)


class CompareFrameDialog(QDialog):
    """Three-way paused-frame compare view: SDR vs HDR GT vs HDR Convert."""

    compare_requested = pyqtSignal(str, int)

    def __init__(self, mpv_available: bool, mpv_widget_factory, best_mpv_scale: str, parent=None):
        # Keep Compare as an independent top-level dialog on Windows. When a
        # parent-owned top-level window is minimized, Qt/Windows can collapse it
        # into a tiny corner stub instead of treating it like a normal window.
        super().__init__(None)
        self.setObjectName("CompareFrameDialog")
        self._owner_widget = parent
        self.setWindowFlag(Qt.WindowType.Dialog, True)
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, True)
        self.setSizeGripEnabled(True)
        self.setWindowTitle("Frame Compare")
        self.resize(1500, 760)
        self._screen_hook_handle = None
        self._compare_surface_refresh_queued = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._lbl_meta = QLabel("Frame: -")
        self._lbl_meta.setProperty("pill", True)
        root.addWidget(self._lbl_meta)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self._lbl_compare_prec = QLabel("Compare precision:")
        self._lbl_compare_prec.setProperty("muted", True)
        self._cmb_compare_prec = QComboBox()
        self._cmb_compare_prec.setMinimumWidth(190)
        self._lbl_compare_frame = QLabel("Frame:")
        self._lbl_compare_frame.setProperty("muted", True)
        self._spn_compare_frame = QSpinBox()
        self._spn_compare_frame.setMinimumWidth(110)
        self._spn_compare_frame.setRange(0, 0)
        self._spn_compare_frame.setAccelerated(True)
        self._spn_compare_frame.setKeyboardTracking(False)
        self._btn_recompare = QPushButton("Refresh")
        self._btn_recompare.setMinimumWidth(110)
        self._btn_recompare.setProperty("role", "primary")
        controls.addWidget(self._lbl_compare_prec)
        controls.addWidget(self._cmb_compare_prec)
        controls.addWidget(self._lbl_compare_frame)
        controls.addWidget(self._spn_compare_frame)
        controls.addWidget(self._btn_recompare)
        controls.addStretch(1)
        root.addLayout(controls)

        self._precision_sync_guard = False
        self._btn_recompare.clicked.connect(self._emit_compare_request)
        self._cmb_compare_prec.currentTextChanged.connect(
            lambda _text: self._emit_compare_request()
        )
        self._spn_compare_frame.editingFinished.connect(self._emit_compare_request)

        self._disp_sdr = _CompareVideoPane(
            "SDR",
            force_hdr_metadata=False,
            mpv_available=mpv_available,
            mpv_widget_factory=mpv_widget_factory,
            best_mpv_scale=best_mpv_scale,
            preview_scale_kernel=best_mpv_scale,
        )
        self._disp_gt = _CompareVideoPane(
            "HDR GT",
            force_hdr_metadata=True,
            mpv_available=mpv_available,
            mpv_widget_factory=mpv_widget_factory,
            best_mpv_scale=best_mpv_scale,
            preview_scale_kernel=best_mpv_scale,
        )
        self._disp_algo = _CompareVideoPane(
            "HDR Convert",
            force_hdr_metadata=True,
            mpv_available=mpv_available,
            mpv_widget_factory=mpv_widget_factory,
            best_mpv_scale=best_mpv_scale,
            preview_scale_kernel=best_mpv_scale,
        )
        self._split_compare = QSplitter(Qt.Orientation.Horizontal)
        self._split_compare.setChildrenCollapsible(False)
        self._split_compare.setHandleWidth(12)
        self._split_compare.addWidget(self._disp_sdr)
        self._split_compare.addWidget(self._disp_gt)
        self._split_compare.addWidget(self._disp_algo)
        self._split_compare.setStretchFactor(0, 1)
        self._split_compare.setStretchFactor(1, 1)
        self._split_compare.setStretchFactor(2, 1)
        root.addWidget(self._split_compare, 1)
        self._reset_splitter_on_show = True

        self._grp_acc = QGroupBox("Accuracy Metrics")
        self._grp_acc.setObjectName("MetricsCard")
        acc_grid = QGridLayout(self._grp_acc)
        acc_grid.setContentsMargins(10, 6, 10, 8)
        acc_grid.setHorizontalSpacing(14)
        acc_grid.setVerticalSpacing(8)
        mono = QFont("Consolas", 9)
        self._acc = {}
        acc_keys = (
            "psnr", "sssim", "deitp",
            "psnr_norm", "sssim_norm", "deitp_norm",
            "hdrvdp3", "obj",
        )
        for idx, key in enumerate(acc_keys):
            lbl = QLabel(f"{key}: -")
            lbl.setFont(mono)
            lbl.setMinimumWidth(170)
            lbl.setProperty("metricChip", True)
            acc_grid.addWidget(lbl, idx // 3, idx % 3)
            self._acc[key] = lbl
        root.addWidget(self._grp_acc)

        self._lbl_note = QLabel("")
        self._lbl_note.setProperty("accentText", True)
        root.addWidget(self._lbl_note)

    def _restore_parent_video_cursor(self):
        parent = getattr(self, "_owner_widget", None)
        targets = [parent]
        if parent is not None:
            for name in (
                "_disp_hdr_mpv",
                "_disp_sdr_mpv",
                "_disp_hdr_cpu",
                "_disp_sdr_cpu",
                "_disp_hdr_stack",
                "_disp_sdr_stack",
            ):
                widget = getattr(parent, name, None)
                if widget is not None:
                    targets.append(widget)
        for widget in targets:
            try:
                widget.setCursor(Qt.CursorShape.ArrowCursor)
            except Exception:
                pass

    def hideEvent(self, event):
        try:
            self.clearFocus()
            self._cmb_compare_prec.clearFocus()
            self._spn_compare_frame.clearFocus()
            self._btn_recompare.clearFocus()
        except Exception:
            pass
        super().hideEvent(event)
        QTimer.singleShot(0, self._restore_parent_video_cursor)

    def _reset_compare_splitter_sizes(self):
        if not hasattr(self, "_split_compare") or self._split_compare is None:
            return
        if self._split_compare.count() < 3:
            return
        total = int(self._split_compare.size().width())
        if total <= 0:
            total = max(3, int(self.size().width()) - 20)
        one = max(1, total // 3)
        self._split_compare.setSizes([one, one, max(1, total - (2 * one))])

    def _compare_panes(self):
        return (self._disp_sdr, self._disp_gt, self._disp_algo)

    def _attach_screen_change_hook(self):
        try:
            handle = self.windowHandle()
        except Exception:
            handle = None
        if handle is None or handle is self._screen_hook_handle:
            return
        if self._screen_hook_handle is not None:
            try:
                self._screen_hook_handle.screenChanged.disconnect(
                    self._on_screen_changed
                )
            except Exception:
                pass
        try:
            handle.screenChanged.connect(self._on_screen_changed)
        except Exception:
            return
        self._screen_hook_handle = handle

    def _schedule_compare_surface_refresh(self, delay_ms: int = 0):
        if self._compare_surface_refresh_queued:
            return
        self._compare_surface_refresh_queued = True

        def _refresh():
            self._compare_surface_refresh_queued = False
            if not self.isVisible():
                return
            self._attach_screen_change_hook()
            for pane in self._compare_panes():
                pane.refresh_surface()

        QTimer.singleShot(max(0, int(delay_ms)), _refresh)

    def _on_screen_changed(self, _screen=None):
        self._schedule_compare_surface_refresh(40)

    def showEvent(self, event):
        super().showEvent(event)
        self._attach_screen_change_hook()
        self._schedule_compare_surface_refresh(0)
        if bool(getattr(self, "_reset_splitter_on_show", False)):
            self._reset_splitter_on_show = False
            QTimer.singleShot(0, self._reset_compare_splitter_sizes)

    def _emit_compare_request(self):
        if self._precision_sync_guard:
            return
        key = str(self._cmb_compare_prec.currentText() or "").strip()
        if key:
            self.compare_requested.emit(key, int(self._spn_compare_frame.value()))

    def set_frame_bounds(
        self,
        min_frame: int,
        max_frame: int,
        current_frame: int | None = None,
    ):
        low = max(0, int(min_frame))
        high = max(low, int(max_frame))
        current = self._spn_compare_frame.value() if current_frame is None else int(current_frame)
        current = max(low, min(high, current))
        self._precision_sync_guard = True
        self._spn_compare_frame.setRange(low, high)
        self._spn_compare_frame.setValue(current)
        self._precision_sync_guard = False

    def set_precision_options(self, options: list[str], selected: str | None = None):
        keys = [str(v).strip() for v in (options or []) if str(v).strip()]
        current = str(selected or "").strip()
        if not keys:
            self._precision_sync_guard = True
            self._cmb_compare_prec.clear()
            self._precision_sync_guard = False
            self._cmb_compare_prec.setEnabled(False)
            self._spn_compare_frame.setEnabled(False)
            self._btn_recompare.setEnabled(False)
            return
        if current not in keys:
            old = str(self._cmb_compare_prec.currentText() or "").strip()
            current = old if old in keys else keys[0]
        self._precision_sync_guard = True
        self._cmb_compare_prec.clear()
        self._cmb_compare_prec.addItems(keys)
        self._cmb_compare_prec.setCurrentText(current)
        self._precision_sync_guard = False
        self._cmb_compare_prec.setEnabled(True)
        self._spn_compare_frame.setEnabled(True)
        self._btn_recompare.setEnabled(True)

    def set_recompare_busy(self, busy: bool):
        is_busy = bool(busy)
        self._cmb_compare_prec.setEnabled(not is_busy and self._cmb_compare_prec.count() > 0)
        self._spn_compare_frame.setEnabled(not is_busy and self._cmb_compare_prec.count() > 0)
        self._btn_recompare.setEnabled(not is_busy and self._cmb_compare_prec.count() > 0)

    def set_frames(self, frame_idx: int, sdr: np.ndarray | None,
                   hdr_gt: np.ndarray | None, hdr_algo: np.ndarray | None,
                   note: str = "", metrics: dict | None = None,
                   hdr_algo_label: str | None = None):
        self.setWindowTitle(f"Frame Compare - frame {int(frame_idx)}")
        self._lbl_meta.setText(f"Frame: {int(frame_idx)}")
        self.set_frame_bounds(
            self._spn_compare_frame.minimum(),
            self._spn_compare_frame.maximum(),
            current_frame=int(frame_idx),
        )

        algo_title = str(hdr_algo_label or "HDR Convert")
        self._disp_algo.set_title(algo_title)

        self._disp_sdr.set_frame(sdr, "SDR\n(unavailable)")
        self._disp_gt.set_frame(hdr_gt, "HDR GT\n(unavailable)")
        self._disp_algo.set_frame(hdr_algo, f"{algo_title}\n(unavailable)")

        metrics = metrics or {}

        def _fmt_metric(v, fmt: str, suffix: str = "") -> str:
            try:
                fv = float(v)
            except Exception:
                return "-"
            if not np.isfinite(fv):
                return "-"
            return f"{format(fv, fmt)}{suffix}"

        self._acc["psnr"].setText(
            f"PSNR: {_fmt_metric(metrics.get('psnr_db'), '.2f', ' dB')}"
        )
        self._acc["sssim"].setText(f"SSIM: {_fmt_metric(metrics.get('sssim'), '.4f')}")
        self._acc["deitp"].setText(f"DeltaEITP: {_fmt_metric(metrics.get('delta_e_itp'), '.2f')}")
        self._acc["psnr_norm"].setText(
            f"PSNR-N: {_fmt_metric(metrics.get('psnr_norm_db'), '.2f', ' dB')}"
        )
        self._acc["sssim_norm"].setText(
            f"SSIM-N: {_fmt_metric(metrics.get('sssim_norm'), '.4f')}"
        )
        self._acc["deitp_norm"].setText(
            f"DeltaEITP-N: {_fmt_metric(metrics.get('delta_e_itp_norm'), '.2f')}"
        )

        hdrvdp3_txt = _fmt_metric(metrics.get("hdr_vdp3"), ".3f")
        if hdrvdp3_txt != "-":
            self._acc["hdrvdp3"].setText(f"HDR-VDP3: {hdrvdp3_txt}")
            self._acc["hdrvdp3"].setToolTip("")
        else:
            vdp_note = str(metrics.get("hdr_vdp3_note", "") or "").strip()

            # Keep a short, visible reason in the row itself so users do not
            # need to hover the tooltip to know why HDR-VDP3 is unavailable.
            vdp_note_short = ""
            if vdp_note:
                low = vdp_note.lower()
                if "octave not found" in low:
                    vdp_note_short = "Octave not found in PATH"
                elif "timed out" in low or "timeout" in low:
                    vdp_note_short = "execution timed out"
                elif "no parsable score" in low:
                    vdp_note_short = "no parsable score"
                elif "command error" in low:
                    vdp_note_short = "bridge command failed"
                elif "unavailable" in low:
                    vdp_note_short = "bridge unavailable"
                else:
                    vdp_note_short = vdp_note
            if len(vdp_note_short) > 44:
                vdp_note_short = f"{vdp_note_short[:41]}..."

            if vdp_note_short:
                self._acc["hdrvdp3"].setText(f"HDR-VDP3: - ({vdp_note_short})")
            else:
                self._acc["hdrvdp3"].setText("HDR-VDP3: -")
            self._acc["hdrvdp3"].setToolTip(vdp_note)

        obj_note = str(metrics.get("obj_note", "") or "").strip()
        if obj_note:
            self._acc["obj"].setText(f"Obj: {obj_note}")
        elif hdr_gt is None:
            self._acc["obj"].setText("Obj: need HDR GT")
        else:
            self._acc["obj"].setText("Obj: -")

        self._lbl_note.setText(str(note or ""))

    def closeEvent(self, event):
        self._reset_splitter_on_show = True
        self._compare_surface_refresh_queued = False
        if self._screen_hook_handle is not None:
            try:
                self._screen_hook_handle.screenChanged.disconnect(
                    self._on_screen_changed
                )
            except Exception:
                pass
            self._screen_hook_handle = None
        self._disp_sdr.stop()
        self._disp_gt.stop()
        self._disp_algo.stop()
        super().closeEvent(event)


class DetachedVideoWindow(QWidget):
    """Floating host window for a video pane (SDR/HDR)."""

    closed = pyqtSignal(str)

    def __init__(self, key: str, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("DetachedVideoWindow")
        self._key = str(key)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, True)
        self.setWindowTitle(title)
        self.resize(960, 540)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

    def set_content(self, widget: QWidget):
        self._layout.addWidget(widget, 1)

    def closeEvent(self, event):
        self.closed.emit(self._key)
        super().closeEvent(event)

