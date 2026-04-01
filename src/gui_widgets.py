"""Reusable GUI widget classes extracted from gui.py."""

from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt, QObject, QTimer, pyqtSignal
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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


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
        self.setMinimumSize(320, 180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background: #111; color: #555; "
                           "font-size: 14px; }")
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

    def __init__(
        self,
        title: str,
        force_hdr_metadata: bool,
        mpv_available: bool,
        mpv_widget_factory,
        best_mpv_scale: str,
        parent=None,
    ):
        super().__init__(parent)
        self._title = str(title)
        self._force_hdr_metadata = bool(force_hdr_metadata)
        self._last_size: tuple[int, int] | None = None
        self._mpv = None
        self._stack: QStackedWidget | None = None
        self._mpv_widget_factory = mpv_widget_factory
        self._best_mpv_scale = str(best_mpv_scale)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self._lbl_title = QLabel(self._title)
        self._lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_title.setStyleSheet("color: #d8e6ff; font-weight: 600;")
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

    def set_title(self, title: str):
        self._title = str(title or "")
        self._lbl_title.setText(self._title)

    def _ensure_mpv(self, w: int, h: int):
        if self._mpv is None:
            return
        size = (int(w), int(h))
        if self._last_size == size:
            return
        self._last_size = size
        self._mpv.start_playback(
            width=int(w),
            height=int(h),
            fps=1.0,
            scale_kernel=self._best_mpv_scale,
            force_hdr_metadata=self._force_hdr_metadata,
            film_grain=False,
        )

    def set_frame(self, bgr: np.ndarray | None, unavailable_text: str):
        if not isinstance(bgr, np.ndarray):
            if self._mpv is not None:
                try:
                    self._mpv.stop_playback()
                except Exception:
                    pass
                self._last_size = None
            self._cpu.clear_display()
            self._cpu.setText(unavailable_text)
            if self._stack is not None:
                self._stack.setCurrentWidget(self._cpu)
            return

        h, w = bgr.shape[:2]
        if self._mpv is not None and self._stack is not None:
            try:
                self._ensure_mpv(w, h)
                rgb16 = np.ascontiguousarray(bgr[:, :, ::-1].astype(np.uint16) * 257)
                self._mpv.feed_frame(rgb16.data)
                self._stack.setCurrentWidget(self._mpv)
                return
            except Exception:
                # Graceful fallback to CPU preview when mpv path fails.
                pass

        self._cpu.update_frame(np.ascontiguousarray(bgr))
        if self._stack is not None:
            self._stack.setCurrentWidget(self._cpu)

    def stop(self):
        if self._mpv is not None:
            try:
                self._mpv.stop_playback()
            except Exception:
                pass
        self._last_size = None


class CompareFrameDialog(QDialog):
    """Three-way paused-frame compare view: SDR vs HDR GT vs HDR Convert."""

    precision_requested = pyqtSignal(str)

    def __init__(self, mpv_available: bool, mpv_widget_factory, best_mpv_scale: str, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, True)
        self.setSizeGripEnabled(True)
        self.setWindowTitle("Frame Compare")
        self.resize(1500, 760)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._lbl_meta = QLabel("Frame: -")
        self._lbl_meta.setStyleSheet("color: #bbb;")
        root.addWidget(self._lbl_meta)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self._lbl_compare_prec = QLabel("Compare precision:")
        self._lbl_compare_prec.setStyleSheet("color: #bbb;")
        self._cmb_compare_prec = QComboBox()
        self._cmb_compare_prec.setMinimumWidth(190)
        self._btn_recompare = QPushButton("Refresh")
        self._btn_recompare.setMinimumWidth(110)
        controls.addWidget(self._lbl_compare_prec)
        controls.addWidget(self._cmb_compare_prec)
        controls.addWidget(self._btn_recompare)
        controls.addStretch(1)
        root.addLayout(controls)

        self._precision_sync_guard = False
        self._btn_recompare.clicked.connect(self._emit_precision_request)
        self._cmb_compare_prec.currentTextChanged.connect(
            lambda _text: self._emit_precision_request()
        )

        self._disp_sdr = _CompareVideoPane(
            "SDR",
            force_hdr_metadata=False,
            mpv_available=mpv_available,
            mpv_widget_factory=mpv_widget_factory,
            best_mpv_scale=best_mpv_scale,
        )
        self._disp_gt = _CompareVideoPane(
            "HDR GT",
            force_hdr_metadata=True,
            mpv_available=mpv_available,
            mpv_widget_factory=mpv_widget_factory,
            best_mpv_scale=best_mpv_scale,
        )
        self._disp_algo = _CompareVideoPane(
            "HDR Convert",
            force_hdr_metadata=True,
            mpv_available=mpv_available,
            mpv_widget_factory=mpv_widget_factory,
            best_mpv_scale=best_mpv_scale,
        )
        self._split_compare = QSplitter(Qt.Orientation.Horizontal)
        self._split_compare.setChildrenCollapsible(False)
        self._split_compare.addWidget(self._disp_sdr)
        self._split_compare.addWidget(self._disp_gt)
        self._split_compare.addWidget(self._disp_algo)
        self._split_compare.setStretchFactor(0, 1)
        self._split_compare.setStretchFactor(1, 1)
        self._split_compare.setStretchFactor(2, 1)
        root.addWidget(self._split_compare, 1)
        self._reset_splitter_on_show = True

        self._grp_acc = QGroupBox("Accuracy Metrics")
        acc_grid = QGridLayout(self._grp_acc)
        acc_grid.setContentsMargins(10, 6, 10, 8)
        acc_grid.setHorizontalSpacing(14)
        acc_grid.setVerticalSpacing(4)
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
            acc_grid.addWidget(lbl, idx // 3, idx % 3)
            self._acc[key] = lbl
        root.addWidget(self._grp_acc)

        self._lbl_note = QLabel("")
        self._lbl_note.setStyleSheet("color: #9ecbff;")
        root.addWidget(self._lbl_note)

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

    def showEvent(self, event):
        super().showEvent(event)
        if bool(getattr(self, "_reset_splitter_on_show", False)):
            self._reset_splitter_on_show = False
            QTimer.singleShot(0, self._reset_compare_splitter_sizes)

    def _emit_precision_request(self):
        if self._precision_sync_guard:
            return
        key = str(self._cmb_compare_prec.currentText() or "").strip()
        if key:
            self.precision_requested.emit(key)

    def set_precision_options(self, options: list[str], selected: str | None = None):
        keys = [str(v).strip() for v in (options or []) if str(v).strip()]
        current = str(selected or "").strip()
        if not keys:
            self._precision_sync_guard = True
            self._cmb_compare_prec.clear()
            self._precision_sync_guard = False
            self._cmb_compare_prec.setEnabled(False)
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
        self._btn_recompare.setEnabled(True)

    def set_recompare_busy(self, busy: bool):
        is_busy = bool(busy)
        self._cmb_compare_prec.setEnabled(not is_busy and self._cmb_compare_prec.count() > 0)
        self._btn_recompare.setEnabled(not is_busy and self._cmb_compare_prec.count() > 0)
    def set_frames(self, frame_idx: int, sdr: np.ndarray | None,
                   hdr_gt: np.ndarray | None, hdr_algo: np.ndarray | None,
                   note: str = "", metrics: dict | None = None,
                   hdr_algo_label: str | None = None):
        self.setWindowTitle(f"Frame Compare - frame {int(frame_idx)}")
        self._lbl_meta.setText(f"Frame: {int(frame_idx)}")

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
        self._acc["sssim"].setText(f"SSSIM: {_fmt_metric(metrics.get('sssim'), '.4f')}")
        self._acc["deitp"].setText(f"DeltaEITP: {_fmt_metric(metrics.get('delta_e_itp'), '.2f')}")
        self._acc["psnr_norm"].setText(
            f"PSNR-N: {_fmt_metric(metrics.get('psnr_norm_db'), '.2f', ' dB')}"
        )
        self._acc["sssim_norm"].setText(
            f"SSSIM-N: {_fmt_metric(metrics.get('sssim_norm'), '.4f')}"
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
        self._disp_sdr.stop()
        self._disp_gt.stop()
        self._disp_algo.stop()
        super().closeEvent(event)


class DetachedVideoWindow(QWidget):
    """Floating host window for a video pane (SDR/HDR)."""

    closed = pyqtSignal(str)

    def __init__(self, key: str, title: str, parent=None):
        super().__init__(parent)
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

