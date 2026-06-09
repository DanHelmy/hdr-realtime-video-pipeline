from __future__ import annotations

import os
import time

import psutil

from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRect,
    QTimer,
    Qt,
)
from PyQt6.QtWidgets import QApplication, QGraphicsOpacityEffect, QVBoxLayout, QWidget

from gui_config import (
    SOURCE_MODE_WINDOW,
    _normalize_source_mode,
    _source_is_below_processing_preset,
)
from gui_scaling import (
    DEFAULT_UPSCALER,
    _is_upscale_required,
    _select_hdr_scale_antiring,
    _select_hdr_scale_kernel,
    _select_mpv_cas_strength,
)
from gui_widgets import DetachedVideoWindow


class VideoTransitionOverlay(QWidget):
    """Top-level black overlay that fades away above native child windows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VideoTransitionOverlay")
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background-color: #000000;")
        self.setWindowOpacity(0.0)
        self.hide()

    def getFadeOpacity(self) -> float:
        return float(self.windowOpacity())

    def setFadeOpacity(self, value: float):
        self.setWindowOpacity(max(0.0, min(1.0, float(value))))


class WindowingMixin:
    """Window/layout/fullscreen/popup behavior helpers for MainWindow."""

    def _current_screen_geometry(self) -> QRect:
        """Get full monitor geometry (not availableGeometry) to cover taskbar."""
        screen = None
        try:
            screen = QApplication.screenAt(self.frameGeometry().center())
        except Exception:
            screen = None
        win_handle = self.windowHandle()
        if screen is None:
            screen = win_handle.screen() if win_handle is not None else None
        if screen is None:
            screen = self.screen()
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is None:
            return QRect(0, 0, 1600, 900)
        return screen.geometry()

    def _safe_borderless_geometry(self) -> QRect:
        """Near-fullscreen rect to avoid exclusive/fullscreen optimizations."""
        g = self._current_screen_geometry()
        # Keep a 1px inset so Windows does not treat it like true fullscreen.
        if g.width() > 4 and g.height() > 4:
            return g.adjusted(1, 1, -1, -1)
        return g

    def _set_view_mode_silently(self, mode: str):
        prev = self._cmb_view.blockSignals(True)
        self._cmb_view.setCurrentText(mode)
        self._cmb_view.blockSignals(prev)
        self._on_view(mode)

    def _set_process_priority(self, high: bool):
        try:
            p = psutil.Process(os.getpid())
            if high:
                if self._proc_priority_saved is None:
                    self._proc_priority_saved = p.nice()
                p.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                if self._proc_priority_saved is not None:
                    p.nice(self._proc_priority_saved)
                self._proc_priority_saved = None
        except Exception:
            pass

    @staticmethod
    def _rehost_widget(widget: QWidget, host_layout: QVBoxLayout):
        old_parent = widget.parentWidget()
        if old_parent is not None and old_parent.layout() is not None:
            old_parent.layout().removeWidget(widget)
        widget.setParent(None)
        host_layout.addWidget(widget)

    def _current_video_tab_label(self) -> str:
        if self._video_tabs is None:
            return ""
        idx = self._video_tabs.currentIndex()
        if idx < 0:
            return ""
        return str(self._video_tabs.tabText(idx) or "")

    def _is_sdr_output_visible(self) -> bool:
        if self._sdr_float_window is not None and self._sdr_float_window.isVisible():
            return True
        if self._video_tabs is None:
            return True
        label = self._current_video_tab_label()
        if not label:
            return True
        return label in {"SDR", "Side by Side"}

    def _sync_worker_sdr_visibility(self) -> bool:
        visible = self._is_sdr_output_visible()
        worker_visible = visible
        if (
            self._playing
            and getattr(self, "_active_use_mpv", False)
            and self._disp_sdr_mpv is not None
        ):
            # Keep SDR mpv fed even on HDR-only view. Switching back to SDR or
            # side-by-side is then a warm pane switch instead of a cold feeder
            # restart with stale frames.
            worker_visible = True
        try:
            self._worker.set_sdr_visible(worker_visible)
        except Exception:
            pass
        return visible

    def _video_transition_target(self) -> QWidget | None:
        return getattr(self, "_video_transition_target_widget", None) or self

    def _position_video_transition_overlay(self):
        overlay = getattr(self, "_video_transition_overlay", None)
        target = self._video_transition_target()
        if overlay is None or target is None:
            return
        if not target.isVisible():
            overlay.hide()
            return
        forced_rect = getattr(self, "_video_transition_forced_rect", None)
        if isinstance(forced_rect, QRect) and forced_rect.isValid():
            rect = forced_rect
        else:
            if target is self:
                rect = self.frameGeometry()
            else:
                try:
                    pos = target.mapToGlobal(QPoint(0, 0))
                    rect = QRect(pos, target.size())
                except Exception:
                    rect = self.frameGeometry()
        if not rect.isValid() or rect.width() <= 1 or rect.height() <= 1:
            rect = self.frameGeometry()
        overlay.setGeometry(rect)
        overlay.raise_()

    def _ensure_video_transition_track_timer(self):
        if getattr(self, "_video_transition_track_timer", None) is not None:
            return
        timer = QTimer(self)
        timer.setSingleShot(False)
        timer.timeout.connect(self._position_video_transition_overlay)
        self._video_transition_track_timer = timer

    def _track_video_transition_overlay_for(self, duration_ms: int):
        self._ensure_video_transition_track_timer()
        timer = getattr(self, "_video_transition_track_timer", None)
        if timer is None:
            return
        timer.stop()
        timer.start(16)
        QTimer.singleShot(
            max(80, int(duration_ms) + 80),
            lambda: timer.stop() if timer is self._video_transition_track_timer else None,
        )

    def _begin_video_surface_fade(
        self,
        duration_ms: int = 1000,
        start_opacity: float = 1.0,
        cover_rect: QRect | None = None,
        target_widget: QWidget | None = None,
    ):
        previous_target = getattr(self, "_video_transition_target_widget", None)
        self._video_transition_target_widget = target_widget
        target = self._video_transition_target()
        if target is None or not target.isVisible():
            self._video_transition_target_widget = previous_target
            return
        overlay = getattr(self, "_video_transition_overlay", None)
        if overlay is None:
            # Keep this parentless: it is a native top-level cover positioned in
            # global desktop coordinates. Parenting it to the main window can
            # leave a stale child-sized square during fullscreen/layout moves.
            overlay = VideoTransitionOverlay(None)
            self._video_transition_overlay = overlay
        anim = getattr(self, "_video_transition_anim", None)
        if anim is not None:
            anim.stop()
        self._video_transition_forced_rect = cover_rect
        self._position_video_transition_overlay()
        overlay.show()
        overlay.raise_()
        overlay.setFadeOpacity(max(0.0, min(1.0, float(start_opacity))))
        overlay.repaint()
        QApplication.processEvents()
        QTimer.singleShot(0, self._position_video_transition_overlay)
        QTimer.singleShot(40, self._position_video_transition_overlay)
        QTimer.singleShot(120, self._position_video_transition_overlay)
        self._track_video_transition_overlay_for(duration_ms)
        anim = QPropertyAnimation(overlay, b"windowOpacity", self)
        anim.setDuration(max(90, int(duration_ms)))
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        anim.setStartValue(overlay.getFadeOpacity())
        anim.setEndValue(0.0)

        def _finish():
            overlay.hide()
            overlay.setFadeOpacity(0.0)
            self._video_transition_forced_rect = None
            self._video_transition_target_widget = None
            track_timer = getattr(self, "_video_transition_track_timer", None)
            if track_timer is not None:
                track_timer.stop()

        anim.finished.connect(_finish)
        self._video_transition_anim = anim
        anim.start()

    def _begin_video_surface_fade_after_layout(
        self,
        duration_ms: int = 1000,
        start_opacity: float = 1.0,
        delay_ms: int = 16,
        cover_rect: QRect | None = None,
        target_widget: QWidget | None = None,
    ):
        def _start():
            try:
                if self._root_layout is not None:
                    self._root_layout.activate()
                if self._video_tabs is not None:
                    self._video_tabs.updateGeometry()
            except Exception:
                pass
            self._begin_video_surface_fade(
                duration_ms=duration_ms,
                start_opacity=start_opacity,
                cover_rect=cover_rect,
                target_widget=target_widget,
            )

        QTimer.singleShot(max(0, int(delay_ms)), _start)

    def _begin_video_pane_fade(
        self,
        duration_ms: int = 1000,
        start_opacity: float = 1.0,
    ):
        self._begin_video_surface_fade(
            duration_ms=duration_ms,
            start_opacity=start_opacity,
            target_widget=self._video_tabs,
        )

    def _schedule_sdr_reactivation_relock(self):
        """Recover metrics after showing SDR; the pane is kept warm already."""
        if not self._playing:
            return

        try:
            self._worker.reset_runtime_metrics()
        except Exception:
            pass
        self._schedule_window_state_refresh(40, soft_only=True)

    def _on_video_tab_changed(self, index: int):
        if self._video_tabs is None or index < 0:
            return
        new_label = str(self._video_tabs.tabText(index) or "")
        old_label = str(getattr(self, "_active_video_tab_label", "") or "")
        reactivate_sdr = old_label == "HDR" and new_label in {"SDR", "Side by Side"}
        self._begin_video_pane_fade(duration_ms=1000, start_opacity=1.0)

        def _apply_tab_switch():
            label = new_label
            if label == "Side by Side":
                # Side-by-side needs both panes docked in the split hosts.
                if self._sdr_float_window is not None:
                    self._dock_video_pane("sdr")
                if self._hdr_float_window is not None:
                    self._dock_video_pane("hdr")
                if (
                    self._side_sdr_host is not None
                    and self._disp_sdr.parentWidget() is not self._side_sdr_host
                ):
                    self._rehost_widget(self._disp_sdr, self._side_sdr_host.layout())
                if (
                    self._side_hdr_host is not None
                    and self._disp_hdr.parentWidget() is not self._side_hdr_host
                ):
                    self._rehost_widget(self._disp_hdr, self._side_hdr_host.layout())
            else:
                # Single tabs own their respective panes when not popped out.
                if self._sdr_float_window is None and self._sdr_tab_host is not None:
                    if self._disp_sdr.parentWidget() is not self._sdr_tab_host:
                        self._rehost_widget(self._disp_sdr, self._sdr_tab_host.layout())
                if self._hdr_float_window is None and self._hdr_tab_host is not None:
                    if self._disp_hdr.parentWidget() is not self._hdr_tab_host:
                        self._rehost_widget(self._disp_hdr, self._hdr_tab_host.layout())
            self._save_user_settings()

        self._with_layout_freeze(_apply_tab_switch, refresh_delay=40)
        self._active_video_tab_label = new_label
        if self._playing:
            self._sync_worker_sdr_visibility()
            if new_label == "Side by Side":
                self._stop_periodic_relock()
            else:
                self._start_periodic_relock()
        if self._playing and reactivate_sdr:
            self._schedule_sdr_reactivation_relock()
        if self._playing:
            self._schedule_state_change_relock(delay_ms=140, drop_frames=3)

    def _on_app_state_changed(self, state: Qt.ApplicationState):
        active = state == Qt.ApplicationState.ApplicationActive
        if self._app_active == active:
            return
        self._app_active = active
        if self._playing:
            self._set_process_priority(True)
            try:
                self._worker.reset_runtime_metrics()
            except Exception:
                pass
            if active:
                self._pause_for_ui_transition()
                self._schedule_state_change_relock(delay_ms=140, drop_frames=2)

    def _toggle_sdr_popout(self):
        if self._sdr_float_window is not None:
            self._dock_video_pane("sdr")
            return
        self._begin_video_pane_fade(duration_ms=1000, start_opacity=1.0)

        def _apply_pop():
            win = DetachedVideoWindow("sdr", "SDR View")
            win.closed.connect(self._on_video_window_closed)
            self._rehost_widget(self._disp_sdr, win.layout())
            win.move(self.frameGeometry().topLeft() + QPoint(40, 40))
            win.show()
            win.raise_()
            win.activateWindow()
            self._sdr_float_window = win
            self._btn_pop_sdr.setText("Dock SDR")

        self._with_layout_freeze(_apply_pop, refresh_delay=40)
        if self._playing:
            self._sync_worker_sdr_visibility()
            self._schedule_state_change_relock(delay_ms=160, drop_frames=3)

    def _toggle_hdr_popout(self):
        if self._hdr_float_window is not None:
            self._dock_video_pane("hdr")
            return
        self._begin_video_pane_fade(duration_ms=1000, start_opacity=1.0)

        def _apply_pop():
            win = DetachedVideoWindow("hdr", "HDR View")
            win.closed.connect(self._on_video_window_closed)
            self._rehost_widget(self._disp_hdr, win.layout())
            win.move(self.frameGeometry().topLeft() + QPoint(80, 80))
            win.show()
            win.raise_()
            win.activateWindow()
            self._hdr_float_window = win
            self._btn_pop_hdr.setText("Dock HDR")

        self._with_layout_freeze(_apply_pop, refresh_delay=40)
        if self._playing:
            self._schedule_state_change_relock(
                delay_ms=160,
                drop_frames=3,
                settle_delay_ms=520,
                settle_drop_frames=1,
            )

    def _dock_video_pane(self, key: str, from_signal: bool = False):
        self._begin_video_pane_fade(duration_ms=1000, start_opacity=1.0)

        def _apply_dock():
            side_mode = (
                self._video_tabs is not None
                and self._video_tabs.currentIndex() >= 0
                and self._video_tabs.tabText(self._video_tabs.currentIndex())
                == "Side by Side"
            )
            if key == "sdr":
                win = self._sdr_float_window
                if win is None:
                    return
                if side_mode and self._side_sdr_host is not None:
                    self._rehost_widget(self._disp_sdr, self._side_sdr_host.layout())
                elif self._sdr_tab_host is not None:
                    self._rehost_widget(self._disp_sdr, self._sdr_tab_host.layout())
                if not from_signal:
                    try:
                        win.closed.disconnect(self._on_video_window_closed)
                    except Exception:
                        pass
                    win.close()
                self._sdr_float_window = None
                self._btn_pop_sdr.setText("Pop SDR")
            elif key == "hdr":
                win = self._hdr_float_window
                if win is None:
                    return
                if side_mode and self._side_hdr_host is not None:
                    self._rehost_widget(self._disp_hdr, self._side_hdr_host.layout())
                elif self._hdr_tab_host is not None:
                    self._rehost_widget(self._disp_hdr, self._hdr_tab_host.layout())
                if not from_signal:
                    try:
                        win.closed.disconnect(self._on_video_window_closed)
                    except Exception:
                        pass
                    win.close()
                self._hdr_float_window = None
                self._btn_pop_hdr.setText("Pop HDR")

        self._with_layout_freeze(_apply_dock, refresh_delay=40)
        if self._playing and str(key).lower() == "sdr":
            self._sync_worker_sdr_visibility()
        if self._playing:
            settle_ms = 520 if str(key).lower() == "hdr" else None
            self._schedule_state_change_relock(
                delay_ms=160,
                drop_frames=3,
                settle_delay_ms=settle_ms,
                settle_drop_frames=1,
            )

    def _on_video_window_closed(self, key: str):
        if self._ui_closing:
            return
        self._dock_video_pane(str(key), from_signal=True)

    @staticmethod
    def _without_fullscreen(state: Qt.WindowState) -> Qt.WindowState:
        return state & ~Qt.WindowState.WindowFullScreen

    def _set_immersive_video_ui(self, enabled: bool):
        """Hide controls/panels and make video surface edge-to-edge."""
        if self._root_layout is None:
            return

        targets = {
            "row0": self._row0_widget,
            "row1": self._row1_widget,
            "row2": self._row2_widget,
            "metrics": self._grp_metrics,
            "hdr": self._grp_hdr,
            "inspector": getattr(self, "_inspector_workspace", None),
        }

        def _ensure_effect(widget: QWidget) -> QGraphicsOpacityEffect:
            eff = self._ui_anim_effects.get(widget)
            if eff is None:
                eff = QGraphicsOpacityEffect(widget)
                eff.setOpacity(1.0)
                widget.setGraphicsEffect(eff)
                self._ui_anim_effects[widget] = eff
            return eff

        def _animate_widget(widget: QWidget, show: bool):
            if widget is None:
                return
            if self._ui_anim_duration_ms <= 0:
                widget.setVisible(bool(show))
                eff = self._ui_anim_effects.get(widget)
                if eff is not None:
                    eff.setOpacity(1.0)
                return
            anim = self._ui_anim_running.get(widget)
            if anim is not None:
                anim.stop()
            eff = _ensure_effect(widget)
            anim = QPropertyAnimation(eff, b"opacity", self)
            anim.setDuration(max(60, int(self._ui_anim_duration_ms)))
            anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            if show:
                widget.setVisible(True)
                eff.setOpacity(0.0)
                anim.setStartValue(0.0)
                anim.setEndValue(1.0)
            else:
                eff.setOpacity(1.0)
                anim.setStartValue(1.0)
                anim.setEndValue(0.0)

            def _done():
                if not show:
                    widget.setVisible(False)
                    eff.setOpacity(1.0)

            anim.finished.connect(_done)
            self._ui_anim_running[widget] = anim
            anim.start()

        if enabled:
            self._immersive_saved_vis = {k: w.isVisible() for k, w in targets.items()}
            m = self._root_layout.contentsMargins()
            self._immersive_saved_margins = (m.left(), m.top(), m.right(), m.bottom())
            self._immersive_saved_spacing = self._root_layout.spacing()
            self._immersive_saved_view_mode = self._cmb_view.currentText()
            self._begin_immersive_timeline_overlay()
            self._begin_immersive_metrics_overlay()

            for w in targets.values():
                if w is not None and w.isVisible():
                    _animate_widget(w, False)
            self._root_layout.setContentsMargins(0, 0, 0, 0)
            self._root_layout.setSpacing(0)
            if self._video_tabs is not None:
                self._video_tabs.tabBar().setVisible(False)
            return

        self._end_immersive_timeline_overlay()
        self._end_immersive_metrics_overlay()
        for key, w in targets.items():
            want = self._immersive_saved_vis.get(key, True)
            if w is not None:
                _animate_widget(w, bool(want))

        if self._immersive_saved_margins is not None:
            l, t, r, b = self._immersive_saved_margins
            self._root_layout.setContentsMargins(l, t, r, b)
        if self._immersive_saved_spacing is not None:
            self._root_layout.setSpacing(self._immersive_saved_spacing)
        if self._video_tabs is not None:
            self._video_tabs.tabBar().setVisible(True)
        if self._immersive_saved_view_mode:
            self._set_view_mode_silently(self._immersive_saved_view_mode)

    def _begin_immersive_timeline_overlay(self):
        row = self._row3_widget
        if row is None or getattr(self, "_immersive_timeline_overlay", False):
            return
        if _normalize_source_mode(getattr(self, "_source_mode", None)) == SOURCE_MODE_WINDOW:
            return
        parent = row.parentWidget()
        layout = parent.layout() if parent is not None else None
        index = layout.indexOf(row) if layout is not None else -1
        self._immersive_row3_parent = parent
        self._immersive_row3_layout = layout
        self._immersive_row3_index = index
        if layout is not None:
            layout.removeWidget(row)
        row.setParent(self)
        row.setVisible(True)
        row.raise_()
        self._immersive_timeline_overlay = True
        self._position_immersive_timeline_overlay()

    def _begin_immersive_metrics_overlay(self):
        metrics = getattr(self, "_grp_metrics", None)
        if metrics is None or getattr(self, "_immersive_metrics_overlay", False):
            return
        chk = getattr(self, "_chk_metrics", None)
        if chk is None or not bool(chk.isChecked()):
            return
        parent = metrics.parentWidget()
        layout = parent.layout() if parent is not None else None
        index = layout.indexOf(metrics) if layout is not None else -1
        self._immersive_metrics_parent = parent
        self._immersive_metrics_layout = layout
        self._immersive_metrics_index = index
        if layout is not None:
            layout.removeWidget(metrics)
        metrics.setParent(self)
        metrics.hide()
        metrics.raise_()
        self._immersive_metrics_overlay = True
        self._position_immersive_metrics_overlay()

    def _end_immersive_timeline_overlay(self):
        row = self._row3_widget
        if row is None or not getattr(self, "_immersive_timeline_overlay", False):
            return
        layout = getattr(self, "_immersive_row3_layout", None)
        parent = getattr(self, "_immersive_row3_parent", None)
        index = int(getattr(self, "_immersive_row3_index", -1))
        row.hide()
        if parent is not None:
            row.setParent(parent)
        if layout is not None:
            if index < 0 or index > layout.count():
                layout.addWidget(row)
            else:
                layout.insertWidget(index, row)
        row.setVisible(
            _normalize_source_mode(getattr(self, "_source_mode", None))
            != SOURCE_MODE_WINDOW
        )
        self._immersive_timeline_overlay = False
        self._immersive_row3_parent = None
        self._immersive_row3_layout = None
        self._immersive_row3_index = -1

    def _end_immersive_metrics_overlay(self):
        metrics = getattr(self, "_grp_metrics", None)
        if metrics is None or not getattr(self, "_immersive_metrics_overlay", False):
            return
        layout = getattr(self, "_immersive_metrics_layout", None)
        parent = getattr(self, "_immersive_metrics_parent", None)
        index = int(getattr(self, "_immersive_metrics_index", -1))
        metrics.hide()
        if parent is not None:
            metrics.setParent(parent)
        if layout is not None:
            if index < 0 or index > layout.count():
                layout.addWidget(metrics)
            else:
                layout.insertWidget(index, metrics)
        metrics.setVisible(bool(self._immersive_saved_vis.get("metrics", True)))
        self._immersive_metrics_overlay = False
        self._immersive_metrics_parent = None
        self._immersive_metrics_layout = None
        self._immersive_metrics_index = -1

    def _position_immersive_timeline_overlay(self):
        row = self._row3_widget
        if row is None or not getattr(self, "_immersive_timeline_overlay", False):
            return
        margin = 16
        hint = row.sizeHint()
        width = max(240, self.width() - (margin * 2))
        height = max(44, hint.height())
        x = margin
        y = max(margin, self.height() - height - margin)
        row.setGeometry(x, y, width, height)
        row.raise_()

    def _immersive_metrics_overlay_width(self) -> int:
        metrics = getattr(self, "_grp_metrics", None)
        if metrics is None:
            return 360
        layout = metrics.layout()
        base_width = max(
            360,
            int(metrics.minimumSizeHint().width()),
            int(metrics.sizeHint().width()),
        )
        if layout is None:
            return base_width
        margins = layout.contentsMargins()
        spacing = max(0, int(layout.horizontalSpacing()))
        paired_rows = (
            ("fps", "latency"),
            ("frame", "res"),
            ("gpu", "cpu"),
        )
        full_rows = ("model", "prec")

        def _label_width(key: str) -> int:
            lbl = getattr(self, "_m", {}).get(key)
            if lbl is None:
                return 0
            text_w = int(lbl.fontMetrics().horizontalAdvance(str(lbl.text() or ""))) + 12
            return max(
                int(lbl.minimumSizeHint().width()),
                int(lbl.sizeHint().width()),
                text_w,
            )

        paired_width = 0
        for left_key, right_key in paired_rows:
            paired_width = max(
                paired_width,
                _label_width(left_key) + spacing + _label_width(right_key),
            )
        full_width = max((_label_width(key) for key in full_rows), default=0)
        content_width = (
            int(margins.left())
            + int(margins.right())
            + max(paired_width, full_width)
            + 24
        )
        return max(base_width, content_width)

    def _position_immersive_metrics_overlay(self):
        metrics = getattr(self, "_grp_metrics", None)
        if metrics is None or not getattr(self, "_immersive_metrics_overlay", False):
            return
        if bool(getattr(self, "_immersive_metrics_positioning", False)):
            return
        self._immersive_metrics_positioning = True
        try:
            margin = 16
            hint = metrics.sizeHint()
            width = min(
                self._immersive_metrics_overlay_width(),
                max(280, self.width() - (margin * 2)),
            )
            height = max(hint.height(), metrics.minimumSizeHint().height())
            metrics.setGeometry(margin, margin, width, height)
            metrics.raise_()
        finally:
            self._immersive_metrics_positioning = False

    def _show_immersive_timeline_overlay(self):
        row = self._row3_widget
        if row is None or not getattr(self, "_immersive_timeline_overlay", False):
            return
        row.setVisible(True)
        self._position_immersive_timeline_overlay()

    def _show_immersive_metrics_overlay(self):
        metrics = getattr(self, "_grp_metrics", None)
        if metrics is None or not getattr(self, "_immersive_metrics_overlay", False):
            return
        chk = getattr(self, "_chk_metrics", None)
        if chk is None or not bool(chk.isChecked()):
            metrics.hide()
            return
        metrics.setVisible(True)
        self._position_immersive_metrics_overlay()

    def _position_ui_overlay(self):
        self._position_video_transition_overlay()
        self._position_immersive_timeline_overlay()
        self._position_immersive_metrics_overlay()
        if self._ui_overlay_btn is None:
            return
        margin = 16
        w = self._ui_overlay_btn.width()
        h = self._ui_overlay_btn.height()
        x = max(margin, self.width() - w - margin)
        y = margin
        self._ui_overlay_btn.move(x, y)

    def _show_ui_overlay_temporarily(self):
        if self._ui_overlay_btn is None:
            return
        if not self._ui_hidden:
            self._ui_overlay_btn.hide()
            return
        self._show_immersive_timeline_overlay()
        self._show_immersive_metrics_overlay()
        self._position_ui_overlay()
        self._ui_overlay_btn.show()
        self._ui_overlay_btn.raise_()
        if self._ui_overlay_timer is None:
            self._ui_overlay_timer = QTimer(self)
            self._ui_overlay_timer.setSingleShot(True)
            self._ui_overlay_timer.timeout.connect(self._hide_ui_overlay)
        self._ui_overlay_timer.stop()
        self._ui_overlay_timer.start(int(self._ui_overlay_hide_ms))

    def _hide_ui_overlay(self):
        if (
            getattr(self, "_immersive_timeline_overlay", False)
            and self._seek_slider is not None
            and self._seek_slider.isSliderDown()
        ):
            if self._ui_overlay_timer is not None:
                self._ui_overlay_timer.start(int(self._ui_overlay_hide_ms))
            return
        if self._ui_overlay_btn is not None:
            self._ui_overlay_btn.hide()
        if getattr(self, "_immersive_timeline_overlay", False) and self._row3_widget is not None:
            self._row3_widget.hide()
        if getattr(self, "_immersive_metrics_overlay", False):
            metrics = getattr(self, "_grp_metrics", None)
            if metrics is not None:
                metrics.hide()

    def _toggle_ui_visibility(self):
        if not self._playing:
            return
        self._begin_video_surface_fade(
            duration_ms=1000,
            start_opacity=1.0,
        )
        self._ui_hidden = not self._ui_hidden
        self._set_immersive_video_ui(self._ui_hidden)
        if self._ui_hidden:
            self.menuBar().setVisible(False)
            self.statusBar().setVisible(False)
            self._show_ui_overlay_temporarily()
        else:
            self.menuBar().setVisible(True)
            self.statusBar().setVisible(True)
            if self._row3_widget is not None:
                self._row3_widget.setVisible(
                    getattr(self, "_source_mode", "video") != "window_capture"
                )
            if self._ui_overlay_btn is not None:
                self._ui_overlay_btn.hide()
        if self._btn_toggle_ui is not None:
            self._btn_toggle_ui.setText("Show UI" if self._ui_hidden else "Hide UI")
        if self._playing:
            self._schedule_state_change_relock(delay_ms=120, drop_frames=2)

    def _set_pause_button_labels(self, paused: bool):
        if paused:
            self._btn_pause.setText("Resume")
        else:
            self._btn_pause.setText("Pause")

    @staticmethod
    def _screen_signature(screen) -> str | None:
        if screen is None:
            return None
        try:
            g = screen.geometry()
            name = str(screen.name() or "?")
            serial = str(screen.serialNumber() or "")
            dpr = float(screen.devicePixelRatio())
            return (
                f"{name}|{serial}|"
                f"{int(g.x())},{int(g.y())},{int(g.width())}x{int(g.height())}|"
                f"{dpr:.3f}"
            )
        except Exception:
            return None

    def _screen_for_widget(self, widget: QWidget | None = None):
        """Resolve the screen that owns a widget's top-level window."""
        screen = None
        if widget is not None:
            try:
                wh = widget.windowHandle()
                if wh is not None:
                    screen = wh.screen()
            except Exception:
                screen = None
            if screen is None:
                try:
                    top = widget.window()
                    wh = top.windowHandle() if top is not None else None
                    if wh is not None:
                        screen = wh.screen()
                except Exception:
                    screen = None
        if screen is None:
            try:
                wh = self.windowHandle()
                if wh is not None:
                    screen = wh.screen()
            except Exception:
                screen = None
        if screen is None:
            screen = QApplication.primaryScreen()
        return screen

    @staticmethod
    def _fit_content_to_bounds(
        bounds_w: int,
        bounds_h: int,
        src_w: int | None = None,
        src_h: int | None = None,
    ) -> tuple[int, int]:
        """Fit the processed frame aspect into a physical-pixel target box."""
        bw = max(2, int(bounds_w))
        bh = max(2, int(bounds_h))
        sw = max(1, int(src_w or 16))
        sh = max(1, int(src_h or 9))
        scale = min(float(bw) / float(sw), float(bh) / float(sh))
        return (
            max(2, int(round(float(sw) * scale))),
            max(2, int(round(float(sh) * scale))),
        )

    def _current_hdr_pane_bounds(self) -> tuple[int, int] | None:
        widget = (
            self._disp_hdr_mpv
            if getattr(self, "_disp_hdr_mpv", None) is not None
            else self
        )
        try:
            size = widget.size()
            dpr = float(widget.devicePixelRatioF())
            w = int(round(float(size.width()) * max(1.0, dpr)))
            h = int(round(float(size.height()) * max(1.0, dpr)))
            if w > 16 and h > 16:
                return max(2, w), max(2, h)
        except Exception:
            pass
        return None

    def _current_monitor_bounds(self) -> tuple[int, int]:
        widget = (
            self._disp_hdr_mpv
            if getattr(self, "_disp_hdr_mpv", None) is not None
            else self
        )
        screen = self._screen_for_widget(widget)
        if screen is None:
            return (1920, 1080)
        try:
            g = screen.geometry()
            dpr = float(screen.devicePixelRatio())
            w = int(round(float(g.width()) * max(1.0, dpr)))
            h = int(round(float(g.height()) * max(1.0, dpr)))
            return max(2, w), max(2, h)
        except Exception:
            return (1920, 1080)

    def _current_actual_upscale_target_dims(
        self,
        proc_w: int | None = None,
        proc_h: int | None = None,
    ) -> tuple[int, int]:
        """Physical pixels currently available to the HDR video image."""
        pane_bounds = self._current_hdr_pane_bounds()
        if pane_bounds is None:
            pane_bounds = self._current_monitor_bounds()
        return self._fit_content_to_bounds(
            pane_bounds[0],
            pane_bounds[1],
            proc_w,
            proc_h,
        )

    def _current_upscale_target_dims(
        self,
        proc_w: int | None = None,
        proc_h: int | None = None,
    ) -> tuple[int, int]:
        """Presentation target: follow the actual HDR pane size."""
        pane_bounds = self._current_hdr_pane_bounds()
        if pane_bounds is None:
            monitor_bounds = self._current_monitor_bounds()
            return self._fit_content_to_bounds(
                monitor_bounds[0],
                monitor_bounds[1],
                proc_w,
                proc_h,
            )
        return self._fit_content_to_bounds(
            pane_bounds[0],
            pane_bounds[1],
            proc_w,
            proc_h,
        )

    def _disable_active_top_preset_sharpen(
        self,
        target_w: int,
        target_h: int,
        proc_w: int,
        proc_h: int,
    ) -> bool:
        """Avoid sharpening source-bucket padding when no monitor upscale exists."""
        source_dims = getattr(self, "_source_video_dims", None)
        if not (isinstance(source_dims, tuple) and len(source_dims) == 2):
            return False
        active_scale_key = str(
            getattr(self, "_active_resolution", None)
            or self._cmb_res.currentText()
            or ""
        ).strip()
        if active_scale_key == "Source":
            active_scale_key = str(
                getattr(self, "_source_max_resolution_key", "1080p") or "1080p"
            )
        if active_scale_key != str(
            getattr(self, "_source_max_resolution_key", "1080p") or "1080p"
        ):
            return False
        if not _source_is_below_processing_preset(
            source_dims[0], source_dims[1], active_scale_key
        ):
            return False
        return not _is_upscale_required(proc_w, proc_h, target_w, target_h)

    def _apply_monitor_upscale_settings(self, announce: bool = False) -> bool:
        """Recompute HDR mpv scaling after monitor or pane-size changes."""
        if not self._playing or not getattr(self, "_active_use_mpv", False):
            return False
        if self._disp_hdr_mpv is None or self._last_res is None:
            return False
        proc_w, proc_h = self._last_res
        target_w, target_h = self._current_upscale_target_dims(proc_w, proc_h)
        actual_w, actual_h = self._current_actual_upscale_target_dims(proc_w, proc_h)
        self._cur_upscale_target_w = int(target_w)
        self._cur_upscale_target_h = int(target_h)
        # Use the applied runtime mode here. The combo box is only a pending
        # preference until the user clicks Apply.
        upscale_choice = str(
            getattr(self, "_active_upscale_mode", None) or DEFAULT_UPSCALER
        )
        kernel = _select_hdr_scale_kernel(
            proc_w,
            proc_h,
            target_w,
            target_h,
            upscale_choice,
        )
        antiring = _select_hdr_scale_antiring(
            proc_w,
            proc_h,
            target_w,
            target_h,
            kernel,
        )
        cas = _select_mpv_cas_strength(
            proc_w,
            proc_h,
            actual_w,
            actual_h,
            using_fsr=(kernel == "fsr"),
            scale_kernel=kernel,
        )
        if self._disable_active_top_preset_sharpen(
            actual_w, actual_h, proc_w, proc_h
        ):
            cas = 0.0

        ok = True
        if (
            str(kernel) != str(getattr(self, "_active_mpv_scale_kernel", ""))
            or abs(
                float(antiring)
                - float(getattr(self, "_active_mpv_scale_antiring", 0.0))
            )
            > 1e-6
        ):
            ok = bool(self._disp_hdr_mpv.set_scale_kernel(kernel, antiring)) and ok
        if (
            ok
            and abs(float(cas) - float(getattr(self, "_active_mpv_cas", 0.0)))
            > 1e-6
        ):
            ok = bool(self._disp_hdr_mpv.set_cas_strength(cas)) and ok
        if not ok:
            return False

        self._active_mpv_scale_kernel = kernel
        self._active_mpv_scale_antiring = float(antiring)
        self._active_mpv_cas = float(cas)
        self._active_upscale_mode = str(upscale_choice)
        if announce and _is_upscale_required(proc_w, proc_h, target_w, target_h):
            self.statusBar().showMessage(
                f"Pane upscale active: {proc_w}x{proc_h} -> "
                f"{target_w}x{target_h} via {upscale_choice}"
            )
        return True

    def _widget_screen_signature(self, widget: QWidget | None) -> str | None:
        if widget is None:
            return None
        screen = self._screen_for_widget(widget)
        return self._screen_signature(screen)

    def _attach_screen_change_hook(self, widget: QWidget | None):
        if widget is None:
            return
        try:
            top = widget.window()
            handle = top.windowHandle() if top is not None else widget.windowHandle()
        except Exception:
            handle = None
        if handle is None:
            return
        token = id(handle)
        if token in self._screen_hooked_handles:
            return
        try:
            handle.screenChanged.connect(self._on_screen_changed)
            self._screen_hooked_handles.add(token)
        except Exception:
            pass

    def _sync_screen_change_hooks(self):
        self._attach_screen_change_hook(self)
        self._attach_screen_change_hook(self._disp_hdr_mpv)
        self._attach_screen_change_hook(self._disp_sdr_mpv)

    def _on_screen_changed(self, _screen=None):
        try:
            self._sync_upscale_controls()
            self._update_apply_button_state()
        except Exception:
            pass
        if not self._playing:
            return
        self._schedule_window_state_refresh(0, soft_only=True)

    def _detect_mpv_screen_change(self) -> bool:
        changed = False
        hdr_sig = self._widget_screen_signature(self._disp_hdr_mpv)
        sdr_sig = self._widget_screen_signature(self._disp_sdr_mpv)
        if hdr_sig is not None and hdr_sig != self._hdr_mpv_screen_sig:
            changed = True
        if sdr_sig is not None and sdr_sig != self._sdr_mpv_screen_sig:
            changed = True
        self._hdr_mpv_screen_sig = hdr_sig
        self._sdr_mpv_screen_sig = sdr_sig
        return changed

    def _apply_mpv_runtime_filters(self, attempts_left: int = 2):
        ok = True
        if self._disp_hdr_mpv is not None:
            ok = bool(self._disp_hdr_mpv.set_cas_strength(self._active_mpv_cas)) and ok
            ok = bool(self._disp_hdr_mpv.set_film_grain(self._active_film_grain)) and ok
        if self._disp_sdr_mpv is not None:
            ok = bool(self._disp_sdr_mpv.set_cas_strength(0.0)) and ok
        if (not ok) and attempts_left > 0:
            QTimer.singleShot(
                120,
                lambda: self._apply_mpv_runtime_filters(attempts_left - 1),
            )

    def _ensure_window_refresh_timer(self):
        if self._window_refresh_timer is not None:
            return
        self._window_refresh_timer = QTimer(self)
        self._window_refresh_timer.setSingleShot(True)
        self._window_refresh_timer.timeout.connect(
            self._refresh_mpv_after_window_state_change
        )

    def _begin_layout_freeze(self):
        return

    def _end_layout_freeze(
        self,
        refresh_delay: int | None = 40,
        refresh_soft_only: bool = False,
    ):
        if refresh_delay is not None:
            self._schedule_window_state_refresh(
                refresh_delay, soft_only=refresh_soft_only
            )

    def _with_layout_freeze(
        self, fn, refresh_delay: int | None = 40, refresh_soft_only: bool = False
    ):
        fn()
        if refresh_delay is not None:
            self._schedule_window_state_refresh(
                refresh_delay, soft_only=refresh_soft_only
            )

    def _schedule_window_state_refresh(self, delay_ms: int = 140, soft_only: bool = False):
        self._ensure_window_refresh_timer()
        self._window_refresh_soft_only = bool(soft_only)
        self._window_refresh_timer.stop()
        self._window_refresh_timer.start(max(0, int(delay_ms)))

    def _ensure_overlay_reposition_timer(self):
        if self._overlay_reposition_timer is not None:
            return
        self._overlay_reposition_timer = QTimer(self)
        self._overlay_reposition_timer.setSingleShot(True)
        self._overlay_reposition_timer.timeout.connect(self._position_ui_overlay)

    def _schedule_overlay_position(self, delay_ms: int = 16):
        self._position_ui_overlay()

    def _ensure_ui_pause_timer(self):
        if self._ui_pause_timer is not None:
            return
        self._ui_pause_timer = QTimer(self)
        self._ui_pause_timer.setSingleShot(True)

    def _ensure_periodic_relock_timer(self):
        if self._periodic_relock_timer is not None:
            return
        self._periodic_relock_timer = QTimer(self)
        self._periodic_relock_timer.setSingleShot(False)
        self._periodic_relock_timer.timeout.connect(self._periodic_relock_tick)

    def _periodic_relock_tick(self):
        if not self._playing:
            return
        if (
            _normalize_source_mode(getattr(self, "_source_mode", None))
            == SOURCE_MODE_WINDOW
        ):
            return
        if self._current_video_tab_label() == "Side by Side":
            self._stop_periodic_relock()
            return
        if self._worker is None or self._worker.is_paused:
            return
        if self._startup_sync_pending:
            return
        if bool(getattr(self, "_video_prebuffer_pending", False)):
            return
        if not self._active_use_mpv:
            return
        if self._audio_available and self._audio_player is not None:
            return
        # Light-touch resync to keep HDR/SDR aligned over time (video-only).
        fps = getattr(self, "_vid_fps", 30.0)
        target_sec = float(self._last_seek_frame) / max(fps, 1e-6)
        audio_clock_ok = not (
            self._startup_audio_gate_active
            or self._auto_muted_low_fps
            or self._scrub_muted
            or self._relock_hold_muted
            or self._pending_playhead_relock_on_unmute
        )
        if self._audio_available and self._audio_player is not None and not audio_clock_ok:
            return
        if self._audio_available and self._audio_player is not None and audio_clock_ok:
            try:
                target_sec = float(self._audio_player.position()) / 1000.0
            except Exception:
                target_sec = float(self._last_seek_frame) / max(fps, 1e-6)

        hdr_t = None
        sdr_t = None
        if self._disp_hdr_mpv is not None:
            try:
                hdr_t = self._disp_hdr_mpv.get_time_seconds()
            except Exception:
                hdr_t = None
        if self._disp_sdr_mpv is not None:
            try:
                sdr_t = self._disp_sdr_mpv.get_time_seconds()
            except Exception:
                sdr_t = None

        drift_threshold_s = float(getattr(self, "_periodic_relock_drift_s", 0.045))
        drifts = [
            abs(float(t) - target_sec) for t in (hdr_t, sdr_t) if t is not None
        ]
        if drifts and max(drifts) <= max(0.0, drift_threshold_s):
            return
        if not drifts:
            return

        if self._worker is not None:
            try:
                self._worker.flush_display_queues(drop_frames=1)
            except AttributeError:
                self._worker.flush_hdr_queue(drop_frames=1)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.seek_seconds(target_sec)
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.seek_seconds(target_sec)

    def _start_periodic_relock(self):
        self._ensure_periodic_relock_timer()
        if self._periodic_relock_timer is None:
            return
        self._periodic_relock_timer.stop()
        if (
            _normalize_source_mode(getattr(self, "_source_mode", None))
            == SOURCE_MODE_WINDOW
        ):
            return
        if self._current_video_tab_label() == "Side by Side":
            return
        if self._audio_available and self._audio_player is not None:
            return
        period_ms = int(getattr(self, "_periodic_relock_ms", 1200))
        first_ms = int(getattr(self, "_periodic_relock_first_ms", 450))
        self._periodic_relock_timer.start(max(200, period_ms))
        # Do one early relock so users do not wait for the full periodic interval.
        QTimer.singleShot(max(100, first_ms), self._periodic_relock_tick)

    def _stop_periodic_relock(self):
        if self._periodic_relock_timer is not None:
            self._periodic_relock_timer.stop()

    def _pause_for_ui_transition(
        self,
        duration_ms: int | None = None,
        wait_for_stable: bool = True,
    ):
        if not self._playing:
            return
        if bool(getattr(self, "_video_prebuffer_pending", False)):
            return
        if self._worker is not None and self._worker.is_paused:
            return
        if self._ui_pause_timer is not None and self._ui_pause_timer.isActive():
            self._ui_pause_timer.stop()
        delay = int(
            duration_ms if duration_ms is not None else self._ui_pause_duration_ms
        )
        # FSR shader refresh can stall mpv a bit longer during UI changes.
        if bool(getattr(self, "_active_use_mpv", False)) and (
            str(getattr(self, "_active_mpv_scale_kernel", "") or "").strip().lower()
            == "fsr"
        ):
            delay += 140
        delay = max(60, delay)

        if self._worker is not None:
            self._worker.pause()
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.set_paused(True)
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.set_paused(True)
        if self._audio_available:
            self._set_audio_paused(True)

        self._ensure_ui_pause_timer()
        if self._ui_pause_timer is None:
            return

        def _resume():
            if not self._playing:
                return
            if bool(getattr(self, "_video_prebuffer_pending", False)):
                return
            if self._worker is not None and self._worker.is_paused:
                self._worker.resume()
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_mpv.set_paused(False)
            if self._disp_sdr_mpv is not None:
                self._disp_sdr_mpv.set_paused(False)
            if (
                self._audio_available
                and not self._startup_sync_pending
                and _normalize_source_mode(getattr(self, "_source_mode", None))
                != SOURCE_MODE_WINDOW
            ):
                # Keep audio paused until FPS stabilizes after the UI change.
                self._startup_audio_gate_active = True
                self._ui_resync_gate_strict = True
                now_t = time.perf_counter()
                self._audio_seek_guard_until = max(
                    self._audio_seek_guard_until, now_t + 0.8
                )

        try:
            self._ui_pause_timer.timeout.disconnect()
        except Exception:
            pass
        self._ui_pause_timer.timeout.connect(_resume)
        self._ui_pause_timer.start(delay)

    def _stabilize_window_capture_surface_after_startup(self):
        if not self._playing or not self._active_use_mpv:
            return
        source_mode = _normalize_source_mode(getattr(self, "_source_mode", None))
        if self._disp_sdr_mpv is not None and self._disp_sdr_stack is not None:
            self._disp_sdr_stack.setCurrentWidget(self._disp_sdr_mpv)
        if self._disp_hdr_mpv is not None and self._disp_hdr_stack is not None:
            self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_mpv)
        try:
            self._sync_worker_sdr_visibility()
        except Exception:
            pass

        if source_mode != SOURCE_MODE_WINDOW:
            # Normal file playback can use the seekable relock path after a
            # soft surface refresh. Live capture has no seekable timeline.
            self._pause_for_ui_transition(duration_ms=100, wait_for_stable=True)

        def _soft_warm():
            if not self._playing or not self._active_use_mpv:
                return
            worker_paused = bool(self._worker is not None and self._worker.is_paused)
            can_unpause_mpv = bool(source_mode == SOURCE_MODE_WINDOW or not worker_paused)
            if can_unpause_mpv and self._disp_hdr_mpv is not None:
                self._disp_hdr_mpv.set_paused(False)
            if can_unpause_mpv and self._disp_sdr_mpv is not None:
                self._disp_sdr_mpv.set_paused(False)
            self._with_layout_freeze(
                lambda: None,
                refresh_delay=40,
                refresh_soft_only=True,
            )
            self._apply_mpv_runtime_filters(2)
            if source_mode != SOURCE_MODE_WINDOW:
                self._schedule_state_change_relock(delay_ms=140, drop_frames=2)

        _soft_warm()
        QTimer.singleShot(220, _soft_warm)
        if source_mode != SOURCE_MODE_WINDOW:
            QTimer.singleShot(420, _soft_warm)

    def _toggle_borderless_full_window(self):
        now_t = time.perf_counter()
        if (now_t - self._window_toggle_last_t) < self._window_toggle_cooldown_s:
            return
        self._window_toggle_last_t = now_t
        if self._borderless_full_window:
            self._exit_borderless_full_window()
        else:
            self._enter_borderless_full_window()

    def _enter_borderless_full_window(self):
        if self._borderless_full_window:
            return
        self._begin_video_surface_fade(
            duration_ms=1000,
            start_opacity=1.0,
            cover_rect=self._current_screen_geometry(),
        )
        self._pause_for_ui_transition()

        def _apply_full():
            self._saved_window_geometry = self.geometry()
            self._saved_window_state = self._without_fullscreen(self.windowState())
            # Borderless fullscreen geometry: hides taskbar/top bar without
            # using true fullscreen (keeps "zoom" feel).
            self.menuBar().setVisible(True)
            self.statusBar().setVisible(True)
            self._ui_hidden = False
            self._set_immersive_video_ui(False)
            if self._btn_toggle_ui is not None:
                self._btn_toggle_ui.setText("Hide UI")
            self.setWindowState(self._without_fullscreen(self.windowState()))
            self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
            self.show()
            self.setGeometry(self._current_screen_geometry())

            self._borderless_full_window = True
            if self._act_borderless_full_window is not None:
                self._act_borderless_full_window.setChecked(True)
            self._schedule_overlay_position(0)

        self._with_layout_freeze(_apply_full, refresh_delay=140, refresh_soft_only=True)
        self._schedule_state_change_relock(delay_ms=180, drop_frames=3)

    def _exit_borderless_full_window(self):
        if not self._borderless_full_window:
            return
        self._begin_video_surface_fade(
            duration_ms=1000,
            start_opacity=1.0,
            cover_rect=self._current_screen_geometry(),
        )
        self._pause_for_ui_transition()

        def _apply_exit():
            restore_state = self._without_fullscreen(self._saved_window_state)
            restore_geom = self._saved_window_geometry
            self._borderless_full_window = False
            if self._act_borderless_full_window is not None:
                self._act_borderless_full_window.setChecked(False)

            self.setWindowState(self._without_fullscreen(self.windowState()))
            self.setWindowFlag(Qt.WindowType.FramelessWindowHint, False)
            self.show()
            self.menuBar().setVisible(True)
            self.statusBar().setVisible(True)
            self._set_immersive_video_ui(False)
            self._ui_hidden = False
            if self._btn_toggle_ui is not None:
                self._btn_toggle_ui.setText("Hide UI")
            if restore_geom is not None:
                self.setGeometry(restore_geom)
            if bool(restore_state & Qt.WindowState.WindowMaximized):
                self.showMaximized()
            else:
                self.showNormal()
            self._schedule_overlay_position(0)

        self._with_layout_freeze(_apply_exit, refresh_delay=140, refresh_soft_only=True)
        self._schedule_state_change_relock(delay_ms=180, drop_frames=3)

    def _should_use_mpv_pipeline(self) -> bool:
        return self._disp_hdr_mpv is not None

    def _on_view(self, mode):
        self._begin_video_pane_fade(duration_ms=1000, start_opacity=1.0)
        if self._video_tabs is not None:
            self._video_tabs.tabBar().setVisible(True)
        if self._disp_sdr_mpv is not None:
            if self._playing:
                self._disp_sdr_stack.setCurrentWidget(self._disp_sdr_mpv)
            else:
                self._disp_sdr_stack.setCurrentWidget(self._disp_sdr_cpu)
        if self._disp_hdr_mpv is not None:
            # Show textual placeholder when idle; switch to mpv only during playback.
            if self._playing:
                self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_mpv)
            else:
                self._disp_hdr_stack.setCurrentWidget(self._disp_hdr_cpu)
        # Let the worker skip unnecessary copies / postprocess
        if self._playing:
            sdr_visible = self._sync_worker_sdr_visibility()
            if sdr_visible and self._last_sdr_frame is not None:
                if self._disp_sdr_cpu is not None:
                    self._disp_sdr_cpu.update_frame(self._last_sdr_frame)
            self._schedule_state_change_relock(delay_ms=140, drop_frames=3)

    def _refresh_mpv_after_window_state_change(self):
        if not self._playing:
            return
        soft_only = bool(self._window_refresh_soft_only)
        self._window_refresh_soft_only = False
        self._sync_screen_change_hooks()
        display_changed = self._detect_mpv_screen_change()
        target_changed = False
        try:
            target_w, target_h = self._current_upscale_target_dims(
                *(self._last_res or (None, None))
            )
            target_changed = (
                int(target_w) != int(getattr(self, "_cur_upscale_target_w", 0))
                or int(target_h) != int(getattr(self, "_cur_upscale_target_h", 0))
            )
        except Exception:
            target_changed = False
        paused = bool(self._worker is not None and self._worker.is_paused)
        if paused:
            # Defer full mpv refresh while paused to avoid blackscreen.
            self._deferred_mpv_refresh = True
            if display_changed or target_changed or soft_only:
                self._apply_monitor_upscale_settings(announce=False)
                self._apply_mpv_runtime_filters(2)
            return
        self._deferred_mpv_refresh = False
        if (
            (not soft_only)
            and self._disp_hdr_mpv is not None
            and self._disp_hdr_mpv.needs_surface_refresh()
        ):
            self._disp_hdr_mpv.refresh_surface()
            if paused:
                self._disp_hdr_mpv.set_paused(True)
        if (
            (not soft_only)
            and self._disp_sdr_mpv is not None
            and self._disp_sdr_mpv.needs_surface_refresh()
        ):
            self._disp_sdr_mpv.refresh_surface()
            if paused:
                self._disp_sdr_mpv.set_paused(True)

        # Recompute the presentation scaler whenever the monitor or pane size
        # changes; small side-by-side panes should not keep fullscreen tuning.
        if display_changed or target_changed or soft_only:
            self._apply_monitor_upscale_settings(announce=display_changed)
            try:
                self._sync_upscale_controls()
                self._update_apply_button_state()
            except Exception:
                pass
            if display_changed:
                self._apply_mpv_runtime_filters(2)
        self._schedule_state_change_relock(drop_frames=3)
