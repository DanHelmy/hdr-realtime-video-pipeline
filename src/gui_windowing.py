from __future__ import annotations

import os
import time

import psutil

from PyQt6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QRect, QTimer, Qt
from PyQt6.QtWidgets import QApplication, QGraphicsOpacityEffect, QVBoxLayout, QWidget

from gui_config import SOURCE_MODE_WINDOW, _normalize_source_mode
from gui_widgets import DetachedVideoWindow


class WindowingMixin:
    """Window/layout/fullscreen/popup behavior helpers for MainWindow."""

    def _current_screen_geometry(self) -> QRect:
        """Get full monitor geometry (not availableGeometry) to cover taskbar."""
        win_handle = self.windowHandle()
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

    def _on_video_tab_changed(self, index: int):
        if self._video_tabs is None or index < 0:
            return
        self._pause_for_ui_transition()

        def _apply_tab_switch():
            label = self._video_tabs.tabText(index)
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
        if self._playing:
            self._relock_timeline(delay_ms=140, drop_frames=3)

    def _on_app_state_changed(self, state: Qt.ApplicationState):
        active = state == Qt.ApplicationState.ApplicationActive
        if self._app_active == active:
            return
        self._app_active = active
        if self._playing:
            self._set_process_priority(True)
            if active:
                self._pause_for_ui_transition()

    def _toggle_sdr_popout(self):
        if self._sdr_float_window is not None:
            self._dock_video_pane("sdr")
            return

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
            self._relock_timeline(delay_ms=160, drop_frames=3)

    def _toggle_hdr_popout(self):
        if self._hdr_float_window is not None:
            self._dock_video_pane("hdr")
            return

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
            self._relock_timeline(delay_ms=160, drop_frames=3)
            # Follow-up relock after next position update to prevent stale-frame audio lag.
            QTimer.singleShot(520, lambda: self._relock_timeline(drop_frames=1))

    def _dock_video_pane(self, key: str, from_signal: bool = False):
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
        if self._playing:
            self._relock_timeline(delay_ms=160, drop_frames=3)
            if str(key).lower() == "hdr":
                # Follow-up relock after next position update to prevent stale-frame audio lag.
                QTimer.singleShot(520, lambda: self._relock_timeline(drop_frames=1))

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

            for w in targets.values():
                if w is not None and w.isVisible():
                    _animate_widget(w, False)
            self._root_layout.setContentsMargins(0, 0, 0, 0)
            self._root_layout.setSpacing(0)
            if self._video_tabs is not None:
                self._video_tabs.tabBar().setVisible(False)
            return

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

    def _position_ui_overlay(self):
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
        if self._row3_widget is not None:
            self._row3_widget.setVisible(
                getattr(self, "_source_mode", "video") != "window_capture"
            )
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
        if self._ui_overlay_btn is not None:
            self._ui_overlay_btn.hide()

    def _toggle_ui_visibility(self):
        if not self._playing:
            return
        self._pause_for_ui_transition()
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
        # Layout changes can momentarily stall video presentation.
        # Re-anchor audio to the current timeline after the UI toggle.
        if self._playing:
            self._relock_timeline(delay_ms=120, drop_frames=2)

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

    def _widget_screen_signature(self, widget: QWidget | None) -> str | None:
        if widget is None:
            return None
        screen = None
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
        if self._worker is None or self._worker.is_paused:
            return
        if self._startup_sync_pending:
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
        if self._audio_available and self._audio_player is not None and audio_clock_ok:
            try:
                target_sec = float(self._audio_player.position()) / 1000.0
            except Exception:
                target_sec = float(self._last_seek_frame) / max(fps, 1e-6)

        drift_threshold_s = float(getattr(self, "_periodic_relock_drift_s", 0.045))
        hdr_drift = None
        sdr_drift = None
        if self._disp_hdr_mpv is not None:
            try:
                hdr_t = self._disp_hdr_mpv.get_time_seconds()
                if hdr_t is not None:
                    hdr_drift = abs(float(hdr_t) - target_sec)
            except Exception:
                hdr_drift = None
        if self._disp_sdr_mpv is not None:
            try:
                sdr_t = self._disp_sdr_mpv.get_time_seconds()
                if sdr_t is not None:
                    sdr_drift = abs(float(sdr_t) - target_sec)
            except Exception:
                sdr_drift = None

        drifts = [d for d in (hdr_drift, sdr_drift) if d is not None]
        if drifts and max(drifts) <= max(0.0, drift_threshold_s):
            return

        if self._worker is not None:
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
        self, duration_ms: int | None = None, wait_for_stable: bool = True
    ):
        if not self._playing:
            return
        if self._worker is not None and self._worker.is_paused:
            return
        if self._ui_pause_timer is not None and self._ui_pause_timer.isActive():
            self._ui_pause_timer.stop()
        delay = int(
            duration_ms if duration_ms is not None else self._ui_pause_duration_ms
        )
        # FSR shader refresh can stall mpv a bit longer during UI changes.
        if self._active_mpv_scale_kernel == "fsr":
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
        if (
            _normalize_source_mode(getattr(self, "_source_mode", None))
            != SOURCE_MODE_WINDOW
        ):
            return
        if not self._playing or not self._active_use_mpv:
            return
        tabs = getattr(self, "_video_tabs", None)
        if tabs is None or tabs.count() < 2:
            self._pause_for_ui_transition(duration_ms=120, wait_for_stable=True)
            self._with_layout_freeze(lambda: None, refresh_delay=40)
            self._relock_timeline(delay_ms=140, drop_frames=3)
            return
        original_index = max(0, int(tabs.currentIndex()))
        candidate_index = None
        for offset in range(1, tabs.count()):
            idx = (original_index + offset) % tabs.count()
            if idx != original_index:
                candidate_index = idx
                break
        if candidate_index is None:
            return

        # The reliable user-discovered fix is an actual pane change, not just
        # a repaint. Mirror that once on startup, then restore the user's tab.
        tabs.setCurrentIndex(candidate_index)

        def _restore_original_tab():
            if (
                _normalize_source_mode(getattr(self, "_source_mode", None))
                != SOURCE_MODE_WINDOW
            ):
                return
            if not self._playing:
                return
            try:
                tabs.setCurrentIndex(original_index)
            except Exception:
                pass

        QTimer.singleShot(120, _restore_original_tab)

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
        self._relock_timeline(delay_ms=180, drop_frames=3)

    def _exit_borderless_full_window(self):
        if not self._borderless_full_window:
            return
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
        self._relock_timeline(delay_ms=180, drop_frames=3)

    def _should_use_mpv_pipeline(self) -> bool:
        return self._disp_hdr_mpv is not None

    def _on_view(self, mode):
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
            # Keep SDR path running so tab switches stay instantaneous.
            self._worker.set_sdr_visible(True)
            if self._last_sdr_frame is not None:
                if self._disp_sdr_cpu is not None:
                    self._disp_sdr_cpu.update_frame(self._last_sdr_frame)
            # Any view change can desync; relock the timeline.
            self._relock_timeline(delay_ms=140, drop_frames=3)

    def _refresh_mpv_after_window_state_change(self):
        if not self._playing:
            return
        soft_only = bool(self._window_refresh_soft_only)
        self._window_refresh_soft_only = False
        self._sync_screen_change_hooks()
        display_changed = self._detect_mpv_screen_change()
        paused = bool(self._worker is not None and self._worker.is_paused)
        if paused:
            # Defer full mpv refresh while paused to avoid blackscreen.
            self._deferred_mpv_refresh = True
            if display_changed:
                self._apply_mpv_runtime_filters(2)
            return
        self._deferred_mpv_refresh = False
        if (not soft_only) and self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.refresh_surface()
            if paused:
                self._disp_hdr_mpv.set_paused(True)
        if (not soft_only) and self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.refresh_surface()
            if paused:
                self._disp_sdr_mpv.set_paused(True)

        # Let mpv handle output colorspace adaptation; only reassert runtime
        # filters when the active display actually changed.
        if display_changed:
            self._apply_mpv_runtime_filters(2)
        # Relock video/audio after any window/layout refresh.
        self._relock_timeline(drop_frames=3)
