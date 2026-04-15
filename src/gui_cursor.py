from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication


class CursorBehaviorMixin:
    """Cursor auto-hide helpers for MainWindow playback UX."""

    def _widget_is_video_surface(self, widget) -> bool:
        cur = widget
        visited = 0
        while cur is not None and visited < 12:
            try:
                if bool(cur.property("videoSurface")):
                    return True
            except Exception:
                pass
            try:
                cur = cur.parentWidget()
            except Exception:
                try:
                    cur = cur.parent()
                except Exception:
                    cur = None
            visited += 1
        return False

    def _cursor_over_video_surface(self) -> bool:
        try:
            widget = QApplication.widgetAt(QCursor.pos())
        except Exception:
            widget = None
        return self._widget_is_video_surface(widget)

    def _show_cursor(self):
        if self._cursor_hidden:
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass
            self._cursor_hidden = False

    def _hide_cursor_if_idle(self):
        if not self._cursor_idle_enabled or not self._playing:
            return
        if self._cursor_hidden:
            return
        if not self._cursor_over_video_surface():
            self._show_cursor()
            return
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.BlankCursor))
        self._cursor_hidden = True

    def _arm_cursor_idle_timer(self, *, video_surface: bool | None = None):
        if self._cursor_idle_timer is None:
            return
        self._cursor_idle_timer.stop()
        should_arm = (
            self._cursor_idle_enabled
            and self._playing
            and (
                self._cursor_over_video_surface()
                if video_surface is None
                else bool(video_surface)
            )
        )
        if should_arm:
            self._cursor_idle_timer.start(int(self._cursor_idle_ms))
        else:
            self._show_cursor()

    def _on_hide_cursor_toggled(self, enabled: bool):
        self._cursor_idle_enabled = bool(enabled)
        if not self._cursor_idle_enabled:
            if self._cursor_idle_timer is not None:
                self._cursor_idle_timer.stop()
            self._show_cursor()
        else:
            self._arm_cursor_idle_timer()

    def _init_cursor_idle_tracking(self):
        self._cursor_idle_timer = QTimer(self)
        self._cursor_idle_timer.setSingleShot(True)
        self._cursor_idle_timer.timeout.connect(self._hide_cursor_if_idle)
        QApplication.instance().installEventFilter(self)
