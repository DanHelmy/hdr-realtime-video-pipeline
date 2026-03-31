from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication


class CursorBehaviorMixin:
    """Cursor auto-hide helpers for MainWindow playback UX."""

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
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.BlankCursor))
        self._cursor_hidden = True

    def _arm_cursor_idle_timer(self):
        if self._cursor_idle_timer is None:
            return
        self._cursor_idle_timer.stop()
        if self._cursor_idle_enabled and self._playing:
            self._cursor_idle_timer.start(int(self._cursor_idle_ms))

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
