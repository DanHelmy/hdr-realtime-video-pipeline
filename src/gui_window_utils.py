from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget


def configure_independent_window(
    widget: QWidget,
    *,
    minimize: bool = True,
    maximize: bool = True,
    close: bool = True,
) -> None:
    """Make a secondary UI surface behave like a normal top-level window.

    Parent-owned dialogs can minimize strangely on Windows, especially after
    resizing/maximizing tool windows. These flags keep the workflow modal when
    callers use exec(), but avoid OS-level child/owned-window behavior.
    """
    widget.setWindowFlag(Qt.WindowType.Window, True)
    widget.setWindowFlag(Qt.WindowType.Dialog, False)
    widget.setWindowFlag(Qt.WindowType.WindowTitleHint, True)
    widget.setWindowFlag(Qt.WindowType.CustomizeWindowHint, False)
    widget.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, bool(minimize))
    widget.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, bool(maximize))
    widget.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, bool(close))
