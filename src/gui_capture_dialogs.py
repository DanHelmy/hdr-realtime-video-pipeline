from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from gui_config import (
    LIVE_CAPTURE_OBSERVE_FPS,
    LIVE_CAPTURE_PROCESS_FPS,
)
from window_capture_source import (
    WindowCaptureTarget,
    attach_best_browser_tab_session,
    enumerate_window_capture_targets,
    find_best_matching_window_capture_target,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_EXTENSION_DIR = os.path.join(_ROOT, "browser_tab_capture_extension")
_SUPPORTED_CHROME_PROCESS = "chrome.exe"


class WindowCaptureDialog(QDialog):
    """Choose a visible Chrome window for native live capture."""

    def __init__(
        self,
        *,
        initial_fps_label: str | None = None,
        initial_session_id: str | None = None,
        initial_target: WindowCaptureTarget | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Browser Window Capture (Experimental)")
        self.resize(820, 320)
        self._initial_session_id = str(initial_session_id or "").strip()
        self._initial_target = initial_target
        self._targets: list[WindowCaptureTarget] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        self._lbl_intro = QLabel(
            "Browser Window Capture is experimental and only supported with Google Chrome.\n\n"
            "1. The browser extension is bundled in this repo under browser_tab_capture_extension.\n"
            "2. Use Open Extension Folder if you need to load it in Chrome from chrome://extensions with Developer mode enabled.\n"
            "3. In Chrome Settings > System, turn off 'Use graphics acceleration when available', then restart Chrome.\n"
            "4. Chrome requires you to press Start Chrome Audio Sync in the extension yourself before Play.\n"
            "5. Adjust the extension delay slider while playback is running until lip-sync looks right.\n"
            "6. HDRTVNet++ captures video from a visible Chrome window and stays silent.\n"
            "7. The extension delays and plays Chrome tab audio locally.\n"
            "8. Chrome Audio Sync stays active until you stop it in the extension."
        )
        self._lbl_intro.setWordWrap(True)
        root.addWidget(self._lbl_intro)

        row_setup = QHBoxLayout()
        self._btn_open_extension_folder = QPushButton("Open Extension Folder")
        row_setup.addWidget(self._btn_open_extension_folder)
        row_setup.addStretch(1)
        root.addLayout(row_setup)

        row_window = QHBoxLayout()
        row_window.addWidget(QLabel("Browser Window:"))
        self._cmb_window = QComboBox()
        self._cmb_window.setMinimumWidth(520)
        row_window.addWidget(self._cmb_window, 1)
        self._btn_refresh = QPushButton("Refresh")
        row_window.addWidget(self._btn_refresh)
        root.addLayout(row_window)

        self._lbl_capture_mode = QLabel(
            f"Video observes Chrome up to {LIVE_CAPTURE_OBSERVE_FPS:g} fps, processes a steady {LIVE_CAPTURE_PROCESS_FPS:g} fps stream, then mpv repeats frames on display vsync."
        )
        self._lbl_capture_mode.setWordWrap(True)
        root.addWidget(self._lbl_capture_mode)

        self._lbl_status = QLabel("")
        self._lbl_status.setProperty("accentText", True)
        self._lbl_status.setWordWrap(True)
        root.addWidget(self._lbl_status)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._btn_refresh.clicked.connect(self._reload_targets)
        self._cmb_window.currentIndexChanged.connect(self._update_status)
        self._btn_open_extension_folder.clicked.connect(self._open_extension_folder)

        self._reload_targets()

    def _show_open_error(self, title: str, detail: str):
        QMessageBox.warning(self, title, detail)

    def _open_extension_folder(self):
        if not os.path.isdir(_EXTENSION_DIR):
            self._show_open_error(
                "Extension Folder Missing",
                f"Could not find the bundled extension folder:\n\n{_EXTENSION_DIR}",
            )
            return
        try:
            os.startfile(_EXTENSION_DIR)
        except Exception as exc:
            self._show_open_error(
                "Could Not Open Extension Folder",
                f"Could not open:\n\n{_EXTENSION_DIR}\n\n{exc}",
            )

    def _reload_targets(self):
        self._targets = [
            attach_best_browser_tab_session(target) or target
            for target in enumerate_window_capture_targets()
            if str(getattr(target, "process_name", "") or "").strip().lower()
            == _SUPPORTED_CHROME_PROCESS
        ]
        self._cmb_window.blockSignals(True)
        self._cmb_window.clear()
        selected_idx = -1
        for idx, target in enumerate(self._targets):
            item_id = str(
                getattr(target, "hwnd", 0) or getattr(target, "session_id", "") or ""
            )
            self._cmb_window.addItem(target.label, item_id)
            if self._initial_session_id and str(getattr(target, "session_id", "") or "") == self._initial_session_id:
                selected_idx = idx
        if selected_idx < 0 and self._initial_target is not None:
            best_target = find_best_matching_window_capture_target(
                self._targets, self._initial_target
            )
            if best_target is not None:
                best_id = str(
                    getattr(best_target, "hwnd", 0)
                    or getattr(best_target, "session_id", "")
                    or ""
                )
                for idx, target in enumerate(self._targets):
                    target_id = str(
                        getattr(target, "hwnd", 0)
                        or getattr(target, "session_id", "")
                        or ""
                    )
                    if target_id == best_id:
                        selected_idx = idx
                        break
        self._cmb_window.blockSignals(False)
        if self._targets:
            self._cmb_window.setEnabled(True)
            self._btn_refresh.setEnabled(True)
            if selected_idx >= 0:
                self._cmb_window.setCurrentIndex(selected_idx)
            else:
                self._cmb_window.setCurrentIndex(0)
        else:
            self._cmb_window.addItem("No visible Google Chrome windows found")
            self._cmb_window.setEnabled(False)
        self._update_status()

    def _update_status(self):
        target = self.selected_target()
        if target is None:
            self._lbl_status.setText(
                "No visible Google Chrome windows were found. Open the Chrome window you want to capture, keep it visible, then refresh."
            )
            return
        title = str(getattr(target, "title", "") or "").strip() or target.label
        details: list[str] = []
        if int(target.width) > 0 and int(target.height) > 0:
            details.append(f"{int(target.width)}x{int(target.height)}")
        browser_name = str(getattr(target, "browser_name", "") or "").strip()
        process_name = str(getattr(target, "process_name", "") or "").strip()
        if browser_name:
            details.append(browser_name)
        elif process_name:
            details.append(process_name)
        session_id = str(getattr(target, "session_id", "") or "").strip()
        if session_id:
            sync_text = (
                "Chrome Audio Sync ready. Keep Chrome graphics acceleration off and adjust delay in the extension popup."
            )
        else:
            sync_text = (
                "Chrome Audio Sync not started. Keep Chrome graphics acceleration off, then start Chrome Audio Sync before Play if you want delayed audio."
            )
        detail_text = " - ".join(details)
        if detail_text:
            self._lbl_status.setText(
                f"Selected: {title}\nWindow: {detail_text}\n{sync_text}"
            )
        else:
            self._lbl_status.setText(f"Selected: {title}\n{sync_text}")

    def selected_target(self) -> WindowCaptureTarget | None:
        idx = int(self._cmb_window.currentIndex())
        if idx < 0 or idx >= len(self._targets):
            return None
        return self._targets[idx]
