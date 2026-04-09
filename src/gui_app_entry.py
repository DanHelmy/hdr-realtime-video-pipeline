from __future__ import annotations

import argparse
import os
import sys

from PyQt6.QtCore import qInstallMessageHandler
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from windows_runtime import ensure_windows_supported

def _apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(30, 30, 30))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(208, 208, 208))
    p.setColor(QPalette.ColorRole.Base,            QColor(22, 22, 22))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(35, 35, 35))
    p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(40, 40, 40))
    p.setColor(QPalette.ColorRole.ToolTipText,     QColor(208, 208, 208))
    p.setColor(QPalette.ColorRole.Text,            QColor(208, 208, 208))
    p.setColor(QPalette.ColorRole.Button,          QColor(45, 45, 45))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(208, 208, 208))
    p.setColor(QPalette.ColorRole.BrightText,      QColor(255, 51, 51))
    p.setColor(QPalette.ColorRole.Link,            QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.Highlight,       QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(p)


def _install_qt_log_filter():
    """Suppress noisy, non-actionable Qt FFmpeg teardown warnings."""
    noisy_prefix = (
        "QObject::disconnect: wildcard call disconnects from destroyed signal of QFFmpeg::"
    )
    dpi_prefixes = (
        "SetProcessDpiAwarenessContext() failed:",
        "Qt's default DPI awareness context is DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2.",
    )

    def _handler(_msg_type, _context, message):
        text = str(message or "")
        if text.startswith(noisy_prefix):
            return
        if any(text.startswith(prefix) for prefix in dpi_prefixes):
            return
        try:
            print(text)
        except Exception:
            pass

    qInstallMessageHandler(_handler)


def _parse_gui_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video", default=None,
                        help="Auto-open this video on launch")
    parser.add_argument("--source-mode", default=None,
                        help="Initial source mode (video/window_capture for live browser-window capture)")
    parser.add_argument("--capture-hwnd", default=None,
                        help="Initial native browser-window handle for live capture mode")
    parser.add_argument("--capture-session-id", default=None,
                        help="Optional Chrome Audio Sync session id to pair with a browser window capture target")
    parser.add_argument("--capture-title", default=None,
                        help="Title hint for live capture mode")
    parser.add_argument("--capture-browser-name", default=None,
                        help="Browser name hint for live capture mode")
    parser.add_argument("--capture-process-name", default=None,
                        help="Browser process-name hint for live capture mode")
    parser.add_argument("--capture-url", default=None,
                        help="Source URL hint for live capture mode")
    parser.add_argument("--capture-fps", default=None,
                        help="Initial capture FPS preset label (e.g. 24 FPS)")
    parser.add_argument("--resolution", default=None,
                        help="Initial resolution preset (1080p/720p/540p/Source)")
    parser.add_argument("--precision", default=None,
                        help="Initial precision preset (GUI label)")
    parser.add_argument("--use-hg", default=None,
                        help="Enable HG (1/0). Default from saved settings.")
    parser.add_argument("--view", default=None,
                        help="Initial view mode (Tabbed)")
    parser.add_argument("--autoplay", default="0",
                        help="Auto-start playback after loading video (0/1)")
    parser.add_argument("--start-frame", default=None,
                        help="Initial frame index to seek to after startup")
    parser.add_argument("--upscale", default=None,
                        help="Initial upscale mode (GUI label)")
    parser.add_argument("--film-grain", default=None,
                        help="Enable film grain shader (0/1)")
    parser.add_argument("--hdr-gt", default=None,
                        help="HDR ground-truth video path for objective metrics")
    if argv is None:
        args, _unknown = parser.parse_known_args()
    else:
        args, _unknown = parser.parse_known_args(argv[1:])
    return args


def run_gui(window_cls, root_dir: str, argv: list[str] | None = None):
    ensure_windows_supported("HDRTVNet++ GUI")
    args = _parse_gui_args(argv)
    os.chdir(root_dir)
    _install_qt_log_filter()
    try:
        from browser_tab_bridge import ensure_browser_tab_bridge_running

        ensure_browser_tab_bridge_running()
    except Exception as exc:
        try:
            print(f"Browser tab bridge startup failed: {exc}")
        except Exception:
            pass
    app_argv = sys.argv if argv is None else argv
    app = QApplication(app_argv)
    _apply_dark_theme(app)
    win = window_cls(
        initial_video=args.video,
        initial_resolution=args.resolution,
        initial_precision=args.precision,
        initial_view=args.view,
        initial_use_hg=args.use_hg,
        initial_autoplay=(str(args.autoplay).strip() == "1"),
        initial_start_frame=args.start_frame,
        initial_upscale=args.upscale,
        initial_film_grain=args.film_grain,
        initial_hdr_gt=args.hdr_gt,
        initial_source_mode=args.source_mode,
        initial_capture_hwnd=args.capture_hwnd,
        initial_capture_session_id=args.capture_session_id,
        initial_capture_title=args.capture_title,
        initial_capture_browser_name=args.capture_browser_name,
        initial_capture_process_name=args.capture_process_name,
        initial_capture_url=args.capture_url,
        initial_capture_fps=args.capture_fps,
    )
    win.show()
    return app.exec()
