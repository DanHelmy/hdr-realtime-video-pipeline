from __future__ import annotations

from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication


_COLORS = {
    "window": "#1a1c1f",
    "window_alt": "#1d2024",
    "surface": "#25282d",
    "surface_alt": "#2b2f35",
    "surface_hover": "#31353d",
    "panel": "#16181b",
    "panel_soft": "#202329",
    "panel_raised": "#2a2d33",
    "border": "#343841",
    "border_strong": "#4a4f5a",
    "text": "#d7dadd",
    "text_soft": "#c2c6cb",
    "text_muted": "#a2a7ae",
    "text_dim": "#757b84",
    "accent": "#5d7ea6",
    "accent_soft": "#4f6886",
    "accent_strong": "#7f97b3",
    "amber": "#b0b3b8",
    "amber_soft": "#2f3338",
    "green": "#6f8d7e",
    "green_soft": "#2a332d",
    "red": "#8f7373",
    "red_soft": "#3a2c2c",
    "track": "#30343a",
    "selection_text": "#f2f4f6",
}


def _build_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(_COLORS["window"]))
    p.setColor(QPalette.ColorRole.WindowText, QColor(_COLORS["text"]))
    p.setColor(QPalette.ColorRole.Base, QColor(_COLORS["panel"]))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(_COLORS["surface"]))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(_COLORS["surface_alt"]))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(_COLORS["text"]))
    p.setColor(QPalette.ColorRole.Text, QColor(_COLORS["text"]))
    p.setColor(QPalette.ColorRole.Button, QColor(_COLORS["surface"]))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(_COLORS["text"]))
    p.setColor(QPalette.ColorRole.BrightText, QColor(_COLORS["red"]))
    p.setColor(QPalette.ColorRole.Link, QColor(_COLORS["accent"]))
    p.setColor(QPalette.ColorRole.Highlight, QColor(_COLORS["accent"]))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(_COLORS["selection_text"]))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(_COLORS["text_dim"]))
    return p


def _build_stylesheet() -> str:
    c = _COLORS
    return f"""
QToolTip {{
    color: {c["text"]};
    background-color: {c["surface_alt"]};
    border: 1px solid {c["border_strong"]};
    padding: 6px 8px;
}}

QMainWindow,
QDialog {{
    background-color: {c["window"]};
}}

QWidget#AppSurface {{
    background-color: {c["window"]};
}}

QMenuBar {{
    background-color: {c["panel"]};
    border-bottom: 1px solid {c["border"]};
    padding: 4px 8px;
}}

QMenuBar::item {{
    color: {c["text_soft"]};
    padding: 6px 10px;
    border-radius: 8px;
    background-color: transparent;
}}

QMenuBar::item:selected {{
    color: {c["text"]};
    background-color: {c["surface"]};
}}

QMenu {{
    color: {c["text"]};
    background-color: {c["panel_soft"]};
    border: 1px solid {c["border"]};
    padding: 6px;
}}

QMenu::item {{
    padding: 7px 14px;
    border-radius: 8px;
}}

QMenu::item:selected {{
    background-color: {c["surface_hover"]};
}}

QStatusBar {{
    color: {c["text_soft"]};
    background-color: {c["panel"]};
    border-top: 1px solid {c["border"]};
}}

QStatusBar::item {{
    border: none;
}}

QWidget#ModeToolbar,
QWidget#PlaybackToolbar,
QWidget#TimelineToolbar {{
    background-color: {c["panel_soft"]};
    border: 1px solid {c["border"]};
    border-radius: 6px;
}}

QWidget#ViewerWorkspace,
QWidget#InspectorWorkspace {{
    background-color: transparent;
}}

QScrollArea#InspectorWorkspace {{
    border: none;
    background-color: transparent;
}}

QWidget#InspectorContent {{
    background-color: transparent;
}}

QSplitter#WorkspaceSplit::handle {{
    background-color: {c["panel"]};
}}

QSplitter#WorkspaceSplit::handle:hover {{
    background-color: {c["surface_alt"]};
}}

QTabWidget::pane {{
    border-left: 1px solid {c["border"]};
    border-right: 1px solid {c["border"]};
    border-bottom: 1px solid {c["border"]};
    border-top: none;
    border-radius: 6px;
    background-color: {c["panel"]};
    top: 0px;
}}

QTabBar::tab {{
    color: {c["text_muted"]};
    background-color: {c["panel_soft"]};
    border: 1px solid {c["border"]};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 8px 14px;
    margin-right: 2px;
}}

QTabBar::tab:selected {{
    color: {c["text"]};
    background-color: {c["surface_alt"]};
    border-color: {c["accent_soft"]};
    border-bottom: 1px solid {c["surface_alt"]};
}}

QTabBar::tab:hover:!selected {{
    color: {c["text_soft"]};
    background-color: {c["surface"]};
}}

QTabWidget#VideoTabs::pane {{
    border: 1px solid {c["border"]};
    border-radius: 6px;
    background-color: {c["panel"]};
    top: -1px;
}}

QTabBar#ViewerTabBar {{
    left: 10px;
}}

QTabBar#ViewerTabBar::tab {{
    padding: 9px 18px;
    margin-right: 1px;
}}

QTabBar#ViewerTabBar::tab:selected {{
    margin-bottom: -1px;
    background-color: {c["panel"]};
    border-color: {c["border_strong"]};
    border-bottom: 1px solid {c["panel"]};
}}

QGroupBox {{
    color: {c["text"]};
    background-color: {c["panel_soft"]};
    border: 1px solid {c["border"]};
    border-radius: 6px;
    margin-top: 14px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {c["text_muted"]};
}}

QGroupBox#InspectorPanel,
QGroupBox#MetricsCard,
QGroupBox#HdrCard {{
    background-color: {c["panel_soft"]};
    border-color: {c["border"]};
}}

QGroupBox#InspectorPanel::title {{
    color: {c["text_soft"]};
}}

QLabel {{
    color: {c["text"]};
    background-color: transparent;
}}

QLabel[muted="true"] {{
    color: {c["text_muted"]};
}}

QLabel[pill="true"] {{
    color: {c["text_soft"]};
    background-color: transparent;
    border: none;
    padding: 0 2px;
}}

QLabel[metricChip="true"] {{
    color: {c["text_soft"]};
    background-color: transparent;
    border: none;
    padding: 0 2px;
}}

QLabel[eyebrow="true"] {{
    color: {c["text_soft"]};
    font-weight: 600;
}}

QLabel[accentText="true"] {{
    color: {c["text_soft"]};
}}

QPushButton {{
    color: {c["text"]};
    background-color: {c["surface"]};
    border: 1px solid {c["border"]};
    border-radius: 4px;
    padding: 6px 12px;
    min-height: 18px;
}}

QPushButton:hover {{
    background-color: {c["surface_hover"]};
    border-color: {c["border_strong"]};
}}

QPushButton:pressed {{
    background-color: {c["panel_raised"]};
}}

QPushButton:disabled {{
    color: {c["text_dim"]};
    background-color: {c["panel"]};
    border-color: {c["border"]};
}}

QPushButton[role="primary"] {{
    color: {c["text"]};
    background-color: {c["surface_alt"]};
    border-color: {c["accent_soft"]};
    font-weight: 600;
}}

QPushButton[role="primary"]:hover {{
    background-color: {c["surface_hover"]};
    border-color: {c["accent"]};
}}

QPushButton[role="success"] {{
    color: {c["text"]};
    background-color: {c["surface"]};
    border-color: {c["border_strong"]};
}}

QPushButton[role="success"]:hover {{
    background-color: {c["surface_hover"]};
}}

QPushButton[role="warning"] {{
    color: {c["text"]};
    background-color: {c["surface"]};
    border-color: {c["border_strong"]};
}}

QPushButton[role="warning"]:hover {{
    background-color: {c["surface_hover"]};
}}

QPushButton[role="ghost"] {{
    color: {c["text_soft"]};
    background-color: {c["surface"]};
}}

QPushButton[toolbar="compact"] {{
    padding: 6px 12px;
}}

QComboBox,
QSpinBox,
QDoubleSpinBox,
QLineEdit,
QTextEdit {{
    color: {c["text"]};
    background-color: {c["panel"]};
    border: 1px solid {c["border"]};
    border-radius: 4px;
    padding: 6px 10px;
    selection-background-color: {c["accent"]};
    selection-color: {c["selection_text"]};
}}

QComboBox:hover,
QSpinBox:hover,
QDoubleSpinBox:hover,
QLineEdit:hover,
QTextEdit:hover {{
    border-color: {c["border_strong"]};
}}

QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QLineEdit:focus,
QTextEdit:focus {{
    border-color: {c["accent_soft"]};
}}

QComboBox::drop-down,
QSpinBox::up-button,
QSpinBox::down-button,
QDoubleSpinBox::up-button,
QDoubleSpinBox::down-button {{
    border: none;
    background-color: transparent;
    width: 18px;
}}

QComboBox QAbstractItemView {{
    color: {c["text"]};
    background-color: {c["panel_soft"]};
    border: 1px solid {c["border"]};
    selection-background-color: {c["accent_soft"]};
    selection-color: {c["text"]};
}}

QCheckBox {{
    color: {c["text_soft"]};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 1px solid {c["border_strong"]};
    background-color: {c["panel"]};
}}

QCheckBox::indicator:hover {{
    border-color: {c["accent_soft"]};
}}

QCheckBox::indicator:checked {{
    background-color: {c["accent"]};
    border-color: {c["accent"]};
}}

QSlider::groove:horizontal {{
    height: 8px;
    border-radius: 4px;
    background-color: {c["track"]};
}}

QSlider::sub-page:horizontal {{
    background-color: {c["accent_soft"]};
    border-radius: 4px;
}}

QSlider::add-page:horizontal {{
    background-color: {c["track"]};
    border-radius: 4px;
}}

QSlider::handle:horizontal {{
    width: 14px;
    margin: -6px 0;
    border-radius: 3px;
    background-color: {c["accent"]};
    border: 1px solid {c["accent_strong"]};
}}

QSlider::handle:horizontal:hover {{
    background-color: {c["accent_strong"]};
}}

QProgressBar {{
    color: {c["text_soft"]};
    background-color: {c["panel"]};
    border: 1px solid {c["border"]};
    border-radius: 6px;
}}

QProgressBar::chunk {{
    background-color: {c["accent"]};
    border-radius: 5px;
}}

QSplitter::handle {{
    background-color: transparent;
}}

QSplitter::handle:hover {{
    background-color: rgba(127, 151, 179, 0.18);
}}

QWidget[videoSurface="true"],
QLabel#VideoDisplay {{
    background-color: #111214;
    border: 1px solid {c["border"]};
    border-radius: 4px;
    color: {c["text_dim"]};
}}
"""


def apply_app_theme(app: QApplication):
    app.setStyle("Fusion")
    app.setPalette(_build_palette())
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(_build_stylesheet())
