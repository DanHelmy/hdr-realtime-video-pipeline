"""
HDRTVNet++ Real-Time Video Pipeline — PyQt6 GUI

Usage:
    python src/gui.py

Features:
    - Browse or drag-and-drop any video file
    - Switch precision (FP16 / FP32 / INT8) at any time, even mid-playback
    - Play / Pause / Stop controls
    - Side-by-side SDR input vs HDR output (or single view)
    - Toggle real-time metrics panel (FPS, latency, GPU/CPU memory, model size)
    - Optional objective metrics with HDR ground truth (PSNR, SSSIM, DeltaEITP, HDR-VDP3 hook)
    - Dark theme
"""

import sys

from gui_bootstrap import prepare_runtime_environment

_HERE, _ROOT = prepare_runtime_environment(__file__)

# Import torch BEFORE PyQt6 — on ROCm-Windows the ROCm SDK DLLs must
# be loaded first; PyQt6 loads its own DLLs which can conflict if torch
# hasn't initialised ROCm yet.
import torch  # noqa: F401

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMainWindow
try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput  # noqa: F401
    _HAS_QT_AUDIO = True
except ImportError:
    _HAS_QT_AUDIO = False

from gui_compare import CompareViewMixin
from gui_audio_mute import AutoMuteMixin
from gui_cursor import CursorBehaviorMixin
from gui_timeline import TimelineSeekMixin
from gui_ground_truth import GroundTruthMixin
from gui_audio_playback import AudioPlaybackMixin
from gui_windowing import WindowingMixin
from gui_lifecycle import LifecycleMixin
from gui_playback_runtime import PlaybackRuntimeMixin
from gui_settings_preview import SettingsPreviewMixin
from gui_worker_slots import WorkerSlotsMixin
from gui_ui_builder import UiBuilderMixin
from gui_signal_wiring import SignalWiringMixin
from gui_state_init import StateInitMixin
from gui_pipeline_worker import PipelineWorker
from gui_app_entry import run_gui

# Precision/resolution config moved to gui_config.py.

# ╔═══════════════════════════════════════════════════════════════╗
# ║  mpv HDR Display (named-pipe writer + embedded widget)        ║
# ╚═══════════════════════════════════════════════════════════════╝

# mpv widget class moved to gui_mpv_widget.py.

# ╔═══════════════════════════════════════════════════════════════╗
# ║  Pipeline Worker Thread                                       ║
# ╚═══════════════════════════════════════════════════════════════╝

# Pipeline worker moved to gui_pipeline_worker.py.

# Widget/dialog classes moved to gui_widgets.py.

# Compile/precompile dialogs moved to gui_compile_dialogs.py.



# Kernel-cache helpers moved to gui_compile_cache.py.
# ╔═══════════════════════════════════════════════════════════════╗
# ║  Main Window                                                  ║
# ╚═══════════════════════════════════════════════════════════════╝

class MainWindow(
    WindowingMixin,
    CompareViewMixin,
    AutoMuteMixin,
    CursorBehaviorMixin,
    TimelineSeekMixin,
    GroundTruthMixin,
    AudioPlaybackMixin,
    UiBuilderMixin,
    StateInitMixin,
    SignalWiringMixin,
    WorkerSlotsMixin,
    SettingsPreviewMixin,
    PlaybackRuntimeMixin,
    LifecycleMixin,
    QMainWindow,
):
    def __init__(self, initial_video=None, initial_resolution=None,
                 initial_precision=None, initial_view=None,
                 initial_use_hg=None,
                 initial_autoplay=False, initial_start_frame=None,
                 initial_upscale=None, initial_film_grain=None,
                 initial_hdr_gt=None):
        super().__init__()
        self.setWindowTitle("HDRTVNet++ — Real-Time SDR → HDR Pipeline")
        self.setMinimumSize(1024, 600)
        self.resize(1600, 900)
        self.setAcceptDrops(True)

        self._worker = PipelineWorker()
        self._init_runtime_state(
            initial_start_frame=initial_start_frame,
            root_dir=_ROOT,
            has_qt_audio=_HAS_QT_AUDIO,
        )

        self._build_ui()
        self._connect_signals()
        self._init_audio_backend()
        self._load_user_settings(
            initial_resolution,
            initial_precision,
            initial_view,
            initial_use_hg,
            initial_upscale,
            initial_film_grain,
            initial_hdr_gt,
        )
        self._init_cursor_idle_tracking()
        QTimer.singleShot(0, self._show_startup_runtime_warnings)

        # Auto-open video passed via --video (used by restart)
        self._queue_initial_video_open(
            initial_video=initial_video,
            initial_resolution=initial_resolution,
            initial_precision=initial_precision,
            initial_view=initial_view,
            initial_autoplay=initial_autoplay,
            initial_upscale=initial_upscale,
        )

    # ── UI construction ──────────────────────────────────────

    # UI construction moved to gui_ui_builder.py

    # Signal wiring moved to gui_signal_wiring.py

    # Settings/preview helpers moved to gui_settings_preview.py

    # ── Slots: file / tools moved to gui_playback_runtime.py ─────

    # Worker signal slots moved to gui_worker_slots.py

# ╔═══════════════════════════════════════════════════════════════╗
# ║  Entry Point                                                  ║
# ╚═══════════════════════════════════════════════════════════════╝

def main():
    sys.exit(run_gui(MainWindow, root_dir=_ROOT, argv=sys.argv))


if __name__ == "__main__":
    main()









