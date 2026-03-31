from __future__ import annotations

import os
import sys
import time
import webbrowser

import cv2
import torch

from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QProgressDialog,
    QDialog,
)

from gui_config import (
    PRECISIONS,
    _select_model_path,
    _available_precision_keys,
    MAX_W,
    MAX_H,
    RESOLUTION_SCALES,
)
from gui_compile_cache import (
    _compiled_marker_path,
    _is_compiled,
    _precision_to_compile_arg,
)
from gui_compile_dialogs import (
    _CompileDialog,
    _PrecompileOptionsDialog,
    _PrecompileDialog,
)
from gui_media_probe import (
    _probe_hdr_input,
    _norm_path,
)
from gui_scaling import (
    BEST_UPSCALE_MODE,
    UPSCALER_CHOICES,
    DEFAULT_UPSCALER,
    _limited_playback_fps,
    _select_hdr_scale_kernel,
    _select_hdr_scale_antiring,
    _select_mpv_cas_strength,
    _normalize_upscale_choice,
    _ensure_fsr_shader,
    _ensure_filmgrain_shader,
)
from gui_widgets import _KernelCacheClearWorker
from windows_runtime import default_cache_root

try:
    from models.hdrtvnet_torch import _HAS_COMPILE, _HAS_HIP_SDK, _HAS_TRITON, _IS_ROCM
except Exception:
    _HAS_COMPILE = False
    _HAS_HIP_SDK = False
    _HAS_TRITON = False
    _IS_ROCM = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_HG_WEIGHTS_PATH = os.path.join(
    _ROOT, "src", "models", "weights", "HG_weights.pth"
)
_LIBMPV_DLL_PATH = os.path.join(_ROOT, "src", "libmpv-2.dll")
_ASSETS_DRIVE_URL = (
    "https://drive.google.com/drive/folders/"
    "1jh8gXBVzqRse-7w_2Dztca1_KVh5eRu1?usp=drive_link"
)
_HIP_SDK_URL = "https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html"


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_predequantize_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m in {"on", "off"}:
        return m
    return "auto"


class PlaybackRuntimeMixin:
    """Playback, restart/apply settings, and compile/file tool flows for MainWindow."""

    def _can_run_autotune_compile(self) -> bool:
        if not torch.cuda.is_available():
            return False
        if not _HAS_COMPILE or not _HAS_TRITON:
            return False
        if os.name == "nt" and _IS_ROCM and not _HAS_HIP_SDK:
            return False
        return True

    def _effective_predequantize_mode_for_precision(
        self,
        precision: str,
        selected_mode: str | None = None,
    ) -> str:
        mode = _normalize_predequantize_mode(
            selected_mode
            if selected_mode is not None
            else getattr(self, "_predequantize_mode", "auto")
        )
        if not str(precision).startswith("int8"):
            return mode
        if mode == "on":
            return "on"
        if mode == "off":
            return "off"
        if _IS_ROCM:
            return "on"
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                has_int8_tc = (
                    props.major > 7
                    or (props.major == 7 and props.minor >= 5)
                )
                return "off" if has_int8_tc else "on"
            except Exception:
                return "auto"
        return "auto"

    def _is_compile_ready_for_runtime(
        self,
        w: int,
        h: int,
        precision: str,
        model_path: str,
        use_hg: bool,
        selected_predequantize_mode: str | None = None,
    ) -> bool:
        effective_pdq_mode = self._effective_predequantize_mode_for_precision(
            precision,
            selected_mode=selected_predequantize_mode,
        )
        if _is_compiled(
            w,
            h,
            precision,
            model_path=model_path,
            use_hg=bool(use_hg),
            predequantize_mode=effective_pdq_mode,
        ):
            return True
        # INT8 with effective pre-dequantize ON reuses FP16 runtime graph
        # shape; treat matching FP16 compile as ready.
        if str(precision).startswith("int8") and effective_pdq_mode == "on":
            fp16_model = _select_model_path("FP16", bool(use_hg))
            if _is_compiled(
                w,
                h,
                "fp16",
                model_path=fp16_model,
                use_hg=bool(use_hg),
                predequantize_mode="auto",
            ):
                return True
        return False

    def _compile_cache_missing_for_any(
        self,
        resolutions: list[str],
        precision: str,
        model_path: str,
        use_hg: bool,
    ) -> bool:
        if not self._can_run_autotune_compile():
            return False
        for res in resolutions:
            try:
                w_s, h_s = str(res).lower().split("x")
                w, h = int(w_s), int(h_s)
            except Exception:
                # If resolution parsing fails, keep warning enabled.
                return True
            if not self._is_compile_ready_for_runtime(
                w,
                h,
                precision,
                model_path=model_path,
                use_hg=bool(use_hg),
            ):
                return True
        return False

    def _show_startup_runtime_warnings(self):
        if not self._enforce_required_clone_assets():
            return
        self._warn_if_hip_sdk_missing_on_rocm_windows()

    def _missing_required_clone_assets(self) -> list[tuple[str, str]]:
        missing: list[tuple[str, str]] = []
        if not os.path.isfile(_LIBMPV_DLL_PATH):
            missing.append(("libmpv-2.dll", _LIBMPV_DLL_PATH))
        if not os.path.isfile(_HG_WEIGHTS_PATH):
            missing.append(("HG_weights.pth", _HG_WEIGHTS_PATH))
        return missing

    def _enforce_required_clone_assets(self) -> bool:
        """Block startup until required external clone assets are present."""
        if not _env_enabled("HDRTVNET_REQUIRE_CLONE_ASSETS", "1"):
            return True

        missing = self._missing_required_clone_assets()
        if not missing:
            return True

        while missing:
            missing_lines = "\n".join(
                [f"- {name}: {path}" for name, path in missing]
            )
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("Required Files Missing")
            box.setText(
                "This Git-clone setup is missing required files, so the app cannot run yet."
            )
            box.setInformativeText(
                "Download from Google Drive, place files exactly at:\n\n"
                "1) libmpv-2.dll -> src/libmpv-2.dll\n"
                "2) HG_weights.pth -> src/models/weights/HG_weights.pth\n\n"
                "Then click Restart App.\n"
                "If files are still missing after restart, this warning will appear again.\n\n"
                f"Missing right now:\n{missing_lines}"
            )
            open_btn = box.addButton(
                "Open Google Drive",
                QMessageBox.ButtonRole.ActionRole,
            )
            restart_btn = box.addButton(
                "Restart App",
                QMessageBox.ButtonRole.AcceptRole,
            )
            exit_btn = box.addButton(
                "Exit",
                QMessageBox.ButtonRole.RejectRole,
            )
            box.setDefaultButton(restart_btn)
            box.setEscapeButton(exit_btn)
            box.exec()

            clicked = box.clickedButton()
            if clicked is open_btn:
                try:
                    webbrowser.open(_ASSETS_DRIVE_URL, new=2)
                except Exception:
                    pass
                missing = self._missing_required_clone_assets()
                continue

            if clicked is restart_btn:
                self._restart_app_clean()
                return False

            self.close()
            return False

        return True

    def _warn_if_hip_sdk_missing_on_rocm_windows(self):
        if getattr(self, "_startup_hip_sdk_warning_shown", False):
            return
        self._startup_hip_sdk_warning_shown = True

        if not _env_enabled("HDRTVNET_WARN_MISSING_HIP_SDK", "1"):
            return
        if os.name != "nt" or not _IS_ROCM:
            return
        if getattr(self, "_suppress_hip_sdk_warning", False):
            return
        if not torch.cuda.is_available():
            return
        hip_sdk_path = r"C:\Program Files\AMD\ROCm"
        if os.path.isdir(hip_sdk_path):
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("AMD HIP SDK Not Detected")
        box.setText(
            f"ROCm-Windows was detected, but AMD HIP SDK was not found at:\n{hip_sdk_path}\n\n"
            "The app can still run, but max-autotune compile may be skipped and "
            "playback performance can be lower.\n\n"
            f"Install HIP SDK for best performance:\n{_HIP_SDK_URL}",
        )
        never_warn = QCheckBox("Do not show this warning again")
        box.setCheckBox(never_warn)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        open_btn = box.addButton(
            "Open HIP SDK Page",
            QMessageBox.ButtonRole.ActionRole,
        )
        while True:
            box.exec()
            if box.clickedButton() is open_btn:
                try:
                    webbrowser.open(_HIP_SDK_URL, new=2)
                except Exception:
                    pass
                continue
            break

        if never_warn.isChecked():
            self._suppress_hip_sdk_warning = True
            try:
                self._save_user_settings()
            except Exception:
                pass

        self.statusBar().showMessage(
            "HIP SDK not detected: compile fallback may reduce AMD performance."
        )

    def _confirm_autotune_precompile_ready(self) -> bool:
        if not getattr(self, "_autotune_warning_needed", False):
            return True
        if not _env_enabled("HDRTVNET_WARN_AUTOTUNE_PREP", "1"):
            return True

        answer = QMessageBox.warning(
            self,
            "First-Time Max-Autotune Compile",
            "This configuration has not been compiled yet.\n\n"
            "Kernel max-autotune is about to run.\n\n"
            "For best kernel selection, close GPU-heavy apps first "
            "(games, video editors, and browser GPU tabs).\n\n"
            "Continue compiling now?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        if answer != QMessageBox.StandardButton.Ok:
            self.statusBar().showMessage("Kernel compile cancelled by user.")
            return False

        return True

    def _predequantize_mode_label(self) -> str:
        mode = _normalize_predequantize_mode(
            getattr(self, "_predequantize_mode", "auto")
        )
        if mode == "on":
            return "On (force pre-dequantize)"
        if mode == "off":
            return "Off (keep runtime dequant)"
        return "Auto (recommended)"

    def _choose_predequantize_mode(self):
        options = [
            "Auto (recommended)",
            "On (force pre-dequantize)",
            "Off (keep runtime dequant)",
        ]
        current_label = self._predequantize_mode_label()
        try:
            current_idx = options.index(current_label)
        except ValueError:
            current_idx = 0

        selected, ok = QInputDialog.getItem(
            self,
            "INT8 Pre-dequantization",
            "Choose pre-dequantization mode:",
            options,
            current_idx,
            False,
        )
        if not ok:
            return

        selected = str(selected or "").strip().lower()
        if selected.startswith("on"):
            mode = "on"
        elif selected.startswith("off"):
            mode = "off"
        else:
            mode = "auto"

        mode = _normalize_predequantize_mode(mode)
        if mode == getattr(self, "_predequantize_mode", "auto"):
            return

        self._predequantize_mode = mode
        try:
            self._save_user_settings()
        except Exception:
            pass

        if self._playing and self._worker is not None:
            active_gui_prec = self._cmb_prec.currentText()
            active_prec_arg = _precision_to_compile_arg(active_gui_prec)
            is_int8_active = str(active_prec_arg).startswith("int8")
            if is_int8_active:
                cur_pw, cur_ph = self._last_res if self._last_res else (MAX_W, MAX_H)
                target_model_path = _select_model_path(
                    active_gui_prec,
                    self._chk_hg.isChecked(),
                )
                if not self._is_compile_ready_for_runtime(
                    cur_pw,
                    cur_ph,
                    active_prec_arg,
                    model_path=target_model_path,
                    use_hg=self._chk_hg.isChecked(),
                    selected_predequantize_mode=mode,
                ):
                    self.statusBar().showMessage(
                        f"INT8 pre-dequantization mode '{mode}' not compiled at "
                        f"{cur_pw}x{cur_ph}; restarting for clean compile."
                    )
                    self._save_user_settings()
                    self._restart_with_video(
                        self._video_path,
                        resolution=self._cmb_res.currentText(),
                        precision=active_gui_prec,
                        view=self._cmb_view.currentText(),
                        use_hg=self._chk_hg.isChecked(),
                        upscale=self._cmb_upscale.currentText()
                        if hasattr(self, "_cmb_upscale")
                        else DEFAULT_UPSCALER,
                        film_grain=self._chk_film_grain.isChecked()
                        if hasattr(self, "_chk_film_grain")
                        else None,
                        hdr_gt=self._hdr_ground_truth_path,
                        autoplay=True,
                        start_frame=int(self._seek_slider.value()),
                    )
                    return
                self._pause_for_precision_swap(active_gui_prec)
                self._worker.request_predequantize_mode(mode)
                self._schedule_precision_audio_resync()
                self.statusBar().showMessage(
                    f"Applying pre-dequantization mode: {mode}"
                )
                return

            # Non-INT8 precision is unaffected by pre-dequantization mode.
            self.statusBar().showMessage(
                f"Pre-dequantization mode set to {mode}. "
                "Will apply when INT8 precision is active."
            )
        else:
            self.statusBar().showMessage(
                f"Pre-dequantization mode set to {mode}. Takes effect on next Play."
            )

    def _schedule_precision_audio_resync(self, delay_ms: int = 240):
        if not self._playing:
            return
        # Precision swaps can stall inference; pause briefly, then re-anchor audio.
        self._audio_resync_pending = True
        self._audio_fps_recovered = False
        if self._audio_available:
            self._set_audio_paused(True)
        now_t = time.perf_counter()
        self._audio_seek_guard_until = max(self._audio_seek_guard_until, now_t + 0.6)

        def _finish():
            if not self._playing:
                return
            self._audio_fps_recovered = True
            self._resync_audio_to_current_timeline()

        QTimer.singleShot(max(0, int(delay_ms)), _finish)

    def _pause_for_precision_swap(self, key: str, timeout_ms: int = 20000):
        if not self._playing or self._worker is None:
            return
        # Track which precision we are waiting to stabilize.
        self._precision_swap_pending = key
        # Only auto-resume if we weren't already paused by the user.
        self._precision_pause_armed = not self._worker.is_paused

        if self._precision_pause_armed:
            self._worker.pause()
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_mpv.set_paused(True)
            if self._disp_sdr_mpv is not None:
                self._disp_sdr_mpv.set_paused(True)
            if self._audio_available:
                self._set_audio_paused(True)

        # Safety timeout in case we miss the "Ready" status message.
        if self._precision_swap_timer is None:
            self._precision_swap_timer = QTimer(self)
            self._precision_swap_timer.setSingleShot(True)
        else:
            self._precision_swap_timer.stop()

        def _timeout_resume():
            self._resume_after_precision_swap(force=True)

        try:
            self._precision_swap_timer.timeout.disconnect()
        except Exception:
            pass
        self._precision_swap_timer.timeout.connect(_timeout_resume)
        self._precision_swap_timer.start(int(timeout_ms))

    def _resume_after_precision_swap(self, force: bool = False):
        if self._precision_swap_pending is None:
            return
        if self._precision_swap_timer is not None:
            self._precision_swap_timer.stop()
        if self._precision_pause_armed:
            if self._worker is not None and self._worker.is_paused:
                self._worker.resume()
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_mpv.set_paused(False)
            if self._disp_sdr_mpv is not None:
                self._disp_sdr_mpv.set_paused(False)
            if self._audio_available and not self._startup_sync_pending:
                # Keep audio paused until FPS stabilizes after the precision swap.
                self._startup_audio_gate_active = True
                self._ui_resync_gate_strict = True
                self._pending_playhead_relock_on_unmute = False
                self._relock_hold_muted = False
                self._arm_mute_until_fps_recovery()
                self._auto_mute_rearm_until = max(
                    float(getattr(self, "_auto_mute_rearm_until", 0.0)),
                    time.perf_counter()
                    + float(getattr(self, "_precision_swap_mute_grace_s", 2.5)),
                )
        self._precision_swap_pending = None
        self._precision_pause_armed = False

    # - Slots: file / tools -----------------------------------

    def _on_compile_ready(self):
        """Called on main thread after Triton compile finishes.
        Safe to start mpv now - GPU is free from autotuning."""
        if self._compile_dlg is not None:
            self._compile_dlg.close()
            self._compile_dlg.deleteLater()
            self._compile_dlg = None

        pending = getattr(self, "_pending_mpv_start", None)
        if pending and self._disp_hdr_mpv is not None:
            (
                pw,
                ph,
                fps,
                scale_kernel,
                scale_antiring,
                cas_strength,
                audio_path,
                film_grain,
                force_hdr_metadata,
            ) = pending
            self._disp_hdr_mpv.start_playback(
                pw,
                ph,
                fps=fps,
                scale_kernel=scale_kernel,
                scale_antiring=scale_antiring,
                cas_strength=cas_strength,
                audio_path=audio_path,
                film_grain=film_grain,
                force_hdr_metadata=force_hdr_metadata,
            )
            # Anchor mpv timeline at 0 on startup to avoid initial drift.
            self._disp_hdr_mpv.seek_seconds(0.0)
            if not self._audio_available:
                self._apply_selected_audio_track_mpv_async()
            self._apply_volume_to_backends()
            if self._startup_sync_pending:
                self._disp_hdr_mpv.set_paused(True)
            self._worker.set_mpv_widget(self._disp_hdr_mpv)
            self._pending_mpv_start = None
            if self._startup_sync_pending:
                QTimer.singleShot(250, self._release_startup_sync)
        pending_sdr = getattr(self, "_pending_sdr_mpv_start", None)
        if pending_sdr and self._disp_sdr_mpv is not None:
            pw, ph, fps, scale_kernel = pending_sdr
            self._disp_sdr_mpv.start_playback(
                pw,
                ph,
                fps=fps,
                scale_kernel=scale_kernel,
                audio_path=None,
                force_hdr_metadata=False,
            )
            if self._startup_sync_pending:
                self._disp_sdr_mpv.set_paused(True)
            self._worker.set_sdr_mpv_widget(self._disp_sdr_mpv)
            self._sdr_mpv_feed_from_worker = True
            self._pending_sdr_mpv_start = None
        self._sync_screen_change_hooks()

    def _precompile_kernels(self):
        """Open the pre-compile dialog - runs compile_kernels.py as a
        completely separate process with zero GPU interference."""

        opts = _PrecompileOptionsDialog(
            initial_precision=self._cmb_prec.currentText(),
            initial_resolution=self._cmb_res.currentText(),
            precision_keys=_available_precision_keys(),
            parent=self,
        )
        if opts.exec() != QDialog.DialogCode.Accepted:
            return

        gui_prec = opts.selected_precision()
        resolutions = opts.selected_resolutions()
        prec_arg = _precision_to_compile_arg(gui_prec)
        model_path = _select_model_path(gui_prec, self._chk_hg.isChecked())
        self._autotune_warning_needed = self._compile_cache_missing_for_any(
            resolutions,
            prec_arg,
            model_path=model_path,
            use_hg=self._chk_hg.isChecked(),
        )
        if not self._confirm_autotune_precompile_ready():
            return

        dlg = _PrecompileDialog(
            resolutions,
            precision=prec_arg,
            model_path=model_path,
            use_hg=self._chk_hg.isChecked(),
            hg_weights=_HG_WEIGHTS_PATH if os.path.isfile(_HG_WEIGHTS_PATH) else None,
            clear_cache=False,
            predequantize_mode=_normalize_predequantize_mode(
                getattr(self, "_predequantize_mode", "auto")
            ),
            parent=self,
        )
        dlg.exec()  # modal - blocks until user closes
        if not dlg.succeeded:
            return

    def _clear_kernel_cache(self):
        """Delete kernel cache folders or selected precision/resolution compile entries."""
        import getpass
        import pathlib
        import re
        import tempfile

        triton_root = pathlib.Path(
            os.environ.get("TRITON_CACHE_DIR", os.path.join(default_cache_root(), "triton"))
        )
        triton_dir = triton_root / "cache"

        all_cache_dirs = []
        if triton_dir.is_dir() and any(triton_dir.iterdir()):
            all_cache_dirs.append(triton_dir)

        inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if not inductor_dir:
            inductor_dir = os.path.join(
                tempfile.gettempdir(),
                f"torchinductor_{getpass.getuser()}",
            )
        inductor_path = pathlib.Path(inductor_dir)
        if inductor_path.is_dir() and any(inductor_path.iterdir()):
            all_cache_dirs.append(inductor_path)

        marker_path = _compiled_marker_path()

        def _read_marker_lines() -> list[str]:
            if not marker_path.is_file():
                return []
            try:
                return [
                    ln.strip()
                    for ln in marker_path.read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                ]
            except Exception:
                return []

        def _write_marker_lines(lines: list[str]) -> bool:
            try:
                if lines:
                    marker_path.parent.mkdir(parents=True, exist_ok=True)
                    marker_path.write_text(
                        "\n".join(sorted(set(lines))) + "\n",
                        encoding="utf-8",
                    )
                elif marker_path.exists():
                    marker_path.unlink()
                return True
            except Exception:
                return False

        marker_lines = _read_marker_lines()
        key_re = re.compile(
            r"^(?P<w>\d+)x(?P<h>\d+)_(?P<precision>[^_]+)_hg[01]_[^_]+_"
            r"(?:(?:auto|on|off)_)?(?:[A-Za-z0-9-]+)$"
        )
        marker_groups: dict[tuple[str, int, int], set[str]] = {}
        for line in marker_lines:
            m = key_re.match(line)
            if not m:
                continue
            precision = m.group("precision")
            w = int(m.group("w"))
            h = int(m.group("h"))
            marker_groups.setdefault((precision, w, h), set()).add(line)

        target_options: list[tuple[str, tuple[str, set[str] | None]]] = []
        if all_cache_dirs or marker_lines:
            target_options.append(("Full wipe (all caches)", ("wipe_all", None)))
        for (precision, w, h), keys in sorted(
            marker_groups.items(),
            key=lambda item: (item[0][0], -item[0][1], -item[0][2]),
        ):
            count = len(keys)
            plural = "entry" if count == 1 else "entries"
            label = f"{precision} @ {w}x{h} ({count} compile {plural})"
            target_options.append((label, ("markers", set(keys))))

        if not target_options:
            QMessageBox.information(
                self,
                "Kernel Cache",
                "No kernel cache entries found.",
            )
            return

        if len(target_options) > 1:
            option_labels = [label for label, _payload in target_options]
            selected_label, ok = QInputDialog.getItem(
                self,
                "Clear Kernel Cache",
                "Choose what to clear:",
                option_labels,
                0,
                False,
            )
            if not ok:
                return
            selected_payload = next(
                (payload for label, payload in target_options if label == selected_label),
                None,
            )
            if selected_payload is None:
                return
        else:
            selected_payload = target_options[0][1]

        mode, payload = selected_payload
        if mode == "markers":
            keys_to_remove = set(payload or set())
            if not keys_to_remove:
                return
            msg = (
                "This will delete compile cache records for the selected "
                "precision/resolution.\n\n"
                "Only matching entries will be removed.\n"
                "Continue?"
            )
            btn = QMessageBox.question(
                self,
                "Clear Kernel Cache",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if btn != QMessageBox.StandardButton.Yes:
                return

            before = _read_marker_lines()
            remaining = [ln for ln in before if ln not in keys_to_remove]
            if not _write_marker_lines(remaining):
                QMessageBox.warning(
                    self,
                    "Kernel Cache",
                    "Failed to update compile cache records.",
                )
                return

            removed = len(before) - len(remaining)
            if removed <= 0:
                QMessageBox.information(
                    self,
                    "Kernel Cache",
                    "No matching compile cache records were found.",
                )
            else:
                QMessageBox.information(
                    self,
                    "Kernel Cache",
                    f"Removed {removed} compile cache record(s).",
                )
            return

        dirs = list(all_cache_dirs)
        if dirs and self._worker is not None and self._worker.isRunning():
            msg = (
                "Kernel cache cannot be cleared while playback is running.\n\n"
                "Stop playback and clear cache now?"
            )
            btn = QMessageBox.question(
                self,
                "Clear Kernel Cache",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if btn != QMessageBox.StandardButton.Yes:
                return
            self._worker.stop()
            self._worker.wait(10000)

        if dirs:
            target_lines = "\n".join(f"  {d}" for d in dirs)
            msg = (
                "This will wipe all kernel caches and compile records:\n\n"
                + target_lines
                + "\n\nKernels will be recompiled on next playback.\n"
                "Continue?"
            )
        else:
            msg = (
                "This will wipe all compile cache records.\n\n"
                "Kernels will be recompiled on next playback.\n"
                "Continue?"
            )
        btn = QMessageBox.question(
            self,
            "Clear Kernel Cache",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if btn != QMessageBox.StandardButton.Yes:
            return

        if dirs:
            dlg = QProgressDialog("Clearing kernel cache...", None, 0, 0, self)
            dlg.setWindowTitle("Clear Kernel Cache")
            dlg.setMinimumDuration(0)
            dlg.setCancelButton(None)
            dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
            dlg.show()

            self._cache_clear_thread = QThread(self)
            self._cache_clear_worker = _KernelCacheClearWorker(dirs)
            self._cache_clear_worker.moveToThread(self._cache_clear_thread)

            def _on_finished(ok: bool):
                dlg.close()
                self._cache_clear_worker = None
                self._cache_clear_thread = None
                marker_ok = _write_marker_lines([])
                if ok and marker_ok:
                    QMessageBox.information(
                        self,
                        "Kernel Cache",
                        "All kernel caches were cleared.",
                    )
                elif ok:
                    QMessageBox.warning(
                        self,
                        "Kernel Cache",
                        "Kernel folders were cleared, but compile records could "
                        "not be fully reset.",
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Kernel Cache",
                        "Cache clear completed with errors. "
                        "Some cache files may remain.",
                    )

            self._cache_clear_thread.started.connect(self._cache_clear_worker.run)
            self._cache_clear_worker.finished.connect(_on_finished)
            self._cache_clear_worker.finished.connect(self._cache_clear_thread.quit)
            self._cache_clear_worker.finished.connect(
                self._cache_clear_worker.deleteLater
            )
            self._cache_clear_thread.finished.connect(
                self._cache_clear_thread.deleteLater
            )
            self._cache_clear_thread.start()
            return

        if _write_marker_lines([]):
            QMessageBox.information(
                self,
                "Kernel Cache",
                "All compile cache records were cleared.",
            )
        else:
            QMessageBox.warning(
                self,
                "Kernel Cache",
                "Failed to clear compile cache records.",
            )

    def _open_file(self):
        start_dir = self._last_open_dir if os.path.isdir(self._last_open_dir) else _ROOT
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            start_dir,
            "Video (*.mp4 *.avi *.mkv *.mov *.webm *.flv);;All (*)",
        )
        if path:
            try:
                open_dir = os.path.dirname(path)
                if open_dir and os.path.isdir(open_dir):
                    self._last_open_dir = open_dir
                    self._save_user_settings()
            except Exception:
                pass
            self._set_video(path)

    def _set_video(self, path, auto_play: bool = False):
        candidate_hdr = _probe_hdr_input(path)
        if bool(candidate_hdr.get("is_hdr", False)):
            reason = str(candidate_hdr.get("reason", "HDR metadata detected")).strip()
            QMessageBox.warning(
                self,
                "Unsupported Input",
                "HDR input videos are not supported for conversion.\n\n"
                f"Selected file appears HDR ({reason}).\n"
                "Please select an SDR source video.",
            )
            self.statusBar().showMessage(
                "Rejected HDR input video. Please choose an SDR source."
            )
            return

        old_norm = _norm_path(self._video_path)
        new_norm = _norm_path(path)
        video_changed = bool(old_norm and new_norm and old_norm != new_norm)

        # Stop current playback if running
        if self._playing:
            self._stop()

        if video_changed:
            self._reset_hdr_ground_truth(
                "HDR GT reset because the input video changed. Select a matching HDR GT file."
            )

        # If playback has already been started once in this process,
        # always restart before loading another video to keep compile/
        # runtime state clean and deterministic.
        if self._last_res is not None:
            self._restart_with_video(
                path,
                resolution=self._cmb_res.currentText(),
                precision=self._cmb_prec.currentText(),
                view=self._cmb_view.currentText(),
                upscale=self._cmb_upscale.currentText()
                if hasattr(self, "_cmb_upscale")
                else None,
                film_grain=self._chk_film_grain.isChecked()
                if hasattr(self, "_chk_film_grain")
                else None,
                hdr_gt=self._hdr_ground_truth_path,
                autoplay=auto_play,
            )
            return

        self._video_path = path
        try:
            vdir = os.path.dirname(path)
            if vdir and os.path.isdir(vdir):
                self._last_open_dir = vdir
        except Exception:
            pass
        self._save_user_settings()
        self._refresh_resolution_options_for_video(path)
        self._refresh_audio_tracks_for_video(path)
        self._lbl_file.setText(os.path.basename(path))
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_compare.setEnabled(False)
        self._btn_apply_settings.setEnabled(False)
        self._prepare_idle_timeline(path)
        self._show_idle_preview(path)
        self.setWindowTitle(f"HDRTVNet++ - {os.path.basename(path)}")
        self.statusBar().showMessage(
            f"Selected: {path} - preview loaded. Press Play to start."
        )
        if auto_play:
            QTimer.singleShot(100, self._play)

    def _restart_with_video(
        self,
        path,
        resolution=None,
        precision=None,
        view=None,
        use_hg=None,
        upscale=None,
        film_grain=None,
        hdr_gt=None,
        autoplay=False,
        start_frame=None,
    ):
        """Restart the GUI process with a new video.

        A fresh process avoids stale torch.compile/dynamo state that
        causes slow in-process re-tracing when the resolution changes.
        """
        self.statusBar().showMessage("Restarting for new resolution ...")
        self._suppress_eof_restart_once = True
        # Clean shutdown
        if self._playing:
            self._worker.stop()
            self._worker.wait(5000)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.stop_playback()
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.stop_playback()

        # Hide the parent window so the user doesn't see two GUIs
        self.hide()
        QApplication.instance().processEvents()

        # Re-exec with --video
        # The parent must wait for the child so the shell stays blocked
        # the entire time. Otherwise the prompt appears immediately and
        # the child's output overwrites it, leaving no usable prompt
        # after the child exits.
        import subprocess as _sp

        args = [sys.executable, sys.argv[0], "--video", path]
        if resolution in RESOLUTION_SCALES:
            args += ["--resolution", resolution]
        elif resolution == "Source":
            args += ["--resolution", "Source"]
        if precision in PRECISIONS:
            args += ["--precision", precision]
        if view == "Tabbed":
            args += ["--view", view]
        if use_hg is not None:
            args += ["--use-hg", "1" if use_hg else "0"]
        if isinstance(upscale, str) and upscale in UPSCALER_CHOICES:
            args += ["--upscale", upscale]
        if film_grain is not None:
            args += ["--film-grain", "1" if film_grain else "0"]
        if isinstance(hdr_gt, str) and hdr_gt.strip():
            args += ["--hdr-gt", hdr_gt.strip()]
        if autoplay:
            args += ["--autoplay", "1"]
        if start_frame is not None:
            args += ["--start-frame", str(max(0, int(start_frame)))]
        rc = _sp.call(args)
        sys.exit(rc)

    # - Slots: playback ---------------------------------------

    def _play(self):
        if self._playing or not self._video_path:
            return

        src_probe = _probe_hdr_input(self._video_path)
        if bool(src_probe.get("is_hdr", False)):
            reason = str(src_probe.get("reason", "HDR metadata detected")).strip()
            QMessageBox.warning(
                self,
                "Unsupported Input",
                "HDR input videos are not supported for conversion.\n\n"
                f"This file appears HDR ({reason}).\n"
                "Please select an SDR source video.",
            )
            self.statusBar().showMessage("Cannot start conversion: input video is HDR.")
            return

        if self._hdr_ground_truth_path:
            ok, note = self._validate_hdr_ground_truth(
                self._hdr_ground_truth_path,
                source_path=self._video_path,
            )
            if not ok:
                bad_name = os.path.basename(self._hdr_ground_truth_path)
                self._reset_hdr_ground_truth(f"HDR GT cleared ({bad_name}): {note}")
            self._objective_metrics_enabled = False
        else:
            self._objective_metrics_enabled = False
            self._update_hdr_ground_truth_label()

        self._source_hdr_info = _probe_hdr_input(self._video_path)

        # Determine processing resolution
        cap = cv2.VideoCapture(self._video_path)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vfps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Output (display) resolution: always 1080p (letterbox handles aspect).
        ow, oh = MAX_W, MAX_H
        self._cur_output_w, self._cur_output_h = ow, oh

        # Processing resolution from scale selector (fixed preset; letterbox handles aspect).
        scale_key = self._cmb_res.currentText()
        scale_dims = RESOLUTION_SCALES.get(scale_key)
        if scale_key == "Source" and self._source_proc_dims is not None:
            scale_dims = self._source_proc_dims
        if scale_dims is not None and (scale_dims[0] < ow or scale_dims[1] < oh):
            pw, ph = scale_dims
        else:
            pw, ph = ow, oh

        # Select upscale kernel choice (only allowed for 540p/720p presets)
        upscale_choice = DEFAULT_UPSCALER
        if scale_key in {"540p", "720p"}:
            upscale_choice = self._cmb_upscale.currentText() or DEFAULT_UPSCALER
        # Resolve FSR availability up-front so we don't report it if it can't load.
        if (
            _normalize_upscale_choice(upscale_choice) == "fsr"
            and not _ensure_fsr_shader()
        ):
            self.statusBar().showMessage(
                "FSR shader unavailable (download failed). Falling back to EWA LanczosSharp."
            )
            upscale_choice = "EWA LanczosSharp"

        # Set up seek slider
        self._vid_fps = vfps if vfps > 0 else 30.0
        display_fps = _limited_playback_fps(self._vid_fps)
        self._seek_slider.setRange(0, max(0, total_frames - 1))
        self._seek_slider.setValue(0)
        self._seek_slider.setEnabled(True)
        self._seek_slider.setToolTip("Seek while paused is queued and applied on Resume.")
        self._lbl_time.setText("0:00")
        dur_secs = total_frames / self._vid_fps if self._vid_fps > 0 else 0
        self._lbl_duration.setText(self._fmt_time(dur_secs))

        # Map GUI precision to compile arg
        gui_prec = self._cmb_prec.currentText()
        prec_arg = _precision_to_compile_arg(gui_prec)

        # Always compile via a clean subprocess - this ensures autotune
        # benchmarks have zero GPU interference from Qt / D3D11 / mpv.
        # If the Triton + Inductor cache is already warm from a previous
        # compile, the subprocess finishes in seconds and auto-closes.
        model_path = _select_model_path(gui_prec, self._chk_hg.isChecked())
        self._autotune_warning_needed = self._compile_cache_missing_for_any(
            [f"{pw}x{ph}"],
            prec_arg,
            model_path=model_path,
            use_hg=self._chk_hg.isChecked(),
        )
        if not self._confirm_autotune_precompile_ready():
            return
        dlg = _PrecompileDialog(
            [f"{pw}x{ph}"],
            precision=prec_arg,
            model_path=model_path,
            use_hg=self._chk_hg.isChecked(),
            hg_weights=_HG_WEIGHTS_PATH if os.path.isfile(_HG_WEIGHTS_PATH) else None,
            predequantize_mode=_normalize_predequantize_mode(
                getattr(self, "_predequantize_mode", "auto")
            ),
            parent=self,
        )
        dlg.exec()  # modal - blocks until done
        if not dlg.succeeded:
            # Compile failed or user closed early - don't start playback
            return

        # Start playback
        self._last_res = (pw, ph)
        self._playing = True
        self._active_precision = self._cmb_prec.currentText()
        self._active_resolution = self._cmb_res.currentText()
        self._active_use_hg = self._chk_hg.isChecked()
        self._active_upscale_mode = upscale_choice
        # Keep mpv initialized whenever available so view switches are UI-only.
        use_mpv_pipeline = self._disp_hdr_mpv is not None
        self._active_use_mpv = use_mpv_pipeline
        self._startup_sync_pending = bool(use_mpv_pipeline)
        self._mpv_start_resync_t = 0.0
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_stack.setCurrentWidget(
                self._disp_sdr_mpv if use_mpv_pipeline else self._disp_sdr_cpu
            )
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_stack.setCurrentWidget(
                self._disp_hdr_mpv if use_mpv_pipeline else self._disp_hdr_cpu
            )
        self._update_apply_button_state()
        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_stop.setEnabled(True)
        self._btn_compare.setEnabled(True)
        self._btn_file.setEnabled(False)
        if self._btn_toggle_ui is not None:
            self._btn_toggle_ui.setEnabled(True)
        self._cmb_prec.setEnabled(True)
        self._set_pause_button_labels(False)

        # Start mpv HDR display AFTER compile finishes (via signal)
        # so that mpv's D3D11 GPU usage doesn't pollute Triton autotuning.
        # mpv receives frames at processing resolution; GPU scaling happens in mpv.
        self._pending_mpv_start = None
        self._pending_sdr_mpv_start = None
        if use_mpv_pipeline and self._disp_hdr_mpv is not None:
            mpv_audio_path = None if self._audio_available else self._video_path
            self._active_mpv_scale_kernel = _select_hdr_scale_kernel(
                pw, ph, ow, oh, upscale_choice
            )
            self._active_mpv_scale_antiring = _select_hdr_scale_antiring(
                pw, ph, ow, oh, self._active_mpv_scale_kernel
            )
            using_fsr = self._active_mpv_scale_kernel == "fsr"
            self._active_mpv_cas = _select_mpv_cas_strength(
                pw, ph, ow, oh, using_fsr, self._active_mpv_scale_kernel
            )
            self._active_film_grain = bool(
                hasattr(self, "_chk_film_grain") and self._chk_film_grain.isChecked()
            )
            if self._active_film_grain and not _ensure_filmgrain_shader():
                self.statusBar().showMessage(
                    "Film grain shader unavailable (download failed)."
                )
                self._active_film_grain = False
                if (
                    hasattr(self, "_chk_film_grain")
                    and self._chk_film_grain is not None
                ):
                    self._chk_film_grain.blockSignals(True)
                    self._chk_film_grain.setChecked(False)
                    self._chk_film_grain.blockSignals(False)
            self._pending_mpv_start = (
                pw,
                ph,
                float(display_fps),
                self._active_mpv_scale_kernel,
                self._active_mpv_scale_antiring,
                self._active_mpv_cas,
                mpv_audio_path,
                self._active_film_grain,
                True,
            )
            if self._disp_sdr_mpv is not None:
                self._pending_sdr_mpv_start = (ow, oh, float(display_fps), "bicubic")
        else:
            self._worker.set_mpv_widget(None)
            self._worker.set_sdr_mpv_widget(None)
            self._sdr_mpv_feed_from_worker = False

        self._worker.configure(
            self._video_path,
            self._cmb_prec.currentText(),
            proc_w=pw,
            proc_h=ph,
            output_w=ow,
            output_h=oh,
            input_is_hdr=False,
            use_hg=self._chk_hg.isChecked(),
            predequantize_mode=_normalize_predequantize_mode(
                getattr(self, "_predequantize_mode", "auto")
            ),
            objective_metrics_enabled=self._objective_metrics_enabled,
            hdr_ground_truth_path=self._hdr_ground_truth_path,
        )

        # Show loading dialog (in-process model load + cache warmup is fast
        # since subprocess already compiled the kernels)
        self._compile_dlg = _CompileDialog(self)
        self._compile_dlg.show()

        self._worker.start()
        self._set_process_priority(True)
        if pw != ow or ph != oh:
            upscale_backend = "mpv GPU" if use_mpv_pipeline else "CPU fallback"
            self.statusBar().showMessage(
                f"Upscale active: {pw}x{ph} -> {ow}x{oh} via {BEST_UPSCALE_MODE} ({upscale_backend})"
            )
        else:
            self.statusBar().showMessage(f"No upscale stage: processing at {ow}x{oh}.")
        self._post_seek_resync_frames = 0
        self._pending_seek_on_resume = None
        if self._startup_sync_pending:
            self._worker.pause()
        # Startup audio gate: release audio only after FPS stabilizes.
        self._startup_audio_gate_active = True
        self._scrub_muted = True
        self._arm_mute_until_fps_recovery()
        if self._audio_available:
            self._start_audio_playback(self._video_path)
            self._set_audio_paused(True)
        else:
            self.statusBar().showMessage(
                "Qt audio backend unavailable; using mpv audio fallback (seek sync may be limited)."
            )

        # Restore timeline position after process restart (resolution change).
        if self._startup_seek_frame is not None:
            target = max(
                0, min(int(self._startup_seek_frame), self._seek_slider.maximum())
            )
            self._worker.request_seek(target)
            self._seek_slider.setValue(target)
            self._lbl_time.setText(self._fmt_time(target / max(self._vid_fps, 1e-6)))
            self._seek_audio_seconds(target / max(self._vid_fps, 1e-6))
            if self._disp_hdr_mpv is not None and not self._audio_available:
                self._disp_hdr_mpv.seek_seconds(target / max(self._vid_fps, 1e-6))
        self._startup_seek_frame = None
        self._arm_cursor_idle_timer()
        self._start_periodic_relock()

    def _toggle_pause(self):
        if not self._playing:
            return
        if self._worker.is_paused:
            self._user_pause_override_startup = False
            queued = self._pending_seek_on_resume
            if queued is not None:
                self._worker.request_seek(int(queued))
                fps = getattr(self, "_vid_fps", 30.0)
                self._seek_audio_seconds(int(queued) / max(fps, 1e-6))
                if self._disp_hdr_mpv is not None and not self._audio_available:
                    self._disp_hdr_mpv.seek_seconds(int(queued) / max(fps, 1e-6))
                now_t = time.perf_counter()
                self._audio_seek_guard_until = now_t + 1.0
                self._audio_resync_pending = True
                self._audio_fps_recovered = False
                self._post_seek_resync_frames = 120
                self._resume_audio_after_seek = bool(self._audio_available)
                self._seek_resume_target = int(queued)
                self._seek_resume_started_t = time.perf_counter()
                if self._audio_available:
                    QTimer.singleShot(420, self._ensure_selected_audio_track_qt)
                self._pending_seek_on_resume = None
            self._worker.resume()
            self._set_pause_button_labels(False)
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_mpv.set_paused(False)
            if self._disp_sdr_mpv is not None:
                self._disp_sdr_mpv.set_paused(False)
            if not self._resume_audio_after_seek:
                audio_release_blocked = (
                    self._startup_audio_gate_active
                    or self._auto_muted_low_fps
                    or self._scrub_muted
                    or self._relock_hold_muted
                    or self._pending_playhead_relock_on_unmute
                )
                self._set_audio_paused(bool(audio_release_blocked))
            if self._active_use_mpv:
                if queued is not None:
                    self._request_playhead_skip_relock_after_unmute(
                        first_delay_ms=20, settle_delay_ms=180
                    )
                else:
                    self._relock_timeline(delay_ms=60, drop_frames=2)
                if queued is None:

                    def _resume_video_resync():
                        if not self._playing or self._worker.is_paused:
                            return
                        fps = getattr(self, "_vid_fps", 30.0)
                        target_frame = int(self._last_seek_frame)
                        resume_dt = time.perf_counter() - float(
                            self._last_user_pause_t or 0.0
                        )
                        if self._audio_available and self._audio_player is not None:
                            try:
                                have_ms = int(self._audio_player.position())
                                target_frame = int(
                                    round((have_ms / 1000.0) * max(fps, 1e-6))
                                )
                            except Exception:
                                target_frame = int(self._last_seek_frame)
                        if resume_dt < 0.6:
                            return
                        if abs(target_frame - int(self._last_seek_frame)) >= 6:
                            self._worker.request_seek(int(target_frame))
                            if (
                                self._disp_hdr_mpv is not None
                                and not self._audio_available
                            ):
                                self._disp_hdr_mpv.seek_seconds(
                                    int(target_frame) / max(fps, 1e-6)
                                )
                            self._audio_seek_guard_until = time.perf_counter() + 1.0
                            self._audio_resync_pending = True
                            self._audio_fps_recovered = False
                            self._post_seek_resync_frames = 60

                    QTimer.singleShot(80, _resume_video_resync)
            self._arm_cursor_idle_timer()
        else:
            self._worker.pause()
            self._last_user_pause_t = time.perf_counter()
            self._set_pause_button_labels(True)
            if self._disp_hdr_mpv is not None:
                self._disp_hdr_mpv.set_paused(True)
            if self._disp_sdr_mpv is not None:
                self._disp_sdr_mpv.set_paused(True)
            self._set_audio_paused(True)
            if self._startup_sync_pending:
                self._user_pause_override_startup = True
            if self._cursor_idle_timer is not None:
                self._cursor_idle_timer.stop()
            self._show_cursor()

    def _stop(self):
        self._suppress_eof_restart_once = True
        self._worker.stop()
        self._worker.wait(10000)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.stop_playback()
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.stop_playback()
        self._stop_audio_playback()
        self._set_process_priority(False)
        # Clear the current video so user must choose a file again.
        self._video_path = None
        self._source_hdr_info = {"is_hdr": False, "reason": "unknown"}
        # No active input video => clear GT binding too.
        self._reset_hdr_ground_truth()
        if self._lbl_file is not None:
            self._lbl_file.setText("No video selected")
        if self._lbl_duration is not None:
            self._lbl_duration.setText("0:00")
        self._reset_controls()

    def _restart_app_clean(self):
        self.statusBar().showMessage("Restarting app ...")
        self._save_user_settings()
        self._suppress_eof_restart_once = True
        # Clean shutdown
        if self._playing:
            self._worker.stop()
            self._worker.wait(5000)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.stop_playback()
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.stop_playback()
        self._stop_audio_playback()

        # Hide the parent window so the user doesn't see two GUIs
        self.hide()
        QApplication.instance().processEvents()

        import subprocess as _sp

        args = [sys.executable, sys.argv[0]]
        rc = _sp.call(args)
        sys.exit(rc)

    def _stop_and_restart(self):
        self._stop()
        self._restart_app_clean()

    # - Slots: settings ---------------------------------------

    def _on_precision(self, key):
        self._chk_hg.setEnabled(True)
        if self._playing:
            self._update_apply_button_state()

    def _on_resolution(self, scale_key):
        self._sync_upscale_controls()
        if self._playing:
            self._update_apply_button_state()

    def _on_upscale_changed(self, _mode: str):
        self._sync_upscale_controls()
        if self._playing:
            self._update_apply_button_state()

    def _on_film_grain_changed(self, _state):
        if self._playing:
            self._update_apply_button_state()
        self._save_user_settings()

    def _on_hg_toggle(self, _state):
        if self._playing:
            self._update_apply_button_state()

    def _apply_runtime_settings(self):
        if not self._playing or not self._video_path:
            return
        if not self._has_pending_setting_changes():
            self.statusBar().showMessage("No pending setting changes.")
            return

        new_prec = self._cmb_prec.currentText()
        new_res = self._cmb_res.currentText()
        current_upscale = (
            self._cmb_upscale.currentText()
            if hasattr(self, "_cmb_upscale")
            else DEFAULT_UPSCALER
        )
        new_upscale = DEFAULT_UPSCALER
        if new_res in {"540p", "720p"}:
            new_upscale = current_upscale or DEFAULT_UPSCALER
        upscale_changed = new_upscale != self._active_upscale_mode
        film_grain_changed = False
        if hasattr(self, "_chk_film_grain") and self._chk_film_grain is not None:
            film_grain_changed = (
                self._chk_film_grain.isChecked() != self._active_film_grain
            )
        needs_restart = new_res != self._active_resolution
        if self._chk_hg.isChecked() != self._active_use_hg:
            needs_restart = True
        notices: list[str] = []

        def _apply_mpv_hot_swap(
            action, pause_ms: int = 260, relock_ms: int = 260, relock_drop: int = 2
        ) -> bool:
            if self._worker is not None and not self._worker.is_paused:
                self._pause_for_ui_transition(duration_ms=pause_ms, wait_for_stable=True)
            ok = bool(action())
            if ok and self._playing:
                self._relock_timeline(delay_ms=relock_ms, drop_frames=relock_drop)
            return ok

        if needs_restart:
            self._save_user_settings()
            self._restart_with_video(
                self._video_path,
                resolution=new_res,
                precision=new_prec,
                view=self._cmb_view.currentText(),
                use_hg=self._chk_hg.isChecked(),
                upscale=current_upscale,
                film_grain=self._chk_film_grain.isChecked()
                if hasattr(self, "_chk_film_grain")
                else None,
                hdr_gt=self._hdr_ground_truth_path,
                autoplay=True,
                start_frame=int(self._seek_slider.value()),
            )
            return

        def _apply_film_grain_toggle() -> bool:
            if not film_grain_changed:
                return True
            if not self._active_use_mpv or self._disp_hdr_mpv is None:
                self.statusBar().showMessage(
                    "Film grain requires mpv; keeping previous setting."
                )
                if (
                    hasattr(self, "_chk_film_grain")
                    and self._chk_film_grain is not None
                ):
                    self._chk_film_grain.blockSignals(True)
                    self._chk_film_grain.setChecked(self._active_film_grain)
                    self._chk_film_grain.blockSignals(False)
                return False
            enabled = bool(self._chk_film_grain.isChecked())
            if enabled and not _ensure_filmgrain_shader():
                self.statusBar().showMessage(
                    "Film grain shader unavailable (download failed)."
                )
                if (
                    hasattr(self, "_chk_film_grain")
                    and self._chk_film_grain is not None
                ):
                    self._chk_film_grain.blockSignals(True)
                    self._chk_film_grain.setChecked(False)
                    self._chk_film_grain.blockSignals(False)
                return False
            # Match upscale behavior: brief pause + relock around shader swap.
            ok = _apply_mpv_hot_swap(lambda: self._disp_hdr_mpv.set_film_grain(enabled))
            if ok:
                self._active_film_grain = enabled
                self._save_user_settings()
                self.statusBar().showMessage(
                    "Film grain enabled." if enabled else "Film grain disabled."
                )
            else:
                self.statusBar().showMessage(
                    "Film grain hot-swap failed; keeping previous setting."
                )
                if (
                    hasattr(self, "_chk_film_grain")
                    and self._chk_film_grain is not None
                ):
                    self._chk_film_grain.blockSignals(True)
                    self._chk_film_grain.setChecked(self._active_film_grain)
                    self._chk_film_grain.blockSignals(False)
            return ok

        if upscale_changed:
            if not self._active_use_mpv or self._disp_hdr_mpv is None:
                self.statusBar().showMessage(
                    "Upscale mode change requires mpv GPU pipeline; keeping previous setting."
                )
                if hasattr(self, "_cmb_upscale"):
                    self._cmb_upscale.blockSignals(True)
                    self._cmb_upscale.setCurrentText(self._active_upscale_mode)
                    self._cmb_upscale.blockSignals(False)
                return
            # Pause both SDR/HDR + audio briefly so the hot-swap doesn't
            # desync the two pipelines.
            cur_pw, cur_ph = self._last_res if self._last_res else (MAX_W, MAX_H)
            ow, oh = self._cur_output_w, self._cur_output_h
            kernel = _select_hdr_scale_kernel(cur_pw, cur_ph, ow, oh, new_upscale)
            antiring = _select_hdr_scale_antiring(cur_pw, cur_ph, ow, oh, kernel)

            def _apply_upscale_hot_swap() -> bool:
                if not self._disp_hdr_mpv.set_scale_kernel(kernel, antiring):
                    return False
                self._active_mpv_scale_kernel = kernel
                self._active_mpv_scale_antiring = antiring
                self._active_mpv_cas = _select_mpv_cas_strength(
                    cur_pw,
                    cur_ph,
                    ow,
                    oh,
                    using_fsr=(kernel == "fsr"),
                    scale_kernel=kernel,
                )
                self._disp_hdr_mpv.set_cas_strength(self._active_mpv_cas)
                self._active_upscale_mode = new_upscale

                def _announce():
                    mode_label = str(self._active_upscale_mode or "")
                    using_shader = kernel == "fsr" or "ssim" in str(kernel).lower()
                    self.statusBar().showMessage(
                        f"Upscale hot-swap: {mode_label} ({'shader active' if using_shader else 'kernel active'})"
                    )

                QTimer.singleShot(80, _announce)
                self._save_user_settings()
                return True

            if not _apply_mpv_hot_swap(_apply_upscale_hot_swap):
                err = getattr(self._disp_hdr_mpv, "_last_scale_error", None)
                if err:
                    self.statusBar().showMessage(f"Upscale hot-swap failed: {err}")
                else:
                    self.statusBar().showMessage(
                        "Upscale hot-swap failed; keeping previous setting."
                    )
            _apply_film_grain_toggle()
            self._update_apply_button_state()
            return

        if new_prec != self._active_precision:
            cur_pw, cur_ph = self._last_res if self._last_res else (MAX_W, MAX_H)
            target_prec_arg = _precision_to_compile_arg(new_prec)
            target_model_path = _select_model_path(new_prec, self._chk_hg.isChecked())
            if not self._is_compile_ready_for_runtime(
                cur_pw,
                cur_ph,
                target_prec_arg,
                model_path=target_model_path,
                use_hg=self._chk_hg.isChecked(),
                selected_predequantize_mode=getattr(self, "_predequantize_mode", "auto"),
            ):
                self.statusBar().showMessage(
                    f"Precision {new_prec} not precompiled at {cur_pw}x{cur_ph}; restarting for clean compile."
                )
                self._save_user_settings()
                self._restart_with_video(
                    self._video_path,
                    resolution=new_res,
                    precision=new_prec,
                    view=self._cmb_view.currentText(),
                    use_hg=self._chk_hg.isChecked(),
                    upscale=current_upscale,
                    film_grain=self._chk_film_grain.isChecked()
                    if hasattr(self, "_chk_film_grain")
                    else None,
                    hdr_gt=self._hdr_ground_truth_path,
                    autoplay=True,
                    start_frame=int(self._seek_slider.value()),
                )
                return
            self._pause_for_precision_swap(new_prec)
            self._worker.request_precision_change(new_prec)
            if self._playing:
                self._schedule_precision_audio_resync()
            notices.append(f"Applying precision change: {new_prec}")
            self._active_precision = new_prec

        _apply_film_grain_toggle()

        self._active_use_hg = self._chk_hg.isChecked()
        self._active_upscale_mode = new_upscale

        self._active_resolution = new_res
        self._save_user_settings()
        self._update_apply_button_state()
        if notices:
            self.statusBar().showMessage(" | ".join(notices))
