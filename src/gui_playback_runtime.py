from __future__ import annotations

import os
import sys
import time
import webbrowser

import cv2
import torch

from PyQt6.QtCore import Qt, QEventLoop, QProcess, QProcessEnvironment, QThread, QTimer
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
    LIVE_CAPTURE_DISPLAY_FPS,
    PRECISIONS,
    SOURCE_MODE_VIDEO,
    SOURCE_MODE_WINDOW,
    _select_model_path,
    _available_precision_keys,
    _normalize_capture_fps_label,
    _normalize_source_mode,
    MAX_W,
    MAX_H,
    RESOLUTION_SCALES,
    _processing_preset_dims,
    _source_is_below_processing_preset,
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
from gui_export import ExportOptionsDialog, VideoExportWorker
from gui_media_probe import (
    _probe_hdr_input,
    _norm_path,
)
from required_clone_assets import (
    ensure_required_clone_assets,
    manual_assets_drive_url,
    missing_required_clone_assets,
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
from gui_capture_dialogs import WindowCaptureDialog
from windows_runtime import project_cache_root
from window_capture_source import (
    WindowCaptureTarget,
    capture_target_to_cli_args,
    probe_window_capture_target,
    resolve_window_capture_target,
)

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
_HIP_SDK_URL = "https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html"


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_predequantize_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m in {"on", "off"}:
        return m
    return "auto"


def _normalize_runtime_execution_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m == "eager":
        return "eager"
    return "compile"


class PlaybackRuntimeMixin:
    """Playback, restart/apply settings, and compile/file tool flows for MainWindow."""

    def _stop_active_browser_tab_session(self) -> None:
        return

    _EXPORT_LOCK_WIDGET_NAMES = (
        "_btn_file",
        "_btn_play",
        "_btn_pause",
        "_btn_stop",
        "_btn_compare",
        "_btn_apply_settings",
        "_seek_slider",
        "_cmb_source_mode",
        "_cmb_capture_fps",
        "_cmb_prec",
        "_chk_hg",
        "_cmb_res",
        "_cmb_upscale",
        "_chk_film_grain",
        "_btn_hdr_gt",
        "_sld_volume",
        "_cmb_audio_track",
        "_btn_pop_sdr",
        "_btn_pop_hdr",
        "_btn_toggle_ui",
        "_cmb_view",
    )

    def _export_controls_locked(self) -> bool:
        return bool(getattr(self, "_export_interaction_locked", False))

    def _show_export_lock_message(self, action: str = "Playback") -> bool:
        if not self._export_controls_locked():
            return False
        self.statusBar().showMessage(
            f"{action} is locked while export is running. Finish or cancel the export first."
        )
        return True

    def _set_export_interaction_locked(self, locked: bool):
        if bool(locked):
            if self._export_controls_locked():
                return
            saved = {}
            for name in self._EXPORT_LOCK_WIDGET_NAMES:
                widget = getattr(self, name, None)
                if widget is None:
                    continue
                try:
                    saved[name] = bool(widget.isEnabled())
                    widget.setEnabled(False)
                except Exception:
                    continue
            self._export_saved_enabled_states = saved
            self._export_interaction_locked = True
            self.statusBar().showMessage(
                "Playback controls are locked while export is running."
            )
            return

        saved = dict(getattr(self, "_export_saved_enabled_states", {}) or {})
        self._export_saved_enabled_states = {}
        self._export_interaction_locked = False
        for name, was_enabled in saved.items():
            widget = getattr(self, name, None)
            if widget is None:
                continue
            try:
                widget.setEnabled(bool(was_enabled))
            except Exception:
                continue
        if self._playing:
            self._btn_apply_settings.setEnabled(self._has_pending_setting_changes())
        else:
            self._btn_apply_settings.setEnabled(False)

    def _can_run_autotune_compile(self) -> bool:
        if not torch.cuda.is_available():
            return False
        if not _HAS_COMPILE or not _HAS_TRITON:
            return False
        if os.name == "nt" and _IS_ROCM and not _HAS_HIP_SDK:
            return False
        return True

    def _runtime_execution_mode_uses_compile(self) -> bool:
        return _normalize_runtime_execution_mode(
            getattr(self, "_runtime_execution_mode", "compile")
        ) == "compile"

    def _runtime_execution_mode_label(self) -> str:
        mode = _normalize_runtime_execution_mode(
            getattr(self, "_runtime_execution_mode", "compile")
        )
        if mode == "eager":
            return "Eager (not recommended)"
        return "Compile (recommended)"

    def _choose_runtime_execution_mode(self):
        options = ["Compile (recommended)", "Eager (not recommended)"]
        current_label = self._runtime_execution_mode_label()
        try:
            current_idx = options.index(current_label)
        except ValueError:
            current_idx = 0

        selected, ok = QInputDialog.getItem(
            self,
            "Runtime Execution Mode",
            "Choose runtime execution mode:",
            options,
            current_idx,
            False,
        )
        if not ok:
            return

        mode = (
            "eager"
            if str(selected or "").strip().lower().startswith("eager")
            else "compile"
        )
        mode = _normalize_runtime_execution_mode(mode)
        if mode == getattr(self, "_runtime_execution_mode", "compile"):
            return

        self._runtime_execution_mode = mode
        try:
            self._save_user_settings()
        except Exception:
            pass

        if self._playing and self._worker is not None:
            self.statusBar().showMessage(
                f"Runtime execution mode -> {mode}. Restarting playback to apply."
            )
            self._restart_with_active_source(
                resolution=self._cmb_res.currentText(),
                precision=self._cmb_prec.currentText(),
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
                start_frame=int(self._seek_slider.value()) if hasattr(self, "_seek_slider") else 0,
            )
            return

        self.statusBar().showMessage(
            f"Runtime execution mode set to {self._runtime_execution_mode_label()}."
        )

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
        selected_pdq_mode = _normalize_predequantize_mode(
            selected_predequantize_mode
            if selected_predequantize_mode is not None
            else getattr(self, "_predequantize_mode", "auto")
        )
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
        if (
            str(precision).startswith("int8")
            and selected_pdq_mode == "auto"
            and effective_pdq_mode != "auto"
        ):
            # Backward compatibility: older compile subprocesses wrote the
            # raw "auto" marker even when runtime auto-resolved to on/off.
            if _is_compiled(
                w,
                h,
                precision,
                model_path=model_path,
                use_hg=bool(use_hg),
                predequantize_mode="auto",
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

    def _effective_precompile_predequantize_mode(
        self,
        precision: str,
        selected_mode: str | None = None,
    ) -> str:
        return self._effective_predequantize_mode_for_precision(
            precision,
            selected_mode=selected_mode,
        )

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

    def _runtime_compile_verified_keys(self) -> set[tuple]:
        verified = getattr(self, "_runtime_compile_verified", None)
        if not isinstance(verified, set):
            verified = set()
            self._runtime_compile_verified = verified
        return verified

    def _runtime_compile_verify_key(
        self,
        w: int,
        h: int,
        precision: str,
        model_path: str,
        use_hg: bool,
        *,
        selected_predequantize_mode: str | None = None,
    ) -> tuple:
        effective_pdq = self._effective_precompile_predequantize_mode(
            precision,
            selected_mode=selected_predequantize_mode,
        )
        return (
            os.path.normcase(project_cache_root(__file__)),
            int(w),
            int(h),
            str(precision),
            os.path.normcase(os.path.abspath(model_path)),
            bool(use_hg),
            str(effective_pdq),
        )

    def _mark_runtime_compile_verified(
        self,
        w: int,
        h: int,
        precision: str,
        model_path: str,
        use_hg: bool,
        *,
        selected_predequantize_mode: str | None = None,
    ) -> None:
        self._runtime_compile_verified_keys().add(
            self._runtime_compile_verify_key(
                w,
                h,
                precision,
                model_path,
                use_hg,
                selected_predequantize_mode=selected_predequantize_mode,
            )
        )

    def _clear_runtime_compile_verified(self) -> None:
        self._runtime_compile_verified = set()

    def _current_project_kernel_cache_targets(self):
        import pathlib

        cache_root = project_cache_root(__file__)
        triton_root = pathlib.Path(
            os.environ.get("TRITON_CACHE_DIR", os.path.join(cache_root, "triton"))
        )
        inductor_root = pathlib.Path(
            os.environ.get(
                "TORCHINDUCTOR_CACHE_DIR",
                os.path.join(cache_root, "torchinductor"),
            )
        )
        marker_path = _compiled_marker_path()
        return [triton_root / "cache", inductor_root], marker_path

    def _clear_current_project_kernel_cache_now(self) -> bool:
        import shutil

        dirs, marker_path = self._current_project_kernel_cache_targets()
        ok = True
        for d in dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                ok = False
        try:
            if marker_path.exists():
                marker_path.unlink()
        except Exception:
            ok = False
        self._clear_runtime_compile_verified()
        return ok

    def _recover_incompatible_runtime_compile_cache(
        self,
        *,
        w: int,
        h: int,
        precision: str,
        model_path: str,
        use_hg: bool,
        selected_predequantize_mode: str | None = None,
        timed_out: bool,
        process_output: str = "",
        workflow_name: str = "Playback",
    ) -> bool:
        reason = (
            "Cached-kernel verification timed out before warmup completed."
            if timed_out
            else "Cached-kernel verification failed before warmup completed."
        )
        details = str(process_output or "").strip()
        detail_tail = ""
        if details:
            lines = [ln.strip() for ln in details.splitlines() if ln.strip()]
            if lines:
                detail_tail = "\n\nLast log lines:\n" + "\n".join(lines[-6:])

        answer = QMessageBox.warning(
            self,
            "Kernel Cache Looks Incompatible",
            f"{workflow_name} detected a cached-kernel problem for {w}x{h} / {precision}.\n\n"
            f"{reason}\n\n"
            "Clear this local project's kernel cache and recompile now?"
            f"{detail_tail}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage(
                "Cached-kernel verification failed. Playback/export was canceled."
            )
            return False

        progress = QProgressDialog("Clearing this project's kernel cache ...", None, 0, 0, self)
        progress.setWindowTitle("Clear Kernel Cache")
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.show()
        QApplication.processEvents()
        try:
            cleared_ok = self._clear_current_project_kernel_cache_now()
        finally:
            progress.close()
            QApplication.processEvents()
        if not cleared_ok:
            QMessageBox.warning(
                self,
                "Kernel Cache",
                "Failed to fully clear this project's kernel cache.",
            )
            return False

        dlg = _PrecompileDialog(
            [f"{int(w)}x{int(h)}"],
            precision=precision,
            model_path=model_path,
            use_hg=bool(use_hg),
            hg_weights=_HG_WEIGHTS_PATH if os.path.isfile(_HG_WEIGHTS_PATH) else None,
            predequantize_mode=self._effective_precompile_predequantize_mode(
                precision,
                selected_predequantize_mode,
            ),
            parent=self,
        )
        dlg.exec()
        if not dlg.succeeded:
            self.statusBar().showMessage(
                "Kernel recompile was canceled after clearing the cache."
            )
            return False

        self._mark_runtime_compile_verified(
            w,
            h,
            precision,
            model_path,
            use_hg,
            selected_predequantize_mode=selected_predequantize_mode,
        )
        return True

    def _ensure_runtime_compile_cache_usable(
        self,
        *,
        w: int,
        h: int,
        precision: str,
        model_path: str,
        use_hg: bool,
        selected_predequantize_mode: str | None = None,
        workflow_name: str = "Playback",
    ) -> bool:
        if not self._can_run_autotune_compile():
            return True
        if not self._is_compile_ready_for_runtime(
            w,
            h,
            precision,
            model_path=model_path,
            use_hg=bool(use_hg),
            selected_predequantize_mode=selected_predequantize_mode,
        ):
            return True

        verify_key = self._runtime_compile_verify_key(
            w,
            h,
            precision,
            model_path,
            use_hg,
            selected_predequantize_mode=selected_predequantize_mode,
        )
        if verify_key in self._runtime_compile_verified_keys():
            return True

        progress = QProgressDialog("Verifying cached kernels ...", None, 0, 0, self)
        progress.setWindowTitle("Verify Kernel Cache")
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.show()
        QApplication.processEvents()

        script = os.path.join(_HERE, "compile_kernels.py")
        args = [
            script,
            f"{int(w)}x{int(h)}",
            "--precision",
            str(precision),
            "--model",
            str(model_path),
            "--use-hg",
            "1" if bool(use_hg) else "0",
            "--predequantize",
            self._effective_precompile_predequantize_mode(
                precision,
                selected_predequantize_mode,
            ),
            "--verify-cache-only",
        ]
        if os.path.isfile(_HG_WEIGHTS_PATH):
            args += ["--hg-weights", _HG_WEIGHTS_PATH]

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = process.processEnvironment()
        if env.isEmpty():
            env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        if os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
            env.insert("TORCHINDUCTOR_CACHE_DIR", os.environ["TORCHINDUCTOR_CACHE_DIR"])
        if os.environ.get("TRITON_CACHE_DIR"):
            env.insert("TRITON_CACHE_DIR", os.environ["TRITON_CACHE_DIR"])
        process.setProcessEnvironment(env)

        output_chunks: list[str] = []

        def _drain_output() -> None:
            data = process.readAllStandardOutput()
            if not data:
                return
            output_chunks.append(bytes(data).decode("utf-8", errors="replace"))

        process.readyReadStandardOutput.connect(_drain_output)

        loop = QEventLoop(self)
        timeout = QTimer(self)
        timeout.setSingleShot(True)
        state = {"timed_out": False}

        def _on_timeout() -> None:
            state["timed_out"] = True
            try:
                if process.state() != QProcess.ProcessState.NotRunning:
                    process.kill()
            except Exception:
                pass
            loop.quit()

        process.finished.connect(loop.quit)
        timeout.timeout.connect(_on_timeout)

        process.start(sys.executable, ["-u"] + args)
        if not process.waitForStarted(5000):
            progress.close()
            QApplication.processEvents()
            return self._recover_incompatible_runtime_compile_cache(
                w=w,
                h=h,
                precision=precision,
                model_path=model_path,
                use_hg=use_hg,
                selected_predequantize_mode=selected_predequantize_mode,
                timed_out=False,
                process_output="Failed to start cache verification subprocess.",
                workflow_name=workflow_name,
            )

        timeout.start(45000)
        loop.exec()
        timeout.stop()
        _drain_output()
        if process.state() != QProcess.ProcessState.NotRunning:
            try:
                process.kill()
            except Exception:
                pass
            process.waitForFinished(3000)
        progress.close()
        QApplication.processEvents()

        exit_ok = (
            not state["timed_out"]
            and process.exitStatus() == QProcess.ExitStatus.NormalExit
            and process.exitCode() == 0
        )
        process_output = "".join(output_chunks).strip()
        process.deleteLater()
        timeout.deleteLater()

        if exit_ok:
            self._runtime_compile_verified_keys().add(verify_key)
            return True

        return self._recover_incompatible_runtime_compile_cache(
            w=w,
            h=h,
            precision=precision,
            model_path=model_path,
            use_hg=use_hg,
            selected_predequantize_mode=selected_predequantize_mode,
            timed_out=bool(state["timed_out"]),
            process_output=process_output,
            workflow_name=workflow_name,
        )

    def _show_startup_runtime_warnings(self):
        if not self._enforce_required_clone_assets():
            return
        self._warn_if_hip_sdk_missing_on_rocm_windows()

    def _missing_required_clone_assets(self) -> list[tuple[str, str, str]]:
        missing: list[tuple[str, str, str]] = []
        for asset in missing_required_clone_assets(_ROOT):
            missing.append(
                (
                    asset.name,
                    str(asset.target_path(_ROOT)),
                    asset.drive_url,
                )
            )
        return missing

    def _try_auto_download_required_clone_assets(self) -> tuple[bool, list[str]]:
        if not _env_enabled("HDRTVNET_AUTO_DOWNLOAD_CLONE_ASSETS", "1"):
            return False, ["Automatic asset download is disabled by environment."]

        dlg = QProgressDialog("Downloading required files ...", None, 0, 0, self)
        dlg.setWindowTitle("Downloading Required Files")
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.show()
        QApplication.processEvents()

        def _progress(message: str):
            dlg.setLabelText(message)
            QApplication.processEvents()

        try:
            results = ensure_required_clone_assets(_ROOT, progress=_progress)
        finally:
            dlg.close()
            QApplication.processEvents()

        errors = [
            f"- {result.asset.name}: {result.detail}"
            for result in results
            if result.status == "failed"
        ]
        return len(self._missing_required_clone_assets()) == 0, errors

    def _launch_setup_and_exit(self) -> bool:
        setup_bat = os.path.join(_ROOT, "setup.bat")
        try:
            os.startfile(setup_bat)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Setup Launch Failed",
                f"Could not open setup.bat automatically.\n\n{exc}",
            )
            return False
        QMessageBox.information(
            self,
            "Setup Started",
            "Setup has been opened in a new window.\n\n"
            "Finish setup, then launch the GUI again.",
        )
        self.close()
        return False

    def _enforce_required_clone_assets(self) -> bool:
        """Block startup until required external clone assets are present."""
        if not _env_enabled("HDRTVNET_REQUIRE_CLONE_ASSETS", "1"):
            return True

        missing = self._missing_required_clone_assets()
        if not missing:
            return True

        auto_ok, auto_errors = self._try_auto_download_required_clone_assets()
        if auto_ok:
            return True
        missing = self._missing_required_clone_assets()

        while missing:
            missing_lines = "\n".join(
                [f"- {name}: {path}" for name, path, _url in missing]
            )
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("Required Files Missing")
            box.setText(
                "This Git-clone setup is missing required files, so the app cannot run yet."
            )
            auto_error_text = ""
            if auto_errors:
                auto_error_text = (
                    "\nAutomatic download did not complete:\n"
                    + "\n".join(auto_errors)
                    + "\n"
                )
            box.setInformativeText(
                "The app already tried to download the missing files automatically.\n\n"
                "Required locations:\n\n"
                "1) libmpv-2.dll -> src/libmpv-2.dll\n"
                "2) HG_weights.pth -> src/models/weights/HG_weights.pth\n\n"
                "Use Retry to try the auto-download again, Run Setup to refresh the environment, "
                "or open the Google Drive links and place the files manually."
                f"{auto_error_text}\n"
                f"Missing right now:\n{missing_lines}"
            )
            retry_btn = box.addButton(
                "Retry Download / Re-check",
                QMessageBox.ButtonRole.AcceptRole,
            )
            open_btn = box.addButton(
                "Open Google Drive",
                QMessageBox.ButtonRole.ActionRole,
            )
            setup_btn = box.addButton(
                "Run Setup",
                QMessageBox.ButtonRole.AcceptRole,
            )
            exit_btn = box.addButton(
                "Exit",
                QMessageBox.ButtonRole.RejectRole,
            )
            box.setDefaultButton(retry_btn)
            box.setEscapeButton(exit_btn)
            box.exec()

            clicked = box.clickedButton()
            if clicked is open_btn:
                try:
                    opened_any = False
                    for _name, _path, url in missing:
                        if url:
                            webbrowser.open(url, new=2)
                            opened_any = True
                    if not opened_any:
                        webbrowser.open(manual_assets_drive_url(), new=2)
                except Exception:
                    pass
                missing = self._missing_required_clone_assets()
                continue

            if clicked is retry_btn:
                auto_ok, auto_errors = self._try_auto_download_required_clone_assets()
                if auto_ok:
                    return True
                missing = self._missing_required_clone_assets()
                continue

            if clicked is setup_btn:
                return self._launch_setup_and_exit()

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
            "If max-autotune still gives poor kernel picks, a fresh PC restart "
            "can help clear leftover GPU/driver state before compiling again.\n\n"
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
                if self._runtime_execution_mode_uses_compile():
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
                        self._restart_with_active_source(
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
        self._note_live_audio_compile_ready()
        self._sync_screen_change_hooks()

    def _precompile_kernels(self):
        """Open the pre-compile dialog - runs compile_kernels.py as a
        completely separate process with zero GPU interference."""
        if self._playing or (self._worker is not None and self._worker.isRunning()):
            QMessageBox.information(
                self,
                "Pre-compile Kernels",
                "Stop playback before pre-compiling kernels.\n\n"
                "This avoids GPU interference and ensures clean autotune results.",
            )
            return

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
            predequantize_mode=self._effective_precompile_predequantize_mode(
                prec_arg,
                getattr(self, "_predequantize_mode", "auto"),
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
        if self._playing or (self._worker is not None and self._worker.isRunning()):
            QMessageBox.information(
                self,
                "Clear Kernel Cache",
                "Stop playback before clearing the kernel cache.\n\n"
                "This prevents mid-playback stalls and avoids GPU contention.",
            )
            return

        triton_root = pathlib.Path(
            os.environ.get(
                "TRITON_CACHE_DIR",
                os.path.join(project_cache_root(__file__), "triton"),
            )
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
            r"^(?:(?P<ns>[^:]+):)?(?P<w>\d+)x(?P<h>\d+)_(?P<precision>[^_]+)_hg[01]_[^_]+_"
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
                self._clear_runtime_compile_verified()
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
                self._clear_runtime_compile_verified()
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
            self._clear_runtime_compile_verified()
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
        action_name = "Choose browser window" if self._source_mode == SOURCE_MODE_WINDOW else "Open video"
        if self._show_export_lock_message(action_name):
            return
        if self._source_mode == SOURCE_MODE_WINDOW:
            self._open_window_capture_dialog()
            return
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

    def _open_window_capture_dialog(self):
        current_session_id = None
        target = getattr(self, "_capture_target", None)
        if target is not None:
            try:
                current_session_id = str(getattr(target, "session_id", "") or "").strip() or None
            except Exception:
                current_session_id = None
        dlg = WindowCaptureDialog(
            initial_session_id=current_session_id,
            initial_target=target if isinstance(target, WindowCaptureTarget) else None,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        target = dlg.selected_target()
        if target is None:
            QMessageBox.information(
                self,
                "Browser Window Capture",
                "Choose a visible browser window first.",
            )
            return
        self._set_window_capture_source(target)

    def _open_export_dialog(self):
        if self._source_mode == SOURCE_MODE_WINDOW:
            QMessageBox.information(
                self,
                "Export Unavailable",
                "Export is only available in Video Player mode.\n\n"
                "Browser Window Capture is a live viewer and does not export the source window.",
            )
            return
        if self._export_thread is not None:
            QMessageBox.information(
                self,
                "Export Running",
                "An export is already running. Wait for it to finish or cancel it first.",
            )
            return

        if self._playing:
            answer = QMessageBox.question(
                self,
                "Export While Playing",
                "Playback will be paused while the export dialog is open.\n\n"
                "If you start a normal export, playback will stay paused and "
                "locked for the export run.\n\n"
                "If you start an export with experimental max-autotune enabled, "
                "the app may use the same full Stop behavior before export begins so kernel "
                "compile/warmup can run cleanly.\n\n"
                "Export uses the same GPU/model stack as playback, so this keeps "
                "the export path stable.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        resume_after_dialog = bool(self._playing and not self._worker.is_paused)
        if resume_after_dialog:
            self._toggle_pause()

        start_dir = self._last_export_dir if os.path.isdir(self._last_export_dir) else _ROOT
        config = None
        try:
            dlg = ExportOptionsDialog(
                initial_source_path=self._video_path if self._video_path else None,
                suggested_dir=start_dir,
                initial_precision_key=self._cmb_prec.currentText(),
                initial_use_hg=self._chk_hg.isChecked(),
                parent=self,
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return

            config = dlg.export_config()
            if config is None:
                return

            try:
                export_dir = os.path.dirname(config.output_path)
                if export_dir and os.path.isdir(export_dir):
                    self._last_export_dir = export_dir
            except Exception:
                pass
            self._save_user_settings()
        finally:
            if (
                config is None
                and resume_after_dialog
                and self._playing
                and self._worker.is_paused
            ):
                self._toggle_pause()

        self._start_export_job(config, resume_if_aborted=resume_after_dialog)

    def _prepare_export_compile_cache(self, config) -> bool:
        if not getattr(config, "use_max_autotune", False):
            return True

        prec_arg = _precision_to_compile_arg(config.precision_key)
        model_path = _select_model_path(config.precision_key, config.use_hg)
        selected_pdq_mode = _normalize_predequantize_mode(
            getattr(config, "predequantize_mode", "auto")
        )
        compile_ready = self._is_compile_ready_for_runtime(
            int(config.width),
            int(config.height),
            prec_arg,
            model_path=model_path,
            use_hg=bool(config.use_hg),
            selected_predequantize_mode=selected_pdq_mode,
        )
        self._autotune_warning_needed = not compile_ready
        if compile_ready:
            return self._ensure_runtime_compile_cache_usable(
                w=int(config.width),
                h=int(config.height),
                precision=prec_arg,
                model_path=model_path,
                use_hg=bool(config.use_hg),
                selected_predequantize_mode=selected_pdq_mode,
                workflow_name="Export",
            )
        if not self._confirm_autotune_precompile_ready():
            return False

        dlg = _PrecompileDialog(
            [f"{int(config.width)}x{int(config.height)}"],
            precision=prec_arg,
            model_path=model_path,
            use_hg=bool(config.use_hg),
            hg_weights=_HG_WEIGHTS_PATH if os.path.isfile(_HG_WEIGHTS_PATH) else None,
            predequantize_mode=self._effective_precompile_predequantize_mode(
                prec_arg,
                selected_pdq_mode,
            ),
            parent=self,
        )
        dlg.exec()
        if dlg.succeeded:
            self._mark_runtime_compile_verified(
                int(config.width),
                int(config.height),
                prec_arg,
                model_path,
                bool(config.use_hg),
                selected_predequantize_mode=selected_pdq_mode,
            )
            return True

        self.statusBar().showMessage("Export kernel compile was canceled.")
        return False

    def _request_cancel_export(self):
        if self._export_worker is not None:
            self._export_worker.cancel()
        if self._export_progress_dlg is not None:
            self._export_progress_dlg.setLabelText("Canceling export ...")
            self._export_progress_dlg.setCancelButton(None)
        if self._export_compile_dlg is not None:
            self._export_compile_dlg.set_status("Canceling export ...")

    def _create_export_progress_dialog(self):
        if self._export_progress_dlg is not None:
            return self._export_progress_dlg

        progress = QProgressDialog("Preparing export ...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Export Video")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.canceled.connect(self._request_cancel_export)
        progress.show()
        self._export_progress_dlg = progress
        return progress

    def _begin_export_job(self, config, resume_if_aborted: bool = False):
        if self._export_thread is not None:
            return

        if self._playing and self._worker is not None and not self._worker.is_paused:
            self._toggle_pause()
        self._set_export_interaction_locked(True)
        if not self._prepare_export_compile_cache(config):
            self._set_export_interaction_locked(False)
            if (
                resume_if_aborted
                and self._playing
                and self._worker is not None
                and self._worker.is_paused
            ):
                self._toggle_pause()
            return
        if config.use_max_autotune:
            self._export_compile_dlg = _CompileDialog(self)
            self._export_compile_dlg.set_status("Loading export model ...")
            self._export_compile_dlg.show()
        else:
            self._create_export_progress_dialog()

        thread = QThread(self)
        worker = VideoExportWorker(config)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.compile_ready.connect(self._on_export_compile_ready)
        worker.progress.connect(self._on_export_progress)
        worker.finished.connect(self._on_export_finished)
        worker.failed.connect(self._on_export_failed)
        worker.canceled.connect(self._on_export_canceled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.canceled.connect(thread.quit)
        thread.finished.connect(self._cleanup_export_job)

        self._export_thread = thread
        self._export_worker = worker
        thread.start()
        self.statusBar().showMessage(
            "Export started. Playback controls are locked until export finishes."
        )

    def _start_export_job(self, config, resume_if_aborted: bool = False):
        if self._export_thread is not None:
            return

        had_active_playback = bool(
            self._playing and self._worker is not None and self._worker.isRunning()
        )

        if had_active_playback and getattr(config, "use_max_autotune", False):
            answer = QMessageBox.question(
                self,
                "Export Autotune Needs Full Stop",
                "Max-autotune export should stop playback first, then start export.\n\n"
                "This uses the same full Stop behavior before any compile/warmup "
                "work begins, which avoids MIOpen/ROCm conflicts from leaving the "
                "playback model and mpv pipeline alive on the GPU.\n\n"
                "Playback will be stopped now, then export will begin.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                if (
                    resume_if_aborted
                    and self._playing
                    and self._worker is not None
                    and self._worker.is_paused
                ):
                    self._toggle_pause()
                return
            self._stop()
            QTimer.singleShot(
                0,
                lambda cfg=config: self._begin_export_job(
                    cfg, resume_if_aborted=False
                ),
            )
            return

        self._begin_export_job(config, resume_if_aborted=resume_if_aborted)

    def _on_export_compile_ready(self):
        if self._export_compile_dlg is not None:
            self._export_compile_dlg.close()
            self._export_compile_dlg.deleteLater()
            self._export_compile_dlg = None
        self._create_export_progress_dialog()

    def _on_export_progress(self, percent: int, message: str):
        if self._export_progress_dlg is not None:
            self._export_progress_dlg.setLabelText(message)
            self._export_progress_dlg.setValue(max(0, min(100, int(percent))))
        elif self._export_compile_dlg is not None:
            self._export_compile_dlg.set_status(message)
        self.statusBar().showMessage(message)

    def _on_export_finished(self, output_path: str):
        if self._export_progress_dlg is not None:
            self._export_progress_dlg.setValue(100)
            self._export_progress_dlg.setLabelText("Export complete.")
        self.statusBar().showMessage(f"Export finished: {output_path}")
        QMessageBox.information(
            self,
            "Export Complete",
            f"Finished exporting:\n{output_path}",
        )

    def _on_export_failed(self, message: str):
        self.statusBar().showMessage("Export failed.")
        QMessageBox.warning(
            self,
            "Export Failed",
            str(message or "The export failed."),
        )

    def _on_export_canceled(self, message: str):
        self.statusBar().showMessage("Export canceled.")
        if message:
            QMessageBox.information(
                self,
                "Export Canceled",
                message,
            )

    def _cleanup_export_job(self):
        if self._export_compile_dlg is not None:
            self._export_compile_dlg.close()
            self._export_compile_dlg.deleteLater()
            self._export_compile_dlg = None
        if self._export_progress_dlg is not None:
            self._export_progress_dlg.close()
            self._export_progress_dlg.deleteLater()
            self._export_progress_dlg = None
        if self._export_worker is not None:
            self._export_worker.deleteLater()
            self._export_worker = None
        if self._export_thread is not None:
            self._export_thread.deleteLater()
            self._export_thread = None
        self._set_export_interaction_locked(False)

    def _set_video(self, path, auto_play: bool = False):
        if self._show_export_lock_message("Loading a new video"):
            return
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
        self._capture_target = None
        self._source_mode = SOURCE_MODE_VIDEO
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
        self._refresh_source_mode_ui()
        self.setWindowTitle(f"HDRTVNet++ - {os.path.basename(path)}")
        self.statusBar().showMessage(
            f"Selected: {path} - preview loaded. Press Play to start."
        )
        if auto_play:
            QTimer.singleShot(100, self._play)

    def _set_window_capture_source(self, target: WindowCaptureTarget, auto_play: bool = False):
        if self._show_export_lock_message("Choosing a browser window"):
            return
        if target is None:
            return

        preview, src_w, src_h = probe_window_capture_target(target)
        if preview is None or src_w <= 0 or src_h <= 0:
            QMessageBox.warning(
                self,
                "Browser Window Capture",
                "Could not capture frames from the selected browser window.\n\n"
                "Make sure the window is visible on screen, then try again.",
            )
            return

        if self._playing:
            self._stop()

        self._source_mode = SOURCE_MODE_WINDOW
        self._video_path = None
        self._capture_target = WindowCaptureTarget(
            title=str(target.title or "").strip(),
            process_name=str(target.process_name or "").strip(),
            pid=int(getattr(target, "pid", 0) or 0),
            width=int(src_w),
            height=int(src_h),
            hwnd=int(getattr(target, "hwnd", 0) or 0),
            session_id=str(getattr(target, "session_id", "") or "").strip(),
            browser_name=str(getattr(target, "browser_name", "") or "").strip(),
            source_url=str(getattr(target, "source_url", "") or "").strip(),
        )
        self._capture_fps_value = float(LIVE_CAPTURE_DISPLAY_FPS)
        self._reset_hdr_ground_truth()

        if self._last_res is not None:
            self._restart_with_capture_source(
                self._capture_target,
                resolution=self._cmb_res.currentText(),
                precision=self._cmb_prec.currentText(),
                view=self._cmb_view.currentText(),
                use_hg=self._chk_hg.isChecked(),
                upscale=self._cmb_upscale.currentText()
                if hasattr(self, "_cmb_upscale")
                else None,
                film_grain=self._chk_film_grain.isChecked()
                if hasattr(self, "_chk_film_grain")
                else None,
                autoplay=auto_play,
            )
            return

        self._save_user_settings()
        self._set_source_resolution_options_for_dims(src_w, src_h)
        self._prepare_live_timeline(LIVE_CAPTURE_DISPLAY_FPS)
        self._show_idle_preview_frame(preview)
        self._refresh_source_mode_ui()
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_compare.setEnabled(False)
        self._btn_apply_settings.setEnabled(False)
        self.setWindowTitle(f"HDRTVNet++ - {self._capture_target.label}")
        has_tab_audio_sync = bool(str(getattr(self._capture_target, "session_id", "") or "").strip())
        self.statusBar().showMessage(
            (
                f"Selected live browser window: {self._capture_target.label}. "
                "Experimental Chrome-only mode. Chrome's 'Use graphics acceleration when available' must be off. Press Play to start video. If Chrome Audio Sync is active, the extension will delay and play the tab audio locally while HDRTVNet++ stays silent."
            )
            if has_tab_audio_sync
            else (
                f"Selected live browser window: {self._capture_target.label}. "
                "Experimental Chrome-only mode. Chrome's 'Use graphics acceleration when available' must be off. Start Chrome Audio Sync in the extension before Play if you want delayed local browser audio. Without it, Chrome keeps playing audio locally and it can lead the video."
            )
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
        try:
            from browser_tab_bridge import close_browser_tab_bridge

            close_browser_tab_bridge()
        except Exception:
            pass

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

    def _restart_with_capture_source(
        self,
        target: WindowCaptureTarget,
        resolution=None,
        precision=None,
        view=None,
        use_hg=None,
        upscale=None,
        film_grain=None,
        autoplay=False,
        start_frame=None,
    ):
        if target is None:
            return
        self.statusBar().showMessage("Restarting for new capture settings ...")
        self._suppress_eof_restart_once = True
        if self._playing:
            self._worker.stop()
            self._worker.wait(5000)
        if self._disp_hdr_mpv is not None:
            self._disp_hdr_mpv.stop_playback()
        if self._disp_sdr_mpv is not None:
            self._disp_sdr_mpv.stop_playback()
        self.hide()
        QApplication.instance().processEvents()
        try:
            from browser_tab_bridge import close_browser_tab_bridge

            close_browser_tab_bridge()
        except Exception:
            pass

        import subprocess as _sp

        args = [sys.executable, sys.argv[0], "--source-mode", SOURCE_MODE_WINDOW]
        args += capture_target_to_cli_args(target)
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
        if autoplay:
            args += ["--autoplay", "1"]
        if start_frame is not None:
            args += ["--start-frame", str(max(0, int(start_frame)))]
        rc = _sp.call(args)
        sys.exit(rc)

    def _restart_with_active_source(
        self,
        *,
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
        if self._source_mode == SOURCE_MODE_WINDOW:
            if self._capture_target is None:
                return
            self._restart_with_capture_source(
                self._capture_target,
                resolution=resolution,
                precision=precision,
                view=view,
                use_hg=use_hg,
                upscale=upscale,
                film_grain=film_grain,
                autoplay=autoplay,
                start_frame=start_frame,
            )
            return
        if self._video_path:
            self._restart_with_video(
                self._video_path,
                resolution=resolution,
                precision=precision,
                view=view,
                use_hg=use_hg,
                upscale=upscale,
                film_grain=film_grain,
                hdr_gt=hdr_gt,
                autoplay=autoplay,
                start_frame=start_frame,
            )

    # - Slots: playback ---------------------------------------

    def _play(self):
        if self._show_export_lock_message("Playback"):
            return
        source_mode = _normalize_source_mode(
            getattr(self, "_source_mode", SOURCE_MODE_VIDEO)
        )
        is_window_source = source_mode == SOURCE_MODE_WINDOW
        if self._playing:
            return

        if is_window_source:
            target = getattr(self, "_capture_target", None)
            if target is None:
                self.statusBar().showMessage(
                    "Choose a browser window source first, then press Play."
                )
                return
            resolved_target = resolve_window_capture_target(target)
            if resolved_target is not None:
                target = resolved_target
                self._capture_target = resolved_target
            preview, vw, vh = probe_window_capture_target(target)
            if preview is None or vw <= 0 or vh <= 0:
                QMessageBox.warning(
                    self,
                    "Browser Window Capture",
                    "Could not capture frames from the selected browser window.\n\n"
                    "Make sure the window is still visible on screen, then try again.",
                )
                self.statusBar().showMessage(
                    "Browser window capture unavailable: no visible window frames were received."
                )
                return
            self._capture_target = WindowCaptureTarget(
                title=str(target.title or "").strip(),
                process_name=str(target.process_name or "").strip(),
                pid=int(getattr(target, "pid", 0) or 0),
                width=int(vw),
                height=int(vh),
                hwnd=int(getattr(target, "hwnd", 0) or 0),
                session_id=str(getattr(target, "session_id", "") or "").strip(),
                browser_name=str(getattr(target, "browser_name", "") or "").strip(),
                source_url=str(getattr(target, "source_url", "") or "").strip(),
            )
            self._capture_fps_value = float(LIVE_CAPTURE_DISPLAY_FPS)
            self._set_source_resolution_options_for_dims(vw, vh)
            self._prepare_live_timeline(LIVE_CAPTURE_DISPLAY_FPS)
            self._source_hdr_info = {"is_hdr": False, "reason": "browser_window_capture"}
            total_frames = 0
            self._vid_fps = float(LIVE_CAPTURE_DISPLAY_FPS)
            if not str(getattr(self._capture_target, "session_id", "") or "").strip():
                self.statusBar().showMessage(
                    "No matching Chrome Audio Sync session was found. "
                    "Chrome's 'Use graphics acceleration when available' must be off. "
                    "Chrome requires you to start Chrome Audio Sync from the extension before Play. "
                    "Until then, HDRTVNet++ stays silent and the browser keeps playing audio locally."
                )
        else:
            if not self._video_path:
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
                self.statusBar().showMessage(
                    "Cannot start conversion: input video is HDR."
                )
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
            self._vid_fps = vfps if vfps > 0 else 30.0

        self._objective_metrics_enabled = False

        if vw <= 0 or vh <= 0:
            self.statusBar().showMessage("Could not determine source dimensions.")
            return

        # Output/display resolution follows the source-limited top preset so
        # SDR and HDR panes share the same effective sharpness ceiling.
        output_key = str(getattr(self, "_source_max_resolution_key", "1080p") or "1080p")
        ow, oh = _processing_preset_dims(output_key)
        self._cur_output_w, self._cur_output_h = ow, oh

        # Processing resolution from scale selector (fixed preset; letterbox handles aspect).
        scale_key = self._cmb_res.currentText()
        if scale_key == "Source":
            scale_key = getattr(self, "_source_max_resolution_key", "1080p")
        scale_dims = RESOLUTION_SCALES.get(scale_key)
        if scale_dims is not None and (scale_dims[0] < ow or scale_dims[1] < oh):
            pw, ph = scale_dims
        else:
            pw, ph = ow, oh

        source_max_key = getattr(self, "_source_max_resolution_key", "1080p")
        source_dims = getattr(self, "_source_video_dims", None)
        disable_top_preset_sharpen = False
        if (
            isinstance(source_dims, tuple)
            and len(source_dims) == 2
            and scale_key == source_max_key
        ):
            disable_top_preset_sharpen = _source_is_below_processing_preset(
                source_dims[0], source_dims[1], scale_key
            )

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
        display_fps = (
            float(self._vid_fps)
            if is_window_source
            else _limited_playback_fps(self._vid_fps)
        )
        if is_window_source:
            self._seek_slider.setRange(0, 0)
            self._seek_slider.setValue(0)
            self._seek_slider.setEnabled(False)
            self._seek_slider.setToolTip("Live capture has no seekable timeline.")
            self._lbl_time.setText("0:00")
            self._lbl_duration.setText("LIVE")
        else:
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
        self._autotune_warning_needed = False
        if self._runtime_execution_mode_uses_compile():
            compile_ready = self._is_compile_ready_for_runtime(
                pw,
                ph,
                prec_arg,
                model_path=model_path,
                use_hg=self._chk_hg.isChecked(),
                selected_predequantize_mode=getattr(self, "_predequantize_mode", "auto"),
            )
            self._autotune_warning_needed = not compile_ready
            if not compile_ready:
                self._autotune_warning_needed = self._compile_cache_missing_for_any(
                    [f"{pw}x{ph}"],
                    prec_arg,
                    model_path=model_path,
                    use_hg=self._chk_hg.isChecked(),
                )
            if not compile_ready:
                if not self._confirm_autotune_precompile_ready():
                    return
                dlg = _PrecompileDialog(
                    [f"{pw}x{ph}"],
                    precision=prec_arg,
                    model_path=model_path,
                    use_hg=self._chk_hg.isChecked(),
                    hg_weights=_HG_WEIGHTS_PATH if os.path.isfile(_HG_WEIGHTS_PATH) else None,
                    predequantize_mode=self._effective_precompile_predequantize_mode(
                        prec_arg,
                        getattr(self, "_predequantize_mode", "auto"),
                    ),
                    parent=self,
                )
                dlg.exec()  # modal - blocks until done
                if not dlg.succeeded:
                    # Compile failed or user closed early - don't start playback
                    return
                self._mark_runtime_compile_verified(
                    pw,
                    ph,
                    prec_arg,
                    model_path,
                    self._chk_hg.isChecked(),
                    selected_predequantize_mode=getattr(self, "_predequantize_mode", "auto"),
                )
            elif not self._ensure_runtime_compile_cache_usable(
                w=pw,
                h=ph,
                precision=prec_arg,
                model_path=model_path,
                use_hg=self._chk_hg.isChecked(),
                selected_predequantize_mode=getattr(self, "_predequantize_mode", "auto"),
                workflow_name="Playback",
            ):
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
        self._startup_sync_pending = bool(use_mpv_pipeline and (not is_window_source))
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
        self._btn_pause.setEnabled(not is_window_source)
        self._btn_stop.setEnabled(True)
        self._btn_compare.setEnabled(not is_window_source)
        self._btn_file.setEnabled(False)
        if self._btn_toggle_ui is not None:
            self._btn_toggle_ui.setEnabled(True)
        self._cmb_prec.setEnabled(True)
        self._set_pause_button_labels(False)
        self._refresh_source_mode_ui()

        # Start mpv HDR display AFTER compile finishes (via signal)
        # so that mpv's D3D11 GPU usage doesn't pollute Triton autotuning.
        # mpv receives frames at processing resolution; GPU scaling happens in mpv.
        self._pending_mpv_start = None
        self._pending_sdr_mpv_start = None
        if use_mpv_pipeline and self._disp_hdr_mpv is not None:
            mpv_audio_path = None
            if (not is_window_source) and (not self._audio_available):
                mpv_audio_path = self._video_path
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
            if disable_top_preset_sharpen:
                self._active_mpv_cas = 0.0
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
            self._video_path if not is_window_source else None,
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
            runtime_execution_mode=_normalize_runtime_execution_mode(
                getattr(self, "_runtime_execution_mode", "compile")
            ),
            objective_metrics_enabled=self._objective_metrics_enabled,
            hdr_ground_truth_path=self._hdr_ground_truth_path,
            capture_target=(
                {
                    "hwnd": int(getattr(self._capture_target, "hwnd", 0) or 0),
                    "session_id": str(getattr(self._capture_target, "session_id", "") or ""),
                    "title": str(self._capture_target.title or ""),
                    "browser_name": str(getattr(self._capture_target, "browser_name", "") or ""),
                    "source_url": str(getattr(self._capture_target, "source_url", "") or ""),
                    "process_name": str(getattr(self._capture_target, "process_name", "") or ""),
                    "pid": int(getattr(self._capture_target, "pid", 0) or 0),
                    "fps": float(LIVE_CAPTURE_DISPLAY_FPS),
                    "capture_w": int(pw),
                    "capture_h": int(ph),
                }
                if is_window_source and self._capture_target is not None
                else None
            ),
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
            self.statusBar().showMessage(
                f"No upscale stage: processing at {ow}x{oh}."
            )
        self._post_seek_resync_frames = 0
        self._pending_seek_on_resume = None
        if self._startup_sync_pending:
            self._worker.pause()
        if is_window_source:
            self._stop_live_audio_capture()
            self._startup_audio_gate_active = False
            self._scrub_muted = False
            self._refresh_source_mode_ui()
            self.statusBar().showMessage(
                "Live browser window capture started. "
                "Experimental Chrome-only mode: Chrome's 'Use graphics acceleration when available' must be off. "
                "Capture runs dynamically from fresh Chrome window frames. HDRTVNet++ stays silent. "
                "If Chrome Audio Sync is active, the extension delays and plays the tab audio locally."
            )
        elif self._audio_available:
            self._stop_live_audio_capture()
            # Startup audio gate: release audio only after FPS stabilizes.
            self._startup_audio_gate_active = True
            self._scrub_muted = True
            self._arm_mute_until_fps_recovery()
            self._start_audio_playback(self._video_path)
            self._set_audio_paused(True)
        else:
            self._stop_live_audio_capture()
            self._startup_audio_gate_active = True
            self._scrub_muted = True
            self._arm_mute_until_fps_recovery()
            self.statusBar().showMessage(
                "Qt audio backend unavailable; using mpv audio fallback (seek sync may be limited)."
            )

        # Restore timeline position after process restart (resolution change).
        if self._startup_seek_frame is not None and not is_window_source:
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
        if self._show_export_lock_message("Playback"):
            return
        if _normalize_source_mode(getattr(self, "_source_mode", SOURCE_MODE_VIDEO)) == SOURCE_MODE_WINDOW:
            self.statusBar().showMessage(
                "Browser Window Capture is a live viewer. Pause is unavailable in this mode."
            )
            return
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
        preserve_capture_target = (
            _normalize_source_mode(getattr(self, "_source_mode", SOURCE_MODE_VIDEO))
            == SOURCE_MODE_WINDOW
        )
        remembered_capture_target = None
        if preserve_capture_target and self._capture_target is not None:
            remembered_capture_target = WindowCaptureTarget(
                title=str(self._capture_target.title or "").strip(),
                process_name=str(getattr(self._capture_target, "process_name", "") or "").strip(),
                pid=int(getattr(self._capture_target, "pid", 0) or 0),
                width=int(getattr(self._capture_target, "width", 0) or 0),
                height=int(getattr(self._capture_target, "height", 0) or 0),
                hwnd=int(getattr(self._capture_target, "hwnd", 0) or 0),
                session_id=str(getattr(self._capture_target, "session_id", "") or "").strip(),
                browser_name=str(getattr(self._capture_target, "browser_name", "") or "").strip(),
                source_url=str(getattr(self._capture_target, "source_url", "") or "").strip(),
            )
        self._video_path = None
        self._capture_target = remembered_capture_target if preserve_capture_target else None
        self._source_hdr_info = {"is_hdr": False, "reason": "unknown"}
        # No active input video => clear GT binding too.
        self._reset_hdr_ground_truth()
        if self._lbl_file is not None:
            self._lbl_file.setText(
                (
                    self._capture_target.label
                    if preserve_capture_target and self._capture_target is not None
                    else "No browser window selected"
                )
                if self._source_mode == SOURCE_MODE_WINDOW
                else "No video selected"
            )
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
        try:
            from browser_tab_bridge import close_browser_tab_bridge

            close_browser_tab_bridge()
        except Exception:
            pass

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

    def _refresh_live_capture_playback_fps(self, fps: float):
        del fps
        return

    def _on_capture_fps_changed(self, label: str):
        self._capture_fps_label = _normalize_capture_fps_label(label)
        self._capture_fps_value = float(LIVE_CAPTURE_DISPLAY_FPS)
        if _normalize_source_mode(getattr(self, "_source_mode", SOURCE_MODE_VIDEO)) != SOURCE_MODE_WINDOW:
            self._refresh_source_mode_ui()
            return
        self._prepare_live_timeline(LIVE_CAPTURE_DISPLAY_FPS)
        self._refresh_source_mode_ui()

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
        if self._show_export_lock_message("Runtime setting changes"):
            return
        if not self._playing or not self._current_source_available():
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
            self._restart_with_active_source(
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
                active_scale_key = str(self._active_resolution or self._cmb_res.currentText() or "").strip()
                if active_scale_key == "Source":
                    active_scale_key = getattr(self, "_source_max_resolution_key", "1080p")
                source_dims = getattr(self, "_source_video_dims", None)
                disable_top_preset_sharpen = False
                if (
                    isinstance(source_dims, tuple)
                    and len(source_dims) == 2
                    and active_scale_key == getattr(self, "_source_max_resolution_key", "1080p")
                ):
                    disable_top_preset_sharpen = _source_is_below_processing_preset(
                        source_dims[0], source_dims[1], active_scale_key
                    )
                self._active_mpv_cas = _select_mpv_cas_strength(
                    cur_pw,
                    cur_ph,
                    ow,
                    oh,
                    using_fsr=(kernel == "fsr"),
                    scale_kernel=kernel,
                )
                if disable_top_preset_sharpen:
                    self._active_mpv_cas = 0.0
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
            if self._runtime_execution_mode_uses_compile():
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
                    self._restart_with_active_source(
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
