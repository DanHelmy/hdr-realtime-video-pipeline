from __future__ import annotations

import os
import pathlib
import queue
import threading
import time

import cv2
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_DLL_DIR_HANDLES = []


def _prepend_dll_search_path(path: pathlib.Path) -> None:
    path_text = str(path)
    if not path_text or not path.is_dir():
        return
    os.environ["PATH"] = path_text + os.pathsep + os.environ.get("PATH", "")
    try:
        _DLL_DIR_HANDLES.append(os.add_dll_directory(path_text))
    except Exception:
        pass


_prepend_dll_search_path(_HERE)


def _bgr_to_rgb48_bytes(frame: np.ndarray, host_state: dict) -> bytes:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected HxWx3 frame.")
    shape = (int(frame.shape[0]), int(frame.shape[1]), 3)
    arr = host_state.get("numpy")
    if host_state.get("shape") != shape or arr is None:
        arr = np.empty(shape, dtype=np.uint16)
        host_state["shape"] = shape
        host_state["numpy"] = arr
    if frame.dtype == np.uint8:
        np.multiply(frame[:, :, ::-1], np.uint16(257), out=arr, casting="unsafe")
    elif frame.dtype == np.uint16:
        np.copyto(arr, frame[:, :, ::-1], casting="unsafe")
    else:
        src = frame.astype(np.float32, copy=False)
        if src.max(initial=0.0) <= 1.0:
            src = src * 65535.0
        arr[:] = np.clip(src[:, :, ::-1], 0.0, 65535.0).astype(np.uint16)
    return arr.tobytes()


class CliDisplaySink:
    def __init__(
        self,
        *,
        enabled: bool,
        width: int,
        height: int,
        fps: float,
        backend: str = "mpv",
        window_name: str = "HDRTVNet++ CLI",
        scale_kernel: str = "bicubic",
        scale_antiring: float | None = 0.0,
        force_hdr_metadata: bool = True,
        vsync_timed: bool = False,
    ):
        self.enabled = bool(enabled)
        self.backend = str(backend or "mpv").strip().lower()
        self.window_name = str(window_name or "HDRTVNet++ CLI")
        self._app = None
        self._mpv_widget = None
        self._rgb48_state: dict = {}
        self._async_queue: queue.Queue | None = None
        self._async_thread: threading.Thread | None = None
        self._async_shutdown = threading.Event()
        self._last_event_pump_s = 0.0
        self._event_pump_interval_s = max(
            0.0,
            float(os.environ.get("HDRTVNET_CLI_MPV_EVENT_INTERVAL_MS", "8.0"))
            / 1000.0,
        )

        if not self.enabled:
            return
        if self.backend == "opencv":
            return
        if self.backend != "mpv":
            raise ValueError(f"Unknown display backend: {self.backend}")
        self._start_mpv(
            int(width),
            int(height),
            float(fps) if fps and fps > 0 else 30.0,
            scale_kernel=str(scale_kernel or "bicubic"),
            scale_antiring=scale_antiring,
            force_hdr_metadata=bool(force_hdr_metadata),
            vsync_timed=bool(vsync_timed),
        )

    def _start_mpv(
        self,
        width: int,
        height: int,
        fps: float,
        *,
        scale_kernel: str,
        scale_antiring: float | None,
        force_hdr_metadata: bool,
        vsync_timed: bool,
    ) -> None:
        _prepend_dll_search_path(_HERE)
        try:
            import mpv as mpv_lib
        except (OSError, ImportError) as exc:
            raise RuntimeError(
                "mpv display requested, but python-mpv/libmpv is unavailable. "
                "Use --display-backend opencv for the old display path."
            ) from exc

        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError as exc:
            raise RuntimeError(
                "mpv display requested, but PyQt6 is unavailable. "
                "Use --display-backend opencv for the old display path."
            ) from exc

        from gui_mpv_widget import MpvHDRWidget
        from gui_scaling import (
            BEST_MPV_SCALE,
            FILMGRAIN_SHADER_PATH,
            FSR_SHADER_PATH,
            SSIM_DOWNSCALER_SHADER_PATH,
            SSIM_SUPERRES_SHADER_PATH,
            _ensure_filmgrain_shader,
            _ensure_fsr_shader,
            _ensure_ssim_downscaler_shader,
            _ensure_ssim_superres_shader,
            _normalize_shader_paths,
        )

        app = QApplication.instance()
        if app is None:
            app = QApplication([self.window_name])
        self._app = app

        mpv_diag = str(os.environ.get("HDRTVNET_MPV_DIAG", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        widget = MpvHDRWidget(
            mpv_lib=mpv_lib,
            mpv_diag=mpv_diag,
            normalize_shader_paths=_normalize_shader_paths,
            ensure_fsr_shader=_ensure_fsr_shader,
            ensure_ssim_superres_shader=_ensure_ssim_superres_shader,
            ensure_filmgrain_shader=_ensure_filmgrain_shader,
            ensure_ssim_downscaler_shader=_ensure_ssim_downscaler_shader,
            best_mpv_scale=BEST_MPV_SCALE,
            fsr_shader_path=FSR_SHADER_PATH,
            ssim_superres_shader_path=SSIM_SUPERRES_SHADER_PATH,
            filmgrain_shader_path=FILMGRAIN_SHADER_PATH,
            ssim_downscaler_shader_path=SSIM_DOWNSCALER_SHADER_PATH,
        )
        widget.setWindowTitle(self.window_name)
        widget.resize(int(width), int(height))
        widget.show()
        app.processEvents()

        started = widget.start_playback(
            int(width),
            int(height),
            fps=float(fps),
            scale_kernel=scale_kernel,
            scale_antiring=scale_antiring,
            force_hdr_metadata=force_hdr_metadata,
            vsync_timed=vsync_timed,
        )
        if not started:
            widget.close()
            app.processEvents()
            raise RuntimeError(
                getattr(widget, "_last_scale_error", None) or "mpv display startup failed."
            )
        self._mpv_widget = widget
        self._start_async_converter(widget)

    def _start_async_converter(self, widget) -> None:
        self._async_shutdown.clear()
        self._async_queue = queue.Queue(maxsize=1)

        def _worker() -> None:
            rgb48_state: dict = {}
            while not self._async_shutdown.is_set():
                q = self._async_queue
                if q is None:
                    return
                try:
                    item = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    return
                try:
                    widget.feed_frame(_bgr_to_rgb48_bytes(item, rgb48_state))
                except Exception:
                    return

        self._async_thread = threading.Thread(
            target=_worker,
            name="hdrtvnet-cli-mpv-converter",
            daemon=True,
        )
        self._async_thread.start()

    def _queue_latest_frame(self, frame: np.ndarray) -> None:
        q = self._async_queue
        if q is None:
            return
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(np.array(frame, copy=True, order="C"))
        except queue.Full:
            pass

    def show(self, frame) -> bool:
        if not self.enabled:
            return True
        if self.backend == "opencv":
            cv2.imshow(self.window_name, frame)
            return (cv2.waitKey(1) & 0xFF) != 27

        widget = self._mpv_widget
        app = self._app
        if widget is None or app is None:
            return False
        self._queue_latest_frame(frame)
        now_s = time.perf_counter()
        if (
            self._last_event_pump_s <= 0.0
            or now_s - self._last_event_pump_s >= self._event_pump_interval_s
        ):
            app.processEvents()
            self._last_event_pump_s = now_s
        return bool(widget.isVisible())

    def close(self) -> None:
        if self.backend == "opencv":
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            return
        self._async_shutdown.set()
        q = self._async_queue
        if q is not None:
            try:
                q.put_nowait(None)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass
        if self._async_thread is not None:
            self._async_thread.join(timeout=1.0)
            self._async_thread = None
        self._async_queue = None
        widget = self._mpv_widget
        app = self._app
        if widget is not None:
            try:
                widget.stop_playback()
            except Exception:
                pass
            try:
                widget.close()
            except Exception:
                pass
        if app is not None:
            try:
                app.processEvents()
            except Exception:
                pass
