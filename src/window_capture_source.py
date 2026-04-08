from __future__ import annotations

import atexit
import ctypes
import importlib
import os
import queue
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass

import numpy as np
import psutil

try:
    import dxcam
except Exception:
    dxcam = None


def _fit_capture_size(
    src_w: int,
    src_h: int,
    max_w: int,
    max_h: int,
) -> tuple[int, int]:
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    max_w = max(1, int(max_w))
    max_h = max(1, int(max_h))
    scale = min(max_w / src_w, max_h / src_h, 1.0)
    out_w = max(2, int(round(src_w * scale)))
    out_h = max(2, int(round(src_h * scale)))
    out_w -= out_w % 2
    out_h -= out_h % 2
    return max(2, out_w), max(2, out_h)


_BROWSER_PROCESS_LABELS = {
    "brave.exe": "Brave",
    "chrome.exe": "Google Chrome",
    "firefox.exe": "Firefox",
    "msedge.exe": "Microsoft Edge",
    "opera.exe": "Opera",
    "opera_gx.exe": "Opera GX",
    "vivaldi.exe": "Vivaldi",
}

_BROWSER_PROCESS_NAMES = frozenset(_BROWSER_PROCESS_LABELS.keys())
_DXCAM_FACTORY = (
    dxcam.__dict__.get("__factory") if dxcam is not None else None
)
if _DXCAM_FACTORY is None and dxcam is not None:
    _DXCAM_FACTORY = dxcam.DXFactory()
_WINRT_WINDOW_CAPTURE_CACHE: dict[int, "_WinRTWindowCaptureSession"] = {}
_WINRT_WINDOW_BINDINGS = None


@dataclass
class WindowCaptureTarget:
    title: str = ""
    process_name: str = ""
    pid: int = 0
    width: int = 0
    height: int = 0
    hwnd: int = 0
    session_id: str = ""
    browser_name: str = ""
    source_url: str = ""

    @property
    def label(self) -> str:
        title = str(self.title or "").strip() or "Window Capture"
        browser = str(self.browser_name or "").strip()
        proc = str(self.process_name or "").strip()
        source = browser or proc
        if source:
            return f"{title} [{source}]"
        return title


if os.name == "nt":
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    try:
        _dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
    except Exception:
        _dwmapi = None

    _DWMWA_CLOAKED = 14
    _MONITOR_DEFAULTTONEAREST = 2

    _WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    _user32.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
    _user32.EnumWindows.restype = wintypes.BOOL
    _user32.IsWindow.argtypes = [wintypes.HWND]
    _user32.IsWindow.restype = wintypes.BOOL
    _user32.IsWindowVisible.argtypes = [wintypes.HWND]
    _user32.IsWindowVisible.restype = wintypes.BOOL
    _user32.IsIconic.argtypes = [wintypes.HWND]
    _user32.IsIconic.restype = wintypes.BOOL
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextLengthW.restype = ctypes.c_int
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetWindowTextW.restype = ctypes.c_int
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.GetWindowRect.restype = wintypes.BOOL
    _user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    _user32.MonitorFromWindow.restype = wintypes.HMONITOR

    if _dwmapi is not None:
        _dwmapi.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        _dwmapi.DwmGetWindowAttribute.restype = getattr(ctypes, "HRESULT", ctypes.c_long)
else:
    _user32 = None
    _dwmapi = None
    _DWMWA_CLOAKED = 0
    _MONITOR_DEFAULTTONEAREST = 0


def _browser_name_from_process_name(process_name: str) -> str:
    text = str(process_name or "").strip().lower()
    if not text:
        return ""
    return _BROWSER_PROCESS_LABELS.get(text, os.path.splitext(text)[0] or text)


def _norm_target_text(value: str) -> str:
    return str(value or "").strip().casefold()


def _safe_process_name(pid: int) -> str:
    try:
        if int(pid) <= 0:
            return ""
        return str(psutil.Process(int(pid)).name() or "").strip()
    except Exception:
        return ""


def _window_exists(hwnd: int) -> bool:
    if _user32 is None:
        return False
    hwnd_i = int(hwnd or 0)
    return hwnd_i > 0 and bool(_user32.IsWindow(wintypes.HWND(hwnd_i)))


def _window_title(hwnd: int) -> str:
    if _user32 is None:
        return ""
    hwnd_w = wintypes.HWND(int(hwnd or 0))
    length = int(_user32.GetWindowTextLengthW(hwnd_w) or 0)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(hwnd_w, buffer, len(buffer))
    return str(buffer.value or "").strip()


def _window_pid(hwnd: int) -> int:
    if _user32 is None:
        return 0
    pid = wintypes.DWORD(0)
    _user32.GetWindowThreadProcessId(wintypes.HWND(int(hwnd or 0)), ctypes.byref(pid))
    return max(0, int(pid.value or 0))


def _window_rect(hwnd: int) -> tuple[int, int, int, int]:
    if _user32 is None:
        return 0, 0, 0, 0
    rect = wintypes.RECT()
    ok = bool(_user32.GetWindowRect(wintypes.HWND(int(hwnd or 0)), ctypes.byref(rect)))
    if not ok:
        return 0, 0, 0, 0
    width = max(0, int(rect.right) - int(rect.left))
    height = max(0, int(rect.bottom) - int(rect.top))
    return int(rect.left), int(rect.top), width, height


def _window_monitor(hwnd: int) -> int:
    if _user32 is None:
        return 0
    try:
        handle = _user32.MonitorFromWindow(
            wintypes.HWND(int(hwnd or 0)),
            wintypes.DWORD(_MONITOR_DEFAULTTONEAREST),
        )
        return max(0, int(handle or 0))
    except Exception:
        return 0


def _dxcam_output_spec_for_monitor(hmonitor: int):
    factory = _DXCAM_FACTORY
    monitor_i = max(0, int(hmonitor or 0))
    if factory is None or monitor_i <= 0:
        return None
    try:
        for device_idx, outputs in enumerate(factory.outputs):
            for output_idx, output in enumerate(outputs):
                try:
                    output.update_desc()
                except Exception:
                    continue
                try:
                    output_monitor = max(0, int(output.hmonitor or 0))
                except Exception:
                    output_monitor = 0
                if output_monitor != monitor_i:
                    continue
                desc = getattr(output, "desc", None)
                coords = getattr(desc, "DesktopCoordinates", None)
                if coords is None:
                    continue
                return (
                    int(device_idx),
                    int(output_idx),
                    int(coords.left),
                    int(coords.top),
                    int(coords.right),
                    int(coords.bottom),
                )
    except Exception:
        return None
    return None


def _load_winrt_window_capture_bindings():
    global _WINRT_WINDOW_BINDINGS
    if _WINRT_WINDOW_BINDINGS is not None:
        return _WINRT_WINDOW_BINDINGS
    if dxcam is None:
        return None
    try:
        from dxcam._libs.d3d11 import ID3D11Texture2D
        from dxcam._libs.dxgi import IDXGIDevice, IDXGISurface
        from dxcam.core.stagesurf import StageSurface
        from dxcam.processor import Processor

        capture_module = importlib.import_module("winrt.windows.graphics.capture")
        capture_interop_module = importlib.import_module(
            "winrt.windows.graphics.capture.interop"
        )
        directx_module = importlib.import_module("winrt.windows.graphics.directx")
        d3d11_interop_module = importlib.import_module(
            "winrt.windows.graphics.directx.direct3d11.interop"
        )
        _WINRT_WINDOW_BINDINGS = {
            "StageSurface": StageSurface,
            "Processor": Processor,
            "ID3D11Texture2D": ID3D11Texture2D,
            "IDXGIDevice": IDXGIDevice,
            "IDXGISurface": IDXGISurface,
            "frame_pool_cls": getattr(
                capture_module,
                "Direct3D11CaptureFramePool",
            ),
            "DirectXPixelFormat": getattr(
                directx_module,
                "DirectXPixelFormat",
            ),
            "create_for_window": getattr(
                capture_interop_module,
                "create_for_window",
            ),
            "create_direct3d11_device_from_dxgi_device": getattr(
                d3d11_interop_module,
                "create_direct3d11_device_from_dxgi_device",
            ),
            "get_dxgi_surface_from_object": getattr(
                d3d11_interop_module,
                "get_dxgi_surface_from_object",
            ),
        }
    except Exception:
        _WINRT_WINDOW_BINDINGS = False
    return _WINRT_WINDOW_BINDINGS or None


def _close_winrt_object(obj) -> None:
    if obj is None:
        return
    close_fn = getattr(obj, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


class _WinRTWindowCaptureSession:
    def __init__(self, hwnd: int):
        self.hwnd = max(0, int(hwnd or 0))
        self._bindings = _load_winrt_window_capture_bindings()
        if self.hwnd <= 0 or self._bindings is None or _DXCAM_FACTORY is None:
            raise RuntimeError("WinRT window capture is unavailable.")
        self._monitor = 0
        self._device = None
        self._output = None
        self._stage_surface = None
        self._processor = None
        self._winrt_device = None
        self._capture_item = None
        self._frame_pool = None
        self._session = None
        self._pixel_format = None
        self._frame_size = (0, 0)
        self._seen_frame = False
        self._open()

    def _open(self) -> None:
        self.release()
        monitor = _window_monitor(self.hwnd)
        spec = _dxcam_output_spec_for_monitor(monitor)
        if spec is None:
            raise RuntimeError("Unable to resolve monitor for WinRT window capture.")
        device_idx, output_idx, *_coords = spec
        output = _DXCAM_FACTORY.outputs[int(device_idx)][int(output_idx)]
        output.update_desc()
        device = _DXCAM_FACTORY.devices[int(device_idx)]

        bindings = self._bindings
        dxgi_device = device.device.QueryInterface(bindings["IDXGIDevice"])
        dxgi_device_ptr = ctypes.cast(dxgi_device, ctypes.c_void_p).value
        if dxgi_device_ptr is None:
            raise RuntimeError("Failed to get IDXGIDevice pointer for WinRT window capture.")

        winrt_device = bindings["create_direct3d11_device_from_dxgi_device"](
            dxgi_device_ptr
        )
        capture_item = bindings["create_for_window"](int(self.hwnd))
        pixel_format = bindings["DirectXPixelFormat"].B8_G8_R8_A8_UINT_NORMALIZED
        frame_pool = bindings["frame_pool_cls"].create_free_threaded(
            winrt_device,
            pixel_format,
            2,
            capture_item.size,
        )
        session = frame_pool.create_capture_session(capture_item)
        try:
            session.is_cursor_capture_enabled = False
        except Exception:
            pass
        try:
            session.is_border_required = False
        except Exception:
            pass
        session.start_capture()

        frame_w = max(1, int(capture_item.size.width))
        frame_h = max(1, int(capture_item.size.height))
        stage_surface = bindings["StageSurface"](output=output, device=device)
        stage_surface.release()
        stage_surface.rebuild(output=output, device=device, dim=(frame_w, frame_h))

        self._monitor = max(0, int(monitor))
        self._device = device
        self._output = output
        self._stage_surface = stage_surface
        self._processor = bindings["Processor"](output_color="BGR", backend="cv2")
        self._winrt_device = winrt_device
        self._capture_item = capture_item
        self._frame_pool = frame_pool
        self._session = session
        self._pixel_format = pixel_format
        self._frame_size = (frame_w, frame_h)
        self._seen_frame = False

    def _rebuild_stage_surface(self, width: int, height: int) -> None:
        if self._stage_surface is None:
            return
        self._stage_surface.release()
        self._stage_surface.rebuild(
            output=self._output,
            device=self._device,
            dim=(int(width), int(height)),
        )

    def _recreate_frame_pool(self, size) -> bool:
        if self._frame_pool is None or self._winrt_device is None or self._pixel_format is None:
            return False
        try:
            self._frame_pool.recreate(
                self._winrt_device,
                self._pixel_format,
                2,
                size,
            )
            return True
        except Exception:
            return False

    def _drain_to_latest_frame(self, *, wait_timeout_s: float = 0.0):
        deadline = time.perf_counter() + max(0.0, float(wait_timeout_s))
        latest = None
        while True:
            saw_frame = False
            while True:
                try:
                    frame = self._frame_pool.try_get_next_frame()
                except Exception:
                    if latest is not None:
                        _close_winrt_object(latest)
                    raise
                if frame is None:
                    break
                saw_frame = True
                if latest is not None:
                    _close_winrt_object(latest)
                latest = frame
            if latest is not None:
                return latest
            if wait_timeout_s <= 0.0 or time.perf_counter() >= deadline:
                return None
            if not saw_frame:
                time.sleep(0.01)

    def grab(self) -> np.ndarray | None:
        if not _window_exists(self.hwnd):
            return None
        current_monitor = _window_monitor(self.hwnd)
        if current_monitor <= 0:
            return None
        if current_monitor != self._monitor or self._session is None:
            self._open()
        wait_timeout_s = 0.15 if not self._seen_frame else 0.0
        frame = self._drain_to_latest_frame(wait_timeout_s=wait_timeout_s)
        if frame is None:
            return None
        try:
            frame_w = max(1, int(frame.content_size.width))
            frame_h = max(1, int(frame.content_size.height))
            if (frame_w, frame_h) != self._frame_size:
                if not self._recreate_frame_pool(frame.content_size):
                    return None
                self._frame_size = (frame_w, frame_h)
                self._rebuild_stage_surface(frame_w, frame_h)

            surface_ptr = self._bindings["get_dxgi_surface_from_object"](frame.surface)
            if not surface_ptr:
                return None
            texture = ctypes.cast(
                surface_ptr,
                ctypes.POINTER(self._bindings["IDXGISurface"]),
            ).QueryInterface(self._bindings["ID3D11Texture2D"])
            self._device.im_context.CopyResource(self._stage_surface.texture, texture)
            rect = self._stage_surface.map()
            try:
                image = self._processor.process(
                    rect,
                    frame_w,
                    frame_h,
                    (0, 0, frame_w, frame_h),
                    0,
                )
            finally:
                self._stage_surface.unmap()
            self._seen_frame = True
            return np.ascontiguousarray(image)
        finally:
            _close_winrt_object(frame)

    def release(self) -> None:
        if self._stage_surface is not None:
            try:
                self._stage_surface.release()
            except Exception:
                pass
        self._stage_surface = None
        self._processor = None
        _close_winrt_object(self._session)
        _close_winrt_object(self._frame_pool)
        _close_winrt_object(self._capture_item)
        _close_winrt_object(self._winrt_device)
        self._session = None
        self._frame_pool = None
        self._capture_item = None
        self._winrt_device = None
        self._frame_size = (0, 0)
        self._seen_frame = False


def _release_cached_winrt_window_capture(hwnd: int) -> None:
    hwnd_i = max(0, int(hwnd or 0))
    session = _WINRT_WINDOW_CAPTURE_CACHE.pop(hwnd_i, None)
    if session is None:
        return
    try:
        session.release()
    except Exception:
        pass


def _cleanup_winrt_window_captures() -> None:
    for hwnd in list(_WINRT_WINDOW_CAPTURE_CACHE.keys()):
        _release_cached_winrt_window_capture(hwnd)


def _capture_window_via_winrt(hwnd: int) -> tuple[bool, np.ndarray | None]:
    hwnd_i = max(0, int(hwnd or 0))
    if hwnd_i <= 0:
        return False, None
    session = _WINRT_WINDOW_CAPTURE_CACHE.get(hwnd_i)
    if session is None:
        try:
            session = _WinRTWindowCaptureSession(hwnd_i)
        except Exception:
            return False, None
        _WINRT_WINDOW_CAPTURE_CACHE[hwnd_i] = session
    try:
        frame = session.grab()
    except Exception:
        _release_cached_winrt_window_capture(hwnd_i)
        return False, None
    if frame is None and not _window_exists(hwnd_i):
        _release_cached_winrt_window_capture(hwnd_i)
        return False, None
    return True, frame


def _window_cloaked(hwnd: int) -> bool:
    if _dwmapi is None:
        return False
    cloaked = wintypes.DWORD(0)
    try:
        hr = int(
            _dwmapi.DwmGetWindowAttribute(
                wintypes.HWND(int(hwnd or 0)),
                wintypes.DWORD(_DWMWA_CLOAKED),
                ctypes.byref(cloaked),
                wintypes.DWORD(ctypes.sizeof(cloaked)),
            )
        )
    except Exception:
        return False
    return hr == 0 and bool(cloaked.value)


def _native_target_from_hwnd(
    hwnd: int,
    *,
    title: str = "",
    process_name: str = "",
    browser_name: str = "",
    source_url: str = "",
) -> WindowCaptureTarget | None:
    hwnd_i = max(0, int(hwnd or 0))
    if hwnd_i <= 0 or not _window_exists(hwnd_i):
        return None
    if not title:
        title = _window_title(hwnd_i)
    pid = _window_pid(hwnd_i)
    if not process_name:
        process_name = _safe_process_name(pid)
    if not browser_name:
        browser_name = _browser_name_from_process_name(process_name)
    _left, _top, width, height = _window_rect(hwnd_i)
    return WindowCaptureTarget(
        title=str(title or "").strip(),
        process_name=str(process_name or "").strip(),
        pid=max(0, int(pid)),
        width=max(0, int(width)),
        height=max(0, int(height)),
        hwnd=hwnd_i,
        session_id="",
        browser_name=str(browser_name or "").strip(),
        source_url=str(source_url or "").strip(),
    )


def _capture_native_window_frame(hwnd: int) -> tuple[np.ndarray | None, int, int]:
    if not _window_exists(hwnd):
        _release_cached_winrt_window_capture(hwnd)
        return None, 0, 0
    _left, _top, width, height = _window_rect(hwnd)
    if width <= 0 or height <= 0:
        return None, max(0, int(width)), max(0, int(height))
    handled, frame = _capture_window_via_winrt(hwnd)
    if not handled:
        return None, int(width), int(height)
    if frame is None:
        return None, int(width), int(height)
    frame_h, frame_w = frame.shape[:2]
    return frame, int(frame_w), int(frame_h)


atexit.register(_cleanup_winrt_window_captures)


def _bridge_target_from_session_info(info) -> WindowCaptureTarget | None:
    if info is None:
        return None
    return WindowCaptureTarget(
        title=str(getattr(info, "title", "") or "").strip(),
        process_name=str(getattr(info, "process_name", "") or "").strip(),
        pid=0,
        width=max(0, int(getattr(info, "width", 0) or 0)),
        height=max(0, int(getattr(info, "height", 0) or 0)),
        hwnd=0,
        session_id=str(getattr(info, "session_id", "") or "").strip(),
        browser_name=str(getattr(info, "browser_name", "") or "").strip(),
        source_url=str(getattr(info, "source_url", "") or "").strip(),
    )


def _target_match_score(
    candidate: WindowCaptureTarget | None,
    reference: WindowCaptureTarget | None,
) -> int:
    if candidate is None or reference is None:
        return -1
    score = 0

    cand_hwnd = max(0, int(getattr(candidate, "hwnd", 0) or 0))
    ref_hwnd = max(0, int(getattr(reference, "hwnd", 0) or 0))
    if cand_hwnd > 0 and ref_hwnd > 0 and cand_hwnd == ref_hwnd:
        score += 20_000

    cand_session_id = str(getattr(candidate, "session_id", "") or "").strip()
    ref_session_id = str(getattr(reference, "session_id", "") or "").strip()
    if cand_session_id and ref_session_id and cand_session_id == ref_session_id:
        score += 10_000

    cand_pid = max(0, int(getattr(candidate, "pid", 0) or 0))
    ref_pid = max(0, int(getattr(reference, "pid", 0) or 0))
    if cand_pid > 0 and ref_pid > 0 and cand_pid == ref_pid:
        score += 800

    cand_url = _norm_target_text(getattr(candidate, "source_url", ""))
    ref_url = _norm_target_text(getattr(reference, "source_url", ""))
    if cand_url and ref_url and cand_url == ref_url:
        score += 1_000

    cand_title = _norm_target_text(getattr(candidate, "title", ""))
    ref_title = _norm_target_text(getattr(reference, "title", ""))
    if cand_title and ref_title:
        if cand_title == ref_title:
            score += 300
        elif cand_title in ref_title or ref_title in cand_title:
            score += 180

    cand_browser = _norm_target_text(getattr(candidate, "browser_name", ""))
    ref_browser = _norm_target_text(getattr(reference, "browser_name", ""))
    if cand_browser and ref_browser and cand_browser == ref_browser:
        score += 80

    cand_proc = _norm_target_text(getattr(candidate, "process_name", ""))
    ref_proc = _norm_target_text(getattr(reference, "process_name", ""))
    if cand_proc and ref_proc and cand_proc == ref_proc:
        score += 60

    cand_w = max(0, int(getattr(candidate, "width", 0) or 0))
    cand_h = max(0, int(getattr(candidate, "height", 0) or 0))
    ref_w = max(0, int(getattr(reference, "width", 0) or 0))
    ref_h = max(0, int(getattr(reference, "height", 0) or 0))
    if cand_w > 0 and cand_h > 0 and ref_w > 0 and ref_h > 0:
        if cand_w == ref_w and cand_h == ref_h:
            score += 20

    return score if score > 0 else -1


def find_best_matching_window_capture_target(
    targets: list[WindowCaptureTarget],
    reference: WindowCaptureTarget | None,
) -> WindowCaptureTarget | None:
    if reference is None:
        return None
    best_target = None
    best_score = -1
    for target in list(targets or []):
        score = _target_match_score(target, reference)
        if score > best_score:
            best_target = target
            best_score = score
    return best_target


def _clone_window_capture_target(target: WindowCaptureTarget | None) -> WindowCaptureTarget | None:
    if target is None:
        return None
    return WindowCaptureTarget(
        title=str(getattr(target, "title", "") or "").strip(),
        process_name=str(getattr(target, "process_name", "") or "").strip(),
        pid=max(0, int(getattr(target, "pid", 0) or 0)),
        width=max(0, int(getattr(target, "width", 0) or 0)),
        height=max(0, int(getattr(target, "height", 0) or 0)),
        hwnd=max(0, int(getattr(target, "hwnd", 0) or 0)),
        session_id=str(getattr(target, "session_id", "") or "").strip(),
        browser_name=str(getattr(target, "browser_name", "") or "").strip(),
        source_url=str(getattr(target, "source_url", "") or "").strip(),
    )


def _list_bridge_session_targets(*, require_audio: bool = False) -> list[WindowCaptureTarget]:
    try:
        from browser_tab_bridge import list_browser_tab_sessions
    except Exception:
        return []

    targets: list[WindowCaptureTarget] = []
    for info in list_browser_tab_sessions():
        if require_audio and not bool(getattr(info, "has_audio", False)):
            continue
        target = _bridge_target_from_session_info(info)
        if target is not None:
            targets.append(target)
    return targets


def attach_best_browser_tab_session(
    target: WindowCaptureTarget | None,
    *,
    preferred_session_id: str | None = None,
) -> WindowCaptureTarget | None:
    clone = _clone_window_capture_target(target)
    if clone is None:
        return None

    preferred = str(
        preferred_session_id
        if preferred_session_id is not None
        else getattr(clone, "session_id", "")
    ).strip()
    sessions = _list_bridge_session_targets(require_audio=True)
    if not sessions:
        clone.session_id = ""
        return clone

    if preferred:
        for session_target in sessions:
            if str(getattr(session_target, "session_id", "") or "").strip() == preferred:
                clone.session_id = preferred
                if not clone.source_url:
                    clone.source_url = str(getattr(session_target, "source_url", "") or "").strip()
                if not clone.browser_name:
                    clone.browser_name = str(getattr(session_target, "browser_name", "") or "").strip()
                if not clone.process_name:
                    clone.process_name = str(getattr(session_target, "process_name", "") or "").strip()
                return clone

    best_target = None
    best_score = -1
    for session_target in sessions:
        score = _target_match_score(session_target, clone)
        if score > best_score:
            best_target = session_target
            best_score = score

    # Require more than just browser/process similarity before attaching tab audio.
    if best_target is None or best_score < 200:
        clone.session_id = ""
        return clone

    clone.session_id = str(getattr(best_target, "session_id", "") or "").strip()
    if not clone.source_url:
        clone.source_url = str(getattr(best_target, "source_url", "") or "").strip()
    if not clone.browser_name:
        clone.browser_name = str(getattr(best_target, "browser_name", "") or "").strip()
    if not clone.process_name:
        clone.process_name = str(getattr(best_target, "process_name", "") or "").strip()
    return clone


def enumerate_window_capture_targets(
    *,
    exclude_pid: int | None = None,
) -> list[WindowCaptureTarget]:
    if _user32 is None:
        return []

    exclude_pid_i = max(0, int(exclude_pid or 0))
    browser_targets: list[WindowCaptureTarget] = []
    other_targets: list[WindowCaptureTarget] = []

    def _callback(hwnd, _lparam):
        hwnd_i = int(hwnd or 0)
        if hwnd_i <= 0:
            return True
        if not bool(_user32.IsWindowVisible(wintypes.HWND(hwnd_i))):
            return True
        if bool(_user32.IsIconic(wintypes.HWND(hwnd_i))):
            return True
        if _window_cloaked(hwnd_i):
            return True

        title = _window_title(hwnd_i)
        if not title:
            return True

        pid = _window_pid(hwnd_i)
        if exclude_pid_i > 0 and pid == exclude_pid_i:
            return True

        process_name = _safe_process_name(pid)
        if not process_name:
            return True

        _left, _top, width, height = _window_rect(hwnd_i)
        if width < 160 or height < 120:
            return True

        target = WindowCaptureTarget(
            title=title,
            process_name=process_name,
            pid=pid,
            width=width,
            height=height,
            hwnd=hwnd_i,
            session_id="",
            browser_name=_browser_name_from_process_name(process_name),
            source_url="",
        )
        proc_key = str(process_name or "").strip().lower()
        if proc_key in _BROWSER_PROCESS_NAMES:
            browser_targets.append(target)
        else:
            other_targets.append(target)
        return True

    _user32.EnumWindows(_WNDENUMPROC(_callback), 0)
    targets = browser_targets if browser_targets else other_targets
    targets.sort(
        key=lambda item: (
            str(item.browser_name or item.process_name).lower(),
            str(item.title).lower(),
            int(item.hwnd or 0),
        )
    )
    return targets


def target_from_hwnd(
    hwnd: int | str,
    *,
    title: str = "",
    process_name: str = "",
    browser_name: str = "",
    source_url: str = "",
    session_id: str = "",
) -> WindowCaptureTarget | None:
    try:
        hwnd_i = int(str(hwnd or "").strip(), 0)
    except Exception:
        hwnd_i = 0
    target = _native_target_from_hwnd(
        hwnd_i,
        title=title,
        process_name=process_name,
        browser_name=browser_name,
        source_url=source_url,
    )
    if target is None:
        return None
    target.session_id = str(session_id or "").strip()
    return attach_best_browser_tab_session(target, preferred_session_id=target.session_id)


def resolve_window_capture_target(
    target: WindowCaptureTarget | None,
) -> WindowCaptureTarget | None:
    if target is None:
        return None

    hwnd_i = max(0, int(getattr(target, "hwnd", 0) or 0))
    if hwnd_i > 0:
        live_target = target_from_hwnd(
            hwnd_i,
            title=str(getattr(target, "title", "") or "").strip(),
            process_name=str(getattr(target, "process_name", "") or "").strip(),
            browser_name=str(getattr(target, "browser_name", "") or "").strip(),
            source_url=str(getattr(target, "source_url", "") or "").strip(),
            session_id=str(getattr(target, "session_id", "") or "").strip(),
        )
        if live_target is not None and live_target.hwnd > 0 and _window_exists(live_target.hwnd):
            return live_target

    targets = enumerate_window_capture_targets()
    best_target = find_best_matching_window_capture_target(targets, target)
    if best_target is None:
        return None
    preferred_session_id = str(getattr(target, "session_id", "") or "").strip()
    return attach_best_browser_tab_session(
        best_target,
        preferred_session_id=preferred_session_id or None,
    )


def probe_window_capture_target(target: WindowCaptureTarget | int) -> tuple[np.ndarray | None, int, int]:
    hwnd_i = 0
    if isinstance(target, WindowCaptureTarget):
        hwnd_i = max(0, int(getattr(target, "hwnd", 0) or 0))
    else:
        try:
            hwnd_i = int(target)
        except Exception:
            hwnd_i = 0

    if hwnd_i > 0:
        return _capture_native_window_frame(hwnd_i)
    width = int(getattr(target, "width", 0) or 0) if isinstance(target, WindowCaptureTarget) else 0
    height = int(getattr(target, "height", 0) or 0) if isinstance(target, WindowCaptureTarget) else 0
    return None, width, height


class WindowCaptureSource:
    """Live source backed by a native browser window handle."""

    def __init__(
        self,
        fps: float,
        title: str = "",
        prefetch: int = 0,
        capture_max_w: int = 0,
        capture_max_h: int = 0,
        *,
        hwnd: int = 0,
        pid: int = 0,
    ):
        self.hwnd = max(0, int(hwnd or 0))
        self.pid = max(0, int(pid or 0))
        if self.hwnd <= 0:
            raise RuntimeError("Window capture requires a native window handle.")
        self.title = str(title or "").strip()
        self.fps = max(1.0, float(fps))
        self.prefetch = max(0, int(prefetch))
        self.capture_max_w = max(0, int(capture_max_w))
        self.capture_max_h = max(0, int(capture_max_h))
        self.frame_count = 0
        self.duration = 0.0
        self.backend = "native_window"
        self.backend_reason = (
            "Browser window captured natively via Windows Graphics Capture"
        )
        self.last_capture_perf_counter = 0.0
        self._stopped = False
        self._last_frame_index = -1
        self._queue = None
        self._thread = None
        self._sentinel = object()
        self._last_native_frame: np.ndarray | None = None
        self._capture_started_perf = time.perf_counter()
        self._next_native_capture_perf = self._capture_started_perf
        self._frame_interval_s = 1.0 / self.fps

        if self.prefetch > 0:
            self._queue = queue.Queue(maxsize=max(1, self.prefetch))
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

    def read(self):
        ok, frame, _idx, _pts = self.read_with_meta()
        return ok, frame

    def read_with_index(self):
        ok, frame, idx, _pts = self.read_with_meta()
        return ok, frame, idx

    def _resize_frame_if_needed(self, frame: np.ndarray) -> np.ndarray:
        if self.capture_max_w <= 0 or self.capture_max_h <= 0:
            return frame
        src_h, src_w = frame.shape[:2]
        dst_w, dst_h = _fit_capture_size(
            int(src_w),
            int(src_h),
            int(self.capture_max_w),
            int(self.capture_max_h),
        )
        if dst_w == src_w and dst_h == src_h:
            return frame
        import cv2

        resized = cv2.resize(frame, (int(dst_w), int(dst_h)), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(resized)

    def _sleep_until_next_native_capture(self) -> bool:
        while not self._stopped:
            now = time.perf_counter()
            if now >= self._next_native_capture_perf:
                return True
            time.sleep(min(self._next_native_capture_perf - now, 0.01))
        return False

    def _native_capture_payload(self) -> tuple[int, float, float, np.ndarray] | None:
        if not self._sleep_until_next_native_capture():
            return None

        frame, width, height = _capture_native_window_frame(self.hwnd)
        captured_t = time.perf_counter()
        self._next_native_capture_perf = max(
            self._next_native_capture_perf + self._frame_interval_s,
            captured_t,
        )

        if frame is None:
            if not _window_exists(self.hwnd):
                return None
            if self._last_native_frame is None:
                return None
            frame = self._last_native_frame
        else:
            frame = self._resize_frame_if_needed(frame)
            self._last_native_frame = frame

        self._last_frame_index += 1
        idx = int(self._last_frame_index)
        pts_sec = max(0.0, captured_t - self._capture_started_perf)
        return idx, float(pts_sec), float(captured_t), frame

    def _reader_loop(self):
        while not self._stopped:
            payload = self._native_capture_payload()
            if payload is None:
                if _window_exists(self.hwnd):
                    continue
                try:
                    self._queue.put_nowait(self._sentinel)
                except Exception:
                    pass
                break
            while not self._stopped:
                try:
                    self._queue.put_nowait(payload)
                    break
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break

    def read_with_meta(self):
        if self._queue is not None:
            item = self._queue.get()
            if item is self._sentinel:
                return False, None, -1, 0.0
            idx, pts_sec, captured_t, frame = item
            self.last_capture_perf_counter = float(captured_t)
            return True, frame, int(idx), float(pts_sec)

        while not self._stopped:
            payload = self._native_capture_payload()
            if payload is None:
                if _window_exists(self.hwnd):
                    continue
                return False, None, -1, 0.0
            idx, pts_sec, captured_t, frame = payload
            self.last_capture_perf_counter = float(captured_t)
            return True, frame, int(idx), float(pts_sec)
        return False, None, -1, 0.0

    def seek(self, frame_number):
        del frame_number
        return

    def position(self):
        return max(0, int(self._last_frame_index + 1))

    def release(self):
        self._stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        _release_cached_winrt_window_capture(self.hwnd)


def capture_target_to_cli_args(target: WindowCaptureTarget | None) -> list[str]:
    if target is None:
        return []
    args: list[str] = []
    hwnd = max(0, int(getattr(target, "hwnd", 0) or 0))
    session_id = str(getattr(target, "session_id", "") or "").strip()
    if hwnd > 0:
        args += ["--capture-hwnd", str(hwnd)]
    if session_id:
        args += ["--capture-session-id", session_id]
    title = str(target.title or "").strip()
    if title:
        args += ["--capture-title", title]
    browser_name = str(getattr(target, "browser_name", "") or "").strip()
    if browser_name:
        args += ["--capture-browser-name", browser_name]
    process_name = str(getattr(target, "process_name", "") or "").strip()
    if process_name:
        args += ["--capture-process-name", process_name]
    source_url = str(getattr(target, "source_url", "") or "").strip()
    if source_url:
        args += ["--capture-url", source_url]
    return args
