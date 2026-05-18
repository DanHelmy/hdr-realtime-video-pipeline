from __future__ import annotations

import atexit
import ctypes
import ctypes.wintypes as wt
import os
import threading
import time


_WINDOWS_TIMER_STATE = threading.local()
_WINDOWS_TIMER_RESOLUTION_LOCK = threading.Lock()
_WINDOWS_TIMER_RESOLUTION_REQUESTED = False
_WINDOWS_TIMER_100NS_PER_SECOND = 10_000_000
_WINDOWS_TIMER_RESOLUTION_100NS = 5_000  # 0.5 ms when the platform allows it.
_WINDOWS_TIMER_INFINITE = 0xFFFFFFFF
_WINDOWS_WAIT_OBJECT_0 = 0x00000000
_WINDOWS_CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
_WINDOWS_TIMER_MODIFY_STATE = 0x0002
_WINDOWS_SYNCHRONIZE = 0x00100000
_WINDOWS_TIMER_ACCESS = _WINDOWS_TIMER_MODIFY_STATE | _WINDOWS_SYNCHRONIZE
_WINDOWS_STATUS_SUCCESS = 0


class _WindowsWaitableTimer:
    __slots__ = ("handle", "kernel32")

    def __init__(self, kernel32, handle):
        self.kernel32 = kernel32
        self.handle = handle

    def __bool__(self) -> bool:
        return bool(self.handle)

    def __del__(self):
        handle = getattr(self, "handle", None)
        if not handle:
            return
        try:
            self.kernel32.CloseHandle(handle)
        except Exception:
            pass
        self.handle = None


class FPSTimer:
    def __init__(self):
        self.last = time.perf_counter()
        self.frames = 0
        self.fps = 0.0

    def update(self):
        self.frames += 1
        now = time.perf_counter()
        if now - self.last >= 1.0:
            self.fps = self.frames / (now - self.last)
            self.frames = 0
            self.last = now
        return self.fps


def sleep_until(
    deadline_s: float,
    *,
    coarse_margin_s: float = 0.0010,
    spin_margin_s: float = 0.0010,
) -> None:
    """Sleep toward a deadline using Windows high-resolution timers."""
    deadline = float(deadline_s)
    coarse_margin = max(0.0002, float(coarse_margin_s))
    spin_margin = max(0.0, min(float(spin_margin_s), coarse_margin))

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            return
        if remaining > coarse_margin:
            _sleep_precise(max(0.0, remaining - coarse_margin))
            continue
        if remaining > spin_margin:
            _sleep_precise(max(0.0, remaining - spin_margin))
            continue
        while time.perf_counter() < deadline:
            pass


def _sleep_precise(seconds: float) -> None:
    duration = max(0.0, float(seconds))
    if duration <= 0.0:
        return
    if _windows_waitable_sleep(duration):
        return
    time.sleep(duration)


def _windows_waitable_sleep(seconds: float) -> bool:
    timer = _windows_waitable_timer()
    if timer is None:
        return False

    kernel32 = timer.kernel32
    ticks = max(1, int(round(float(seconds) * _WINDOWS_TIMER_100NS_PER_SECOND)))
    due_time = ctypes.c_longlong(-ticks)
    try:
        ok = kernel32.SetWaitableTimerEx(
            timer.handle,
            ctypes.byref(due_time),
            0,
            None,
            None,
            None,
            0,
        )
        if not ok:
            return False
        result = kernel32.WaitForSingleObject(
            timer.handle,
            _WINDOWS_TIMER_INFINITE,
        )
        return result == _WINDOWS_WAIT_OBJECT_0
    except Exception:
        return False


def _windows_waitable_timer():
    if os.name != "nt":
        return None
    if bool(getattr(_WINDOWS_TIMER_STATE, "disabled", False)):
        return None

    _ensure_windows_timer_resolution()

    timer = getattr(_WINDOWS_TIMER_STATE, "timer", None)
    if timer:
        return timer

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateWaitableTimerExW.argtypes = [
            wt.LPVOID,
            wt.LPCWSTR,
            wt.DWORD,
            wt.DWORD,
        ]
        kernel32.CreateWaitableTimerExW.restype = wt.HANDLE
        kernel32.SetWaitableTimerEx.argtypes = [
            wt.HANDLE,
            ctypes.POINTER(ctypes.c_longlong),
            wt.LONG,
            wt.LPVOID,
            wt.LPVOID,
            wt.LPVOID,
            wt.ULONG,
        ]
        kernel32.SetWaitableTimerEx.restype = wt.BOOL
        kernel32.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
        kernel32.WaitForSingleObject.restype = wt.DWORD
        kernel32.CloseHandle.argtypes = [wt.HANDLE]
        kernel32.CloseHandle.restype = wt.BOOL

        handle = kernel32.CreateWaitableTimerExW(
            None,
            None,
            _WINDOWS_CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
            _WINDOWS_TIMER_ACCESS,
        )
        if not handle:
            handle = kernel32.CreateWaitableTimerExW(
                None,
                None,
                0,
                _WINDOWS_TIMER_ACCESS,
            )
        if not handle:
            _WINDOWS_TIMER_STATE.disabled = True
            return None
        timer = _WindowsWaitableTimer(kernel32, handle)
        _WINDOWS_TIMER_STATE.timer = timer
        return timer
    except Exception:
        _WINDOWS_TIMER_STATE.disabled = True
        return None


def _ensure_windows_timer_resolution() -> None:
    global _WINDOWS_TIMER_RESOLUTION_REQUESTED
    if os.name != "nt" or _WINDOWS_TIMER_RESOLUTION_REQUESTED:
        return

    with _WINDOWS_TIMER_RESOLUTION_LOCK:
        if _WINDOWS_TIMER_RESOLUTION_REQUESTED:
            return
        _WINDOWS_TIMER_RESOLUTION_REQUESTED = True
        if _request_nt_timer_resolution():
            return
        _request_winmm_timer_resolution()


def _request_nt_timer_resolution() -> bool:
    try:
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        ntdll.NtQueryTimerResolution.argtypes = [
            ctypes.POINTER(wt.ULONG),
            ctypes.POINTER(wt.ULONG),
            ctypes.POINTER(wt.ULONG),
        ]
        ntdll.NtQueryTimerResolution.restype = wt.LONG
        ntdll.NtSetTimerResolution.argtypes = [
            wt.ULONG,
            wt.BOOLEAN,
            ctypes.POINTER(wt.ULONG),
        ]
        ntdll.NtSetTimerResolution.restype = wt.LONG

        min_resolution = wt.ULONG()
        max_resolution = wt.ULONG()
        current_resolution = wt.ULONG()
        status = ntdll.NtQueryTimerResolution(
            ctypes.byref(min_resolution),
            ctypes.byref(max_resolution),
            ctypes.byref(current_resolution),
        )
        if status != _WINDOWS_STATUS_SUCCESS:
            requested = _WINDOWS_TIMER_RESOLUTION_100NS
        else:
            requested = int(max_resolution.value or _WINDOWS_TIMER_RESOLUTION_100NS)

        actual_resolution = wt.ULONG()
        status = ntdll.NtSetTimerResolution(
            wt.ULONG(requested),
            wt.BOOLEAN(True),
            ctypes.byref(actual_resolution),
        )
        if status != _WINDOWS_STATUS_SUCCESS:
            return False

        def _release() -> None:
            try:
                released_resolution = wt.ULONG()
                ntdll.NtSetTimerResolution(
                    wt.ULONG(requested),
                    wt.BOOLEAN(False),
                    ctypes.byref(released_resolution),
                )
            except Exception:
                pass

        atexit.register(_release)
        return True
    except Exception:
        return False


def _request_winmm_timer_resolution() -> bool:
    try:
        winmm = ctypes.WinDLL("winmm", use_last_error=True)
        winmm.timeBeginPeriod.argtypes = [wt.UINT]
        winmm.timeBeginPeriod.restype = wt.UINT
        winmm.timeEndPeriod.argtypes = [wt.UINT]
        winmm.timeEndPeriod.restype = wt.UINT
        if winmm.timeBeginPeriod(1) != 0:
            return False

        def _release() -> None:
            try:
                winmm.timeEndPeriod(1)
            except Exception:
                pass

        atexit.register(_release)
        return True
    except Exception:
        return False
