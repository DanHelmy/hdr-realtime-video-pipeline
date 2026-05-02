"""Embedded mpv HDR widget extracted from gui.py."""

from __future__ import annotations

import os
import queue as _queue
import threading

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget

from gui_config import LIVE_CAPTURE_MPV_BUFFER_FRAMES


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _env_float(name: str, default: float, *, min_value: float, max_value: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except Exception:
        value = float(default)
    return max(float(min_value), min(float(max_value), float(value)))


def _env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = int(default)
    return max(int(min_value), min(int(max_value), int(value)))


def _env_bool_any(names: tuple[str, ...], default: bool = False) -> bool:
    for name in names:
        if os.environ.get(name) is not None:
            return _env_bool(name, default)
    return bool(default)


def _env_float_any(
    names: tuple[str, ...],
    default: float,
    *,
    min_value: float,
    max_value: float,
) -> float:
    for name in names:
        if os.environ.get(name) is not None:
            return _env_float(name, default, min_value=min_value, max_value=max_value)
    return max(float(min_value), min(float(max_value), float(default)))


def _env_int_any(
    names: tuple[str, ...],
    default: int,
    *,
    min_value: int,
    max_value: int,
) -> int:
    for name in names:
        if os.environ.get(name) is not None:
            return _env_int(name, default, min_value=min_value, max_value=max_value)
    return max(int(min_value), min(int(max_value), int(default)))


def _mpv_deband_options(*, live_capture: bool = False) -> dict:
    env_prefix = "HDRTVNET_BROWSER_MPV_" if live_capture else "HDRTVNET_MPV_"
    if not _env_bool_any((f"{env_prefix}DEBAND", "HDRTVNET_MPV_DEBAND"), True):
        return {}

    # Keep video-player defaults aligned with browser-window capture so
    # quality characteristics match between both source modes.
    default_iterations = 3
    default_threshold = 100
    default_range = 32
    default_grain = 8

    return {
        "deband": "yes",
        "deband_iterations": str(_env_int_any(
            (f"{env_prefix}DEBAND_ITERATIONS", "HDRTVNET_BROWSER_MPV_DEBAND_ITERATIONS", "HDRTVNET_MPV_DEBAND_ITERATIONS"),
            default_iterations, min_value=1, max_value=16
        )),
        "deband_threshold": str(_env_float_any(
            (f"{env_prefix}DEBAND_THRESHOLD", "HDRTVNET_BROWSER_MPV_DEBAND_THRESHOLD", "HDRTVNET_MPV_DEBAND_THRESHOLD"),
            default_threshold, min_value=0.0, max_value=4096.0
        )),
        "deband_range": str(_env_int_any(
            (f"{env_prefix}DEBAND_RANGE", "HDRTVNET_BROWSER_MPV_DEBAND_RANGE", "HDRTVNET_MPV_DEBAND_RANGE"),
            default_range, min_value=1, max_value=64
        )),
        "deband_grain": str(_env_float_any(
            (f"{env_prefix}DEBAND_GRAIN", "HDRTVNET_BROWSER_MPV_DEBAND_GRAIN", "HDRTVNET_MPV_DEBAND_GRAIN"),
            default_grain, min_value=0.0, max_value=4096.0
        )),
    }


def _mpv_dither_options(*, live_capture: bool = False) -> dict:
    env_prefix = "HDRTVNET_BROWSER_MPV_" if live_capture else "HDRTVNET_MPV_"
    if not _env_bool_any((f"{env_prefix}DITHER", "HDRTVNET_MPV_DITHER"), True):
        return {}
    algo = str(
        os.environ.get(
            f"{env_prefix}DITHER_ALGO",
            os.environ.get("HDRTVNET_MPV_DITHER_ALGO", "error-diffusion"),
        )
    ).strip().lower().replace("_", "-")
    if algo not in {"fruit", "ordered", "error-diffusion", "no"}:
        algo = "error-diffusion"
    depth = str(
        os.environ.get(
            f"{env_prefix}DITHER_DEPTH",
            os.environ.get(
                "HDRTVNET_BROWSER_MPV_DITHER_DEPTH",
                os.environ.get("HDRTVNET_MPV_DITHER_DEPTH", "auto"),
            )
        )
    ).strip().lower()
    if depth not in {"auto", "no"}:
        try:
            depth = str(max(1, min(16, int(depth))))
        except Exception:
            depth = "auto"
    if algo == "no" or depth == "no":
        return {
            "dither": "no",
            "dither_depth": "no",
        }

    options = {
        "dither": algo,
        "dither_depth": depth,
    }
    if algo == "fruit":
        options["dither_size_fruit"] = str(_env_int_any((f"{env_prefix}DITHER_SIZE_FRUIT", "HDRTVNET_MPV_DITHER_SIZE_FRUIT"), 6, min_value=2, max_value=8))
    if _env_bool_any((f"{env_prefix}TEMPORAL_DITHER", "HDRTVNET_MPV_TEMPORAL_DITHER"), True):
        options["temporal_dither"] = "yes"
        options["temporal_dither_period"] = str(_env_int_any((f"{env_prefix}TEMPORAL_DITHER_PERIOD", "HDRTVNET_MPV_TEMPORAL_DITHER_PERIOD"), 1, min_value=1, max_value=128))
    return options

def _env_hdr_target_peak_nits() -> str:
    raw = str(os.environ.get("HDRTVNET_MPV_HDR_TARGET_PEAK_NITS", "auto") or "auto").strip().lower()
    if raw == "auto":
        return "auto"
    try:
        val = float(raw)
    except Exception:
        return "auto"
    if not np.isfinite(val) or val <= 0.0:
        return "auto"
    return f"{val:.0f}"

def _mpv_live_interpolation_enabled() -> bool:
    # Favor lower latency for live capture; interpolation can add delay.
    return False


def _without_deband_options(kwargs: dict) -> dict:
    clean = dict(kwargs)
    for key in (
        "deband",
        "deband_iterations",
        "deband_threshold",
        "deband_range",
        "deband_grain",
    ):
        clean.pop(key, None)
    return clean


def _without_dither_options(kwargs: dict) -> dict:
    clean = dict(kwargs)
    for key in (
        "dither",
        "dither_depth",
        "dither_size_fruit",
        "temporal_dither",
        "temporal_dither_period",
    ):
        clean.pop(key, None)
    return clean


def _without_display_quality_options(kwargs: dict) -> dict:
    return _without_dither_options(_without_deband_options(kwargs))


class MpvHDRWidget(QWidget):
    """QWidget that embeds an mpv player for real-time HDR frame display.

    Frames are fed as raw RGB48LE over a Windows named pipe to avoid GIL-heavy
    per-frame Python read callbacks.
    """

    hdr_info_ready = pyqtSignal(dict)  # emitted once VO params are populated
    runtime_notice = pyqtSignal(str)

    def __init__(
        self,
        *,
        mpv_lib,
        mpv_diag: bool,
        normalize_shader_paths,
        ensure_fsr_shader,
        ensure_ssim_superres_shader,
        ensure_filmgrain_shader,
        best_mpv_scale: str,
        fsr_shader_path: str,
        ssim_superres_shader_path: str,
        filmgrain_shader_path: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setMinimumSize(240, 135)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setProperty("videoSurface", True)
        self._player = None
        self._pipe_name: str | None = None
        self._pipe_handle = None
        self._feeder: threading.Thread | None = None
        self._queue: _queue.Queue | None = None
        self._shutdown = threading.Event()
        self._fps = 30.0
        self._force_hdr_metadata = True
        self._sdr_transfer = "bt1886"
        self._diag_enabled = bool(mpv_diag)
        self._seek_warned = False
        self._last_playback_cfg: dict | None = None
        self._last_scale_error: str | None = None
        self._active_vo: str = "gpu-next"
        self._target_colorspace_hint: str = "no"
        self._requested_gpu_api: str = "vulkan"

        self._mpv_lib = mpv_lib
        self._mpv_diag = bool(mpv_diag)
        self._normalize_shader_paths = normalize_shader_paths
        self._ensure_fsr_shader = ensure_fsr_shader
        self._ensure_ssim_superres_shader = ensure_ssim_superres_shader
        self._ensure_filmgrain_shader = ensure_filmgrain_shader
        self._best_mpv_scale = str(best_mpv_scale)
        self._fsr_shader_path = str(fsr_shader_path)
        self._ssim_superres_shader_path = str(ssim_superres_shader_path)
        self._filmgrain_shader_path = str(filmgrain_shader_path)

    @staticmethod
    def _normalize_sdr_transfer(value: str | None) -> str:
        text = str(value or "").strip().lower().replace("-", "").replace("_", "")
        if text in {"srgb", "iec6196621", "iec61966"}:
            return "srgb"
        return "bt1886"

    def _sdr_format_filter(self, cas_strength: float | None = None) -> str:
        gamma = "srgb" if self._sdr_transfer == "srgb" else "bt.1886"
        vf_chain = f"format=colorlevels=full:primaries=bt.709:gamma={gamma}"
        if cas_strength is not None and cas_strength > 0.0:
            vf_chain += f",cas={cas_strength}"
        return vf_chain

    def _build_shader_paths(
        self,
        *,
        use_fsr: bool,
        use_ssim: bool,
        use_film_grain: bool,
    ) -> list[str]:
        shader_paths = []
        if use_fsr:
            shader_paths.append(self._fsr_shader_path)
        if use_ssim:
            shader_paths.append(self._ssim_superres_shader_path)
        if use_film_grain:
            shader_paths.append(self._filmgrain_shader_path)
        return shader_paths

    def _build_hdr_info_snapshot(self, p) -> tuple[dict, dict, dict]:
        vp = {}
        vop = {}
        try:
            vp = p.video_params or {}
        except Exception:
            pass
        try:
            vop = p.video_out_params or {}
        except Exception:
            pass

        def _g(d, *keys):
            for k in keys:
                v = d.get(k)
                if v is not None:
                    return str(v)
            return "?"

        def _prop(name):
            try:
                v = getattr(p, name.replace("-", "_"), None)
                if v is None:
                    return "?"
                if isinstance(v, (dict, list, tuple)):
                    return v
                return str(v)
            except Exception:
                return "?"

        def _fmt_prop_text(value) -> str:
            if value is None:
                return "?"
            if isinstance(value, str):
                text = value.strip()
                return text if text else "?"
            return str(value)

        def _fmt_gpu_api(value) -> str:
            if value in (None, "?"):
                return "?"
            if isinstance(value, dict):
                name = value.get("name")
                return str(name).strip() if name is not None else str(value)
            if isinstance(value, (list, tuple)):
                names: list[str] = []
                for item in value:
                    if isinstance(item, dict):
                        if item.get("enabled", True):
                            name = str(item.get("name", "")).strip()
                            if name:
                                names.append(name)
                    else:
                        text = str(item).strip()
                        if text:
                            names.append(text)
                if names:
                    return ",".join(names)
                return str(value)
            return _fmt_prop_text(value)

        out_prim = _g(vop, "primaries", "colormatrix-primaries")
        out_trc = _g(vop, "gamma", "transfer")
        out_lvl = _g(vop, "levels", "colorlevels")
        t_trc = _fmt_prop_text(_prop("target_trc"))
        t_prim = _fmt_prop_text(_prop("target_prim"))
        gpu_api = _fmt_gpu_api(_prop("gpu_api"))
        requested_gpu_api = str(self._requested_gpu_api or "?")
        if gpu_api == "?":
            gpu_api = requested_gpu_api

        hdr_metadata_forced = ("pq" in t_trc and "bt.2020" in t_prim)
        hdr_vo_confirmed = ("bt.2020" in out_prim and "pq" in out_trc)
        hdr_vo_unknown = (out_prim == "?" or out_trc == "?")
        hdr_info = {
            "primaries": out_prim,
            "transfer": out_trc,
            "levels": out_lvl,
            "sig_peak": _g(vop, "sig-peak", "sig_peak"),
            "max_cll": _g(vop, "max-cll", "max_cll"),
            "max_fall": _g(vop, "max-fall", "max_fall"),
            "vo": _fmt_prop_text(_prop("current_vo")),
            "gpu_api": gpu_api,
            "gpu_api_requested": requested_gpu_api,
            "hdr_metadata_forced": hdr_metadata_forced,
            "hdr_vo_confirmed": hdr_vo_confirmed,
            "hdr_vo_unknown": hdr_vo_unknown,
            "hdr_active": hdr_metadata_forced and (hdr_vo_confirmed or hdr_vo_unknown),
            "target_trc": t_trc,
            "target_prim": t_prim,
        }
        aux = {"vp": vp, "vop": vop}
        props = {"t_trc": t_trc, "t_prim": t_prim}
        return hdr_info, aux, props

    @staticmethod
    def _coerce_shader_prop_list(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value).strip()
        if not text or text.lower() in {"none", "no", "[]"}:
            return []
        if os.pathsep in text:
            parts = [part.strip() for part in text.split(os.pathsep) if part.strip()]
            if len(parts) > 1:
                return parts
        return [text]

    @staticmethod
    def _shader_path_keys(paths: list[str]) -> list[str]:
        return [
            os.path.basename(str(path or "")).strip().lower()
            for path in paths
            if str(path or "").strip()
        ]

    def get_active_shader_paths(self) -> list[str]:
        p = self._player
        if p is None:
            return []
        try:
            return self._coerce_shader_prop_list(getattr(p, "glsl_shaders", None))
        except Exception:
            return []

    def _verify_glsl_shaders(self, expected_paths: list[str]) -> bool:
        import time as _t

        expected_keys = self._shader_path_keys(expected_paths)
        last_active_paths: list[str] = []
        last_active_keys: list[str] = []
        for _ in range(5):
            last_active_paths = self.get_active_shader_paths()
            last_active_keys = self._shader_path_keys(last_active_paths)
            if last_active_keys == expected_keys:
                if self._diag_enabled and expected_keys:
                    print(f"[mpv] glsl-shaders active: {last_active_paths}")
                return True
            _t.sleep(0.02)
        self._last_scale_error = (
            "glsl-shaders verification failed: "
            f"expected={expected_keys or []}, got={last_active_keys or []}"
        )
        print(f"[mpv] {self._last_scale_error}")
        return False

    def _set_glsl_shaders(self, shader_paths: list[str]) -> bool:
        p = self._player
        if p is None:
            return False
        self._last_scale_error = None
        paths = self._normalize_shader_paths(shader_paths)
        try:
            try:
                p.glsl_shaders = paths
                return self._verify_glsl_shaders(paths)
            except Exception:
                pass
            joined = os.pathsep.join(paths)
            p.command("set", "glsl-shaders", joined)
            return self._verify_glsl_shaders(paths)
        except Exception as exc:
            self._last_scale_error = str(exc)
            print(f"[mpv] glsl-shaders set failed: {exc}")
            return False

    def _set_shader_chain(self, shader_paths: list[str]) -> bool:
        return self._set_glsl_shaders(self._normalize_shader_paths(shader_paths))

    @staticmethod
    def _kernel_antiring(scale_kernel: str) -> tuple[str, float]:
        k = str(scale_kernel or "bicubic").strip().lower()
        if not k:
            k = "bicubic"
        if k in {"fsr"}:
            return "fsr", 0.0
        if k in {"ssim_superres", "ssim"}:
            return "spline36", 0.0
        if k in {"ewa_lanczossharp", "ewa_lanczos"}:
            return "ewa_lanczossharp", 0.20
        return k, 0.0

    def _attach_audio_async(self, audio_path: str):
        def _worker():
            import time as _t
            last_exc = None
            for _ in range(15):
                if self._shutdown.is_set():
                    return
                p = self._player
                if p is None:
                    return
                try:
                    p.command("audio-add", audio_path, "select")
                    _t.sleep(0.05)
                    aid = getattr(p, "aid", None)
                    if aid not in (None, "no"):
                        print(f"[mpv audio] attached (aid={aid})")
                        return
                except Exception as exc:
                    last_exc = exc
                _t.sleep(0.2)
            if last_exc is not None:
                print(f"[mpv audio] attach failed after retries: {last_exc}")
            else:
                print("[mpv audio] attach failed: no active audio track")
        threading.Thread(target=_worker, daemon=True).start()

    @staticmethod
    def _poke_named_pipe_client(pipe_name: str | None):
        if not pipe_name:
            return
        try:
            import ctypes
            import ctypes.wintypes as wt

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            GENERIC_WRITE = 0x40000000
            OPEN_EXISTING = 3
            INVALID_HANDLE_VALUE = wt.HANDLE(-1).value

            h = kernel32.CreateFileW(
                pipe_name,
                GENERIC_WRITE,
                0,
                None,
                OPEN_EXISTING,
                0,
                None,
            )
            if h != INVALID_HANDLE_VALUE:
                kernel32.CloseHandle(h)
        except Exception:
            pass

    @staticmethod
    def _pipe_feeder_fn(
        pipe_name: str,
        frame_queue: _queue.Queue,
        shutdown: threading.Event,
        ready_event: threading.Event | None = None,
    ):
        import ctypes
        import ctypes.wintypes as wt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        PIPE_ACCESS_OUTBOUND = 0x00000002
        PIPE_TYPE_BYTE = 0x00000000
        PIPE_WAIT = 0x00000000
        INVALID_HANDLE_VALUE = wt.HANDLE(-1).value
        PIPE_BUF = 1 << 22

        h = kernel32.CreateNamedPipeW(
            pipe_name,
            PIPE_ACCESS_OUTBOUND,
            PIPE_TYPE_BYTE | PIPE_WAIT,
            1,
            PIPE_BUF,
            0,
            0,
            None,
        )
        if h == INVALID_HANDLE_VALUE:
            if ready_event is not None:
                ready_event.set()
            return

        if ready_event is not None:
            ready_event.set()
        kernel32.ConnectNamedPipe(h, None)

        written = wt.DWORD(0)
        while not shutdown.is_set():
            try:
                item = frame_queue.get(timeout=0.2)
            except _queue.Empty:
                continue
            if item is None:
                break
            buf = bytes(item) if not isinstance(item, bytes) else item
            off = 0
            while off < len(buf):
                chunk = buf[off:off + PIPE_BUF]
                ok = kernel32.WriteFile(
                    h,
                    chunk,
                    len(chunk),
                    ctypes.byref(written),
                    None,
                )
                if not ok:
                    shutdown.set()
                    break
                off += written.value

        kernel32.FlushFileBuffers(h)
        kernel32.DisconnectNamedPipe(h)
        kernel32.CloseHandle(h)

    def start_playback(
        self,
        width: int,
        height: int,
        fps: float = 30.0,
        scale_kernel: str = "bicubic",
        scale_antiring: float | None = None,
        cas_strength: float | None = None,
        audio_path: str | None = None,
        film_grain: bool = False,
        force_hdr_metadata: bool = True,
        vsync_timed: bool = False,
        sdr_transfer: str = "bt1886",
        live_capture: bool | None = None,
    ) -> bool:
        self.stop_playback()
        self._shutdown.clear()
        self._fps = float(fps) if fps and fps > 0 else 30.0
        self._force_hdr_metadata = bool(force_hdr_metadata)
        self._sdr_transfer = self._normalize_sdr_transfer(sdr_transfer)
        vsync_timed = bool(vsync_timed)
        live_capture = bool(vsync_timed if live_capture is None else live_capture)
        buffer_frames = (
            max(1, int(LIVE_CAPTURE_MPV_BUFFER_FRAMES))
            if vsync_timed
            else 1
        )
        self._queue = _queue.Queue(maxsize=buffer_frames)
        self._diag_enabled = bool(self._mpv_diag and self._force_hdr_metadata)
        self._last_scale_error = None
        requested_kernel = str(scale_kernel or "").strip().lower()
        kernel_name, antiring = self._kernel_antiring(scale_kernel)
        if scale_antiring is not None:
            antiring = max(0.0, min(1.0, float(scale_antiring)))

        remembered_scale_kernel = (
            "ssim_superres" if requested_kernel == "ssim_superres" else str(kernel_name)
        )

        self._last_playback_cfg = {
            "width": int(width),
            "height": int(height),
            "fps": float(self._fps),
            "scale_kernel": remembered_scale_kernel,
            "scale_antiring": None if scale_antiring is None else float(scale_antiring),
            "cas_strength": None if cas_strength is None else float(cas_strength),
            "audio_path": audio_path,
            "film_grain": bool(film_grain),
            "force_hdr_metadata": self._force_hdr_metadata,
            "vsync_timed": vsync_timed,
            "buffer_frames": int(buffer_frames),
            "sdr_transfer": self._sdr_transfer,
            "live_capture": live_capture,
        }

        pipe_id = id(self)
        self._pipe_name = rf"\\.\pipe\hdrtvnet_mpv_{pipe_id}"

        wid = str(int(self.winId()))
        pipe_url = f"lavf://file:{self._pipe_name}"

        frame_bytes = width * height * 6
        max_demux = str(frame_bytes * max(1, int(buffer_frames)))
        readahead_secs = 0.0
        if vsync_timed:
            readahead_secs = min(
                0.5,
                max(0.0, (float(buffer_frames) - 1.0) / max(self._fps, 1.0)),
            )

        use_fsr = (requested_kernel == "fsr" and self._ensure_fsr_shader())
        if requested_kernel == "fsr" and not use_fsr:
            print(f"[mpv] FSR shader unavailable (download failed). Falling back to {self._best_mpv_scale}.")
            kernel_name = self._best_mpv_scale
            if self._last_playback_cfg is not None:
                self._last_playback_cfg["scale_kernel"] = str(kernel_name)
        use_ssim = (
            requested_kernel == "ssim_superres" and self._ensure_ssim_superres_shader()
        )
        if requested_kernel == "ssim_superres" and not use_ssim:
            print(f"[mpv] SSimSuperRes shader unavailable (download failed). Falling back to {self._best_mpv_scale}.")
            kernel_name = self._best_mpv_scale
            if self._last_playback_cfg is not None:
                self._last_playback_cfg["scale_kernel"] = str(kernel_name)
        use_fsr = (requested_kernel == "fsr" and use_fsr)
        use_film_grain = bool(film_grain and self._ensure_filmgrain_shader())
        if film_grain and not use_film_grain:
            print("[mpv] film grain shader unavailable (download failed).")
            if self._last_playback_cfg is not None:
                self._last_playback_cfg["film_grain"] = False

        # Use gpu-next + colorspace hint so mpv can adapt target output while
        # dragging between HDR/SDR displays (with a compatibility fallback).
        self._target_colorspace_hint = "auto"
        hdr_target_peak = _env_hdr_target_peak_nits()
        deband_options = _mpv_deband_options(live_capture=live_capture)
        dither_options = _mpv_dither_options(live_capture=live_capture)
        mpv_kwargs = dict(
            wid=wid,
            vo="gpu-next",
            gpu_api="vulkan",
            demuxer="rawvideo",
            demuxer_rawvideo_w=str(width),
            demuxer_rawvideo_h=str(height),
            demuxer_rawvideo_mp_format="rgb48le",
            demuxer_rawvideo_fps=str(self._fps),
            untimed=(not vsync_timed),
            audio="auto",
            audio_file_auto="no",
            osc="no",
            input_default_bindings="no",
            input_vo_keyboard="no",
            cache="no",
            demuxer_max_bytes=max_demux,
            demuxer_readahead_secs=str(readahead_secs),
            video_sync="display-resample" if vsync_timed else "desync",
            scale="bilinear" if use_fsr else str(kernel_name),
            cscale="bilinear" if use_fsr else str(kernel_name),
            scale_antiring=str(antiring),
            cscale_antiring=str(antiring),
        )

        mpv_kwargs.update(deband_options)
        mpv_kwargs.update(dither_options)
        deband_active = bool(deband_options)
        dither_active = bool(dither_options)
        use_interpolation = bool(vsync_timed and (not live_capture or _mpv_live_interpolation_enabled()))
        if use_interpolation:
            mpv_kwargs.update(
                interpolation=True,
                tscale="oversample",
            )
        shader_paths = self._build_shader_paths(
            use_fsr=use_fsr,
            use_ssim=use_ssim,
            use_film_grain=use_film_grain,
        )
        if self._force_hdr_metadata:
            # ✅ Both GT and Convert use same HDR visualization pipeline
            vf_chain = "format=colorlevels=full:primaries=bt.2020:gamma=pq"
            if cas_strength is not None and cas_strength > 0.0:
                vf_chain += f",cas={cas_strength}"

            mpv_kwargs.update(
                target_colorspace_hint=self._target_colorspace_hint,
                hdr_compute_peak="yes",
                tone_mapping="bt.2390",
                tone_mapping_param=0.8,
                vf=vf_chain,
            )

        else:
            # ✅ SDR
            vf_chain = self._sdr_format_filter(cas_strength)
            mpv_kwargs.update(
                target_colorspace_hint=self._target_colorspace_hint,
                vf=vf_chain,
            )
        use_external_audio = bool(audio_path and os.path.isfile(audio_path))
        if not use_external_audio:
            mpv_kwargs["audio"] = "no"

        player = None
        startup_exc = None
        try:
            player = self._mpv_lib.MPV(**mpv_kwargs)
            self._active_vo = "gpu-next"
        except Exception as exc:
            startup_exc = exc

        if player is None and dither_active:
            try:
                player = self._mpv_lib.MPV(**_without_dither_options(mpv_kwargs))
                self._active_vo = "gpu-next"
                dither_active = False
                note = (
                    "mpv dither unavailable; continuing without display dither: "
                    f"{startup_exc}"
                )
                print(f"[mpv] {note}")
                try:
                    self.runtime_notice.emit(note)
                except Exception:
                    pass
            except Exception as exc:
                startup_exc = exc

        if player is None and deband_active:
            try:
                player = self._mpv_lib.MPV(**_without_display_quality_options(mpv_kwargs))
                self._active_vo = "gpu-next"
                deband_active = False
                dither_active = False
                note = (
                    "mpv deband/dither unavailable; continuing without display quality filters: "
                    f"{startup_exc}"
                )
                print(f"[mpv] {note}")
                try:
                    self.runtime_notice.emit(note)
                except Exception:
                    pass
            except Exception as exc:
                startup_exc = exc

        if player is None and vsync_timed:
            fallback_kwargs = _without_display_quality_options(mpv_kwargs)
            fallback_kwargs["untimed"] = True
            fallback_kwargs["video_sync"] = "desync"
            fallback_kwargs.pop("interpolation", None)
            fallback_kwargs.pop("tscale", None)
            try:
                player = self._mpv_lib.MPV(**fallback_kwargs)
                self._active_vo = "gpu-next"
                vsync_timed = False
                deband_active = False
                dither_active = False
                if self._last_playback_cfg is not None:
                    self._last_playback_cfg["vsync_timed"] = False
                note = (
                    "mpv vsync/display-quality startup fallback active; using "
                    f"low-latency untimed display: {startup_exc}"
                )
                print(f"[mpv] {note}")
                try:
                    self.runtime_notice.emit(note)
                except Exception:
                    pass
            except Exception as exc:
                startup_exc = exc

        if player is None:
            self._last_scale_error = f"mpv startup failed with gpu-next/vulkan: {startup_exc}"
            print(f"[mpv] {self._last_scale_error}")
            try:
                self.runtime_notice.emit(self._last_scale_error)
            except Exception:
                pass
            self._queue = None
            self._pipe_name = None
            return False
        self._player = player
        pipe_ready = threading.Event()
        self._feeder = threading.Thread(
            target=self._pipe_feeder_fn,
            args=(self._pipe_name, self._queue, self._shutdown, pipe_ready),
            daemon=True,
        )
        self._feeder.start()
        if not pipe_ready.wait(timeout=1.0):
            self._last_scale_error = "mpv named-pipe feeder failed to initialize."
            print(f"[mpv] {self._last_scale_error}")
            try:
                self.runtime_notice.emit(self._last_scale_error)
            except Exception:
                pass
            self.stop_playback()
            return False
        try:
            player.play(pipe_url)
        except Exception as exc:
            self._last_scale_error = f"mpv play failed: {exc}"
            print(f"[mpv] {self._last_scale_error}")
            try:
                self.runtime_notice.emit(self._last_scale_error)
            except Exception:
                pass
            self.stop_playback()
            return False
        if shader_paths and not self._set_shader_chain(shader_paths):
            try:
                self.runtime_notice.emit(
                    f"mpv shader load check failed: {self._last_scale_error or 'unknown error'}"
                )
            except Exception:
                pass
        if use_external_audio:
            self._attach_audio_async(audio_path)

        def _hdr_monitor():
            import time as _t
            printed_once = False
            while not self._shutdown.is_set():
                _t.sleep(0.5)
                p = self._player
                if p is None:
                    return
                try:
                    hdr_info, aux, props = self._build_hdr_info_snapshot(p)
                    self.hdr_info_ready.emit(hdr_info)

                    if self._diag_enabled and not printed_once and len(aux.get("vop", {})) > 2:
                        print("\n╔══════════ mpv HDR diagnostic ══════════╗")
                        print(f"║  video-params keys : {list(aux['vp'].keys())}")
                        print(f"║  video-out-params  : {list(aux['vop'].keys())}")
                        print(f"║  VO output prims   : {hdr_info.get('primaries', '?')}")
                        print(f"║  VO output TRC     : {hdr_info.get('transfer', '?')}")
                        print(f"║  VO output levels  : {hdr_info.get('levels', '?')}")
                        print(f"║  target-trc        : {props.get('t_trc', '?')}")
                        print(f"║  target-prim       : {props.get('t_prim', '?')}")
                        print(f"║  target-peak       : {getattr(p, 'target_peak', '?')}")
                        print(f"║  current-vo        : {getattr(p, 'current_vo', '?')}")
                        print(
                            "║  gpu-api           : "
                            f"{hdr_info.get('gpu_api', '?')} "
                            f"(requested {hdr_info.get('gpu_api_requested', '?')})"
                        )
                        hint_mode = str(self._target_colorspace_hint or "?")
                        print(f"║  colorspace-hint   : {hint_mode}")
                        print("╠════════════════════════════════════════╣")
                        if hdr_info.get("hdr_vo_confirmed", False):
                            print("║  ✓ VO confirms BT.2020 + PQ output")
                        else:
                            print("║  ⚠ VO not yet confirming BT.2020 + PQ")
                        print("╚════════════════════════════════════════╝\n")
                        printed_once = True
                except Exception:
                    pass

        threading.Thread(target=_hdr_monitor, daemon=True).start()
        return True

    def feed_frame(self, rgb48_bytes):
        q = self._queue
        if q is None:
            return
        try:
            q.put_nowait(rgb48_bytes)
        except _queue.Full:
            try:
                q.get_nowait()
            except _queue.Empty:
                pass
            try:
                q.put_nowait(rgb48_bytes)
            except _queue.Full:
                pass

    def set_paused(self, paused: bool):
        p = self._player
        if p is None:
            return
        try:
            p.pause = bool(paused)
        except Exception:
            pass

    def seek_seconds(self, sec: float):
        p = self._player
        if p is None:
            return
        try:
            p.command("seek", max(0.0, float(sec)), "absolute")
        except Exception:
            if not self._seek_warned:
                self._seek_warned = True
                pass

    def set_muted(self, muted: bool):
        p = self._player
        if p is None:
            return
        try:
            p.mute = bool(muted)
        except Exception:
            try:
                p.command("set", "mute", "yes" if muted else "no")
            except Exception:
                pass

    def set_volume_percent(self, volume_percent: int):
        p = self._player
        if p is None:
            return
        v = max(0, min(100, int(volume_percent)))
        try:
            p.volume = v
        except Exception:
            try:
                p.command("set", "volume", str(v))
            except Exception:
                pass

    def set_audio_track_ordinal(self, ordinal: int) -> bool:
        p = self._player
        if p is None:
            return False
        try:
            tracks = getattr(p, "track_list", None)
        except Exception:
            tracks = None
        if isinstance(tracks, list):
            aud = [t for t in tracks if str(t.get("type", "")).lower() == "audio"]
            if aud:
                idx = max(0, min(int(ordinal), len(aud) - 1))
                aid = aud[idx].get("id")
                if aid is not None:
                    try:
                        p.command("set", "aid", str(aid))
                        return True
                    except Exception:
                        pass
        try:
            p.command("set", "aid", str(max(1, int(ordinal) + 1)))
            return True
        except Exception:
            return False

    def set_scale_kernel(self, scale_kernel: str, scale_antiring: float | None = None) -> bool:
        p = self._player
        if p is None:
            return False
        self._last_scale_error = None
        requested_kernel = str(scale_kernel or "").strip().lower()
        kernel, antiring = self._kernel_antiring(scale_kernel)
        if scale_antiring is not None:
            antiring = max(0.0, min(1.0, float(scale_antiring)))
        film_on = False
        if self._last_playback_cfg is not None:
            film_on = bool(self._last_playback_cfg.get("film_grain", False))
        try:
            use_fsr = (kernel == "fsr" and self._ensure_fsr_shader())
            if kernel == "fsr" and not use_fsr:
                kernel = self._best_mpv_scale
            use_ssim = (requested_kernel == "ssim_superres" and self._ensure_ssim_superres_shader())
            if requested_kernel == "ssim_superres" and not use_ssim:
                use_ssim = False
            if use_fsr:
                p.command("set", "scale", "bilinear")
                p.command("set", "cscale", "bilinear")
            else:
                p.command("set", "scale", kernel)
                p.command("set", "cscale", kernel)
            use_film_grain = bool(film_on and self._ensure_filmgrain_shader())
            if film_on and not use_film_grain:
                print("[mpv] film grain shader unavailable (download failed).")
            shader_paths = self._build_shader_paths(
                use_fsr=use_fsr,
                use_ssim=use_ssim,
                use_film_grain=use_film_grain,
            )
            if not self._set_shader_chain(shader_paths):
                raise RuntimeError("Failed to set glsl-shaders.")
            p.command("set", "scale-antiring", str(antiring))
            p.command("set", "cscale-antiring", str(antiring))
            if self._last_playback_cfg is not None:
                if use_ssim:
                    self._last_playback_cfg["scale_kernel"] = "ssim_superres"
                else:
                    self._last_playback_cfg["scale_kernel"] = str(kernel)
                self._last_playback_cfg["scale_antiring"] = float(antiring)
                self._last_playback_cfg["film_grain"] = bool(use_film_grain)
            return True
        except Exception as exc:
            self._last_scale_error = str(exc)
            print(f"[mpv] scale hot-swap failed: {exc}")
            return False

    def set_cas_strength(self, cas_strength: float | None) -> bool:
        p = self._player
        if p is None:
            return False
        try:
            cas_val = float(cas_strength or 0.0)
        except Exception:
            cas_val = 0.0
        if self._force_hdr_metadata:
            vf_chain = "format=colorlevels=full:primaries=bt.2020:gamma=pq"
        else:
            vf_chain = self._sdr_format_filter(cas_val)
        if self._force_hdr_metadata and cas_val > 0.0:
            vf_chain += f",cas={cas_val}"
        try:
            p.command("set", "vf", vf_chain)
            if self._force_hdr_metadata:
                p.command("set", "tone-mapping", "bt.2390")
            if self._last_playback_cfg is not None:
                self._last_playback_cfg["cas_strength"] = float(cas_val)
            return True
        except Exception as exc:
            self._last_scale_error = str(exc)
            print(f"[mpv] cas hot-swap failed: {exc}")
            return False

    def set_film_grain(self, enabled: bool) -> bool:
        p = self._player
        if p is None:
            return False
        use_film = bool(enabled and self._ensure_filmgrain_shader())
        if enabled and not use_film:
            print("[mpv] film grain shader unavailable (download failed).")
        use_fsr = False
        use_ssim = False
        if self._last_playback_cfg is not None:
            k = str(self._last_playback_cfg.get("scale_kernel", "")).lower()
            use_fsr = (k == "fsr" and self._ensure_fsr_shader())
            use_ssim = (
                k == "ssim_superres" and self._ensure_ssim_superres_shader()
            )
        shader_paths = self._build_shader_paths(
            use_fsr=use_fsr,
            use_ssim=use_ssim,
            use_film_grain=use_film,
        )
        try:
            if not self._set_shader_chain(shader_paths):
                raise RuntimeError("Failed to set glsl-shaders.")
            if self._last_playback_cfg is not None:
                self._last_playback_cfg["film_grain"] = bool(use_film)
            return True
        except Exception as exc:
            self._last_scale_error = str(exc)
            print(f"[mpv] film grain hot-swap failed: {exc}")
            return False

    def is_fsr_active(self) -> bool:
        try:
            return any(
                "fsr" in str(path).lower() for path in self.get_active_shader_paths()
            )
        except Exception:
            return False

    def get_time_seconds(self) -> float | None:
        p = self._player
        if p is None:
            return None
        for prop in ("time_pos", "playback_time"):
            try:
                v = getattr(p, prop, None)
                if v is not None:
                    return float(v)
            except Exception:
                pass
        return None

    def stop_playback(self):
        self._shutdown.set()
        q = self._queue
        if q is not None:
            try:
                q.put_nowait(None)
            except _queue.Full:
                pass
        if self._feeder is not None and self._feeder.is_alive():
            self._poke_named_pipe_client(self._pipe_name)
        if self._player is not None:
            try:
                self._player.terminate()
            except Exception:
                pass
            self._player = None
        if self._feeder is not None:
            self._feeder.join(timeout=3)
            self._feeder = None
        self._queue = None

    def refresh_surface(self):
        cfg = self._last_playback_cfg
        if not cfg:
            return
        allowed = {
            "width",
            "height",
            "fps",
            "scale_kernel",
            "scale_antiring",
            "cas_strength",
            "audio_path",
            "force_hdr_metadata",
            "film_grain",
            "vsync_timed",
            "sdr_transfer",
            "live_capture",
        }
        safe_cfg = {k: v for k, v in cfg.items() if k in allowed}
        self.start_playback(**safe_cfg)
