"""Embedded mpv HDR widget extracted from gui.py."""

from __future__ import annotations

import os
import queue as _queue
import threading

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget


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
        self.setMinimumSize(320, 180)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setStyleSheet("background: #111;")
        self._player = None
        self._pipe_name: str | None = None
        self._pipe_handle = None
        self._feeder: threading.Thread | None = None
        self._queue: _queue.Queue | None = None
        self._shutdown = threading.Event()
        self._fps = 30.0
        self._force_hdr_metadata = True
        self._diag_enabled = bool(mpv_diag)
        self._seek_warned = False
        self._last_playback_cfg: dict | None = None
        self._last_scale_error: str | None = None
        self._active_vo: str = "gpu"
        self._target_colorspace_hint: str = "no"

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
                return str(v) if v is not None else "?"
            except Exception:
                return "?"

        out_prim = _g(vop, "primaries", "colormatrix-primaries")
        out_trc = _g(vop, "gamma", "transfer")
        out_lvl = _g(vop, "levels", "colorlevels")
        t_trc = _prop("target_trc")
        t_prim = _prop("target_prim")

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
            "vo": _prop("current_vo"),
            "gpu_api": "d3d11",
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

    def _set_glsl_shaders(self, shader_paths: list[str]) -> bool:
        p = self._player
        if p is None:
            return False
        paths = self._normalize_shader_paths(shader_paths)
        try:
            try:
                p.glsl_shaders = paths
                return True
            except Exception:
                pass
            joined = os.pathsep.join(paths)
            p.command("set", "glsl-shaders", joined)
            return True
        except Exception as exc:
            self._last_scale_error = str(exc)
            print(f"[mpv] glsl-shaders set failed: {exc}")
            return False

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
    def _pipe_feeder_fn(pipe_name: str, frame_queue: _queue.Queue, shutdown: threading.Event):
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
            return

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
    ):
        self.stop_playback()
        self._shutdown.clear()
        self._queue = _queue.Queue(maxsize=1)
        self._fps = float(fps) if fps and fps > 0 else 30.0
        self._force_hdr_metadata = bool(force_hdr_metadata)
        self._diag_enabled = bool(self._mpv_diag and self._force_hdr_metadata)
        kernel_name, antiring = self._kernel_antiring(scale_kernel)
        if scale_antiring is not None:
            antiring = max(0.0, min(1.0, float(scale_antiring)))

        self._last_playback_cfg = {
            "width": int(width),
            "height": int(height),
            "fps": float(self._fps),
            "scale_kernel": str(kernel_name),
            "scale_antiring": None if scale_antiring is None else float(scale_antiring),
            "cas_strength": None if cas_strength is None else float(cas_strength),
            "audio_path": audio_path,
            "film_grain": bool(film_grain),
            "force_hdr_metadata": self._force_hdr_metadata,
        }

        pipe_id = id(self)
        self._pipe_name = rf"\\.\pipe\hdrtvnet_mpv_{pipe_id}"

        self._feeder = threading.Thread(
            target=self._pipe_feeder_fn,
            args=(self._pipe_name, self._queue, self._shutdown),
            daemon=True,
        )
        self._feeder.start()

        wid = str(int(self.winId()))
        pipe_url = f"lavf://file:{self._pipe_name}"

        frame_bytes = width * height * 6
        max_demux = str(frame_bytes)

        use_fsr = (kernel_name == "fsr" and self._ensure_fsr_shader())
        if kernel_name == "fsr" and not use_fsr:
            print(f"[mpv] FSR shader unavailable (download failed). Falling back to {self._best_mpv_scale}.")
            kernel_name = self._best_mpv_scale
            if self._last_playback_cfg is not None:
                self._last_playback_cfg["scale_kernel"] = str(kernel_name)
        use_ssim = (kernel_name == "ssim_superres" and self._ensure_ssim_superres_shader())
        if kernel_name == "ssim_superres" and not use_ssim:
            print(f"[mpv] SSimSuperRes shader unavailable (download failed). Falling back to {self._best_mpv_scale}.")
            kernel_name = self._best_mpv_scale
            if self._last_playback_cfg is not None:
                self._last_playback_cfg["scale_kernel"] = str(kernel_name)
        use_fsr = (kernel_name == "fsr")
        use_ssim = (kernel_name == "ssim_superres")
        use_film_grain = bool(film_grain and self._ensure_filmgrain_shader())
        if film_grain and not use_film_grain:
            print("[mpv] film grain shader unavailable (download failed).")
            if self._last_playback_cfg is not None:
                self._last_playback_cfg["film_grain"] = False

        # Use gpu-next + colorspace hint so mpv can adapt target output while
        # dragging between HDR/SDR displays (with a compatibility fallback).
        self._target_colorspace_hint = "auto"
        mpv_kwargs = dict(
            wid=wid,
            vo="gpu-next",
            gpu_api="d3d11",
            demuxer="rawvideo",
            demuxer_rawvideo_w=str(width),
            demuxer_rawvideo_h=str(height),
            demuxer_rawvideo_mp_format="rgb48le",
            demuxer_rawvideo_fps=str(self._fps),
            untimed=True,
            audio="auto",
            audio_file_auto="no",
            osc="no",
            input_default_bindings="no",
            input_vo_keyboard="no",
            cache="no",
            demuxer_max_bytes=max_demux,
            demuxer_readahead_secs=0,
            video_sync="desync",
            scale="bilinear" if use_fsr else str(kernel_name),
            cscale="bilinear" if use_fsr else str(kernel_name),
            scale_antiring=str(antiring),
            cscale_antiring=str(antiring),
        )
        shader_paths = []
        if use_fsr:
            shader_paths.append(self._fsr_shader_path)
        if use_ssim:
            shader_paths.append(self._ssim_superres_shader_path)
        if use_film_grain:
            shader_paths.append(self._filmgrain_shader_path)
        if self._force_hdr_metadata:
            vf_chain = "format=colorlevels=full:primaries=bt.2020:gamma=pq"
            if cas_strength is not None and cas_strength > 0.0:
                vf_chain += f",cas={cas_strength}"
            mpv_kwargs.update(
                target_colorspace_hint=self._target_colorspace_hint,
                vf=vf_chain,
            )
        else:
            vf_chain = "format=colorlevels=full:primaries=bt.709:gamma=bt.1886"
            if cas_strength is not None and cas_strength > 0.0:
                vf_chain += f",cas={cas_strength}"
            mpv_kwargs.update(
                target_colorspace_hint=self._target_colorspace_hint,
                vf=vf_chain,
            )
        use_external_audio = bool(audio_path and os.path.isfile(audio_path))
        if not use_external_audio:
            mpv_kwargs["audio"] = "no"

        try:
            player = self._mpv_lib.MPV(**mpv_kwargs)
            self._active_vo = "gpu-next"
        except Exception as exc:
            # Older libmpv builds may not support gpu-next/target hint.
            fallback_kwargs = dict(mpv_kwargs)
            fallback_kwargs.update(
                vo="gpu",
                target_colorspace_hint="no",
            )
            self._target_colorspace_hint = "no"
            player = self._mpv_lib.MPV(**fallback_kwargs)
            self._active_vo = "gpu"
            print(
                f"[mpv] gpu-next unavailable ({exc}); "
                "falling back to vo=gpu."
            )
            try:
                self.runtime_notice.emit(
                    "mpv fallback: gpu-next unavailable, using compatibility vo=gpu."
                )
            except Exception:
                pass
        self._player = player
        player.play(pipe_url)
        if shader_paths:
            self._set_glsl_shaders(shader_paths)
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
                        print("║  gpu-api           : d3d11 (forced)")
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
            shader_paths = []
            if use_fsr:
                shader_paths.append(self._fsr_shader_path)
            if use_ssim:
                shader_paths.append(self._ssim_superres_shader_path)
            if use_film_grain:
                shader_paths.append(self._filmgrain_shader_path)
            if not self._set_glsl_shaders(shader_paths):
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
            vf_chain = "format=colorlevels=full:primaries=bt.709:gamma=bt.1886"
        if cas_val > 0.0:
            vf_chain += f",cas={cas_val}"
        try:
            p.command("set", "vf", vf_chain)
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
        if self._last_playback_cfg is not None:
            k = str(self._last_playback_cfg.get("scale_kernel", "")).lower()
            use_fsr = (k == "fsr" and self._ensure_fsr_shader())
        shader_paths = []
        if use_fsr:
            shader_paths.append(self._fsr_shader_path)
        if use_film:
            shader_paths.append(self._filmgrain_shader_path)
        try:
            if not self._set_glsl_shaders(shader_paths):
                raise RuntimeError("Failed to set glsl-shaders.")
            if self._last_playback_cfg is not None:
                self._last_playback_cfg["film_grain"] = bool(use_film)
            return True
        except Exception as exc:
            self._last_scale_error = str(exc)
            print(f"[mpv] film grain hot-swap failed: {exc}")
            return False

    def is_fsr_active(self) -> bool:
        p = self._player
        if p is None:
            return False
        try:
            val = getattr(p, "glsl_shaders", None)
            if not val:
                return False
            if isinstance(val, (list, tuple)):
                return any("fsr" in str(v).lower() for v in val)
            return "fsr" in str(val).lower()
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
        }
        safe_cfg = {k: v for k, v in cfg.items() if k in allowed}
        self.start_playback(**safe_cfg)
