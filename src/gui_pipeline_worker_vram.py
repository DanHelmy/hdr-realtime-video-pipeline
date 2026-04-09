from __future__ import annotations

import os
import subprocess
import threading


class PipelineWorkerVramMixin:
    """Windows app-VRAM polling helpers for PipelineWorker."""

    _APP_VRAM_POLL_INTERVAL_S = 2.0

    @staticmethod
    def _query_app_vram_mb_windows(pid: int) -> float | None:
        """Return dedicated GPU process memory (MB) for this process on Windows."""
        if os.name != "nt":
            return None
        try:
            pid_i = int(pid)
        except Exception:
            return None
        if pid_i <= 0:
            return None

        ps_cmd = (
            "$tag='pid_" + str(pid_i) + "'; "
            "$d=(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' "
            "| Select-Object -ExpandProperty CounterSamples "
            "| Where-Object { $_.InstanceName -like \"*$tag*\" } "
            "| Measure-Object -Property CookedValue -Sum).Sum; "
            "$td=if ($null -eq $d) { 0 } else { [double]$d }; "
            "[string]$td"
        )
        try:
            cp = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            raw = (cp.stdout or "").strip()
            if not raw:
                return 0.0
            bytes_used = float(raw)
            return max(0.0, bytes_used / (1024.0 * 1024.0))
        except Exception:
            return None

    def _start_app_vram_poll(self):
        """Poll app VRAM off the render loop to avoid FPS stalls."""
        if os.name != "nt":
            return
        self._app_vram_poll_stop.clear()

        def _poll():
            pid = os.getpid()
            while not self._app_vram_poll_stop.is_set():
                q = self._query_app_vram_mb_windows(pid)
                if q is not None:
                    self._app_vram_mb = q
                self._app_vram_poll_stop.wait(self._APP_VRAM_POLL_INTERVAL_S)

        self._app_vram_poll_thread = threading.Thread(target=_poll, daemon=True)
        self._app_vram_poll_thread.start()

    def _stop_app_vram_poll(self):
        self._app_vram_poll_stop.set()
        t = self._app_vram_poll_thread
        if t is not None:
            t.join(timeout=1.0)
        self._app_vram_poll_thread = None
