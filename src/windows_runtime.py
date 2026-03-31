from __future__ import annotations

import os
import tempfile


def ensure_windows_supported(component: str) -> None:
    """Exit early with a clear message when launched on non-Windows hosts."""
    if os.name == "nt":
        return
    raise SystemExit(
        f"{component} is Windows-only. Unsupported platform: {os.name}."
    )


def default_cache_root() -> str:
    """Return a stable per-user cache root for Windows builds."""
    local_app = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if local_app:
        return os.path.join(local_app, "HDRTVNetCache")
    return os.path.join(tempfile.gettempdir(), "HDRTVNetCache")
