from __future__ import annotations

import atexit
import ctypes
import hashlib
import os
import pathlib
import threading
import tempfile

_timer_lock = threading.Lock()
_timer_refcount = 0
_winmm = None
try:
    if os.name == "nt":
        _winmm = ctypes.WinDLL("winmm")
except Exception:
    _winmm = None


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


_COMPILE_NAMESPACE_VERSION = "v2"
_COMPILE_NAMESPACE_CACHE: dict[str, str] = {}


def project_root_from_path(current_file: str | None = None) -> str:
    """Best-effort repo root discovery from a file inside this project."""
    if current_file:
        p = pathlib.Path(current_file).resolve()
        start = p.parent if p.is_file() else p
    else:
        start = pathlib.Path(__file__).resolve().parent

    for parent in (start, *start.parents):
        if (parent / "src" / "models" / "hdrtvnet_torch.py").is_file():
            return str(parent)
        if (
            parent.name == "src"
            and (parent / "models" / "hdrtvnet_torch.py").is_file()
        ):
            return str(parent.parent)

    # Fallback for unusual layouts: assume this file lives under src/.
    here = pathlib.Path(__file__).resolve().parent
    return str(here.parent)


def _compile_signature_files(project_root: str) -> list[pathlib.Path]:
    root = pathlib.Path(project_root)
    files: list[pathlib.Path] = []

    models_dir = root / "src" / "models"
    if models_dir.is_dir():
        files.extend(sorted(models_dir.rglob("*.py")))

    for rel in (
        ("src", "gui_compile_cache.py"),
        ("src", "compile_kernels.py"),
    ):
        p = root.joinpath(*rel)
        if p.is_file():
            files.append(p)

    # Deduplicate while preserving sort order.
    uniq: list[pathlib.Path] = []
    seen: set[str] = set()
    for p in files:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


def compile_cache_namespace(current_file: str | None = None) -> str:
    """Return a repo-specific compile ABI fingerprint for cache namespacing."""
    project_root = project_root_from_path(current_file)
    cached = _COMPILE_NAMESPACE_CACHE.get(project_root)
    if cached:
        return cached

    digest = hashlib.sha256()
    digest.update(_COMPILE_NAMESPACE_VERSION.encode("utf-8"))
    digest.update(b"\0")

    for p in _compile_signature_files(project_root):
        rel = p.relative_to(project_root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(p.read_bytes())
        except Exception:
            digest.update(b"<missing>")
        digest.update(b"\0")

    ns = f"{_COMPILE_NAMESPACE_VERSION}_{digest.hexdigest()[:16]}"
    _COMPILE_NAMESPACE_CACHE[project_root] = ns
    return ns


def _compile_profiles_root(base: str) -> str:
    return os.path.join(base, "compile_profiles")


def _profile_marker_path(profile_root: str) -> str:
    return os.path.join(profile_root, "triton", "cache", "hdrtvnet_compiled.txt")


def _profile_has_compiled_markers(profile_root: str) -> bool:
    marker = _profile_marker_path(profile_root)
    return os.path.isfile(marker) and os.path.getsize(marker) > 0


def _profile_has_any_entries(profile_root: str) -> bool:
    try:
        with os.scandir(profile_root) as it:
            for _entry in it:
                return True
    except Exception:
        return False
    return False


def _resolve_compile_profile_root(base: str, namespace: str) -> str:
    profiles_root = _compile_profiles_root(base)
    current_root = os.path.join(profiles_root, namespace)
    if _profile_has_compiled_markers(current_root):
        return current_root
    if _profile_has_any_entries(current_root):
        return current_root

    candidates: list[tuple[float, str]] = []
    try:
        with os.scandir(profiles_root) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                if entry.name == namespace:
                    continue
                if not entry.name.startswith(f"{_COMPILE_NAMESPACE_VERSION}_"):
                    continue
                if not _profile_has_compiled_markers(entry.path):
                    continue
                try:
                    mtime = float(entry.stat().st_mtime)
                except Exception:
                    mtime = 0.0
                candidates.append((mtime, entry.path))
    except Exception:
        return current_root

    if not candidates:
        return current_root
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def project_cache_root(current_file: str | None = None) -> str:
    """Return the default cache root for this repo's compile ABI."""
    explicit = os.environ.get("HDRTVNET_CACHE_DIR")
    if explicit:
        return explicit
    base = default_cache_root()
    ns = compile_cache_namespace(current_file)
    return _resolve_compile_profile_root(base, ns)


def enable_high_resolution_timer(period_ms: int = 1) -> bool:
    """Request a 1 ms Windows timer period for steadier frame pacing."""
    global _timer_refcount
    if os.name != "nt" or _winmm is None:
        return False
    period = max(1, int(period_ms))
    with _timer_lock:
        if _timer_refcount == 0:
            try:
                rc = int(_winmm.timeBeginPeriod(period))
            except Exception:
                return False
            if rc != 0:
                return False
        _timer_refcount += 1
    return True


def disable_high_resolution_timer(period_ms: int = 1) -> None:
    """Release a previous high-resolution timer request."""
    global _timer_refcount
    if os.name != "nt" or _winmm is None:
        return
    period = max(1, int(period_ms))
    with _timer_lock:
        if _timer_refcount <= 0:
            _timer_refcount = 0
            return
        _timer_refcount -= 1
        if _timer_refcount != 0:
            return
        try:
            _winmm.timeEndPeriod(period)
        except Exception:
            pass


atexit.register(disable_high_resolution_timer)
