from __future__ import annotations

import atexit
import ctypes
import hashlib
import logging
import os
import pathlib
import subprocess
import sys
import threading

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


_TORCH_WINDOWS_LOG_FILTER_INSTALLED = False


def install_torch_windows_warning_filter() -> None:
    """Hide known harmless Torch startup noise on Windows.

    PyTorch imports can emit two scary-looking warnings on Windows even when
    the app is working normally: elastic stdout/stderr redirects are unsupported
    on Windows/macOS, and cpp_extension cannot query `cl` when MSVC is not in
    PATH. Keep real Torch warnings visible and filter only those exact messages.
    """
    global _TORCH_WINDOWS_LOG_FILTER_INSTALLED
    if _TORCH_WINDOWS_LOG_FILTER_INSTALLED:
        return
    _TORCH_WINDOWS_LOG_FILTER_INSTALLED = True

    class _TorchWindowsNoiseFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                text = str(record.getMessage() or "")
            except Exception:
                return True
            if "Redirects are currently not supported in Windows or MacOs" in text:
                return False
            if (
                "Error checking compiler version for cl:" in text
                and "cannot find the file specified" in text.lower()
            ):
                return False
            return True

    noise_filter = _TorchWindowsNoiseFilter()
    for logger_name in (
        "torch.distributed.elastic.multiprocessing.redirects",
        "torch.utils.cpp_extension",
    ):
        try:
            logging.getLogger(logger_name).addFilter(noise_filter)
        except Exception:
            pass


def _has_hip_runtime_header(root: str | os.PathLike[str]) -> bool:
    try:
        return (pathlib.Path(root) / "include" / "hip" / "hip_runtime.h").is_file()
    except Exception:
        return False


def _rocm_env_candidate_roots() -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    for base in {sys.prefix, getattr(sys, "base_prefix", sys.prefix)}:
        if not base:
            continue
        site = pathlib.Path(base) / "Lib" / "site-packages"
        # Some ROCm wheels put hip_runtime.h under _rocm_sdk_core while
        # Triton may otherwise discover _rocm_sdk_devel first.
        roots.append(site / "_rocm_sdk_core")
        roots.append(site / "_rocm_sdk_devel")

    for env_key in ("HIP_PATH", "ROCM_PATH", "ROCM_HOME"):
        value = os.environ.get(env_key)
        if value:
            roots.append(pathlib.Path(value))

    rocm_root = pathlib.Path(r"C:\Program Files\AMD\ROCm")
    roots.append(rocm_root)
    if rocm_root.is_dir():
        try:
            roots.extend(p for p in sorted(rocm_root.iterdir()) if p.is_dir())
        except Exception:
            pass

    uniq: list[pathlib.Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            normalized = os.path.normcase(str(root.resolve()))
        except Exception:
            normalized = os.path.normcase(str(root))
        if normalized in seen:
            continue
        seen.add(normalized)
        uniq.append(root)
    return uniq


def configure_rocm_sdk_environment() -> str | None:
    """Point ROCm/Triton discovery at a root with HIP headers when available."""
    if os.name != "nt":
        return None
    selected: pathlib.Path | None = None
    for root in _rocm_env_candidate_roots():
        if _has_hip_runtime_header(root):
            selected = root
            break
    if selected is None:
        return None

    selected_str = str(selected)
    for env_key in ("HIP_PATH", "ROCM_PATH", "ROCM_HOME"):
        os.environ[env_key] = selected_str
    return selected_str


def rocm_sdk_include_dirs() -> list[str]:
    """Return ROCm include dirs that contain HIP headers, ordered by preference."""
    includes: list[str] = []
    seen: set[str] = set()
    for root in _rocm_env_candidate_roots():
        include_dir = pathlib.Path(root) / "include"
        if not (include_dir / "hip" / "hip_runtime.h").is_file():
            continue
        try:
            key = os.path.normcase(str(include_dir.resolve()))
        except Exception:
            key = os.path.normcase(str(include_dir))
        if key in seen:
            continue
        seen.add(key)
        includes.append(str(include_dir))
    return includes


def _has_cuda_toolkit_root(root: str | os.PathLike[str]) -> bool:
    try:
        p = pathlib.Path(root)
    except Exception:
        return False
    if not p.is_dir():
        return False
    exe_name = "nvcc.exe" if os.name == "nt" else "nvcc"
    return (p / "bin" / exe_name).is_file()


def _cuda_env_candidate_roots() -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    for env_key in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        value = os.environ.get(env_key)
        if value:
            roots.append(pathlib.Path(value))

    base = pathlib.Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    cuda_base = base / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if cuda_base.is_dir():
        try:
            children = [p for p in cuda_base.iterdir() if p.is_dir()]
        except Exception:
            children = []

        def _version_key(path: pathlib.Path) -> tuple[int, ...]:
            text = path.name.lower().lstrip("v")
            parts: list[int] = []
            for item in text.split("."):
                try:
                    parts.append(int(item))
                except Exception:
                    parts.append(0)
            return tuple(parts)

        roots.extend(sorted(children, key=_version_key, reverse=True))
        roots.append(cuda_base)

    uniq: list[pathlib.Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            normalized = os.path.normcase(str(root.resolve()))
        except Exception:
            normalized = os.path.normcase(str(root))
        if normalized in seen:
            continue
        seen.add(normalized)
        uniq.append(root)
    return uniq


def configure_cuda_environment() -> str | None:
    """Populate CUDA_HOME/CUDA_PATH from an installed CUDA Toolkit when found."""
    if os.name != "nt":
        return None
    selected: pathlib.Path | None = None
    for root in _cuda_env_candidate_roots():
        if _has_cuda_toolkit_root(root):
            selected = root
            break
    if selected is None:
        return None

    selected_str = str(selected)
    os.environ["CUDA_HOME"] = selected_str
    os.environ["CUDA_PATH"] = selected_str
    bin_dir = selected / "bin"
    if bin_dir.is_dir():
        bin_text = str(bin_dir)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        normalized = {os.path.normcase(os.path.normpath(p)) for p in path_parts if p}
        key = os.path.normcase(os.path.normpath(bin_text))
        if key not in normalized:
            os.environ["PATH"] = bin_text + os.pathsep + os.environ.get("PATH", "")
    return selected_str


_MSVC_ENV_CONFIGURED = False


def _exe_on_path(name: str) -> bool:
    try:
        search = name
        if not search.lower().endswith(".exe"):
            search += ".exe"
        for folder in os.environ.get("PATH", "").split(os.pathsep):
            if folder and (pathlib.Path(folder) / search).is_file():
                return True
    except Exception:
        return False
    return False


def _vswhere_path() -> pathlib.Path | None:
    candidates = []
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    program_files = os.environ.get("ProgramFiles")
    if program_files_x86:
        candidates.append(
            pathlib.Path(program_files_x86)
            / "Microsoft Visual Studio"
            / "Installer"
            / "vswhere.exe"
        )
    if program_files:
        candidates.append(
            pathlib.Path(program_files)
            / "Microsoft Visual Studio"
            / "Installer"
            / "vswhere.exe"
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _visual_studio_installations() -> list[pathlib.Path]:
    installs: list[pathlib.Path] = []
    vswhere = _vswhere_path()
    if vswhere is not None:
        try:
            out = subprocess.check_output(
                [
                    str(vswhere),
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=8,
            )
            installs.extend(pathlib.Path(line.strip()) for line in out.splitlines() if line.strip())
        except Exception:
            pass

    for base_env in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(base_env)
        if not base:
            continue
        vs_base = pathlib.Path(base) / "Microsoft Visual Studio"
        if not vs_base.is_dir():
            continue
        for year in ("2022", "2019", "2017"):
            year_dir = vs_base / year
            if not year_dir.is_dir():
                continue
            try:
                installs.extend(p for p in year_dir.iterdir() if p.is_dir())
            except Exception:
                pass

    uniq: list[pathlib.Path] = []
    seen: set[str] = set()
    for install in installs:
        try:
            key = os.path.normcase(str(install.resolve()))
        except Exception:
            key = os.path.normcase(str(install))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(install)
    return uniq


def _vcvars_candidates() -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    for install in _visual_studio_installations():
        candidates.append(install / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")
        candidates.append(install / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat")
    return [p for p in candidates if p.is_file()]


def _apply_msvc_env_from_vcvars(vcvars: pathlib.Path) -> bool:
    try:
        if vcvars.name.lower() == "vcvarsall.bat":
            cmd = f'cmd.exe /s /c ""{vcvars}" x64 >nul && set"'
        else:
            cmd = f'cmd.exe /s /c ""{vcvars}" >nul && set"'
        out = subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except Exception:
        return False

    env: dict[str, str] = {}
    for line in out.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            env[key] = value

    if not env.get("PATH"):
        return False
    os.environ.update(env)
    return _exe_on_path("cl")


def _add_latest_msvc_bin_to_path() -> bool:
    installs = _visual_studio_installations()
    tool_dirs: list[pathlib.Path] = []
    for install in installs:
        tools_root = install / "VC" / "Tools" / "MSVC"
        if not tools_root.is_dir():
            continue
        try:
            tool_dirs.extend(p for p in tools_root.iterdir() if p.is_dir())
        except Exception:
            pass
    if not tool_dirs:
        return False

    def _version_key(path: pathlib.Path) -> tuple[int, ...]:
        parts: list[int] = []
        for part in path.name.split("."):
            try:
                parts.append(int(part))
            except Exception:
                parts.append(0)
        return tuple(parts)

    for tool_dir in sorted(tool_dirs, key=_version_key, reverse=True):
        for host in ("Hostx64", "HostX64", "Hostx86", "HostX86"):
            bin_dir = tool_dir / "bin" / host / "x64"
            if (bin_dir / "cl.exe").is_file():
                bin_text = str(bin_dir)
                os.environ["PATH"] = bin_text + os.pathsep + os.environ.get("PATH", "")
                return _exe_on_path("cl")
    return False


def configure_msvc_build_environment() -> str | None:
    """Initialize MSVC Build Tools so Torch/ModelOpt extensions can find `cl`.

    A normal GUI launch does not run from a Visual Studio Developer Prompt, so
    Windows often has Build Tools installed but leaves `cl.exe`, INCLUDE, and
    LIB undiscoverable. This mirrors vcvars64 for the current process only.
    """
    global _MSVC_ENV_CONFIGURED
    if os.name != "nt":
        return None
    if _MSVC_ENV_CONFIGURED and _exe_on_path("cl"):
        return os.environ.get("VCINSTALLDIR") or "PATH"
    if _exe_on_path("cl"):
        _MSVC_ENV_CONFIGURED = True
        return os.environ.get("VCINSTALLDIR") or "PATH"

    for vcvars in _vcvars_candidates():
        if _apply_msvc_env_from_vcvars(vcvars):
            _MSVC_ENV_CONFIGURED = True
            return str(vcvars)

    if _add_latest_msvc_bin_to_path():
        _MSVC_ENV_CONFIGURED = True
        return "PATH"
    return None


def configure_torch_msvc_extension_flags() -> bool:
    """Patch Torch's Windows extension flags for ModelOpt CUDA builds."""
    if os.name != "nt":
        return False
    try:
        from torch.utils import cpp_extension
    except Exception:
        return False
    flags = getattr(cpp_extension, "COMMON_MSVC_FLAGS", None)
    if not isinstance(flags, list):
        return False
    if not any(str(flag).lower() == "/zc:preprocessor" for flag in flags):
        flags.append("/Zc:preprocessor")
    return True


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
        generated_dirs = {"__pycache__", "compile_cache", "engines", "onnx"}
        for p in sorted(models_dir.rglob("*.py")):
            try:
                rel_parts = set(p.relative_to(models_dir).parts)
            except ValueError:
                rel_parts = set(p.parts)
            if rel_parts & generated_dirs:
                continue
            files.append(p)

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


def _project_cache_root_base(current_file: str | None = None) -> str:
    """Return the repo-local cache base for this checkout."""
    project_root = pathlib.Path(project_root_from_path(current_file)).resolve()
    return str(project_root / "src" / "models" / "compile_cache")


def _compile_profiles_root(base: str) -> str:
    return os.path.join(base, "profiles")


def project_cache_profiles_root(current_file: str | None = None) -> str:
    """Return the repo-local parent folder that contains compile profiles."""
    explicit = os.environ.get("HDRTVNET_CACHE_DIR")
    if explicit:
        return explicit
    return _compile_profiles_root(_project_cache_root_base(current_file))


def project_cache_root(current_file: str | None = None) -> str:
    """Return the cache root for this local checkout and compile namespace."""
    explicit = os.environ.get("HDRTVNET_CACHE_DIR")
    if explicit:
        return explicit
    ns = compile_cache_namespace(current_file)
    return os.path.join(project_cache_profiles_root(current_file), ns)


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
