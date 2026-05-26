from __future__ import annotations

import codecs
import os
import sys
import threading
from contextlib import contextmanager
from typing import Callable, Iterator


def _shorten_gui_line(text: str, limit: int = 260) -> str:
    line = str(text or "").strip()
    if not line:
        return ""
    if len(line) <= limit:
        return line
    head = max(40, (limit - 5) // 2)
    tail = max(40, limit - head - 5)
    return f"{line[:head]} ... {line[-tail:]}"


class _CallbackTextStream:
    encoding = "utf-8"
    errors = "replace"

    def __init__(self, callback: Callable[[str], None], fd: int):
        self._callback = callback
        self._fd = int(fd)
        self._buffer = ""
        self._lock = threading.Lock()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._fd

    def write(self, text) -> int:
        raw = str(text)
        if not raw:
            return 0
        with self._lock:
            self._buffer += raw.replace("\r\n", "\n").replace("\r", "\n")
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = _shorten_gui_line(line)
                if line:
                    self._callback(line)
        return len(raw)

    def flush(self) -> None:
        with self._lock:
            line = _shorten_gui_line(self._buffer)
            self._buffer = ""
        if line:
            self._callback(line)


def _reader_loop(read_fd: int, callback: Callable[[str], None]) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buffer = ""
    try:
        while True:
            try:
                chunk = os.read(read_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buffer += decoder.decode(chunk, final=False)
            buffer = buffer.replace("\r\n", "\n").replace("\r", "\n")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = _shorten_gui_line(line)
                if line:
                    callback(line)
        buffer += decoder.decode(b"", final=True)
        line = _shorten_gui_line(buffer)
        if line:
            callback(line)
    finally:
        try:
            os.close(read_fd)
        except OSError:
            pass


@contextmanager
def capture_output_to_gui(callback: Callable[[str], None]) -> Iterator[None]:
    """Route noisy in-process build output to a GUI callback.

    TensorRT emits a mix of Python print output and native stdout/stderr logs.
    GUI builds should surface those lines in the app instead of requiring the
    user to watch the launching terminal.
    """

    if callback is None:
        yield
        return

    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    reader = threading.Thread(
        target=_reader_loop,
        args=(read_fd, callback),
        name="gui-output-capture",
        daemon=True,
    )
    reader.start()
    stdout_proxy = _CallbackTextStream(callback, 1)
    stderr_proxy = _CallbackTextStream(callback, 2)
    try:
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        sys.stdout = stdout_proxy
        sys.stderr = stderr_proxy
        yield
    finally:
        try:
            stdout_proxy.flush()
            stderr_proxy.flush()
        except Exception:
            pass
        try:
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
        finally:
            for fd in (write_fd, saved_stdout_fd, saved_stderr_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass
            reader.join(timeout=1.0)
