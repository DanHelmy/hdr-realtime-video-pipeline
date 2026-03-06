import cv2
import queue
import threading


class VideoSource:
    def __init__(self, path, prefetch=0):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video")

        # Reduce decoder queueing latency when backend supports it.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.prefetch = max(0, int(prefetch))
        self._queue = None
        self._thread = None
        self._stopped = False
        self._sentinel = object()
        self._pending_frame = None

        # Video metadata
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0.0
        self.backend = "opencv"
        self.backend_reason = "OpenCV backend"

        # Seek support
        self._seek_lock = threading.Lock()
        self._seek_target = None
        self._last_frame_index = -1
        self._allow_backward_pos = False

        if self.prefetch > 0:
            self._queue = queue.Queue(maxsize=self.prefetch)
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

    def _seek_exact(self, frame_number: int):
        """Stricter seek: backtrack then decode-forward to target neighborhood."""
        if self.frame_count > 0:
            target = max(0, min(int(frame_number), self.frame_count - 1))
        else:
            target = max(0, int(frame_number))

        backtrack = max(8, int(round(self.fps * 0.5)))
        start = max(0, target - backtrack)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        self._last_frame_index = max(-1, start - 1)
        self._allow_backward_pos = True
        self._pending_frame = None

        best = None  # (abs_diff, idx, pts_sec, frame)
        max_probe = max(backtrack * 4, 120)
        for _ in range(max_probe):
            ret, frame, idx, pts_sec = self._read_frame_internal()
            if not ret:
                break
            diff = abs(int(idx) - target)
            if best is None or diff < best[0]:
                best = (diff, int(idx), float(pts_sec), frame)
            if idx >= target:
                break

        if best is not None:
            _, idx, pts_sec, frame = best
            self._pending_frame = (idx, pts_sec, frame)
            self._last_frame_index = max(-1, idx - 1)
            self._allow_backward_pos = True

    def _read_frame_internal(self):
        """Read one frame and return (ok, frame, decoded_index_0_based, pts_sec)."""
        if self._pending_frame is not None:
            idx, pts_sec, frame = self._pending_frame
            self._pending_frame = None
            self._last_frame_index = int(idx)
            self._allow_backward_pos = False
            return True, frame, int(idx), float(pts_sec)

        ret, frame = self.cap.read()
        if not ret:
            return False, None, -1, 0.0

        try:
            # OpenCV reports next frame index after read.
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        except Exception:
            pos = self._last_frame_index + 1

        if pos < 0:
            pos = self._last_frame_index + 1
        if (pos < self._last_frame_index) and (not self._allow_backward_pos):
            pos = self._last_frame_index + 1

        self._allow_backward_pos = False
        self._last_frame_index = int(pos)
        pts_sec = float(pos / self.fps) if self.fps > 0 else 0.0
        return True, frame, int(pos), pts_sec

    def _drain_prefetch_queue(self):
        if self._queue is None:
            return
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _reader_loop(self):
        while not self._stopped:
            with self._seek_lock:
                target = self._seek_target
                self._seek_target = None
            if target is not None:
                self._seek_exact(target)
                self._drain_prefetch_queue()

            ret, frame, idx, pts_sec = self._read_frame_internal()
            if not ret:
                self._queue.put(self._sentinel)
                break
            self._queue.put((idx, pts_sec, frame))

    def read(self):
        """Read one frame and return (ok, frame)."""
        if self._queue is None:
            ret, frame, _idx, _pts = self._read_frame_internal()
            return ret, frame

        item = self._queue.get()
        if item is self._sentinel:
            return False, None
        _idx, _pts, frame = item
        return True, frame

    def read_with_index(self):
        """Read one frame and return (ok, frame, decoded_index_0_based)."""
        ret, frame, idx, _pts = self.read_with_meta()
        return ret, frame, idx

    def read_with_meta(self):
        """Read one frame and return (ok, frame, decoded_index_0_based, pts_sec)."""
        if self._queue is None:
            return self._read_frame_internal()

        item = self._queue.get()
        if item is self._sentinel:
            return False, None, -1, 0.0
        idx, pts_sec, frame = item
        return True, frame, int(idx), float(pts_sec)

    def seek(self, frame_number):
        """Seek to frame_number. Thread-safe for prefetch mode."""
        if self.frame_count > 0:
            frame_number = max(0, min(int(frame_number), self.frame_count - 1))
        else:
            frame_number = max(0, int(frame_number))
        if self._queue is not None:
            self._drain_prefetch_queue()
            with self._seek_lock:
                self._seek_target = frame_number
        else:
            self._seek_exact(frame_number)

    def position(self):
        """Return the current decoded frame number (approximate with prefetch)."""
        return max(0, int(self._last_frame_index + 1))

    def release(self):
        self._stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self.cap.release()

