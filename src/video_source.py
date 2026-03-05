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

        # Video metadata
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0.0

        # Seek support — when a seek is requested we drain the prefetch
        # queue and let the reader thread pick up the new position.
        self._seek_lock = threading.Lock()
        self._seek_target = None           # frame number to seek to

        if self.prefetch > 0:
            self._queue = queue.Queue(maxsize=self.prefetch)
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

    def _reader_loop(self):
        while not self._stopped:
            # Check for pending seek
            with self._seek_lock:
                target = self._seek_target
                self._seek_target = None
            if target is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                # Drain buffered frames so the consumer gets fresh data
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break

            ret, frame = self.cap.read()
            if not ret:
                self._queue.put(self._sentinel)
                break
            self._queue.put(frame)

    def read(self):
        if self._queue is None:
            return self.cap.read()

        item = self._queue.get()
        if item is self._sentinel:
            return False, None
        return True, item

    def seek(self, frame_number):
        """Seek to *frame_number*.  Thread-safe for prefetch mode."""
        frame_number = max(0, min(frame_number, self.frame_count - 1))
        if self._queue is not None:
            with self._seek_lock:
                self._seek_target = frame_number
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def position(self):
        """Return the current frame number (approximate with prefetch)."""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def release(self):
        self._stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self.cap.release()
