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

        if self.prefetch > 0:
            self._queue = queue.Queue(maxsize=self.prefetch)
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

    def _reader_loop(self):
        while not self._stopped:
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

    def release(self):
        self._stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self.cap.release()
