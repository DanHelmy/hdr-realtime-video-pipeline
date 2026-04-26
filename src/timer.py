from __future__ import annotations

import time


class FPSTimer:
    def __init__(self):
        self.last = time.perf_counter()
        self.frames = 0
        self.fps = 0.0

    def update(self):
        self.frames += 1
        now = time.perf_counter()
        if now - self.last >= 1.0:
            self.fps = self.frames / (now - self.last)
            self.frames = 0
            self.last = now
        return self.fps


def sleep_until(
    deadline_s: float,
    *,
    coarse_margin_s: float = 0.0020,
) -> None:
    """Sleep toward a deadline without busy-spinning the GIL."""
    while True:
        remaining = float(deadline_s) - time.perf_counter()
        if remaining <= 0.0:
            return
        if remaining > coarse_margin_s:
            time.sleep(remaining - coarse_margin_s)
            continue
        # Yield rather than busy-spin so feeder/UI threads can keep running.
        time.sleep(0)
