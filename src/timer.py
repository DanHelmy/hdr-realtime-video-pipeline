import time

class FPSTimer:
    def __init__(self):
        self.last = time.time()
        self.frames = 0
        self.fps = 0.0

    def update(self):
        self.frames += 1
        now = time.time()
        if now - self.last >= 1.0:
            self.fps = self.frames / (now - self.last)
            self.frames = 0
            self.last = now
        return self.fps
