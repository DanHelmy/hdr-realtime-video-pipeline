import cv2

class VideoSource:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
