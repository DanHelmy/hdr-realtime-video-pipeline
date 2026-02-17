import cv2
from video_source import VideoSource
from processor import DummyProcessor
from timer import FPSTimer

VIDEO_PATH = "input.mp4"  # change this

def main():
    source = VideoSource(VIDEO_PATH)
    processor = DummyProcessor()
    timer = FPSTimer()

    while True:
        ret, frame = source.read()
        if not ret:
            break

        output = processor.process(frame)
        fps = timer.update()

        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cv2.imshow("Real-Time Video Pipeline (Dummy)", output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
