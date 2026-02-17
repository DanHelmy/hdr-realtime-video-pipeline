import cv2
from video_source import VideoSource
from timer import FPSTimer

# Import the real model instead of dummy
from models.hdrtvnet_fp32 import HDRTVNetFP32


VIDEO_PATH = "input.mp4"  # change this
WEIGHT_PATH = "src/models/weights/Ensemble_AGCM_LE.pth"


def main():
    source = VideoSource(VIDEO_PATH)

    # Replace DummyProcessor with real HDR model
    processor = HDRTVNetFP32(WEIGHT_PATH)

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

        cv2.imshow("Real-Time Video Pipeline (HDRTVNet FP32)", output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
