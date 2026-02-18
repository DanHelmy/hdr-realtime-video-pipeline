import cv2
from video_source import VideoSource
from timer import FPSTimer
from models.hdrtvnet_onnx import HDRTVNetONNX

VIDEO_PATH = "input.mp4"

def main():
    source = VideoSource(VIDEO_PATH)
    processor = HDRTVNetONNX("hdrtvnet_fp32.onnx")
    timer = FPSTimer()

    while True:
        ret, frame = source.read()
        if not ret:
            break

        # Limit to max 1080p without stretching
        h, w = frame.shape[:2]
        max_width = 1920
        max_height = 1080

        scale = min(max_width / w, max_height / h, 1.0)

        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

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

        cv2.imshow("Real-Time Video Pipeline (HDRTVNet ONNX)", output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
