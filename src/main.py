import cv2
import argparse
import time
from video_source import VideoSource
from timer import FPSTimer
from models.hdrtvnet_onnx import HDRTVNetONNX

VIDEO_PATH = "input.mp4"
MODEL_PATH = "hdrtvnet_fp16.onnx"

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time HDRTVNet ONNX video pipeline")
    parser.add_argument("--video", default=VIDEO_PATH, help="Input video path")
    parser.add_argument("--model", default=MODEL_PATH, help="ONNX model path")
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "dml", "cpu"],
        help="Execution provider preference"
    )
    parser.add_argument("--device-id", type=int, default=0, help="DirectML device id")
    parser.add_argument("--max-width", type=int, default=TARGET_WIDTH, help="Max processing width")
    parser.add_argument("--max-height", type=int, default=TARGET_HEIGHT, help="Max processing height")
    parser.add_argument("--warmup", type=int, default=30, help="Frames to ignore in timing stats")
    parser.add_argument("--timing-interval", type=int, default=120, help="Frames between timing reports")
    parser.add_argument("--no-display", action="store_true", help="Disable cv2.imshow for pure throughput testing")
    parser.add_argument("--ort-profile", action="store_true", help="Enable ONNX Runtime profiling output")
    return parser.parse_args()


def main():
    args = parse_args()

    source = VideoSource(args.video)
    processor = HDRTVNetONNX(
        args.model,
        provider=args.provider,
        device_id=args.device_id,
        enable_profiling=args.ort_profile
    )
    timer = FPSTimer()
    frame_idx = 0
    stats_frames = 0
    decode_ms = 0.0
    resize_ms = 0.0
    infer_ms = 0.0
    render_ms = 0.0

    while True:
        t0 = time.perf_counter()
        ret, frame = source.read()
        t1 = time.perf_counter()
        if not ret:
            break

        # Downscale ONLY if larger than 1080p
        h, w = frame.shape[:2]

        if w > args.max_width or h > args.max_height:
            frame = cv2.resize(
                frame,
                (args.max_width, args.max_height),
                interpolation=cv2.INTER_AREA
            )
        t2 = time.perf_counter()

        output = processor.process(frame)
        t3 = time.perf_counter()
        fps = timer.update()

        if not args.no_display:
            cv2.putText(
                output,
                f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
            cv2.imshow("HDRTVNet ONNX", output)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        t4 = time.perf_counter()

        frame_idx += 1
        if frame_idx > args.warmup:
            stats_frames += 1
            decode_ms += (t1 - t0) * 1000.0
            resize_ms += (t2 - t1) * 1000.0
            infer_ms += (t3 - t2) * 1000.0
            render_ms += (t4 - t3) * 1000.0

            if stats_frames % args.timing_interval == 0:
                print(
                    f"[timing] frames={stats_frames} "
                    f"decode={decode_ms / stats_frames:.2f}ms "
                    f"resize={resize_ms / stats_frames:.2f}ms "
                    f"infer={infer_ms / stats_frames:.2f}ms "
                    f"render={render_ms / stats_frames:.2f}ms"
                )

    source.release()
    cv2.destroyAllWindows()
    if args.ort_profile:
        profile_path = processor.end_profiling()
        print(f"ONNX Runtime profile saved to: {profile_path}")


if __name__ == "__main__":
    main()
