import cv2
import argparse
import time
from collections import deque
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
        choices=["auto", "dml", "cuda", "rocm", "tensorrt", "coreml", "openvino", "cpu"],
        help="Execution provider preference (auto picks first available GPU backend)"
    )
    parser.add_argument("--device-id", type=int, default=0, help="DirectML device id")
    parser.add_argument("--max-width", type=int, default=TARGET_WIDTH, help="Max processing width")
    parser.add_argument("--max-height", type=int, default=TARGET_HEIGHT, help="Max processing height")
    parser.add_argument(
        "--static-input",
        action="store_true",
        help="Always resize input to (max-width, max-height) for static-shape model benchmarking"
    )
    parser.add_argument("--warmup", type=int, default=30, help="Frames to ignore in timing stats")
    parser.add_argument("--timing-interval", type=int, default=120, help="Frames between timing reports")
    parser.add_argument("--prefetch", type=int, default=8, help="Reader prefetch queue size (0 disables)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = full video)")
    parser.add_argument("--target-fps", type=float, default=0.0, help="Target playback FPS for pacing stats")
    parser.add_argument(
        "--model-stage-timing",
        action="store_true",
        help="Report preprocess/session/postprocess timing breakdown"
    )
    parser.add_argument("--no-display", action="store_true", help="Disable cv2.imshow for pure throughput testing")
    parser.add_argument("--ort-profile", action="store_true", help="Enable ONNX Runtime profiling output")
    return parser.parse_args()


def main():
    args = parse_args()

    source = VideoSource(args.video, prefetch=args.prefetch)
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
    pre_ms = 0.0
    run_ms = 0.0
    post_ms = 0.0
    frame_ms_sum = 0.0
    fps_samples = deque(maxlen=10000)
    late_frames = 0
    overrun_ms = 0.0

    while True:
        t0 = time.perf_counter()
        ret, frame = source.read()
        t1 = time.perf_counter()
        if not ret:
            break

        # Optional static-input mode for fixed-shape ONNX benchmarking.
        h, w = frame.shape[:2]
        if args.static_input:
            if w != args.max_width or h != args.max_height:
                frame = cv2.resize(
                    frame,
                    (args.max_width, args.max_height),
                    interpolation=cv2.INTER_AREA
                )
        elif w > args.max_width or h > args.max_height:
            frame = cv2.resize(
                frame,
                (args.max_width, args.max_height),
                interpolation=cv2.INTER_AREA
            )

        if processor.is_static_input_model:
            exp_h, exp_w = processor.expected_hw
            fh, fw = frame.shape[:2]
            if (fh, fw) != (exp_h, exp_w):
                raise RuntimeError(
                    f"Static ONNX expects {exp_w}x{exp_h} but frame is {fw}x{fh}. "
                    f"Use --static-input --max-width {exp_w} --max-height {exp_h}, "
                    "or switch to a dynamic-shape ONNX."
                )
        t2 = time.perf_counter()

        if args.model_stage_timing:
            output, pre_t, run_t, post_t = processor.process_timed(frame)
        else:
            output = processor.process(frame)
            pre_t = 0.0
            run_t = 0.0
            post_t = 0.0
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
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        if frame_idx > args.warmup:
            stats_frames += 1
            decode_ms += (t1 - t0) * 1000.0
            resize_ms += (t2 - t1) * 1000.0
            infer_ms += (t3 - t2) * 1000.0
            render_ms += (t4 - t3) * 1000.0
            pre_ms += pre_t
            run_ms += run_t
            post_ms += post_t
            frame_ms = (t4 - t0) * 1000.0
            frame_ms_sum += frame_ms
            if frame_ms > 0:
                fps_samples.append(1000.0 / frame_ms)

            if args.target_fps > 0:
                budget_ms = 1000.0 / args.target_fps
                if frame_ms > budget_ms:
                    late_frames += 1
                    overrun_ms += (frame_ms - budget_ms)

            if stats_frames % args.timing_interval == 0:
                avg_frame_ms = frame_ms_sum / stats_frames
                avg_fps = 1000.0 / avg_frame_ms if avg_frame_ms > 0 else 0.0
                if fps_samples:
                    sorted_fps = sorted(fps_samples)
                    k = max(1, int(len(sorted_fps) * 0.01))
                    one_percent_low = sum(sorted_fps[:k]) / k
                else:
                    one_percent_low = 0.0

                if args.model_stage_timing:
                    msg = (
                        f"[timing] frames={stats_frames} "
                        f"decode={decode_ms / stats_frames:.2f}ms "
                        f"resize={resize_ms / stats_frames:.2f}ms "
                        f"infer={infer_ms / stats_frames:.2f}ms "
                        f"pre={pre_ms / stats_frames:.2f}ms "
                        f"run={run_ms / stats_frames:.2f}ms "
                        f"post={post_ms / stats_frames:.2f}ms "
                        f"render={render_ms / stats_frames:.2f}ms "
                        f"fps={avg_fps:.2f} "
                        f"fps_1p_low={one_percent_low:.2f}"
                    )
                else:
                    msg = (
                        f"[timing] frames={stats_frames} "
                        f"decode={decode_ms / stats_frames:.2f}ms "
                        f"resize={resize_ms / stats_frames:.2f}ms "
                        f"infer={infer_ms / stats_frames:.2f}ms "
                        f"render={render_ms / stats_frames:.2f}ms "
                        f"fps={avg_fps:.2f} "
                        f"fps_1p_low={one_percent_low:.2f}"
                    )

                if args.target_fps > 0:
                    late_pct = (100.0 * late_frames / stats_frames) if stats_frames > 0 else 0.0
                    budget_ms = 1000.0 / args.target_fps
                    drop_est = int(overrun_ms / budget_ms) if budget_ms > 0 else 0
                    msg += (
                        f" target={args.target_fps:.2f} "
                        f"late={late_frames}/{stats_frames}({late_pct:.1f}%) "
                        f"drop_est={drop_est}"
                    )

                print(msg)

    if stats_frames > 0 and stats_frames % args.timing_interval != 0:
        avg_frame_ms = frame_ms_sum / stats_frames
        avg_fps = 1000.0 / avg_frame_ms if avg_frame_ms > 0 else 0.0
        if fps_samples:
            sorted_fps = sorted(fps_samples)
            k = max(1, int(len(sorted_fps) * 0.01))
            one_percent_low = sum(sorted_fps[:k]) / k
        else:
            one_percent_low = 0.0

        if args.model_stage_timing:
            msg = (
                f"[timing] frames={stats_frames} "
                f"decode={decode_ms / stats_frames:.2f}ms "
                f"resize={resize_ms / stats_frames:.2f}ms "
                f"infer={infer_ms / stats_frames:.2f}ms "
                f"pre={pre_ms / stats_frames:.2f}ms "
                f"run={run_ms / stats_frames:.2f}ms "
                f"post={post_ms / stats_frames:.2f}ms "
                f"render={render_ms / stats_frames:.2f}ms "
                f"fps={avg_fps:.2f} "
                f"fps_1p_low={one_percent_low:.2f}"
            )
        else:
            msg = (
                f"[timing] frames={stats_frames} "
                f"decode={decode_ms / stats_frames:.2f}ms "
                f"resize={resize_ms / stats_frames:.2f}ms "
                f"infer={infer_ms / stats_frames:.2f}ms "
                f"render={render_ms / stats_frames:.2f}ms "
                f"fps={avg_fps:.2f} "
                f"fps_1p_low={one_percent_low:.2f}"
            )

        if args.target_fps > 0:
            late_pct = (100.0 * late_frames / stats_frames) if stats_frames > 0 else 0.0
            budget_ms = 1000.0 / args.target_fps
            drop_est = int(overrun_ms / budget_ms) if budget_ms > 0 else 0
            msg += (
                f" target={args.target_fps:.2f} "
                f"late={late_frames}/{stats_frames}({late_pct:.1f}%) "
                f"drop_est={drop_est}"
            )

        print(msg)

    source.release()
    cv2.destroyAllWindows()
    if args.ort_profile:
        profile_path = processor.end_profiling()
        print(f"ONNX Runtime profile saved to: {profile_path}")


if __name__ == "__main__":
    main()
