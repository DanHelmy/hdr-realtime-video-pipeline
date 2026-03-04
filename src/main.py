import cv2
import argparse
import os
import time

import numpy as np
import psutil
import torch
from collections import deque
from video_source import VideoSource
from timer import FPSTimer
from models.hdrtvnet_torch import HDRTVNetTorch

VIDEO_PATH = "input.mp4"
MODEL_PATH = "src/models/weights/Ensemble_AGCM_LE.pth"

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

# Pre-allocated letterbox canvas to avoid allocation every frame
_letterbox_canvas = None
_letterbox_shape = None

def letterbox_frame(frame, target_w, target_h):
    global _letterbox_canvas, _letterbox_shape
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return frame

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if new_w != w or new_h != h:
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = frame

    shape_key = (target_h, target_w, frame.dtype)
    if _letterbox_canvas is None or _letterbox_shape != shape_key:
        _letterbox_canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
        _letterbox_shape = shape_key
    else:
        _letterbox_canvas[:] = 0  # clear is cheaper than allocating

    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    _letterbox_canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return _letterbox_canvas

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time HDRTVNet video pipeline")
    parser.add_argument("--video", default=VIDEO_PATH, help="Input video path")
    parser.add_argument("--model", default=MODEL_PATH, help="Model path (.pth/.pt)")
    parser.add_argument("--max-width", type=int, default=TARGET_WIDTH, help="Max processing width")
    parser.add_argument("--max-height", type=int, default=TARGET_HEIGHT, help="Max processing height")
    parser.add_argument(
        "--letterbox",
        action="store_true",
        help="Preserve aspect ratio by fitting frame into target size with black bars"
    )
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
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device selection"
    )
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "fp16", "fp32", "int8-full", "int8-mixed"],
        help="Inference precision: int8-full = W8A8 (all layers), int8-mixed = selective W8A8/W8A16"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile() even if available (PyTorch engine only)"
    )
    parser.add_argument(
        "--force-compile",
        action="store_true",
        help="Force torch.compile() on ROCm-Windows when HIP SDK auto-detection fails"
    )
    parser.add_argument(
        "--compile-mode",
        default="auto",
        choices=["auto", "default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (auto = 'max-autotune' on all GPUs)"
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Use CUDA graph replay for static-shape inputs (PyTorch engine only)"
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Force channels_last memory format (PyTorch engine only). "
             "Auto-enabled on NVIDIA; use this flag to test on ROCm+Triton."
    )
    parser.add_argument(
        "--predequantize",
        default="auto",
        choices=["auto", "on", "off"],
        help="Pre-dequantize INT8 weights to FP16 at load time. "
             "'auto' enables on GPUs without INT8 tensor cores (e.g. AMD RDNA3), "
             "giving FP16 inference speed with compressed checkpoint storage. "
             "'on' forces pre-dequantization, 'off' keeps INT8 runtime dequant."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    source = VideoSource(args.video, prefetch=args.prefetch)
    predeq = {"auto": "auto", "on": True, "off": False}[args.predequantize]
    processor = HDRTVNetTorch(
        args.model,
        device=args.device,
        precision=args.precision,
        compile_model=not args.no_compile,
        force_compile=args.force_compile,
        compile_mode=args.compile_mode,
        use_cuda_graphs=args.cuda_graphs,
        force_channels_last=args.channels_last,
        predequantize=predeq,
    )
    # torch.compile is lazy — first inference triggers compilation.
    # Print a message so the user knows it's working.
    if hasattr(processor, '_compiled') and processor._compiled:
        print("First frame will trigger torch.compile — this may take several minutes...")

    timer = FPSTimer()
    frame_idx = 0
    stats_frames = 0

    # ---- Static metrics (computed once) --------------------------------
    model_size_mb = os.path.getsize(args.model) / (1024 * 1024)
    process = psutil.Process(os.getpid())
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
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

        h, w = frame.shape[:2]
        if args.static_input:
            if w != args.max_width or h != args.max_height:
                if args.letterbox:
                    frame = letterbox_frame(frame, args.max_width, args.max_height)
                else:
                    frame = cv2.resize(
                        frame,
                        (args.max_width, args.max_height),
                        interpolation=cv2.INTER_AREA
                    )
        elif w > args.max_width or h > args.max_height:
            if args.letterbox:
                frame = letterbox_frame(frame, args.max_width, args.max_height)
            else:
                frame = cv2.resize(
                    frame,
                    (args.max_width, args.max_height),
                    interpolation=cv2.INTER_AREA
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
            # ---- Gather live metrics ------------------------------------
            latency_ms = (t3 - t0) * 1000.0  # decode + resize + infer
            cpu_mem_mb = process.memory_info().rss / (1024 * 1024)
            if use_cuda:
                gpu_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            else:
                gpu_mem_mb = 0.0

            # ---- Draw metrics overlay (ROI-only, avoids full-frame copy) --
            overlay_lines = [
                f"FPS: {fps:.2f}",
                f"Latency: {latency_ms:.1f} ms/frame",
                f"GPU Memory: {gpu_mem_mb:.1f} MB",
                f"CPU Memory: {cpu_mem_mb:.1f} MB",
                f"Model Size: {model_size_mb:.2f} MB",
            ]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            color = (0, 255, 0)
            y_start = 30
            line_height = 30

            # Compute box dimensions from text
            max_text_w = 0
            for line in overlay_lines:
                (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
                max_text_w = max(max_text_w, tw)
            box_x, box_y = 10, 5
            box_w = max_text_w + 30
            box_h = len(overlay_lines) * line_height + 10

            # Alpha-blend only the small overlay ROI (not the entire frame)
            # ~300x fewer pixels than full-frame addWeighted at 1080p
            roi = output[box_y:box_y + box_h, box_x:box_x + box_w]
            roi //= 3  # fast integer darken (~33% brightness, no float/alloc)

            for i, line in enumerate(overlay_lines):
                cv2.putText(output, line, (20, y_start + i * line_height),
                            font, font_scale, color, thickness)

            cv2.imshow("HDRTVNet PyTorch", output)
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

if __name__ == "__main__":
    main()

