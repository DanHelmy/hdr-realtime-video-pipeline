import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime


TIMING_RE = re.compile(r"^\[timing\]\s+(.*)$")
KV_RE = re.compile(r"([a-zA-Z0-9_]+)=([^\s]+)")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark matrix runner for HDRTVNet PyTorch pipeline")
    parser.add_argument("--video", default="input.mp4")
    parser.add_argument(
        "--models",
        default="src/models/weights/Ensemble_AGCM_LE.pth",
        help="Comma-separated .pth model paths"
    )
    parser.add_argument(
        "--precisions",
        default="auto,fp16,fp32",
        help="Comma-separated precisions: auto,fp16,fp32,int8-full,int8-mixed"
    )
    parser.add_argument(
        "--int8-full-model",
        default="src/models/weights/Ensemble_AGCM_LE_int8_full.pt",
        help="Path to INT8 W8A8 model (used when precision=int8-full)"
    )
    parser.add_argument(
        "--int8-mixed-model",
        default="src/models/weights/Ensemble_AGCM_LE_int8_mixed.pt",
        help="Path to INT8 mixed model (used when precision=int8-mixed)"
    )
    parser.add_argument(
        "--compile-modes",
        default="no-compile,force-compile",
        help="Comma-separated compile modes: no-compile,compile,force-compile"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device selection"
    )
    parser.add_argument("--prefetch-values", default="0,8")
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--max-height", type=int, default=1080)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--timing-interval", type=int, default=120)
    parser.add_argument("--target-fps", type=float, default=0.0)
    parser.add_argument("--output-csv", default="")
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Include --channels-last flag in benchmark cases"
    )
    return parser.parse_args()


def parse_list(values):
    return [x.strip() for x in values.split(",") if x.strip()]


def parse_prefetch(values):
    return [int(x.strip()) for x in values.split(",") if x.strip()]


def run_case(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    lines = (proc.stdout or "").splitlines()
    timing_lines = [ln for ln in lines if TIMING_RE.match(ln)]
    if proc.returncode != 0:
        return {"status": "error", "returncode": proc.returncode, "stderr": proc.stderr.strip()}
    if not timing_lines:
        return {"status": "error", "returncode": 0, "stderr": "No [timing] output found"}

    last = timing_lines[-1]
    values = {}
    for key, val in KV_RE.findall(last):
        values[key] = val
    values["status"] = "ok"
    values["timing_line"] = last
    return values


def print_table(rows):
    headers = ["case", "status", "infer", "fps", "fps_1p_low", "pre", "run", "post", "late", "drop_est"]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        print(" | ".join(str(row.get(h, "")) for h in headers))


def main():
    args = parse_args()
    models = parse_list(args.models)
    precisions = parse_list(args.precisions)
    compile_modes = parse_list(args.compile_modes)
    prefetch_values = parse_prefetch(args.prefetch_values)

    # Map INT8 precision variants to their model paths
    int8_model_map = {
        "int8-full": (args.int8_full_model, "quantize_int8_full.py"),
        "int8-mixed": (args.int8_mixed_model, "quantize_int8_mixed.py"),
    }

    cases = []
    for model in models:
        for precision in precisions:
            # INT8 variants use dedicated model paths
            if precision in int8_model_map:
                int8_path, quant_script = int8_model_map[precision]
                if not os.path.isfile(int8_path):
                    print(f"SKIP {precision}: {int8_path} not found (run {quant_script} first)")
                    continue
                for compile_mode in compile_modes:
                    for prefetch in prefetch_values:
                        tag = (f"torch:{os.path.basename(int8_path)}:{precision}"
                               f":{compile_mode}:prefetch={prefetch}")
                        if args.channels_last:
                            tag += ":cl"
                        cases.append({
                            "case": tag,
                            "model": int8_path,
                            "prefetch": prefetch,
                            "precision": precision,
                            "compile_mode": compile_mode,
                        })
                continue

            for compile_mode in compile_modes:
                for prefetch in prefetch_values:
                    tag = (f"torch:{os.path.basename(model)}:{precision}"
                           f":{compile_mode}:prefetch={prefetch}")
                    if args.channels_last:
                        tag += ":cl"
                    cases.append({
                        "case": tag,
                        "model": model,
                        "prefetch": prefetch,
                        "precision": precision,
                        "compile_mode": compile_mode,
                    })

    rows = []
    for case in cases:
        cmd = [
            sys.executable,
            "src/main.py",
            "--video", args.video,
            "--model", case["model"],
            "--device", args.device,
            "--precision", case["precision"],
            "--prefetch", str(case["prefetch"]),
            "--warmup", str(args.warmup),
            "--timing-interval", str(args.timing_interval),
            "--max-frames", str(args.max_frames),
            "--max-width", str(args.max_width),
            "--max-height", str(args.max_height),
            "--model-stage-timing",
            "--no-display",
        ]
        compile_mode = case["compile_mode"]
        if compile_mode == "no-compile":
            cmd.append("--no-compile")
        elif compile_mode == "force-compile":
            cmd.append("--force-compile")
        if args.channels_last:
            cmd.append("--channels-last")
        if args.target_fps > 0:
            cmd.extend(["--target-fps", str(args.target_fps)])

        print(f"Running: {case['case']}")
        result = run_case(cmd)
        row = {
            "case": case["case"],
            "status": result.get("status", "error"),
            "infer": result.get("infer", ""),
            "fps": result.get("fps", ""),
            "fps_1p_low": result.get("fps_1p_low", ""),
            "pre": result.get("pre", ""),
            "run": result.get("run", ""),
            "post": result.get("post", ""),
            "late": result.get("late", ""),
            "drop_est": result.get("drop_est", ""),
            "timing_line": result.get("timing_line", ""),
            "stderr": result.get("stderr", ""),
        }
        rows.append(row)

    print()
    print_table(rows)

    output_csv = args.output_csv
    if not output_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"benchmark_results_{stamp}.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "status", "infer", "fps", "fps_1p_low", "pre", "run", "post", "late", "drop_est", "timing_line", "stderr"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()
