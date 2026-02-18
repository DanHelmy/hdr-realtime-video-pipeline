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
    parser = argparse.ArgumentParser(description="Benchmark matrix runner for HDRTVNet pipeline")
    parser.add_argument("--video", default="input.mp4")
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--prefetch-values", default="0,8")
    parser.add_argument("--models", default="hdrtvnet_fp32.onnx,hdrtvnet_fp16.onnx")
    parser.add_argument("--static-models", default="hdrtvnet_fp32_1080_static.onnx,hdrtvnet_fp16_1080_static.onnx")
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--max-height", type=int, default=1080)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--timing-interval", type=int, default=120)
    parser.add_argument("--target-fps", type=float, default=0.0)
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def parse_prefetch(values):
    out = []
    for item in values.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def parse_models(values):
    return [x.strip() for x in values.split(",") if x.strip()]


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
    prefetch_values = parse_prefetch(args.prefetch_values)
    dynamic_models = parse_models(args.models)
    static_models = parse_models(args.static_models)

    cases = []
    for model in dynamic_models:
        for prefetch in prefetch_values:
            cases.append({
                "case": f"dynamic:{os.path.basename(model)}:prefetch={prefetch}",
                "model": model,
                "prefetch": prefetch,
                "static_input": False,
            })
    for model in static_models:
        if os.path.exists(model):
            for prefetch in prefetch_values:
                cases.append({
                    "case": f"static:{os.path.basename(model)}:prefetch={prefetch}",
                    "model": model,
                    "prefetch": prefetch,
                    "static_input": True,
                })

    rows = []
    for case in cases:
        cmd = [
            sys.executable,
            "src/main.py",
            "--video", args.video,
            "--model", case["model"],
            "--provider", args.provider,
            "--prefetch", str(case["prefetch"]),
            "--warmup", str(args.warmup),
            "--timing-interval", str(args.timing_interval),
            "--max-frames", str(args.max_frames),
            "--max-width", str(args.max_width),
            "--max-height", str(args.max_height),
            "--model-stage-timing",
            "--no-display",
        ]
        if args.target_fps > 0:
            cmd.extend(["--target-fps", str(args.target_fps)])
        if case["static_input"]:
            cmd.append("--static-input")

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
