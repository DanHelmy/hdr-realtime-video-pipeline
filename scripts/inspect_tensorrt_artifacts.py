"""Inspect TensorRT INT8 ONNX and engine artifacts.

ONNX inspection works on any machine with onnx installed. TensorRT engine
inspection and profiling require NVIDIA TensorRT Python bindings.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


def _inspect_onnx(path: Path) -> dict[str, object]:
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(path), load_external_data=True)
    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)
    producers = {output: node for node in graph.node for output in node.output}
    consumers = defaultdict(list)
    for node in graph.node:
        for index, input_name in enumerate(node.input):
            consumers[input_name].append((node, index))

    initializer_types = Counter()
    initializer_names = {init.name for init in graph.initializer}
    fp32_initializers = []
    for init in graph.initializer:
        dtype_name = TensorProto.DataType.Name(init.data_type)
        initializer_types[dtype_name] += 1
        if dtype_name == "FLOAT":
            fp32_initializers.append(init.name)

    qdq_boundary_casts = []
    harmless_casts = []
    compute_ops = {"Conv", "Gemm", "MatMul"}
    for node in graph.node:
        if node.op_type != "Cast" or not node.input or not node.output:
            continue
        producer = producers.get(node.input[0])
        node_consumers = consumers.get(node.output[0], [])
        qdq_boundary = False
        if producer is not None and producer.op_type == "DequantizeLinear":
            qdq_boundary = any(
                consumer.op_type in compute_ops for consumer, _ in node_consumers
            )
        if any(consumer.op_type == "QuantizeLinear" for consumer, _ in node_consumers):
            qdq_boundary = True
        entry = {
            "name": node.name,
            "input": list(node.input),
            "output": list(node.output),
            "producer": producer.op_type if producer is not None else None,
            "consumers": [
                {
                    "op_type": consumer.op_type,
                    "name": consumer.name,
                    "input_index": index,
                }
                for consumer, index in node_consumers
            ],
        }
        if qdq_boundary:
            qdq_boundary_casts.append(entry)
        else:
            harmless_casts.append(entry)

    qdq_consumers = Counter()
    qdq_producers = Counter()
    for node in graph.node:
        if node.op_type == "DequantizeLinear" and node.output:
            for consumer, _ in consumers.get(node.output[0], []):
                qdq_consumers[consumer.op_type] += 1
        if node.op_type == "QuantizeLinear" and node.input:
            producer = producers.get(node.input[0])
            qdq_producers[producer.op_type if producer is not None else "<input>"] += 1

    conv_nodes = [node for node in graph.node if node.op_type == "Conv"]
    add_inputs_with_dq = 0
    add_inputs_total = 0
    mul_inputs_with_dq = 0
    mul_inputs_total = 0
    add_outputs_to_q = 0
    for node in graph.node:
        if node.op_type == "Add":
            if any(
                consumer.op_type == "QuantizeLinear"
                for output_name in node.output
                for consumer, _ in consumers.get(output_name, [])
            ):
                add_outputs_to_q += 1
        if node.op_type not in {"Add", "Mul"}:
            continue
        for input_name in node.input:
            if input_name in initializer_names:
                continue
            producer = producers.get(input_name)
            if node.op_type == "Add":
                add_inputs_total += 1
                if producer is not None and producer.op_type == "DequantizeLinear":
                    add_inputs_with_dq += 1
            else:
                mul_inputs_total += 1
                if producer is not None and producer.op_type == "DequantizeLinear":
                    mul_inputs_with_dq += 1

    return {
        "path": str(path),
        "opsets": [(op.domain, op.version) for op in model.opset_import],
        "graph": {
            "nodes": len(graph.node),
            "initializers": len(graph.initializer),
            "inputs": [value.name for value in graph.input],
            "outputs": [value.name for value in graph.output],
        },
        "op_counts": dict(sorted(op_counts.items())),
        "initializer_types": dict(sorted(initializer_types.items())),
        "fp32_initializers": {
            "count": len(fp32_initializers),
            "sample": fp32_initializers[:24],
        },
        "conv": {
            "nodes": len(conv_nodes),
            "activation_from_dq": sum(
                1
                for node in conv_nodes
                if node.input
                and producers.get(node.input[0])
                and producers[node.input[0]].op_type == "DequantizeLinear"
            ),
            "weight_from_dq": sum(
                1
                for node in conv_nodes
                if len(node.input) > 1
                and producers.get(node.input[1])
                and producers[node.input[1]].op_type == "DequantizeLinear"
            ),
        },
        "qdq": {
            "quantize_linear": int(op_counts.get("QuantizeLinear", 0)),
            "dequantize_linear": int(op_counts.get("DequantizeLinear", 0)),
            "dq_consumers": dict(sorted(qdq_consumers.items())),
            "q_producers": dict(sorted(qdq_producers.items())),
            "add_inputs_with_dq": int(add_inputs_with_dq),
            "add_inputs_total": int(add_inputs_total),
            "add_outputs_to_q": int(add_outputs_to_q),
            "mul_inputs_with_dq": int(mul_inputs_with_dq),
            "mul_inputs_total": int(mul_inputs_total),
            "qdq_boundary_casts": qdq_boundary_casts,
        },
        "casts": {
            "total": int(op_counts.get("Cast", 0)),
            "qdq_boundary": len(qdq_boundary_casts),
            "other": harmless_casts,
        },
    }


def _extract_engine_layers(info_text: str) -> list[dict[str, object]] | None:
    try:
        parsed = json.loads(info_text)
    except Exception:
        return None
    layers = None
    if isinstance(parsed, list):
        layers = parsed
    elif isinstance(parsed, dict):
        layers = parsed.get("Layers") or parsed.get("layers")
    if isinstance(layers, dict):
        layers = list(layers.values())
    if not isinstance(layers, list):
        return None
    normalized = []
    for item in layers:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str):
            try:
                parsed_item = json.loads(item)
            except Exception:
                parsed_item = {"Name": item}
            normalized.append(
                parsed_item if isinstance(parsed_item, dict) else {"Name": item}
            )
    return normalized


def _inspect_engine(path: Path, output_dir: Path | None = None, tag: str = "engine"):
    try:
        import tensorrt as trt
    except Exception as exc:
        return None, {
            "path": str(path),
            "available": False,
            "error": f"TensorRT import failed: {exc}",
        }

    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        return None, {
            "path": str(path),
            "available": False,
            "error": "deserialize_cuda_engine returned None",
        }

    inspector = engine.create_engine_inspector()
    info_text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{tag}_engine_inspector.json").write_text(
            info_text,
            encoding="utf-8",
        )
    layers = _extract_engine_layers(info_text)
    summary: dict[str, object] = {
        "path": str(path),
        "available": True,
        "tensorrt_version": str(getattr(trt, "__version__", "unknown")),
        "num_layers": int(getattr(engine, "num_layers", -1)),
        "num_io_tensors": int(getattr(engine, "num_io_tensors", -1)),
    }
    if layers is None:
        summary["raw_info"] = info_text[:20000]
        return engine, summary

    layer_types = Counter()
    precision_tokens = Counter()
    output_type_tokens = Counter()
    output_dtypes = Counter()
    conv_patterns = Counter()
    reformat_layers = []
    int8_like_layers = []
    for layer in layers:
        layer_type = str(layer.get("LayerType") or layer.get("type") or "")
        name = str(layer.get("Name") or layer.get("name") or "")
        text = json.dumps(layer, sort_keys=True)
        layer_types[layer_type or "<unknown>"] += 1
        for token in ("Int8", "Half", "Float", "FP8", "BF16"):
            if token in text:
                precision_tokens[token] += 1
        inputs = [
            str(item.get("Format/Datatype", ""))
            for item in layer.get("Inputs", [])
            if isinstance(item, dict)
        ]
        outputs = [
            str(item.get("Format/Datatype", ""))
            for item in layer.get("Outputs", [])
            if isinstance(item, dict)
        ]
        for dtype in outputs:
            output_dtypes[dtype] += 1
            for token in ("Int8", "Half", "Float", "FP8", "BF16"):
                if token in dtype:
                    output_type_tokens[token] += 1
        if "Conv" in layer_type:
            weight_type = ""
            if isinstance(layer.get("Weights"), dict):
                weight_type = str(layer["Weights"].get("Type", ""))
            conv_patterns[("|".join(inputs), "|".join(outputs), weight_type)] += 1
        if "reformat" in name.lower() or "reformat" in layer_type.lower() or "Reformat" in text:
            reformat_layers.append({"name": name, "type": layer_type})
        if "Int8" in text:
            int8_like_layers.append({"name": name, "type": layer_type})

    summary.update(
        {
            "layer_types": dict(sorted(layer_types.items())),
            "precision_token_counts": dict(sorted(precision_tokens.items())),
            "output_type_token_counts": dict(sorted(output_type_tokens.items())),
            "output_dtypes": output_dtypes.most_common(),
            "conv_io_weight_patterns": [
                {"inputs": key[0], "outputs": key[1], "weights": key[2], "count": count}
                for key, count in conv_patterns.most_common()
            ],
            "reformat_layers": reformat_layers,
            "int8_like_layer_count": len(int8_like_layers),
            "sample_int8_layers": int8_like_layers[:30],
        }
    )
    return engine, summary


def _profile_engine(
    engine,
    output_dir: Path,
    tag: str,
    runs: int,
    width: int,
    height: int,
) -> dict[str, object]:
    import tensorrt as trt
    import torch

    class _Profiler(trt.IProfiler):
        def __init__(self):
            trt.IProfiler.__init__(self)
            self.records = []

        def report_layer_time(self, layer_name, ms):
            self.records.append((str(layer_name), float(ms)))

    context = engine.create_execution_context()
    profiler = _Profiler()
    context.profiler = profiler
    tensors = {}
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        if name == "input":
            shape = (1, 3, int(height), int(width))
        elif name == "cond":
            shape = (1, 3, max(1, int(height) // 4), max(1, int(width) // 4))
        else:
            shape = tuple(engine.get_tensor_shape(name))
        context.set_input_shape(name, shape)
        tensors[name] = torch.zeros(shape, device="cuda", dtype=torch.float16)
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        tensors[name] = torch.empty(
            tuple(context.get_tensor_shape(name)),
            device="cuda",
            dtype=torch.float16,
        )
    for name, tensor in tensors.items():
        context.set_tensor_address(name, int(tensor.data_ptr()))

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(10):
            context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    profiler.records.clear()
    start = time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(max(1, int(runs))):
            context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000.0 / max(1, int(runs))

    aggregate = collections.defaultdict(lambda: {"count": 0, "total_ms": 0.0})
    for name, ms in profiler.records:
        aggregate[name]["count"] += 1
        aggregate[name]["total_ms"] += ms
    rows = []
    for name, values in aggregate.items():
        rows.append(
            {
                "name": name,
                "count": values["count"],
                "avg_ms": values["total_ms"] / max(1, values["count"]),
            }
        )
    rows.sort(key=lambda row: row["avg_ms"], reverse=True)
    summary = {
        "wall_avg_ms": wall_ms,
        "profile_layers": len(rows),
        "top_layers": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{tag}_profile.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect TensorRT ONNX/engine artifacts.")
    parser.add_argument("--onnx", nargs="*", type=Path, default=[], help="ONNX file(s) to inspect.")
    parser.add_argument("--engine", nargs="*", type=Path, default=[], help="TensorRT engine file(s) to inspect.")
    parser.add_argument("--output", type=Path, default=None, help="Optional combined JSON output path.")
    parser.add_argument("--tag", default="artifact", help="Output filename prefix for per-artifact files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/tensorrt_inspection"),
        help="Directory for raw inspector/profile JSON files.",
    )
    parser.add_argument("--profile", action="store_true", help="Run dummy TensorRT profile for engine artifacts.")
    parser.add_argument("--profile-runs", type=int, default=30)
    parser.add_argument("--profile-resolution", default="1280x720", help="Profile input resolution, WxH.")
    args = parser.parse_args()

    if not args.onnx and not args.engine:
        parser.error("Provide at least one --onnx or --engine artifact.")

    try:
        profile_width, profile_height = (
            int(part) for part in str(args.profile_resolution).lower().split("x", 1)
        )
    except Exception as exc:
        raise SystemExit(f"Invalid --profile-resolution: {args.profile_resolution}") from exc

    result: dict[str, object] = {"onnx": [], "engine": []}
    for index, path in enumerate(args.onnx):
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = _inspect_onnx(path)
        result["onnx"].append(summary)
        artifact_tag = args.tag if len(args.onnx) == 1 else f"{args.tag}_onnx_{index}"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"{artifact_tag}_onnx_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    for index, path in enumerate(args.engine):
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact_tag = args.tag if len(args.engine) == 1 else f"{args.tag}_engine_{index}"
        engine, summary = _inspect_engine(path, args.output_dir, artifact_tag)
        if args.profile and engine is not None:
            summary["profile"] = _profile_engine(
                engine,
                args.output_dir,
                artifact_tag,
                args.profile_runs,
                profile_width,
                profile_height,
            )
        result["engine"].append(summary)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"{artifact_tag}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"[inspect] wrote {args.output}")
    else:
        print(text[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
