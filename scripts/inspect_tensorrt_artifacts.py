from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time

import onnx
import tensorrt as trt
from onnx import TensorProto


def _dtype_name_map() -> dict[int, str]:
    return {value: key for key, value in TensorProto.DataType.items()}


def _load_onnx_summary(path: pathlib.Path) -> dict[str, object]:
    dtype_names = _dtype_name_map()
    model = onnx.load(str(path), load_external_data=True)
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    producers = {}
    consumers = collections.defaultdict(list)
    for node in graph.node:
        for output in node.output:
            producers[output] = node
        for input_name in node.input:
            consumers[input_name].append(node)

    node_types = collections.Counter(node.op_type for node in graph.node)
    initializer_types = collections.Counter(
        dtype_names.get(tensor.data_type, str(tensor.data_type))
        for tensor in initializers.values()
    )
    conv_nodes = [node for node in graph.node if node.op_type == "Conv"]
    add_nodes = [node for node in graph.node if node.op_type == "Add"]
    cast_nodes = []
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        cast_nodes.append(
            {
                "name": node.name,
                "input_from": producers.get(node.input[0]).op_type
                if node.input and producers.get(node.input[0])
                else None,
                "users": [
                    user.op_type
                    for output in node.output
                    for user in consumers.get(output, [])
                ],
            }
        )

    fp32_initializers = [
        tensor.name
        for tensor in initializers.values()
        if dtype_names.get(tensor.data_type) == "FLOAT"
    ]
    return {
        "path": str(path),
        "opsets": [(op.domain, op.version) for op in model.opset_import],
        "nodes": len(graph.node),
        "node_types": node_types.most_common(),
        "initializer_dtypes": dict(initializer_types),
        "quantize_nodes": node_types.get("QuantizeLinear", 0),
        "dequantize_nodes": node_types.get("DequantizeLinear", 0),
        "conv_nodes": len(conv_nodes),
        "conv_activation_from_dq": sum(
            1
            for node in conv_nodes
            if node.input
            and producers.get(node.input[0])
            and producers[node.input[0]].op_type == "DequantizeLinear"
        ),
        "conv_weight_from_dq": sum(
            1
            for node in conv_nodes
            if len(node.input) > 1
            and producers.get(node.input[1])
            and producers[node.input[1]].op_type == "DequantizeLinear"
        ),
        "add_nodes": len(add_nodes),
        "add_inputs_from_dq": sum(
            1
            for node in add_nodes
            for input_name in node.input
            if producers.get(input_name)
            and producers[input_name].op_type == "DequantizeLinear"
        ),
        "add_outputs_to_q": sum(
            1
            for node in add_nodes
            if any(
                user.op_type == "QuantizeLinear"
                for output in node.output
                for user in consumers.get(output, [])
            )
        ),
        "cast_nodes": cast_nodes,
        "fp32_initializer_count": len(fp32_initializers),
        "fp32_initializer_sample": fp32_initializers[:24],
    }


def _normalize_layers(info: object) -> list[dict[str, object]]:
    layers = info.get("Layers") if isinstance(info, dict) else []
    if isinstance(layers, dict):
        layers = list(layers.values())
    normalized = []
    for item in layers if isinstance(layers, list) else []:
        if isinstance(item, dict):
            normalized.append(item)
            continue
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
            except Exception:
                parsed = {"Name": item}
            normalized.append(parsed if isinstance(parsed, dict) else {"Name": item})
    return normalized


def _inspect_engine(path: pathlib.Path, output_dir: pathlib.Path, tag: str):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize TensorRT engine: {path}")
    raw = engine.create_engine_inspector().get_engine_information(
        trt.LayerInformationFormat.JSON
    )
    (output_dir / f"{tag}_engine_inspector.json").write_text(raw, encoding="utf-8")
    try:
        layers = _normalize_layers(json.loads(raw))
    except Exception:
        layers = []

    layer_types = collections.Counter()
    output_dtypes = collections.Counter()
    conv_patterns = collections.Counter()
    for layer in layers:
        layer_type = str(layer.get("LayerType", "unknown"))
        layer_types[layer_type] += 1
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
        if "Conv" in layer_type:
            weight_type = ""
            if isinstance(layer.get("Weights"), dict):
                weight_type = str(layer["Weights"].get("Type", ""))
            conv_patterns[
                ("|".join(inputs), "|".join(outputs), weight_type)
            ] += 1

    return engine, {
        "path": str(path),
        "layers": len(layers),
        "layer_types": layer_types.most_common(),
        "output_dtypes": output_dtypes.most_common(),
        "conv_io_weight_patterns": [
            {"inputs": key[0], "outputs": key[1], "weights": key[2], "count": count}
            for key, count in conv_patterns.most_common()
        ],
    }


class _Profiler(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.records = []

    def report_layer_time(self, layer_name, ms):
        self.records.append((str(layer_name), float(ms)))


def _profile_engine(engine, output_dir: pathlib.Path, tag: str, runs: int) -> dict[str, object]:
    import torch

    context = engine.create_execution_context()
    profiler = _Profiler()
    context.profiler = profiler
    tensors = {}
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        if name == "input":
            shape = (1, 3, 720, 1280)
        elif name == "cond":
            shape = (1, 3, 180, 320)
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
    (output_dir / f"{tag}_profile.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect ONNX and TensorRT artifacts.")
    parser.add_argument("--engine", required=True, help="Path to .engine file.")
    parser.add_argument("--onnx", default=None, help="Optional matching .onnx file.")
    parser.add_argument("--tag", default="engine", help="Output filename prefix.")
    parser.add_argument(
        "--output-dir",
        default="logs/tensorrt_inspection",
        help="Directory for JSON inspection output.",
    )
    parser.add_argument("--profile", action="store_true", help="Run dummy TensorRT profile.")
    parser.add_argument("--profile-runs", type=int, default=30)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    if args.onnx:
        result["onnx"] = _load_onnx_summary(pathlib.Path(args.onnx))
        (output_dir / f"{args.tag}_onnx_summary.json").write_text(
            json.dumps(result["onnx"], indent=2),
            encoding="utf-8",
        )
    engine, engine_summary = _inspect_engine(
        pathlib.Path(args.engine),
        output_dir,
        args.tag,
    )
    result["engine"] = engine_summary
    if args.profile:
        result["profile"] = _profile_engine(
            engine,
            output_dir,
            args.tag,
            args.profile_runs,
        )
    (output_dir / f"{args.tag}_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
