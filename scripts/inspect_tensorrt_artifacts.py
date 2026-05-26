"""Inspect TensorRT INT8 ONNX and engine artifacts.

The ONNX inspection works anywhere. Engine inspection requires NVIDIA
TensorRT Python bindings and a compatible CUDA machine.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _inspect_onnx(path: Path) -> dict[str, object]:
    import onnx
    from onnx import TensorProto

    model = onnx.load(path)
    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)
    producers = {output: node for node in graph.node for output in node.output}
    consumers = defaultdict(list)
    for node in graph.node:
        for index, input_name in enumerate(node.input):
            consumers[input_name].append((node, index))

    initializer_types = Counter()
    initializer_names = {init.name for init in graph.initializer}
    for init in graph.initializer:
        dtype_name = TensorProto.DataType.Name(init.data_type)
        initializer_types[dtype_name] += 1

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
            qdq_boundary = any(consumer.op_type in compute_ops for consumer, _ in node_consumers)
        if any(consumer.op_type == "QuantizeLinear" for consumer, _ in node_consumers):
            qdq_boundary = True
        entry = {
            "name": node.name,
            "input": list(node.input),
            "output": list(node.output),
            "producer": producer.op_type if producer is not None else None,
            "consumers": [
                {"op_type": consumer.op_type, "name": consumer.name, "input_index": index}
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

    add_inputs_with_dq = 0
    add_inputs_total = 0
    mul_inputs_with_dq = 0
    mul_inputs_total = 0
    for node in graph.node:
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
        "graph": {
            "nodes": len(graph.node),
            "initializers": len(graph.initializer),
            "inputs": [value.name for value in graph.input],
            "outputs": [value.name for value in graph.output],
        },
        "op_counts": dict(sorted(op_counts.items())),
        "initializer_types": dict(sorted(initializer_types.items())),
        "qdq": {
            "quantize_linear": int(op_counts.get("QuantizeLinear", 0)),
            "dequantize_linear": int(op_counts.get("DequantizeLinear", 0)),
            "dq_consumers": dict(sorted(qdq_consumers.items())),
            "q_producers": dict(sorted(qdq_producers.items())),
            "add_inputs_with_dq": int(add_inputs_with_dq),
            "add_inputs_total": int(add_inputs_total),
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


def _extract_engine_layers(info_text: str):
    try:
        parsed = json.loads(info_text)
    except Exception:
        return None
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("Layers", "layers"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
    return None


def _inspect_engine(path: Path) -> dict[str, object]:
    try:
        import tensorrt as trt
    except Exception as exc:
        return {"path": str(path), "available": False, "error": f"TensorRT import failed: {exc}"}

    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    data = path.read_bytes()
    engine = runtime.deserialize_cuda_engine(data)
    if engine is None:
        return {"path": str(path), "available": False, "error": "deserialize_cuda_engine returned None"}

    inspector = engine.create_engine_inspector()
    info_text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
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
        return summary

    layer_types = Counter()
    precision_tokens = Counter()
    output_type_tokens = Counter()
    reformat_layers = []
    int8_like_layers = []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        layer_type = str(layer.get("LayerType") or layer.get("type") or "")
        name = str(layer.get("Name") or layer.get("name") or "")
        text = json.dumps(layer, sort_keys=True)
        layer_types[layer_type or "<unknown>"] += 1
        for token in ("Int8", "Half", "Float", "FP8", "BF16"):
            if token in text:
                precision_tokens[token] += 1
        for key in ("Outputs", "outputs"):
            outputs = layer.get(key)
            if isinstance(outputs, list):
                for output in outputs:
                    output_text = json.dumps(output, sort_keys=True)
                    for token in ("Int8", "Half", "Float", "FP8", "BF16"):
                        if token in output_text:
                            output_type_tokens[token] += 1
        if "reformat" in name.lower() or "reformat" in layer_type.lower() or "Reformat" in text:
            reformat_layers.append({"name": name, "type": layer_type})
        if "Int8" in text:
            int8_like_layers.append({"name": name, "type": layer_type})

    summary.update(
        {
            "layer_types": dict(sorted(layer_types.items())),
            "precision_token_counts": dict(sorted(precision_tokens.items())),
            "output_type_token_counts": dict(sorted(output_type_tokens.items())),
            "reformat_layers": reformat_layers,
            "int8_like_layer_count": len(int8_like_layers),
            "sample_int8_layers": int8_like_layers[:30],
        }
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect TensorRT ONNX/engine artifacts.")
    parser.add_argument("--onnx", nargs="*", type=Path, default=[], help="ONNX file(s) to inspect.")
    parser.add_argument("--engine", nargs="*", type=Path, default=[], help="TensorRT engine file(s) to inspect.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    result: dict[str, object] = {"onnx": [], "engine": []}
    for path in args.onnx:
        if not path.is_file():
            raise FileNotFoundError(path)
        result["onnx"].append(_inspect_onnx(path))
    for path in args.engine:
        if not path.is_file():
            raise FileNotFoundError(path)
        result["engine"].append(_inspect_engine(path))

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"[inspect] wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
