import numpy as np
import cv2
import onnxruntime as ort


class HDRTVNetONNX:
    def __init__(self, onnx_path, provider="auto", device_id=0, enable_profiling=False):

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if enable_profiling:
            so.enable_profiling = True

        providers = self._build_providers(provider, device_id)

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=providers
        )

        self.input_name, self.cond_name = self._resolve_input_names()
        self.output_name = self.session.get_outputs()[0].name

        # Detect precision
        input_type = self.session.get_inputs()[0].type
        if "float16" in input_type:
            self.dtype = np.float16
            print("Running in FP16 mode")
        else:
            self.dtype = np.float32
            print("Running in FP32 mode")

    def end_profiling(self):
        if hasattr(self.session, "end_profiling"):
            return self.session.end_profiling()
        return None

    def _build_providers(self, provider, device_id):
        available = set(ort.get_available_providers())
        mode = provider.lower()

        if mode not in {"auto", "dml", "cpu"}:
            raise ValueError("provider must be one of: auto, dml, cpu")

        resolved = []

        if mode in {"auto", "dml"} and "DmlExecutionProvider" in available:
            resolved.append(("DmlExecutionProvider", {"device_id": device_id}))
        elif mode == "dml":
            raise RuntimeError("DmlExecutionProvider is not available in this environment.")

        if "CPUExecutionProvider" in available:
            resolved.append("CPUExecutionProvider")

        if not resolved:
            raise RuntimeError(
                "No compatible execution provider found. "
                f"Available providers: {sorted(available)}"
            )

        print(f"ONNX providers: {resolved}")
        return resolved

    def _resolve_input_names(self):
        inputs = self.session.get_inputs()
        if len(inputs) < 2:
            raise RuntimeError(
                f"Expected 2 inputs (image + condition), found {len(inputs)}: "
                f"{[x.name for x in inputs]}"
            )

        names = [item.name for item in inputs]
        lowered = {name.lower(): name for name in names}

        input_name = lowered.get("input") or names[0]
        cond_name = lowered.get("condition") or names[1]
        return input_name, cond_name

    def preprocess(self, frame_bgr):

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # CPU preprocessing is typically faster in FP32; cast to FP16 at the edge.
        frame = frame_rgb.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = np.transpose(frame, (2, 0, 1))[None, :]

        cond_w = max(1, frame_rgb.shape[1] // 4)
        cond_h = max(1, frame_rgb.shape[0] // 4)
        cond_small = cv2.resize(
            frame_rgb,
            (cond_w, cond_h),
            interpolation=cv2.INTER_LINEAR
        )

        cond = cond_small.astype(np.float32)
        cond *= (1.0 / 255.0)
        cond = np.transpose(cond, (2, 0, 1))[None, :]

        if self.dtype == np.float16:
            frame = frame.astype(np.float16, copy=False)
            cond = cond.astype(np.float16, copy=False)

        return frame, cond

    def postprocess(self, output):

        output = output[0]
        output = np.transpose(output, (1, 2, 0))
        if output.dtype == np.float16:
            output = output.astype(np.float32, copy=False)

        output = np.clip(output, 0.0, 1.0)
        output = (output * 255.0).astype(np.uint8)

        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    def process(self, frame_bgr):

        input_tensor, cond_tensor = self.preprocess(frame_bgr)

        outputs = self.session.run(
            None,
            {
                self.input_name: input_tensor,
                self.cond_name: cond_tensor
            }
        )

        return self.postprocess(outputs[0])
