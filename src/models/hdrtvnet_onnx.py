import numpy as np
import cv2
import onnxruntime as ort
import time


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
        self.expected_hw = self._resolve_expected_hw()
        self.is_static_input_model = self.expected_hw is not None
        if self.is_static_input_model:
            exp_h, exp_w = self.expected_hw
            print(f"Static model input detected: {exp_w}x{exp_h}")

        # Detect precision
        input_type = self.session.get_inputs()[0].type
        if "float16" in input_type:
            self.dtype = np.float16
            print("Running in FP16 mode")
        else:
            self.dtype = np.float32
            print("Running in FP32 mode")

        self._scale = np.float32(1.0 / 255.0)
        self._cached_hw = None
        self._cond_hw = None
        self._frame_rgb = None
        self._cond_rgb = None
        self._input_fp32 = None
        self._cond_fp32 = None
        self._input_fp16 = None
        self._cond_fp16 = None

    def end_profiling(self):
        if hasattr(self.session, "end_profiling"):
            return self.session.end_profiling()
        return None

    def _build_providers(self, provider, device_id):
        available = set(ort.get_available_providers())
        mode = provider.lower()
        valid = {"auto", "dml", "cuda", "rocm", "tensorrt", "coreml", "openvino", "cpu"}
        if mode not in valid:
            raise ValueError(f"provider must be one of: {sorted(valid)}")

        resolved = []
        gpu_priority = [
            ("TensorrtExecutionProvider", None),
            ("CUDAExecutionProvider", None),
            ("ROCMExecutionProvider", None),
            ("DmlExecutionProvider", {"device_id": device_id}),
            ("CoreMLExecutionProvider", None),
            ("OpenVINOExecutionProvider", None),
        ]
        mode_to_ep = {
            "dml": ("DmlExecutionProvider", {"device_id": device_id}),
            "cuda": ("CUDAExecutionProvider", None),
            "rocm": ("ROCMExecutionProvider", None),
            "tensorrt": ("TensorrtExecutionProvider", None),
            "coreml": ("CoreMLExecutionProvider", None),
            "openvino": ("OpenVINOExecutionProvider", None),
        }

        if mode == "auto":
            for ep_name, ep_opts in gpu_priority:
                if ep_name in available:
                    resolved.append((ep_name, ep_opts) if ep_opts is not None else ep_name)
                    break
        elif mode == "cpu":
            pass
        else:
            ep_name, ep_opts = mode_to_ep[mode]
            if ep_name not in available:
                raise RuntimeError(
                    f"{ep_name} is not available in this environment. "
                    f"Available providers: {sorted(available)}"
                )
            resolved.append((ep_name, ep_opts) if ep_opts is not None else ep_name)

        # Always keep CPU fallback unless user explicitly requests CPU only.
        if mode != "cpu" and "CPUExecutionProvider" in available:
            resolved.append("CPUExecutionProvider")
        elif mode == "cpu":
            if "CPUExecutionProvider" not in available:
                raise RuntimeError(
                    f"CPUExecutionProvider is not available. Available providers: {sorted(available)}"
                )
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

    def _resolve_expected_hw(self):
        shape = self.session.get_inputs()[0].shape
        if len(shape) < 4:
            return None
        h = shape[2]
        w = shape[3]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return (h, w)
        return None

    def _ensure_preprocess_buffers(self, h, w):
        if self._cached_hw == (h, w):
            return

        cond_h = max(1, h // 4)
        cond_w = max(1, w // 4)

        self._frame_rgb = np.empty((h, w, 3), dtype=np.uint8)
        self._cond_rgb = np.empty((cond_h, cond_w, 3), dtype=np.uint8)

        self._input_fp32 = np.empty((1, 3, h, w), dtype=np.float32)
        self._cond_fp32 = np.empty((1, 3, cond_h, cond_w), dtype=np.float32)

        if self.dtype == np.float16:
            self._input_fp16 = np.empty((1, 3, h, w), dtype=np.float16)
            self._cond_fp16 = np.empty((1, 3, cond_h, cond_w), dtype=np.float16)
        else:
            self._input_fp16 = None
            self._cond_fp16 = None

        self._cached_hw = (h, w)
        self._cond_hw = (cond_h, cond_w)

    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self._ensure_preprocess_buffers(h, w)

        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB, dst=self._frame_rgb)

        frame = self._input_fp32
        frame[0, 0, :, :] = self._frame_rgb[:, :, 0]
        frame[0, 1, :, :] = self._frame_rgb[:, :, 1]
        frame[0, 2, :, :] = self._frame_rgb[:, :, 2]
        np.multiply(frame, self._scale, out=frame)

        cond_h, cond_w = self._cond_hw
        cv2.resize(
            self._frame_rgb,
            (cond_w, cond_h),
            dst=self._cond_rgb,
            interpolation=cv2.INTER_LINEAR
        )

        cond = self._cond_fp32
        cond[0, 0, :, :] = self._cond_rgb[:, :, 0]
        cond[0, 1, :, :] = self._cond_rgb[:, :, 1]
        cond[0, 2, :, :] = self._cond_rgb[:, :, 2]
        np.multiply(cond, self._scale, out=cond)

        if self.dtype == np.float16:
            self._input_fp16[...] = frame
            self._cond_fp16[...] = cond
            return self._input_fp16, self._cond_fp16

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

    def process_timed(self, frame_bgr):
        t0 = time.perf_counter()
        input_tensor, cond_tensor = self.preprocess(frame_bgr)
        t1 = time.perf_counter()

        outputs = self.session.run(
            None,
            {
                self.input_name: input_tensor,
                self.cond_name: cond_tensor
            }
        )
        t2 = time.perf_counter()

        output = self.postprocess(outputs[0])
        t3 = time.perf_counter()

        return output, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0
