import numpy as np
import cv2
import onnxruntime as ort


class HDRTVNetONNX:
    def __init__(self, onnx_path):

        so = ort.SessionOptions()
        so.intra_op_num_threads = 0
        so.inter_op_num_threads = 0
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=["DmlExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.cond_name = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, frame_bgr):
        # Convert once
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Normalize in-place
        frame = frame_rgb.astype(np.float32)
        frame *= (1.0 / 255.0)

        # Rearrange
        frame = np.transpose(frame, (2, 0, 1))[None, :]

        # Condition branch (smaller)
        cond_small = cv2.resize(
            frame_rgb,
            (frame_rgb.shape[1] // 4, frame_rgb.shape[0] // 4),
            interpolation=cv2.INTER_LINEAR
        )

        cond = cond_small.astype(np.float32)
        cond *= (1.0 / 255.0)
        cond = np.transpose(cond, (2, 0, 1))[None, :]

        return frame, cond

    def postprocess(self, output):
        # output shape: (1,3,H,W)
        output = output[0]
        output = np.transpose(output, (1, 2, 0))

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
