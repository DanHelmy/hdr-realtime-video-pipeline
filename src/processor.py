import numpy as np
from models.base_processor import BaseProcessor


class DummyProcessor(BaseProcessor):
    """
    Placeholder for HDR / ML inference.
    This will later be replaced by FP32 / FP16 / INT8 models.
    """

    def preprocess(self, frame_bgr):
        # Convert to float [0,1]
        return frame_bgr.astype(np.float32) / 255.0

    def infer(self, tensor):
        # Dummy "HDR-like" expansion (gamma curve)
        return np.power(tensor, 2.2)

    def postprocess(self, output):
        # Back to displayable range
        output = np.clip(output, 0.0, 1.0)
        return (output * 255.0).astype(np.uint8)
