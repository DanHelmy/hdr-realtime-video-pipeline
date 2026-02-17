import numpy as np

class DummyProcessor:
    """
    Placeholder for HDR / ML inference.
    This will later be replaced by FP32 / FP16 / INT8 models.
    """

    def process(self, frame_bgr):
        # Convert to float
        img = frame_bgr.astype(np.float32) / 255.0

        # Dummy "HDR-like" expansion (gamma curve)
        img = np.power(img, 2.2)

        # Back to displayable range
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)

        return img
