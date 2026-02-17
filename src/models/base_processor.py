class BaseProcessor:
    """
    Abstract processor interface.
    All HDR models (FP32, FP16, INT8, ONNX) must implement this.
    """

    def __init__(self):
        pass

    def preprocess(self, frame):
        """
        Convert raw frame (numpy array) into model-ready tensor.
        """
        raise NotImplementedError

    def infer(self, tensor):
        """
        Run model inference.
        """
        raise NotImplementedError

    def postprocess(self, output):
        """
        Convert model output back to displayable frame.
        """
        raise NotImplementedError

    def process(self, frame):
        """
        Full pipeline: frame → tensor → inference → frame
        """
        tensor = self.preprocess(frame)
        output = self.infer(tensor)
        return self.postprocess(output)
