import torch
import torch.nn.functional as F
import numpy as np
import cv2

from models.base_processor import BaseProcessor
from models.hdrtvnet_modules.Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE


class HDRTVNetFP32(BaseProcessor):
    """
    FP32 inference wrapper for HDRTVNet (Ensemble_AGCM_LE).
    Clean inference-only implementation.
    """

    def __init__(self, weight_path):
        super().__init__()

        self.device = torch.device("cpu")  # CPU for now
        self.weight_path = weight_path

        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Load architecture and pretrained weights.
        """

        # Instantiate architecture
        self.model = Ensemble_AGCM_LE(
            classifier='color_condition',
            cond_c=6,
            in_nc=3,
            out_nc=3,
            nf=32,
            act_type='relu',
            weighting_network=False
        ).to(self.device)

        # Load weights
        state_dict = torch.load(self.weight_path, map_location=self.device)

        # Remove 'module.' prefix if present
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                cleaned[k[7:]] = v
            else:
                cleaned[k] = v

        self.model.load_state_dict(cleaned, strict=True)
        self.model.eval()

        print("HDRTVNet FP32 model loaded successfully.")

    def preprocess(self, frame_bgr):
        """
        Convert BGR uint8 frame → torch tensor (1,3,H,W) in RGB [0,1]
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = frame_rgb.astype(np.float32) / 255.0

        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def infer(self, tensor):
        """
        Run inference.
        Generate condition input via 1/4 downscale.
        """

        # Condition image: 1/4 resolution
        cond = F.interpolate(
            tensor,
            scale_factor=1/4,
            mode="bilinear",
            align_corners=False
        )

        with torch.no_grad():
            output, _ = self.model((tensor, cond))

        return output

    def postprocess(self, output):
        """
        Convert model output → BGR uint8 frame
        """

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output, 0.0, 1.0)
        output = (output * 255.0).astype(np.uint8)

        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output_bgr
