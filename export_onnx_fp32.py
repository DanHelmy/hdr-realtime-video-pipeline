import torch
import torch.nn.functional as F
import numpy as np

from src.models.hdrtvnet_modules.Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE

WEIGHT_PATH = "src/models/weights/Ensemble_AGCM_LE.pth"
ONNX_OUTPUT = "hdrtvnet_fp32.onnx"

device = torch.device("cpu")

# Build model
model = Ensemble_AGCM_LE(
    classifier='color_condition',
    cond_c=6,
    in_nc=3,
    out_nc=3,
    nf=32,
    act_type='relu',
    weighting_network=False
).to(device)

# Load weights
state_dict = torch.load(WEIGHT_PATH, map_location=device)

cleaned = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        cleaned[k[7:]] = v
    else:
        cleaned[k] = v

model.load_state_dict(cleaned, strict=True)
model.eval()

print("Model loaded.")

# Dummy input (1,3,1080,1920) — use smaller than 4K for export safety
dummy_input = torch.randn(1, 3, 720, 1280)
dummy_cond = F.interpolate(dummy_input, scale_factor=1/4, mode="bilinear", align_corners=False)

print("Exporting to ONNX...")

torch.onnx.export(
    model,
    ((dummy_input, dummy_cond),),
    ONNX_OUTPUT,
    input_names=["input", "condition"],
    output_names=["output", "condition_out"],
    opset_version=17,
    dynamic_axes={
        "input": {2: "height", 3: "width"},
        "condition": {2: "cond_height", 3: "cond_width"},
        "output": {2: "height", 3: "width"}
    }
)

print("ONNX export complete.")
