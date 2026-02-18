import argparse
import torch
import torch.nn.functional as F

from src.models.hdrtvnet_modules.Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE


def parse_args():
    parser = argparse.ArgumentParser(description="Export HDRTVNet FP16 ONNX")
    parser.add_argument("--weights", default="src/models/weights/Ensemble_AGCM_LE.pth")
    parser.add_argument("--output", default="hdrtvnet_fp16.onnx")
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--static",
        action="store_true",
        help="Export fixed-shape ONNX (no dynamic axes)"
    )
    return parser.parse_args()


def load_model(weight_path, device):
    model = Ensemble_AGCM_LE(
        classifier="color_condition",
        cond_c=6,
        in_nc=3,
        out_nc=3,
        nf=32,
        act_type="relu",
        weighting_network=False,
    ).to(device)

    state_dict = torch.load(weight_path, map_location=device)
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(cleaned, strict=True)
    model.half()
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cpu")
    model = load_model(args.weights, device)

    dummy_input = torch.randn(1, 3, args.height, args.width).half()
    dummy_cond = F.interpolate(dummy_input, scale_factor=1 / 4, mode="bilinear", align_corners=False).half()

    export_kwargs = {
        "input_names": ["input", "condition"],
        "output_names": ["output", "condition_out"],
        "opset_version": args.opset,
    }
    if not args.static:
        export_kwargs["dynamic_axes"] = {
            "input": {2: "height", 3: "width"},
            "condition": {2: "cond_height", 3: "cond_width"},
            "output": {2: "height", 3: "width"},
        }

    print("Exporting FP16 ONNX...")
    torch.onnx.export(model, ((dummy_input, dummy_cond),), args.output, **export_kwargs)
    print(f"ONNX export complete: {args.output}")


if __name__ == "__main__":
    main()
