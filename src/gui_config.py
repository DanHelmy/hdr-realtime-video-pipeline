"""Static GUI configuration/constants (precision + resolution presets)."""

from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _weight(name: str) -> str:
    return os.path.join(_HERE, "models", "weights", name)


PRECISIONS = {
    "FP16": {
        "precision": "fp16",
        "model": _weight("Ensemble_AGCM_LE.pth"),
    },
    "FP32": {
        "precision": "fp32",
        "model": _weight("Ensemble_AGCM_LE.pth"),
    },
    "INT8 Mixed (PTQ)": {
        "precision": "int8-mixed",
        "model": _weight("Ensemble_AGCM_LE_int8_mixed.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_mixed_nohg.pt"),
    },
    "INT8 Mixed (QAT)": {
        "precision": "int8-mixed",
        "model": _weight("Ensemble_AGCM_LE_int8_mixed_qat.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_mixed_qat_nohg.pt"),
    },
    "INT8 Full (PTQ)": {
        "precision": "int8-full",
        "model": _weight("Ensemble_AGCM_LE_int8_full.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_full_nohg.pt"),
    },
    "INT8 Full (QAT)": {
        "precision": "int8-full",
        "model": _weight("Ensemble_AGCM_LE_int8_full_qat.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_full_qat_nohg.pt"),
    },
}


def _select_model_path(precision_key: str, use_hg: bool) -> str:
    cfg = PRECISIONS.get(precision_key, {})
    model_path = cfg.get("model", "")
    if cfg.get("precision", "").startswith("int8") and not use_hg:
        model_path = cfg.get("model_nohg") or model_path
    return model_path


def _precision_is_available(precision_key: str) -> bool:
    cfg = PRECISIONS.get(precision_key, {})
    model_path = cfg.get("model")
    if model_path and os.path.isfile(model_path):
        return True
    if cfg.get("precision", "").startswith("int8"):
        alt_path = cfg.get("model_nohg")
        if alt_path and os.path.isfile(alt_path):
            return True
    return False


def _available_precision_keys() -> list[str]:
    keys = [k for k in PRECISIONS.keys() if _precision_is_available(k)]
    return keys or list(PRECISIONS.keys())


MAX_W, MAX_H = 1920, 1080


# Resolution-scale presets (process lower resolution).
RESOLUTION_SCALES = {
    "1080p": None,  # full output resolution path (no upscale stage)
    "720p": (1280, 720),
    "540p": (960, 540),
}

