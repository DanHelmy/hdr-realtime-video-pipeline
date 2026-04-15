"""Objective image/video metrics helpers for the GUI compare flow."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Objective quality metrics tuning.
_OBJECTIVE_METRIC_SAMPLE_EVERY = 6
_OBJECTIVE_HDRVDP3_SAMPLE_EVERY = 24
_OBJECTIVE_METRIC_MAX_SIDE = 512
_OBJECTIVE_HDRVDP3_MAX_SIDE = 256
_HDRVDP3_CMD_ENV = "HDRTVNET_HDRVDP3_CMD"


def _frame_peak_value(frame: np.ndarray) -> float:
    if frame is None:
        return 255.0
    if frame.dtype == np.uint16:
        return 65535.0
    if np.issubdtype(frame.dtype, np.integer):
        return float(np.iinfo(frame.dtype).max)
    max_val = float(np.max(frame)) if np.size(frame) else 1.0
    return 1.0 if max_val <= 1.0 else 65535.0


def _to_unit_float(frame: np.ndarray) -> np.ndarray:
    peak = max(1.0, _frame_peak_value(frame))
    return frame.astype(np.float32) / peak


def _default_hdrvdp3_cmd_template() -> str:
    """Return built-in HDR-VDP3 bridge command template if available."""
    bridge = os.path.join(_ROOT, "scripts", "hdrvdp3_bridge.py")
    if not os.path.isfile(bridge):
        return ""
    py = str(sys.executable or "python")
    return f'"{py}" "{bridge}" --test "{{test}}" --reference "{{reference}}"'


class _RunningAverage:
    def __init__(self):
        self.count = 0
        self.value = None

    def update(self, x: float):
        try:
            x = float(x)
        except Exception:
            return
        if not np.isfinite(x):
            return
        self.count += 1
        if self.value is None:
            self.value = x
        else:
            self.value += (x - self.value) / float(self.count)


def _read_video_frame_at(path: str, frame_idx: int) -> np.ndarray | None:
    """Best-effort random access to a single frame from a video path."""
    if not path or not os.path.isfile(path):
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        idx = max(0, int(frame_idx))
    except Exception:
        idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return frame


def _prepare_metric_pair(
    pred_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    max_side: int = _OBJECTIVE_METRIC_MAX_SIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Align and optionally downscale two BGR uint8 frames for metric evaluation."""
    if pred_bgr is None or ref_bgr is None:
        raise ValueError("Missing frame(s) for metric evaluation.")
    if pred_bgr.shape[:2] != ref_bgr.shape[:2]:
        ref_bgr = cv2.resize(
            ref_bgr, (pred_bgr.shape[1], pred_bgr.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    h, w = pred_bgr.shape[:2]
    if max(h, w) > max(1, int(max_side)):
        scale = float(max_side) / float(max(h, w))
        nw = max(2, int(round(w * scale)))
        nh = max(2, int(round(h * scale)))
        pred_bgr = cv2.resize(pred_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        ref_bgr = cv2.resize(ref_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return pred_bgr, ref_bgr


def _grade_normalize_pred_to_ref(
    pred_bgr: np.ndarray, ref_bgr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match prediction global channel statistics to reference grade."""
    pred = pred_bgr.astype(np.float32)
    ref = ref_bgr.astype(np.float32)
    out = pred.copy()
    for c in range(3):
        p = pred[:, :, c]
        r = ref[:, :, c]
        mp = float(np.mean(p))
        mr = float(np.mean(r))
        sp = float(np.std(p))
        sr = float(np.std(r))
        if sp < 1e-6:
            gain = 1.0
        else:
            gain = sr / sp
        bias = mr - (gain * mp)
        out[:, :, c] = (p * gain) + bias
    peak = max(_frame_peak_value(pred_bgr), _frame_peak_value(ref_bgr))
    dtype = np.uint16 if peak > 255.0 else np.uint8
    out = np.clip(out, 0.0, peak).astype(dtype)
    ref = np.clip(ref, 0.0, peak).astype(dtype)
    return out, ref


def _psnr_bgr(pred_bgr: np.ndarray, ref_bgr: np.ndarray) -> float:
    a = _to_unit_float(pred_bgr)
    b = _to_unit_float(ref_bgr)
    mse = float(np.mean((a - b) ** 2, dtype=np.float64))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _ssim_single_channel(a: np.ndarray, b: np.ndarray) -> float:
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    num = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    ssim_map = num / (den + 1e-12)
    return float(np.mean(ssim_map, dtype=np.float64))


def _ssim_bgr(pred_bgr: np.ndarray, ref_bgr: np.ndarray) -> float:
    a = _to_unit_float(pred_bgr)
    b = _to_unit_float(ref_bgr)
    vals = [
        _ssim_single_channel(a[:, :, 0], b[:, :, 0]),
        _ssim_single_channel(a[:, :, 1], b[:, :, 1]),
        _ssim_single_channel(a[:, :, 2], b[:, :, 2]),
    ]
    return float(np.mean(vals))


def _rgb_to_ictcp(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate RGB -> ICtCp transform (BT.2100 matrix form)."""
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    l = 0.3592 * r + 0.6976 * g - 0.0358 * b
    m = -0.1922 * r + 1.1004 * g + 0.0755 * b
    s = 0.0069 * r + 0.0749 * g + 0.8434 * b
    i = 0.5 * l + 0.5 * m
    t = 1.6137 * l - 3.3234 * m + 1.7097 * s
    p = 4.3781 * l - 4.2455 * m - 0.1325 * s
    return i, t, p


def _delta_e_itp_bgr(pred_bgr: np.ndarray, ref_bgr: np.ndarray) -> float:
    pr = _to_unit_float(pred_bgr[:, :, ::-1])
    rr = _to_unit_float(ref_bgr[:, :, ::-1])
    i1, t1, p1 = _rgb_to_ictcp(pr)
    i2, t2, p2 = _rgb_to_ictcp(rr)
    di = i1 - i2
    dt = t1 - t2
    dp = p1 - p2
    de = 720.0 * np.sqrt(di * di + 0.25 * dt * dt + dp * dp + 1e-12)
    return float(np.mean(de, dtype=np.float64))


def _hdrvdp3_cli_score(pred_bgr: np.ndarray, ref_bgr: np.ndarray) -> tuple[float | None, str]:
    """Run optional HDR-VDP3 CLI command from HDRTVNET_HDRVDP3_CMD."""
    cmd_tpl = str(os.environ.get(_HDRVDP3_CMD_ENV, "") or "").strip()
    if not cmd_tpl:
        cmd_tpl = _default_hdrvdp3_cmd_template()
    if not cmd_tpl:
        return None, (
            f"HDR-VDP3 unavailable. Set {_HDRVDP3_CMD_ENV} "
            "with placeholders {test} and {reference}, or keep scripts/hdrvdp3_bridge.py."
        )

    pred_small, ref_small = _prepare_metric_pair(
        pred_bgr, ref_bgr, max_side=_OBJECTIVE_HDRVDP3_MAX_SIDE
    )
    with tempfile.TemporaryDirectory(prefix="hdrtvnet_vdp3_") as td:
        test_path = os.path.join(td, "test.tiff")
        ref_path = os.path.join(td, "reference.tiff")
        if not cv2.imwrite(test_path, pred_small):
            return None, "HDR-VDP3 write failed (test frame)."
        if not cv2.imwrite(ref_path, ref_small):
            return None, "HDR-VDP3 write failed (reference frame)."
        try:
            cmd = cmd_tpl.format(
                test=test_path,
                pred=test_path,
                reference=ref_path,
                ref=ref_path,
            )
        except Exception:
            return None, (
                f"Invalid {_HDRVDP3_CMD_ENV} template. "
                "Use placeholders {test}/{pred} and {reference}/{ref}."
            )
        try:
            cp = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=240,
                check=False,
            )
        except Exception as exc:
            return None, f"HDR-VDP3 command failed: {exc}"
        if cp.returncode != 0:
            stderr = str(cp.stderr or "").strip()
            if stderr:
                stderr = stderr.splitlines()[-1]
            return None, f"HDR-VDP3 command error (rc={cp.returncode}): {stderr or 'no stderr'}"
        merged = f"{cp.stdout or ''}\n{cp.stderr or ''}"
        marker = re.search(r"HDRVDP3_SCORE\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", merged, re.IGNORECASE)
        if marker:
            try:
                return float(marker.group(1)), ""
            except Exception:
                pass
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", merged)
        if not nums:
            return None, "HDR-VDP3 command returned no parsable score."
        try:
            # Use the last numeric token to avoid banners/version numbers.
            return float(nums[-1]), ""
        except Exception:
            return None, "HDR-VDP3 parse failed."
