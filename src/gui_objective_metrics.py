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
try:
    _OBJECTIVE_HDR_METRIC_PEAK_NITS = float(
        os.environ.get("HDRTVNET_OBJECTIVE_HDR_PEAK_NITS", "1000.0")
    )
    if not np.isfinite(_OBJECTIVE_HDR_METRIC_PEAK_NITS) or _OBJECTIVE_HDR_METRIC_PEAK_NITS <= 0.0:
        raise ValueError
except Exception:
    _OBJECTIVE_HDR_METRIC_PEAK_NITS = 1000.0

_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = 2523.0 / 32.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = 2413.0 / 128.0
_PQ_C3 = 2392.0 / 128.0


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
    return (
        f'"{py}" "{bridge}" --test "{{test}}" --reference "{{reference}}" '
        '--input-encoding "{encoding}"'
    )


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
    """Match prediction global channel statistics to reference grade in linear signal space."""
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


def _grade_normalize_absolute_rgb_to_ref(
    pred_rgb_abs: np.ndarray,
    ref_rgb_abs: np.ndarray,
    *,
    peak_nits: float = _OBJECTIVE_HDR_METRIC_PEAK_NITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Match prediction to reference grade in absolute linear RGB (cd/m^2)."""
    pred = pred_rgb_abs.astype(np.float32, copy=False)
    ref = ref_rgb_abs.astype(np.float32, copy=False)
    out = pred.copy()
    peak = max(1.0, float(peak_nits))
    for c in range(3):
        p = pred[:, :, c]
        r = ref[:, :, c]
        mp = float(np.mean(p))
        mr = float(np.mean(r))
        sp = float(np.std(p))
        sr = float(np.std(r))
        gain = 1.0 if sp < 1e-6 else (sr / sp)
        bias = mr - (gain * mp)
        out[:, :, c] = (p * gain) + bias
    out = np.clip(out, 0.0, peak).astype(np.float32, copy=False)
    ref = np.clip(ref, 0.0, peak).astype(np.float32, copy=False)
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


def _linear_bgr_to_absolute_rgb(
    frame: np.ndarray,
    *,
    peak_nits: float = _OBJECTIVE_HDR_METRIC_PEAK_NITS,
) -> np.ndarray:
    rgb = _to_unit_float(frame[:, :, ::-1])
    return np.clip(rgb, 0.0, 1.0) * float(peak_nits)


def _pq_oetf_absolute(luminance_rgb: np.ndarray) -> np.ndarray:
    y = np.clip(luminance_rgb.astype(np.float32, copy=False) / 10000.0, 0.0, 1.0)
    y_m1 = np.power(y, _PQ_M1).astype(np.float32, copy=False)
    num = _PQ_C1 + (_PQ_C2 * y_m1)
    den = 1.0 + (_PQ_C3 * y_m1)
    return np.power(num / np.maximum(den, 1e-12), _PQ_M2).astype(np.float32, copy=False)


def _linear_rgb_to_itp(
    luminance_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BT.2124 Annex 1 from display-referred linear RGB (cd/m^2) to ITP."""
    rgb = luminance_rgb.astype(np.float32, copy=False)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    l = ((1688.0 * r) + (2146.0 * g) + (262.0 * b)) / 4096.0
    m = ((683.0 * r) + (2951.0 * g) + (462.0 * b)) / 4096.0
    s = ((99.0 * r) + (309.0 * g) + (3688.0 * b)) / 4096.0

    l_p = _pq_oetf_absolute(l)
    m_p = _pq_oetf_absolute(m)
    s_p = _pq_oetf_absolute(s)

    i = 0.5 * l_p + 0.5 * m_p
    ct = ((6610.0 * l_p) - (13613.0 * m_p) + (7003.0 * s_p)) / 4096.0
    cp = ((17933.0 * l_p) - (17390.0 * m_p) - (543.0 * s_p)) / 4096.0
    t = 0.5 * ct
    return i, t, cp


def _delta_e_itp_absolute_rgb(
    pred_rgb_abs: np.ndarray,
    ref_rgb_abs: np.ndarray,
) -> float:
    i1, t1, p1 = _linear_rgb_to_itp(pred_rgb_abs)
    i2, t2, p2 = _linear_rgb_to_itp(ref_rgb_abs)
    di = i1 - i2
    dt = t1 - t2
    dp = p1 - p2
    de = 720.0 * np.sqrt(di * di + dt * dt + dp * dp + 1e-12)
    return float(np.mean(de, dtype=np.float64))


def _linear_bgr_to_bt2100_pq_bgr_u16(
    frame: np.ndarray,
    *,
    peak_nits: float = _OBJECTIVE_HDR_METRIC_PEAK_NITS,
) -> np.ndarray:
    abs_rgb = _linear_bgr_to_absolute_rgb(frame, peak_nits=peak_nits)
    pq_rgb = _pq_oetf_absolute(abs_rgb)
    pq_u16 = np.clip((pq_rgb * 65535.0) + 0.5, 0.0, 65535.0).astype(np.uint16)
    return np.ascontiguousarray(pq_u16[:, :, ::-1])


def _delta_e_itp_bgr(pred_bgr: np.ndarray, ref_bgr: np.ndarray) -> float:
    pr = _linear_bgr_to_absolute_rgb(pred_bgr)
    rr = _linear_bgr_to_absolute_rgb(ref_bgr)
    return _delta_e_itp_absolute_rgb(pr, rr)


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
    pred_small = _linear_bgr_to_bt2100_pq_bgr_u16(pred_small)
    ref_small = _linear_bgr_to_bt2100_pq_bgr_u16(ref_small)
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
                encoding="bt2100-pq",
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


def _compute_full_reference_metrics(
    pred_bgr: np.ndarray,
    ref_bgr: np.ndarray,
) -> dict:
    metrics = {
        "psnr_db": None,
        "sssim": None,
        "delta_e_itp": None,
        "psnr_norm_db": None,
        "sssim_norm": None,
        "delta_e_itp_norm": None,
        "hdr_vdp3": None,
        "obj_note": "",
        "hdr_vdp3_note": "",
    }
    try:
        eval_pred, eval_ref = _prepare_metric_pair(
            pred_bgr,
            ref_bgr,
            max_side=_OBJECTIVE_METRIC_MAX_SIDE,
        )
        metrics["psnr_db"] = float(_psnr_bgr(eval_pred, eval_ref))
        metrics["sssim"] = float(_ssim_bgr(eval_pred, eval_ref))
        metrics["delta_e_itp"] = float(_delta_e_itp_bgr(eval_pred, eval_ref))

        norm_pred, norm_ref = _grade_normalize_pred_to_ref(eval_pred, eval_ref)
        metrics["psnr_norm_db"] = float(_psnr_bgr(norm_pred, norm_ref))
        metrics["sssim_norm"] = float(_ssim_bgr(norm_pred, norm_ref))
        norm_pred_abs, norm_ref_abs = _grade_normalize_absolute_rgb_to_ref(
            _linear_bgr_to_absolute_rgb(eval_pred),
            _linear_bgr_to_absolute_rgb(eval_ref),
        )
        metrics["delta_e_itp_norm"] = float(
            _delta_e_itp_absolute_rgb(norm_pred_abs, norm_ref_abs)
        )
        metrics["obj_note"] = (
            "Computed (raw + normalized; PSNR/SSIM linear, "
            "DeltaEITP/HDR-VDP3 color-managed; DeltaEITP-N normalized pre-PQ)"
        )
    except Exception as exc:
        metrics["obj_note"] = f"Metric error: {exc}"

    try:
        vdp_score, vdp_note = _hdrvdp3_cli_score(pred_bgr, ref_bgr)
        if vdp_score is not None:
            metrics["hdr_vdp3"] = float(vdp_score)
        elif vdp_note:
            metrics["hdr_vdp3_note"] = str(vdp_note)
    except Exception as exc:
        metrics["hdr_vdp3_note"] = f"HDR-VDP3 error: {exc}"

    return metrics
