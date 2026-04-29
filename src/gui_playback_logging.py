from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from datetime import datetime

from gui_config import SOURCE_MODE_WINDOW, _normalize_source_mode


class PlaybackLoggingMixin:
    """Session logging helpers for playback metrics and compare requests."""

    def _reset_playback_logging_buffers(self):
        self._playback_log_compare_events = []
        self._playback_log_pending_compare_index = None
        self._playback_log_runtime_samples = []
        self._playback_log_started_wall_time = ""
        self._playback_log_started_perf = 0.0

    @staticmethod
    def _normalize_playback_log_value(value):
        if value is None:
            return None
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return float(value)
        if isinstance(value, str):
            return str(value)
        return str(value)

    def _normalize_playback_log_dict(self, payload: dict | None) -> dict:
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): self._normalize_playback_log_value(value)
            for key, value in payload.items()
        }

    @staticmethod
    def _playback_log_stats(samples: list[dict], key: str) -> dict | None:
        vals = []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            value = sample.get(key)
            try:
                num = float(value)
            except Exception:
                continue
            if not math.isfinite(num):
                continue
            vals.append(num)
        if not vals:
            return None
        return {
            "count": len(vals),
            "avg": sum(vals) / float(len(vals)),
            "min": min(vals),
            "max": max(vals),
            "last": vals[-1],
        }

    @staticmethod
    def _format_playback_log_stats(label: str, stats: dict | None, suffix: str = "") -> str:
        if not stats:
            return f"{label}: n/a"
        return (
            f"{label}: avg {stats['avg']:.3f}{suffix}, "
            f"min {stats['min']:.3f}{suffix}, "
            f"max {stats['max']:.3f}{suffix}, "
            f"last {stats['last']:.3f}{suffix}"
        )

    @staticmethod
    def _sanitize_playback_log_slug(text: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
        slug = slug.strip("._-")
        if not slug:
            slug = "session"
        return slug[:48]

    def _capture_playback_log_settings(self) -> dict:
        return {
            "source_mode": _normalize_source_mode(getattr(self, "_source_mode", None)),
            "precision": str(
                getattr(self, "_active_precision", "")
                or self._cmb_prec.currentText()
                or ""
            ).strip(),
            "resolution": str(self._cmb_res.currentText() or "").strip()
            if hasattr(self, "_cmb_res") and self._cmb_res is not None
            else "",
            "upscale_mode": str(self._cmb_upscale.currentText() or "").strip()
            if hasattr(self, "_cmb_upscale") and self._cmb_upscale is not None
            else "",
            "use_hg": bool(self._chk_hg.isChecked())
            if hasattr(self, "_chk_hg") and self._chk_hg is not None
            else False,
            "film_grain": bool(self._chk_film_grain.isChecked())
            if hasattr(self, "_chk_film_grain") and self._chk_film_grain is not None
            else False,
            "runtime_execution_mode": str(
                getattr(self, "_runtime_execution_mode", "") or ""
            ).strip(),
            "predequantize_mode": str(
                getattr(self, "_predequantize_mode", "") or ""
            ).strip(),
            "objective_metrics_enabled": bool(
                getattr(self, "_objective_metrics_enabled", False)
            ),
            "hdr_ground_truth_path": str(
                getattr(self, "_hdr_ground_truth_path", "") or ""
            ).strip()
            or None,
        }

    def _start_playback_logging_session(self):
        self._playback_log_session_started = True
        self._playback_log_source_label = self._current_source_label_for_logging()
        self._reset_playback_logging_buffers()
        self._playback_log_started_wall_time = (
            datetime.now().astimezone().isoformat(timespec="seconds")
        )
        self._playback_log_started_perf = time.perf_counter()
        try:
            self._worker.start_session_logging()
        except Exception:
            pass
        self._update_playback_log_button()

    def _update_playback_log_button(self):
        btn = getattr(self, "_btn_log", None)
        if btn is None:
            return
        active = bool(getattr(self, "_playback_log_enabled", False))
        btn.blockSignals(True)
        btn.setChecked(active)
        btn.setText("Logging On" if active else "Log Session")
        btn.blockSignals(False)
        btn.setEnabled(active or self._current_source_available())

    def _current_source_label_for_logging(self) -> str:
        mode = _normalize_source_mode(getattr(self, "_source_mode", None))
        if mode == SOURCE_MODE_WINDOW:
            target = getattr(self, "_capture_target", None)
            return str(getattr(target, "label", "") or "").strip()
        path = str(getattr(self, "_video_path", "") or "").strip()
        if not path:
            return ""
        return os.path.basename(path)

    def _toggle_playback_logging(self, enabled: bool):
        if enabled:
            self._playback_log_enabled = True
            self._playback_log_source_label = self._current_source_label_for_logging()
            self._playback_log_session_started = False
            self._reset_playback_logging_buffers()
            if bool(getattr(self, "_playing", False)):
                self._start_playback_logging_session()
            self._update_playback_log_button()
            self.statusBar().showMessage(
                "Playback logging armed. Full runtime and compare metrics will be saved when the session ends.",
                6000,
            )
            return

        printed = self._finalize_playback_logging("logging turned off")
        if not printed:
            self.statusBar().showMessage("Playback logging disabled.", 4000)

    def _prepare_playback_logging_for_start(self):
        if not bool(getattr(self, "_playback_log_enabled", False)):
            return
        self._start_playback_logging_session()

    def _record_runtime_metrics_for_logging(self, payload: dict):
        if not bool(getattr(self, "_playback_log_enabled", False)):
            return
        if not bool(getattr(self, "_playback_log_session_started", False)):
            return
        sample = self._normalize_playback_log_dict(payload)
        now_local = datetime.now().astimezone().isoformat(timespec="seconds")
        start_perf = float(getattr(self, "_playback_log_started_perf", 0.0) or 0.0)
        sample["logged_at_local"] = now_local
        sample["elapsed_s"] = (
            round(max(0.0, time.perf_counter() - start_perf), 3)
            if start_perf > 0.0
            else 0.0
        )
        self._playback_log_runtime_samples.append(sample)

    def _note_compare_request_for_logging(
        self,
        *,
        precision_key: str | None,
        frame_number: int | None,
    ):
        if not bool(getattr(self, "_playback_log_enabled", False)):
            return
        requested_frame = frame_number
        if requested_frame is None:
            requested_frame = getattr(
                self,
                "_audio_sync_frame_hint",
                getattr(self, "_last_seek_frame", 0),
            )
        try:
            requested_frame = max(0, int(requested_frame))
        except Exception:
            requested_frame = 0
        precision = str(precision_key or "").strip()
        if not precision:
            precision = str(
                getattr(self, "_active_precision", "")
                or self._cmb_prec.currentText()
                or ""
            ).strip()
        self._playback_log_compare_events.append(
            {
                "requested_frame": requested_frame,
                "resolved_frame": None,
                "precision": precision or "unknown",
            }
        )
        self._playback_log_pending_compare_index = (
            len(self._playback_log_compare_events) - 1
        )

    def _resolve_compare_request_for_logging(
        self,
        *,
        frame_number: int,
        precision_key: str | None = None,
        note: str | None = None,
        metrics: dict | None = None,
    ):
        if not bool(getattr(self, "_playback_log_enabled", False)):
            return
        idx = getattr(self, "_playback_log_pending_compare_index", None)
        precision = str(precision_key or "").strip() or "unknown"
        event = {
            "requested_frame": int(frame_number),
            "resolved_frame": int(frame_number),
            "precision": precision,
            "note": str(note or "").strip(),
            "metrics": self._normalize_playback_log_dict(metrics),
        }
        if (
            idx is not None
            and 0 <= int(idx) < len(getattr(self, "_playback_log_compare_events", []))
        ):
            event = self._playback_log_compare_events[int(idx)]
            event["resolved_frame"] = int(frame_number)
            if precision:
                event["precision"] = precision
            event["note"] = str(note or "").strip()
            event["metrics"] = self._normalize_playback_log_dict(metrics)
        else:
            self._playback_log_compare_events.append(event)
        self._playback_log_pending_compare_index = None

    def _build_playback_log_session_dir(self, source_label: str) -> str:
        root_dir = str(getattr(self, "_playback_log_dir", "") or "").strip()
        if not root_dir:
            root_dir = os.path.join(os.getcwd(), "logs", "playback_sessions")
        os.makedirs(root_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = self._sanitize_playback_log_slug(source_label)
        base = os.path.join(root_dir, f"{stamp}_{slug}")
        session_dir = base
        suffix = 2
        while os.path.exists(session_dir):
            session_dir = f"{base}_{suffix}"
            suffix += 1
        os.makedirs(session_dir, exist_ok=False)
        return session_dir

    def _write_playback_runtime_csv(self, path: str, samples: list[dict]):
        if not samples:
            return
        preferred = [
            "elapsed_s",
            "logged_at_local",
            "fps",
            "latency_ms",
            "model_latency_ms",
            "live_video_latency_ms",
            "frame",
            "cpu_mb",
            "gpu_mb",
            "model_mb",
            "model_size_label",
            "precision",
            "proc_res",
            "psnr_db",
            "sssim",
            "delta_e_itp",
            "hdr_vdp3",
            "objective_enabled",
            "objective_note",
            "hdr_vdp3_note",
            "is_live_capture",
        ]
        extras = []
        seen = set(preferred)
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            for key in sample.keys():
                if key in seen:
                    continue
                seen.add(key)
                extras.append(key)
        fieldnames = preferred + sorted(extras)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for sample in samples:
                writer.writerow(
                    {key: sample.get(key) for key in fieldnames}
                )

    def _write_playback_compare_csv(self, path: str, events: list[dict]):
        if not events:
            return
        fieldnames = [
            "index",
            "requested_frame",
            "resolved_frame",
            "precision",
            "note",
            "psnr_db",
            "sssim",
            "delta_e_itp",
            "psnr_norm_db",
            "sssim_norm",
            "delta_e_itp_norm",
            "hdr_vdp3",
            "obj_note",
            "hdr_vdp3_note",
        ]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for idx, event in enumerate(events, start=1):
                metrics = event.get("metrics") if isinstance(event, dict) else {}
                metrics = metrics if isinstance(metrics, dict) else {}
                writer.writerow(
                    {
                        "index": idx,
                        "requested_frame": event.get("requested_frame"),
                        "resolved_frame": event.get("resolved_frame"),
                        "precision": event.get("precision"),
                        "note": event.get("note"),
                        "psnr_db": metrics.get("psnr_db"),
                        "sssim": metrics.get("sssim"),
                        "delta_e_itp": metrics.get("delta_e_itp"),
                        "psnr_norm_db": metrics.get("psnr_norm_db"),
                        "sssim_norm": metrics.get("sssim_norm"),
                        "delta_e_itp_norm": metrics.get("delta_e_itp_norm"),
                        "hdr_vdp3": metrics.get("hdr_vdp3"),
                        "obj_note": metrics.get("obj_note"),
                        "hdr_vdp3_note": metrics.get("hdr_vdp3_note"),
                    }
                )

    def _build_playback_log_text(
        self,
        *,
        reason: str,
        source_label: str,
        settings: dict,
        runtime_samples: list[dict],
        compare_events: list[dict],
        worker_summary: dict,
        runtime_summary: dict,
        session_dir: str,
        started_at: str,
        ended_at: str,
    ) -> str:
        lines = [
            "HDRTVNet++ Playback Log",
            f"Reason: {reason}",
            f"Saved To: {session_dir}",
            f"Started: {started_at or 'n/a'}",
            f"Ended: {ended_at or 'n/a'}",
            f"Source: {source_label or 'n/a'}",
            f"Source Mode: {settings.get('source_mode', '') or 'n/a'}",
            "",
            "Settings:",
            f"  Precision: {settings.get('precision') or 'n/a'}",
            f"  Resolution: {settings.get('resolution') or 'n/a'}",
            f"  Upscale: {settings.get('upscale_mode') or 'n/a'}",
            f"  Use HG: {settings.get('use_hg')}",
            f"  Film Grain: {settings.get('film_grain')}",
            f"  Runtime Mode: {settings.get('runtime_execution_mode') or 'n/a'}",
            f"  Predequantize: {settings.get('predequantize_mode') or 'n/a'}",
            f"  HDR GT: {settings.get('hdr_ground_truth_path') or 'none'}",
            "",
            "Runtime Metrics:",
            f"  Samples Saved: {len(runtime_samples)}",
            "  "
            + self._format_playback_log_stats(
                "FPS", runtime_summary.get("fps"), ""
            ),
            "  "
            + self._format_playback_log_stats(
                "Latency", runtime_summary.get("latency_ms"), " ms"
            ),
            "  "
            + self._format_playback_log_stats(
                "Inference Latency (sampled UI)",
                runtime_summary.get("model_latency_ms"),
                " ms",
            ),
            "  "
            + self._format_playback_log_stats(
                "GPU Memory", runtime_summary.get("gpu_mb"), " MB"
            ),
            "  "
            + self._format_playback_log_stats(
                "CPU Memory", runtime_summary.get("cpu_mb"), " MB"
            ),
        ]
        model_size_stats = runtime_summary.get("model_mb")
        if model_size_stats:
            size_label = str(
                runtime_summary.get("model_size_label") or "Model Artifact"
            ).strip()
            for sample in reversed(runtime_samples):
                if not isinstance(sample, dict):
                    continue
                label = str(sample.get("model_size_label") or "").strip()
                if label:
                    size_label = label
                    break
            lines.append(
                "  "
                + self._format_playback_log_stats(
                    f"{size_label} Size",
                    model_size_stats,
                    " MB",
                )
            )
        live_latency_stats = runtime_summary.get("live_video_latency_ms")
        if live_latency_stats:
            lines.append(
                "  "
                + self._format_playback_log_stats(
                    "Live Video Latency", live_latency_stats, " ms"
                )
            )
        for key, label, suffix in (
            ("psnr_db", "PSNR", " dB"),
            ("sssim", "SSIM", ""),
            ("delta_e_itp", "DeltaE ITP", ""),
            ("hdr_vdp3", "HDR-VDP3", ""),
        ):
            stats = runtime_summary.get(key)
            if stats:
                lines.append(
                    "  " + self._format_playback_log_stats(label, stats, suffix)
                )
        exact_infer_avg = worker_summary.get("avg_model_latency_ms")
        exact_infer_count = int(worker_summary.get("model_latency_samples", 0) or 0)
        if exact_infer_avg is None:
            lines.append("  Exact Inference Average: n/a")
        else:
            lines.append(
                "  Exact Inference Average: "
                f"{float(exact_infer_avg):.3f} ms over {exact_infer_count} frames"
            )

        lines.extend(
            [
                "",
                f"Compare Events: {len(compare_events)}",
            ]
        )
        if not compare_events:
            lines.append("  None")
        else:
            for idx, event in enumerate(compare_events, start=1):
                metrics = event.get("metrics") if isinstance(event, dict) else {}
                metrics = metrics if isinstance(metrics, dict) else {}
                frame_number = event.get("resolved_frame")
                if frame_number is None:
                    frame_number = event.get("requested_frame")
                lines.append(
                    f"  {idx}. Frame {frame_number if frame_number is not None else 'unknown'} "
                    f"({event.get('precision') or 'unknown'})"
                )
                if event.get("note"):
                    lines.append(f"     Note: {event.get('note')}")
                for metric_key, label, suffix in (
                    ("psnr_db", "PSNR", " dB"),
                    ("sssim", "SSIM", ""),
                    ("delta_e_itp", "DeltaEITP", ""),
                    ("psnr_norm_db", "PSNR-N", " dB"),
                    ("sssim_norm", "SSIM-N", ""),
                    ("delta_e_itp_norm", "DeltaEITP-N", ""),
                    ("hdr_vdp3", "HDR-VDP3", ""),
                ):
                    value = metrics.get(metric_key)
                    if value is None:
                        continue
                    try:
                        value_text = f"{float(value):.4f}{suffix}"
                    except Exception:
                        value_text = str(value)
                    lines.append(f"     {label}: {value_text}")
                obj_note = str(metrics.get("obj_note", "") or "").strip()
                hdr_vdp3_note = str(metrics.get("hdr_vdp3_note", "") or "").strip()
                if obj_note:
                    lines.append(f"     Objective Note: {obj_note}")
                if hdr_vdp3_note:
                    lines.append(f"     HDR-VDP3 Note: {hdr_vdp3_note}")
        lines.append("")
        return "\n".join(lines)

    def _finalize_playback_logging(self, reason: str) -> bool:
        if not bool(getattr(self, "_playback_log_enabled", False)):
            return False

        try:
            summary = self._worker.consume_session_logging_summary()
        except Exception:
            summary = {}

        compare_events = list(getattr(self, "_playback_log_compare_events", []))
        runtime_samples = list(getattr(self, "_playback_log_runtime_samples", []))
        source_label = str(
            getattr(self, "_playback_log_source_label", "") or ""
        ).strip() or self._current_source_label_for_logging()
        sample_count = int(summary.get("model_latency_samples", 0) or 0)
        has_content = (
            bool(runtime_samples)
            or bool(compare_events)
            or sample_count > 0
            or bool(getattr(self, "_playback_log_session_started", False))
        )
        started_at = str(getattr(self, "_playback_log_started_wall_time", "") or "").strip()
        started_perf = float(getattr(self, "_playback_log_started_perf", 0.0) or 0.0)
        ended_at = datetime.now().astimezone().isoformat(timespec="seconds")
        settings = self._capture_playback_log_settings()
        runtime_summary = {
            key: self._playback_log_stats(runtime_samples, key)
            for key in (
                "fps",
                "latency_ms",
                "model_latency_ms",
                "live_video_latency_ms",
                "gpu_mb",
                "cpu_mb",
                "model_mb",
                "psnr_db",
                "sssim",
                "delta_e_itp",
                "hdr_vdp3",
            )
        }
        for sample in reversed(runtime_samples):
            if not isinstance(sample, dict):
                continue
            label = str(sample.get("model_size_label") or "").strip()
            if label:
                runtime_summary["model_size_label"] = label
                break

        self._playback_log_enabled = False
        self._playback_log_session_started = False
        self._playback_log_source_label = ""
        self._reset_playback_logging_buffers()
        self._update_playback_log_button()

        if not has_content:
            return False

        session_dir = self._build_playback_log_session_dir(source_label or "session")
        text_report = self._build_playback_log_text(
            reason=reason,
            source_label=source_label,
            settings=settings,
            runtime_samples=runtime_samples,
            compare_events=compare_events,
            worker_summary=summary,
            runtime_summary=runtime_summary,
            session_dir=session_dir,
            started_at=started_at,
            ended_at=ended_at,
        )

        session_payload = {
            "reason": reason,
            "saved_at_local": ended_at,
            "started_at_local": started_at or None,
            "session_elapsed_s": (
                round(
                    max(
                        0.0,
                        time.perf_counter() - started_perf,
                    ),
                    3,
                )
                if started_perf > 0.0
                else 0.0
            ),
            "source_label": source_label or None,
            "settings": settings,
            "worker_summary": summary,
            "runtime_metric_summary": runtime_summary,
            "runtime_metrics": runtime_samples,
            "compare_events": compare_events,
            "files": {
                "summary_txt": "summary.txt",
                "session_json": "session.json",
                "runtime_metrics_csv": "runtime_metrics.csv",
                "compare_events_csv": (
                    "compare_events.csv" if compare_events else None
                ),
            },
        }

        with open(os.path.join(session_dir, "session.json"), "w", encoding="utf-8") as handle:
            json.dump(session_payload, handle, indent=2)
        with open(os.path.join(session_dir, "summary.txt"), "w", encoding="utf-8") as handle:
            handle.write(text_report)
        self._write_playback_runtime_csv(
            os.path.join(session_dir, "runtime_metrics.csv"),
            runtime_samples,
        )
        if compare_events:
            self._write_playback_compare_csv(
                os.path.join(session_dir, "compare_events.csv"),
                compare_events,
            )

        print()
        print("=== HDRTVNet++ Playback Log Saved ===")
        print(f"Reason: {reason}")
        print(f"Folder: {session_dir}")
        print(f"Runtime samples: {len(runtime_samples)}")
        print(f"Compare events: {len(compare_events)}")
        if summary.get("avg_model_latency_ms") is not None:
            print(
                "Exact inference average: "
                f"{float(summary['avg_model_latency_ms']):.3f} ms "
                f"over {sample_count} frames"
            )
        print()

        try:
            self.statusBar().showMessage(
                f"Playback log saved to {session_dir}",
                8000,
            )
        except Exception:
            pass
        return True
