"""
scripts/check_c960_webcam_live.py
================================
Live host-side diagnostic utility for EMEET SmartCam C960 Ultra (or any local USB camera)
against the active GDUT-HWD 5-Class deploy checkpoint (`blue, white, yellow, red, none`).

Reports:
  - Connected Windows PnP Camera devices
  - Actual opened DSHOW resolution / FPS and measured end-to-end latency
  - Per-class detections (`blue, white, yellow, red, none`) and `none` confidence spread
  - Clear troubleshooting guidance if EMEET Studio holds exclusive UVC camera access
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DEFAULT_MODEL_PATH
from src.engine.detector import SafetyObjectDetector


def list_windows_cameras() -> List[str]:
    """Queries Plug-and-Play Camera devices on Windows hosts."""
    if os.name != "nt":
        return []
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-PnpDevice -Class Camera -Status OK | Select-Object -ExpandProperty FriendlyName",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if proc.returncode == 0:
            return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    return []


def _open_camera(index: int, width: int, height: int, fps: int) -> Optional[cv2.VideoCapture]:
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


def _read_non_blank_probe(cap: cv2.VideoCapture, warmup: int = 5) -> Optional[np.ndarray]:
    last_good = None
    for _ in range(max(1, warmup)):
        ok, frame = cap.read()
        if (
            ok
            and frame is not None
            and frame.size > 0
            and float(frame.mean()) > 2.0
            and float(frame.std()) > 1.0
        ):
            last_good = frame
    return last_good


def run_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    cameras = list_windows_cameras()
    candidate_indices = [args.device_index] if args.device_index >= 0 else [0, 1, 2, 3]

    opened_idx: Optional[int] = None
    cap: Optional[cv2.VideoCapture] = None
    probe_frame: Optional[np.ndarray] = None
    selected_role: str = "Explicit Index" if args.device_index >= 0 else "Auto Probe"

    if args.device_index >= 0:
        c = _open_camera(args.device_index, args.width, args.height, args.fps)
        if c is not None:
            frame = _read_non_blank_probe(c, warmup=5)
            if frame is not None:
                cap = c
                opened_idx = args.device_index
                probe_frame = frame
            else:
                c.release()
    else:
        fallback_idx: Optional[int] = None
        preferred_1080p_idx: Optional[int] = None
        for idx in candidate_indices:
            c = _open_camera(idx, 1920, 1080, args.fps)
            if c is None:
                continue
            max_w = int(round(c.get(cv2.CAP_PROP_FRAME_WIDTH)))
            max_h = int(round(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            frame = _read_non_blank_probe(c, warmup=4)
            c.release()
            if frame is None:
                continue
            if fallback_idx is None:
                fallback_idx = idx
            if max_w >= 1900 and max_h >= 1000 and preferred_1080p_idx is None:
                preferred_1080p_idx = idx
                break

        chosen = preferred_1080p_idx if preferred_1080p_idx is not None else fallback_idx
        if chosen is not None:
            c = _open_camera(chosen, args.width, args.height, args.fps)
            if c is not None:
                frame = _read_non_blank_probe(c, warmup=5)
                if frame is not None:
                    cap = c
                    opened_idx = chosen
                    probe_frame = frame
                    selected_role = (
                        "1080p External Camera (C960 Ultra Priority)"
                        if preferred_1080p_idx is not None
                        else "720p Fallback Camera"
                    )
                else:
                    c.release()

    if cap is None or probe_frame is None:
        return {
            "status": "camera_busy_or_unavailable",
            "pnp_cameras": cameras,
            "probed_indices": candidate_indices,
            "opened_index": None,
            "message": (
                "Could not capture live non-blank frames from C960 Ultra via DirectShow. "
                "Exit EMEET Studio completely (including its Windows system tray icon) and close "
                "other camera apps before opening AEGIS-CCTV or re-running this check."
            ),
        }

    actual_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    actual_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    reported_fps = round(float(cap.get(cv2.CAP_PROP_FPS)), 2)

    detector = SafetyObjectDetector(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=0.45,
        device=(args.device.strip() or None),
        use_half=False,
        use_selective_amp=False,
        none_conf_thresh=args.none_conf,
    )
    # Warmup inference once
    detector.predict(probe_frame, conf=args.conf, none_conf=args.none_conf, imgsz=640)

    class_counts: Dict[str, int] = {"blue": 0, "white": 0, "yellow": 0, "red": 0, "none": 0}
    none_confs: List[float] = []
    in_split_band = 0
    latencies_ms: List[float] = []
    captured_frames = 0

    t_loop_start = time.perf_counter()
    for _ in range(args.frames):
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            continue
        captured_frames += 1
        dets, _, lat_ms = detector.predict(frame, conf=args.conf, none_conf=args.none_conf, imgsz=640)
        latencies_ms.append(float(lat_ms))
        for d in dets:
            cname = str(d.get("name", "")).lower()
            conf_val = float(d.get("conf", 0.0))
            if cname in class_counts:
                class_counts[cname] += 1
            if detector.is_bare_head_class(cname):
                none_confs.append(conf_val)
                if args.none_conf <= conf_val < args.conf:
                    in_split_band += 1

    wall_sec = max(1e-6, time.perf_counter() - t_loop_start)
    cap.release()

    return {
        "status": "ok",
        "pnp_cameras": cameras,
        "opened_index": opened_idx,
        "camera_selection_mode": selected_role,
        "actual_resolution": f"{actual_w}x{actual_h}",
        "reported_camera_fps": reported_fps,
        "captured_frames": captured_frames,
        "measured_e2e_fps": round(captured_frames / wall_sec, 2),
        "detector_device": detector.device,
        "mean_infer_ms": round(statistics.mean(latencies_ms), 2) if latencies_ms else 0.0,
        "thresholds": {"helmet_conf": args.conf, "none_conf": args.none_conf},
        "class_detection_counts": class_counts,
        "none_confidence_stats": {
            "count": len(none_confs),
            "in_split_band_count": in_split_band,
            "min": round(min(none_confs), 4) if none_confs else None,
            "mean": round(statistics.mean(none_confs), 4) if none_confs else None,
            "max": round(max(none_confs), 4) if none_confs else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="EMEET SmartCam C960 Ultra Live Diagnostic")
    parser.add_argument("--device-index", type=int, default=-1, help="Camera index (-1 auto-probes 0..3)")
    parser.add_argument("--frames", type=int, default=30, help="Number of live frames to evaluate")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS")
    parser.add_argument("--conf", type=float, default=0.35, help="Positive helmet confidence threshold")
    parser.add_argument("--none-conf", type=float, default=0.25, help="Bare-head (none) confidence threshold")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="", help="Torch device override (e.g. cpu or cuda:0)")
    parser.add_argument("--strict", action="store_true", help="Return non-zero exit code if camera unavailable")
    args = parser.parse_args()

    result = run_diagnostic(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.strict and result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
