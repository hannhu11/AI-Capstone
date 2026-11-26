"""
src/engine/pipeline.py
======================
CCTV Simulation Pipeline Engine (GDUT-HWD 5-Class Enterprise VMS v2.1).
Features:
- Selective CUDA AMP FP16 inference (FP32 guards inside CoordConv/BiFormerBlockLite).
- Head-Level ByteTrack + Anti-Wall-Bleed Hungarian Upper-Torso Re-ID (`gdut_reid.py`).
- Lock-free Telemetry & Registry Snapshot Cache (`_read_lock`) eliminating GPU lock contention.
- High-speed Metadata-only Binary WebSocket support for C960 Ultra webcam.
- Video Transport Controls, Role-by-Color Hierarchy, Multi-Filter, Solo Focus & CSV Export.
"""

from __future__ import annotations
import csv
import io
import os
import cv2
import queue
import time
import copy
import torch
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.config import (
    config,
    load_role_policy,
    save_role_policy,
)
from src.engine.detector import SafetyObjectDetector
from src.engine.bytetrack import BYTETracker
from src.engine.spatial_safety import SpatialSafetyEngine
from src.alerting.alert_manager import alert_manager
from src.engine.reid_tracker import reid_registry
from src.engine.gdut_reid import UpperTorsoReIDGallery


CLASS_COLORS = {
    "person": (180, 180, 180),
    "safety_vest": (21, 128, 61),
    "blue_helmet": (129, 76, 15),
    "green_helmet": (21, 128, 61),
    "red_helmet": (27, 27, 153),
    "white_helmet": (105, 85, 71),
    "yellow_helmet": (9, 83, 180),
    "blue": (129, 76, 15),
    "green": (21, 128, 61),
    "red": (27, 27, 153),
    "white": (105, 85, 71),
    "yellow": (9, 83, 180),
    "none": (38, 38, 220),
}

_RESOLUTION_HEIGHTS = {
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
}


def compute_box_intersection_ratio(
    box_a: List[float] | np.ndarray,
    box_b: List[float] | np.ndarray,
    use_upper_torso: bool = True,
    upper_torso_ratio: float = 0.60,
) -> float:
    r"""
    Computes fraction of box_a's area that overlaps with box_b (or box_b's upper torso).
    Overlap ratio = |box_a \cap box_b| / |box_a|
    """
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    aw = max(0.0, float(ax2 - ax1))
    ah = max(0.0, float(ay2 - ay1))
    a_area = aw * ah
    if a_area <= 1e-7:
        return 0.0

    if use_upper_torso:
        by2_eff = float(by1) + max(0.0, float(by2 - by1)) * upper_torso_ratio
    else:
        by2_eff = float(by2)

    ix1 = max(float(ax1), float(bx1))
    iy1 = max(float(ay1), float(by1))
    ix2 = min(float(ax2), float(bx2))
    iy2 = min(float(ay2), by2_eff)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    return (inter_w * inter_h) / a_area


def _apply_target_resolution(frame: np.ndarray, target_resolution: str) -> np.ndarray:
    """Resizes frame to target vertical resolution (480p/720p/1080p) or caps >1920px UHD when original."""
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    res_key = str(target_resolution or "original").lower().strip()
    if res_key in _RESOLUTION_HEIGHTS:
        target_h = _RESOLUTION_HEIGHTS[res_key]
        if h != target_h and h > 0:
            scale = float(target_h) / float(h)
            target_w = max(16, int(round(w * scale)))
            if target_w % 2 != 0:
                target_w += 1
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            return cv2.resize(frame, (target_w, target_h), interpolation=interp)
        return frame

    max_dim = max(h, w)
    if max_dim > 1920:
        scale = 1920.0 / float(max_dim)
        return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return frame


class CCTVSurveillancePipeline:
    """Thread-safe Enterprise CCTV Surveillance, Tracking, Re-ID & Video Control Pipeline."""

    def __init__(self, video_source: Optional[str] = None):
        self.source_path = video_source or config.video_source
        self.detector = SafetyObjectDetector(
            model_path=config.model_path,
            conf_thresh=config.conf_threshold,
            iou_thresh=config.iou_threshold,
            device=config.device,
            use_half=config.use_half,
            use_selective_amp=getattr(config, "use_selective_amp", False),
        )
        self.tracker = BYTETracker(
            track_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            new_track_thresh=config.new_track_thresh,
            track_buffer=config.track_buffer,
            match_thresh=config.match_thresh,
            grace_period=getattr(config, "track_grace_period", 2),
        )
        self.safety_engine = SpatialSafetyEngine(
            head_box_ratio=config.head_box_ratio,
            torso_min_ratio=config.torso_min_ratio,
            torso_max_ratio=config.torso_max_ratio,
            helmet_overlap_thresh=config.helmet_overlap_thresh,
            vest_overlap_thresh=config.vest_overlap_thresh,
            consecutive_frames_thresh=config.consecutive_violation_frames,
            cooldown_seconds=config.alert_cooldown_seconds,
            min_tracklet_len=getattr(config, "min_tracklet_len", 15),
        )

        # Role-by-Color Hierarchy & GDUT Head-Anchored Upper-Torso Re-ID Gallery
        self.role_policy: Dict[str, Dict[str, Any]] = load_role_policy()
        self.gdut_reid = UpperTorsoReIDGallery(
            ttl_seconds=getattr(config, "gdut_reid_ttl_sec", 600.0),
            sim_thresh=getattr(config, "gdut_reid_sim_thresh", 0.70),
            use_deep=getattr(config, "gdut_reid_use_deep", True),
            deep_refresh_interval=30,
            min_confirm_hits=6,
            min_confirm_frames=24,
            device=self.detector.device,
        )

        # Legacy compatibility fields
        self.stationary_furniture_tracker: Dict[int, Dict[str, Any]] = {}
        self.confirmed_furniture_zones: List[List[float]] = []
        self.stationary_next_id: int = 1
        reid_registry.reset()

        # Focus & Multi-Dimensional Follow / Filter State
        self.focus_track_id: Optional[int] = None
        self.filter_colors: List[str] = []
        self.filter_roles: List[str] = []
        self.violators_only: bool = False

        # Video Transport Controls State
        self.is_paused: bool = False
        self.playback_speed: float = 1.0
        self.target_resolution: str = "original"
        self.loop_enabled: bool = True
        self.current_frame_index: int = 0
        self.total_frames: int = 0
        self.source_type: str = "video"
        self._pending_seek_frame: Optional[int] = None
        self._last_raw_frame: Optional[np.ndarray] = None
        self._last_worker_reports: List[Dict[str, Any]] = []
        self._last_detections: List[Dict[str, Any]] = []

        alert_manager.clear_alerts()

        self.cap: Optional[cv2.VideoCapture] = None
        self.stream_fps: float = 30.0
        self.frame_stride: int = 1
        self.frame_width: int = 1920
        self.frame_height: int = 1080
        self.latest_active_track_ids: List[int] = []
        self.latest_active_tracks: List[Dict[str, Any]] = []
        self._init_video_capture()

        # Telemetry state (EMA alpha = 0.25)
        self.fps_rolling = 0.0
        self.inference_latency_ms = 0.0
        self.total_latency_ms = 0.0
        self.workers_count_ema = 0.0
        self.compliance_rate_ema = 100.0
        self.active_workers_count = 0
        self.compliant_workers_count = 0
        self.active_violations_count = 0
        self.compliance_rate = 100.0
        self.helmet_color_distribution: Dict[str, int] = {
            "blue": 0, "white": 0, "yellow": 0, "red": 0, "none": 0, "green": 0
        }
        self.role_distribution: Dict[str, int] = {}
        self.total_frames_processed = 0

        self.lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._cached_telemetry: Dict[str, Any] = {}
        self._cached_workers: List[Dict[str, Any]] = []

        self.is_running = True
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_encoded_jpeg: Optional[bytes] = None
        self.frame_idx = 0
        self.frame_condition = threading.Condition()
        self._encode_queue: queue.Queue = queue.Queue(maxsize=1)

        self._refresh_read_caches_unlocked()
        alert_manager.clear_alerts()

        # Pre-warm deep Re-ID extractor outside self.lock so frame-1 never blocks GPU loop
        threading.Thread(target=self._prewarm_reid_background, daemon=True).start()
        self._encode_thread = threading.Thread(target=self._jpeg_encode_worker, daemon=True)
        self._encode_thread.start()
        self.worker_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.worker_thread.start()

    def _prewarm_reid_background(self):
        try:
            self.gdut_reid.prewarm()
        except Exception:
            pass

    def _refresh_read_caches_unlocked(self):
        """Updates lightweight lock-free read snapshots without blocking HTTP readers on GPU inference."""
        r_lock = getattr(self, "_read_lock", None)
        if r_lock is None:
            return
        tele = self._build_telemetry_unlocked()
        workers = list(tele.get("workers", []))
        with r_lock:
            self._cached_telemetry = tele
            self._cached_workers = workers

    def _detect_source_type(self, src_str: str) -> str:
        s = src_str.strip()
        if s == "browser_webcam":
            return "browser_webcam"
        if s.startswith("webcam:") or s.isdigit():
            return "webcam"
        if s.startswith("rtsp://") or s.startswith("http://") or s.startswith("https://"):
            return "rtsp"
        return "video"

    def _open_capture_device(self, src_str: str) -> Optional[cv2.VideoCapture]:
        if src_str == "browser_webcam":
            return None
        if src_str.startswith("webcam:") or src_str.isdigit():
            idx_part = src_str.split(":", 1)[1] if ":" in src_str else src_str
            try:
                cam_idx = int(idx_part)
            except ValueError:
                cam_idx = 0
            if os.name == "nt":
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                if cap is not None and cap.isOpened():
                    return cap
                if cap is not None:
                    cap.release()
            return cv2.VideoCapture(cam_idx)
        return cv2.VideoCapture(src_str)

    def _init_video_capture(self):
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None

        src_str = str(self.source_path).strip()
        self.source_type = self._detect_source_type(src_str)
        self.current_frame_index = 0
        self.total_frames = 0

        if self.source_type == "browser_webcam":
            self.stream_fps = 25.0
            self.frame_stride = 1
            self.frame_width = 1280
            self.frame_height = 720
            return

        self.cap = self._open_capture_device(src_str)
        if self.cap is None or not self.cap.isOpened():
            print(f"⚠️ Warning: Could not open video source: {src_str}")
            self.stream_fps = 30.0
            self.frame_stride = 1
            self.frame_width = 1920
            self.frame_height = 1080
        else:
            if self.source_type == "webcam":
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if str(getattr(self, "target_resolution", "original")) in _RESOLUTION_HEIGHTS:
                        th = _RESOLUTION_HEIGHTS[str(self.target_resolution)]
                        tw = {480: 640, 720: 1280, 1080: 1920}.get(th, 1280)
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, tw)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, th)
                except Exception:
                    pass

            src_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.stream_fps = src_fps if (src_fps and src_fps > 0) else 30.0
            src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_width = src_w if src_w > 0 else 1920
            self.frame_height = src_h if src_h > 0 else 1080
            fc = int(max(0.0, float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)))
            self.total_frames = fc if self.source_type == "video" else 0
            self.frame_stride = 2 if self.stream_fps >= 50.0 else 1
            print(
                f"📹 Opened {self.source_type.upper()} Source: {src_str} "
                f"({self.frame_width}x{self.frame_height} @ {self.stream_fps:.1f} FPS, Frames={self.total_frames})"
            )

    def _reset_session_metrics(self, clear_focus: bool = True):
        self.tracker.reset()
        self.safety_engine.reset()
        if hasattr(self, "stationary_furniture_tracker"):
            self.stationary_furniture_tracker.clear()
        if hasattr(self, "confirmed_furniture_zones"):
            self.confirmed_furniture_zones.clear()
        self.stationary_next_id = 1
        if clear_focus:
            self.focus_track_id = None
        if hasattr(self, "gdut_reid") and self.gdut_reid is not None:
            self.gdut_reid.reset()
        reid_registry.reset()
        self.active_workers_count = 0
        self.workers_count_ema = 0.0
        self.compliant_workers_count = 0
        self.active_violations_count = 0
        self.compliance_rate = 100.0
        self.compliance_rate_ema = 100.0
        self.total_frames_processed = 0
        self.fps_rolling = 0.0
        self.helmet_color_distribution = {"blue": 0, "white": 0, "yellow": 0, "red": 0, "none": 0, "green": 0}
        self.role_distribution = {}
        self.latest_active_track_ids = []
        self.latest_active_tracks = []
        self._last_raw_frame = None
        self._last_worker_reports = []
        self._last_detections = []
        self._pending_seek_frame = None
        self.is_paused = False
        self.latest_encoded_jpeg = None
        alert_manager.clear_alerts()

    def switch_source(self, new_source: str):
        """Switches video/webcam source dynamically with a clean slate."""
        with self.lock:
            self.source_path = new_source
            config.video_source = new_source
            self._reset_session_metrics(clear_focus=True)
            self._init_video_capture()
            self._refresh_read_caches_unlocked()
            print(f"📹 Switched Source to: {new_source} (Mode={getattr(self, 'source_type', 'video')})")

    def switch_model(self, new_model_path: str):
        """Hot-swaps detector weights dynamically, resetting trackers and alerts."""
        with self.lock:
            self.detector.load_model(new_model_path)
            config.model_path = new_model_path
            self._reset_session_metrics(clear_focus=True)
            self._refresh_read_caches_unlocked()
            print(f"🔄 Switched Model Weights to: {new_model_path} (Clean Slate Active)")

    def reset_tracking_registry(self):
        with self.lock:
            self.tracker.reset()
            self.focus_track_id = None
            if hasattr(self, "gdut_reid") and self.gdut_reid is not None:
                self.gdut_reid.reset()
            reid_registry.reset()
            self.latest_active_track_ids = []
            self.latest_active_tracks = []
            self._refresh_read_caches_unlocked()

    # ------------------------------------------------------------------
    # Video Transport Controls
    # ------------------------------------------------------------------
    def set_paused(self, paused: Optional[bool] = None) -> bool:
        with self.lock:
            self.is_paused = (not self.is_paused) if paused is None else bool(paused)
            self._refresh_read_caches_unlocked()
            return self.is_paused

    def seek_to(
        self,
        frame: Optional[int] = None,
        seconds: Optional[float] = None,
        delta_seconds: Optional[float] = None,
        percent: Optional[float] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            fps = max(1.0, float(getattr(self, "stream_fps", 30.0)))
            total = int(getattr(self, "total_frames", 0))
            cur = int(getattr(self, "current_frame_index", 0))

            if frame is not None:
                target = int(frame)
            elif seconds is not None:
                target = int(round(float(seconds) * fps))
            elif delta_seconds is not None:
                target = int(round(cur + float(delta_seconds) * fps))
            elif percent is not None:
                pct = max(0.0, min(100.0, float(percent)))
                target = int(round((pct / 100.0) * max(0, total - 1))) if total > 0 else 0
            else:
                target = cur

            target = max(0, min(total - 1, target)) if total > 0 else max(0, target)
            self._pending_seek_frame = target
            self.current_frame_index = target
            self.tracker.reset()
            self.safety_engine.reset()
            if hasattr(self, "gdut_reid") and self.gdut_reid is not None:
                self.gdut_reid.on_video_loop()
            reid_registry.on_video_loop()

            if self.cap is not None and self.cap.isOpened() and self.source_type == "video":
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                self._pending_seek_frame = None
                if self.is_paused:
                    ret, frm = self.cap.read()
                    if ret and frm is not None:
                        self._last_raw_frame = _apply_target_resolution(frm, self.target_resolution)
                        annotated = self._process_frame_unlocked(self._last_raw_frame, config.camera_id)
                        self._publish_encoded_frame(annotated)

            self._refresh_read_caches_unlocked()
            return self._build_playback_state_unlocked()

    def set_speed(self, speed: float) -> float:
        with self.lock:
            val = max(0.25, min(4.0, float(speed)))
            self.playback_speed = round(val, 2)
            self._refresh_read_caches_unlocked()
            return self.playback_speed

    def set_resolution(self, resolution: str) -> str:
        with self.lock:
            key = str(resolution or "original").lower().strip()
            if key not in ("480p", "720p", "1080p", "original"):
                key = "original"
            self.target_resolution = key
            if self.cap is not None and self.cap.isOpened() and self.source_type == "webcam" and key in _RESOLUTION_HEIGHTS:
                th = _RESOLUTION_HEIGHTS[key]
                tw = {480: 640, 720: 1280, 1080: 1920}.get(th, 1280)
                try:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, tw)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, th)
                except Exception:
                    pass
            self._refresh_read_caches_unlocked()
            return self.target_resolution

    def _build_playback_state_unlocked(self) -> Dict[str, Any]:
        fps = max(1.0, float(getattr(self, "stream_fps", 30.0)))
        total = int(getattr(self, "total_frames", 0))
        cur = int(getattr(self, "current_frame_index", 0))
        if total > 0:
            cur = max(0, min(total - 1, cur))
        cur_sec = round(cur / fps, 2)
        dur_sec = round(total / fps, 2) if total > 0 else 0.0
        pct = round((cur / max(1, total - 1)) * 100.0, 2) if total > 1 else 0.0
        return {
            "is_paused": bool(getattr(self, "is_paused", False)),
            "current_frame": cur,
            "total_frames": total,
            "current_sec": cur_sec,
            "duration_sec": dur_sec,
            "progress_pct": pct,
            "speed": float(getattr(self, "playback_speed", 1.0)),
            "resolution": str(getattr(self, "target_resolution", "original")),
            "source_type": str(getattr(self, "source_type", "video")),
            "loop_enabled": bool(getattr(self, "loop_enabled", True)),
            "source_name": Path(str(getattr(self, "source_path", ""))).name,
        }

    def get_playback_state(self) -> Dict[str, Any]:
        with self._read_lock:
            pb = self._cached_telemetry.get("playback_state")
            if pb:
                return dict(pb)
        with self.lock:
            return self._build_playback_state_unlocked()

    # ------------------------------------------------------------------
    # Multi-Dimensional Follow, Filter & Role Policy Controls
    # ------------------------------------------------------------------
    def set_focus_track(self, track_id: Optional[int]) -> Optional[int]:
        with self.lock:
            self.focus_track_id = int(track_id) if track_id is not None else None
            if self.is_paused and self._last_raw_frame is not None:
                vis = self._render_annotations(self._last_raw_frame, self._last_detections, self._last_worker_reports)
                self._publish_encoded_frame(vis)
            self._refresh_read_caches_unlocked()
            return self.focus_track_id

    def set_filter(
        self,
        colors: Optional[List[str]] = None,
        roles: Optional[List[str]] = None,
        violators_only: Optional[bool] = None,
        clear: bool = False,
    ) -> Dict[str, Any]:
        with self.lock:
            if clear:
                self.filter_colors = []
                self.filter_roles = []
                self.violators_only = False
            else:
                if colors is not None:
                    self.filter_colors = [str(c).strip().lower() for c in colors if str(c).strip()]
                if roles is not None:
                    self.filter_roles = [str(r).strip().lower() for r in roles if str(r).strip()]
                if violators_only is not None:
                    self.violators_only = bool(violators_only)

            if self.is_paused and self._last_raw_frame is not None:
                vis = self._render_annotations(self._last_raw_frame, self._last_detections, self._last_worker_reports)
                self._publish_encoded_frame(vis)
            self._refresh_read_caches_unlocked()
            return self._get_filter_state_unlocked()

    def _get_filter_state_unlocked(self) -> Dict[str, Any]:
        return {
            "colors": list(getattr(self, "filter_colors", [])),
            "roles": list(getattr(self, "filter_roles", [])),
            "violators_only": bool(getattr(self, "violators_only", False)),
            "focus_track_id": getattr(self, "focus_track_id", None),
        }

    def get_filter_state(self) -> Dict[str, Any]:
        with self._read_lock:
            fs = self._cached_telemetry.get("filter_state")
            if fs:
                return dict(fs)
        with self.lock:
            return self._get_filter_state_unlocked()

    def set_track_label(self, track_id: int, label: str) -> bool:
        with self.lock:
            ok = False
            if hasattr(self, "gdut_reid") and self.gdut_reid is not None:
                ok = self.gdut_reid.set_label(track_id, label)
            if self._last_worker_reports:
                for r in self._last_worker_reports:
                    if r.get("track_id") == int(track_id):
                        r["custom_label"] = str(label).strip()[:40] or None
            if self.is_paused and self._last_raw_frame is not None:
                vis = self._render_annotations(self._last_raw_frame, self._last_detections, self._last_worker_reports)
                self._publish_encoded_frame(vis)
            self._refresh_read_caches_unlocked()
            return ok

    def enroll_track(self, track_id: int, label: Optional[str] = None) -> bool:
        with self.lock:
            if self._last_raw_frame is None or self.gdut_reid is None:
                return False
            box = None
            for r in self._last_worker_reports:
                if r.get("track_id") == int(track_id):
                    box = r.get("bbox")
                    if label is not None and str(label).strip():
                        r["custom_label"] = str(label).strip()[:40]
                    break
            ok = self.gdut_reid.enroll_reference(track_id, self._last_raw_frame, head_box=box, label=label)
            self._refresh_read_caches_unlocked()
            return ok

    def get_role_policy(self) -> Dict[str, Dict[str, Any]]:
        with self._read_lock:
            rp = self._cached_telemetry.get("role_policy")
            if rp:
                return copy.deepcopy(rp)
        return copy.deepcopy(self.role_policy)

    def set_role_policy(self, updates: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            self.role_policy = save_role_policy(updates)
            self._refresh_read_caches_unlocked()
            return copy.deepcopy(self.role_policy)

    def export_personnel_csv(self) -> str:
        workers = self.get_registered_workers(filter_ghosts=True)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "track_id",
            "custom_label",
            "current_class",
            "helmet_color",
            "historical_helmet_color",
            "role_title",
            "status",
            "is_active",
            "total_frames_seen",
            "first_frame",
            "last_seen_frame",
            "seconds_since_seen",
        ])
        for w in workers:
            writer.writerow([
                w.get("track_id"),
                w.get("custom_label") or "",
                w.get("current_class") or "",
                w.get("helmet_color") or "",
                w.get("historical_helmet_color") or "",
                w.get("historical_role_title") or w.get("role_title") or "",
                w.get("status") or "",
                w.get("is_active"),
                w.get("total_frames_seen"),
                w.get("first_frame"),
                w.get("last_seen_frame"),
                w.get("seconds_since_seen"),
            ])
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Publishing & Background Worker Loop
    # ------------------------------------------------------------------
    def _publish_encoded_frame(self, frame_bgr: Optional[np.ndarray]):
        if frame_bgr is None or frame_bgr.size == 0:
            return
        self.latest_annotated_frame = frame_bgr
        enc_q = getattr(self, "_encode_queue", None)
        if enc_q is None:
            self._encode_and_notify(frame_bgr, self.target_resolution)
            return
        while not enc_q.empty():
            try:
                enc_q.get_nowait()
            except Exception:
                break
        try:
            enc_q.put_nowait((frame_bgr.copy(), self.target_resolution))
        except Exception:
            pass

    def _encode_and_notify(self, frame_bgr: np.ndarray, res_key: str):
        h, w = frame_bgr.shape[:2]
        target_w = 1280 if res_key == "1080p" else 960
        if w > target_w:
            target_h = int(h * (target_w / w))
            display_frame = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            display_frame = frame_bgr

        ret, buffer = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ret:
            jpeg_bytes = buffer.tobytes()
            with self.frame_condition:
                self.latest_encoded_jpeg = jpeg_bytes
                self.frame_idx += 1
                self.frame_condition.notify_all()

    def _jpeg_encode_worker(self):
        """Offloads CPU JPEG compression from GPU inference stream worker."""
        while getattr(self, "is_running", False):
            try:
                frame_bgr, res_key = self._encode_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._encode_and_notify(frame_bgr, res_key)
            except Exception:
                pass

    def _stream_worker(self):
        """Dedicated background loop with self-healing recovery and speed-aware pacing."""
        consecutive_errors = 0
        last_error_log_time = 0.0

        while self.is_running:
            t0 = time.perf_counter()
            if self.source_type == "browser_webcam":
                time.sleep(0.05)
                continue

            if self.is_paused:
                time.sleep(0.08)
                continue

            try:
                frame = self.process_next_frame()
                if frame is not None:
                    consecutive_errors = 0
                    self._publish_encoded_frame(frame)
            except Exception as e:
                consecutive_errors += 1
                curr_t = time.time()
                if curr_t - last_error_log_time >= 3.0:
                    import traceback
                    print(f"⚠️ Stream worker exception (streak={consecutive_errors}): {e}")
                    traceback.print_exc()
                    last_error_log_time = curr_t

                err_msg = str(e).lower()
                if any(
                    sig in err_msg
                    for sig in (
                        "illegal memory access",
                        "device-side assert",
                        "misaligned address",
                        "illegal instruction",
                        "unspecified launch failure",
                        "hardware stack error",
                        "context is destroyed",
                    )
                ):
                    os._exit(1)

                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                if consecutive_errors >= 3:
                    try:
                        with self.lock:
                            self._init_video_capture()
                            self.detector.load_model(self.detector.model_path)
                        consecutive_errors = 0
                    except Exception:
                        try:
                            with self.lock:
                                self.detector.device = "cpu"
                                self.detector.use_half = False
                                self.detector.use_selective_amp = False
                                self.detector.load_model(self.detector.model_path)
                            consecutive_errors = 0
                        except Exception:
                            pass
                time.sleep(0.1)

            elapsed = time.perf_counter() - t0
            speed = max(0.25, float(getattr(self, "playback_speed", 1.0)))
            target_delay = (float(self.frame_stride) / max(1.0, float(self.stream_fps))) / speed
            if elapsed < target_delay:
                time.sleep(target_delay - elapsed)

    def get_latest_frame(self, last_frame_idx: int = -1, timeout: float = 0.5) -> Tuple[Optional[bytes], int]:
        with self.frame_condition:
            if self.frame_idx == last_frame_idx or self.latest_encoded_jpeg is None:
                self.frame_condition.wait(timeout=timeout)
            return self.latest_encoded_jpeg, self.frame_idx

    # ------------------------------------------------------------------
    # Core Frame Processing (Shared by Video Thread & WebSocket Webcam)
    # ------------------------------------------------------------------
    def process_next_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.is_paused and self._last_raw_frame is not None:
                return self._render_annotations(self._last_raw_frame, self._last_detections, self._last_worker_reports)

            if self.cap is None or not self.cap.isOpened():
                self._init_video_capture()
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.05)
                    return None

            if self._pending_seek_frame is not None and self.source_type == "video":
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._pending_seek_frame)
                self._pending_seek_frame = None

            ret, frame = self.cap.read()
            if not ret:
                if self.source_type == "video" and self.loop_enabled:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    self.tracker.reset()
                    self.safety_engine.reset()
                    self.stationary_furniture_tracker.clear()
                    self.confirmed_furniture_zones.clear()
                    self.stationary_next_id = 1
                    self.gdut_reid.on_video_loop()
                    reid_registry.on_video_loop()
                if not ret:
                    time.sleep(0.03)
                    return None

            if self.frame_stride > 1 and self.source_type == "video":
                for _ in range(self.frame_stride - 1):
                    self.cap.grab()

            if self.source_type == "video":
                pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.current_frame_index = int(max(0, round(pos) - 1))

            frame = _apply_target_resolution(frame, self.target_resolution)
            return self._process_frame_unlocked(frame, config.camera_id, render_image=True)

    def process_external_frame(
        self,
        frame_bgr: np.ndarray,
        camera_id: str = "WEBCAM-C960",
        resolution: Optional[str] = None,
        return_annotated: bool = False,
    ) -> Dict[str, Any]:
        """
        Thread-safe entry point for Browser Direct WebSocket `/ws/webcam`.
        Skips expensive OpenCV JPEG drawing & re-encoding when `return_annotated=False`
        so the browser can overlay crisp vector boxes at 25-30 FPS over live `<video>`.
        """
        with self.lock:
            if self.source_type != "browser_webcam":
                self.tracker.reset()
                self.safety_engine.reset()
                if hasattr(self, "gdut_reid") and self.gdut_reid is not None:
                    self.gdut_reid.reset()
                reid_registry.reset()
                self.focus_track_id = None
                self.latest_active_track_ids = []
                self.latest_active_tracks = []
                self._last_worker_reports = []
                self._last_detections = []
            self.source_type = "browser_webcam"
            if resolution:
                res_k = str(resolution).lower().strip()
                if res_k in ("480p", "720p", "1080p", "original"):
                    self.target_resolution = res_k
            resized = _apply_target_resolution(frame_bgr, self.target_resolution)
            annotated = self._process_frame_unlocked(resized, camera_id=camera_id, render_image=return_annotated)
            if return_annotated and annotated is not None:
                self._publish_encoded_frame(annotated)

        tele = self.get_telemetry()
        workers = self.get_registered_workers(filter_ghosts=True)
        return {
            "annotated_frame": annotated,
            "telemetry": tele,
            "personnel": workers,
            "active_tracks": list(tele.get("active_tracks", [])),
        }

    def _process_frame_unlocked(
        self,
        frame: np.ndarray,
        camera_id: str,
        render_image: bool = True,
    ) -> Optional[np.ndarray]:
        t_start = time.perf_counter()
        h_frame, w_frame = frame.shape[:2]
        self.frame_height = h_frame
        self.frame_width = w_frame

        detections, person_dets, infer_ms = self.detector.predict(frame)
        self.inference_latency_ms = infer_ms
        self.stationary_furniture_tracker.clear()
        self.confirmed_furniture_zones.clear()

        is_helmet_only = getattr(self.detector, "is_helmet_only", False)
        has_person_class = getattr(self.detector, "has_person_class", False)

        filtered_detections: List[Dict[str, Any]] = []
        for d in detections:
            d_name_l = str(d.get("name", "")).lower().strip()
            is_person_cls = "person" in d_name_l or "worker" in d_name_l or (has_person_class and d["cls"] == 0)
            if is_person_cls:
                filtered_detections.append(d)
                continue

            if not is_helmet_only:
                is_blue_cand = (
                    "blue" in d_name_l
                    or d.get("color") == "blue"
                    or (has_person_class and d["cls"] == 2 and (not d_name_l or d_name_l.isdigit()))
                )
                if is_blue_cand:
                    bx1, by1, bx2, by2 = [int(round(v)) for v in d["box"][:4]]
                    bx1 = max(0, min(w_frame - 1, bx1))
                    by1 = max(0, min(h_frame - 1, by1))
                    bx2 = max(bx1 + 1, min(w_frame, bx2))
                    by2 = max(by1 + 1, min(h_frame, by2))
                    crop = frame[by1:by2, bx1:bx2]
                    if crop.size > 0 and np.any(crop > 0):
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        if float(np.mean(hsv[:, :, 2])) < 95.0:
                            continue

            is_any_helmet = (
                "vest" not in d_name_l
                and "person" not in d_name_l
                and (
                    "helmet" in d_name_l
                    or d_name_l in ("blue", "white", "yellow", "red", "green", "none")
                    or (has_person_class and d["cls"] in (2, 3, 4, 5, 6) and (not d_name_l or d_name_l.isdigit()))
                )
            )
            if is_any_helmet:
                bx1, by1, bx2, by2 = d["box"][:4]
                bw = max(1.0, bx2 - bx1)
                bh = max(1.0, by2 - by1)
                if (bw / bh) > 2.35:
                    continue

            orig_conf = d["conf"]
            if is_helmet_only:
                effective_conf = orig_conf
            else:
                max_overlap = 0.0
                if len(person_dets) > 0:
                    for p_box in person_dets:
                        ov = compute_box_intersection_ratio(
                            d["box"], p_box[:4], use_upper_torso=is_any_helmet, upper_torso_ratio=0.60
                        )
                        if ov > max_overlap:
                            max_overlap = ov
                effective_conf = orig_conf if max_overlap >= 0.25 else orig_conf * 0.15

            if effective_conf >= config.conf_threshold:
                d_c = dict(d)
                d_c["conf"] = round(effective_conf, 4)
                filtered_detections.append(d_c)

        detections = filtered_detections

        if is_helmet_only:
            head_dets = self.detector.get_head_tracks()
            if len(head_dets) > 0:
                valid_head = []
                for hd in head_dets:
                    bw = float(hd[2] - hd[0])
                    bh = float(hd[3] - hd[1])
                    valid_head.append(bw >= 8.0 and bh >= 8.0 and (bw / max(1.0, bh)) <= 2.4)
                head_dets = head_dets[np.array(valid_head, dtype=bool)] if any(valid_head) else np.empty((0, 6), dtype=np.float32)

            active_tracks = self.tracker.update(head_dets, frame=None)
            now_ts = time.time()
            batch_items = [
                (
                    track.track_id,
                    track.tlbr,
                    str(self.detector.class_names.get(int(track.cls_id), "none")).lower().strip(),
                )
                for track in active_tracks
            ]
            profiles = self.gdut_reid.associate_batch(
                batch_items,
                frame_bgr=frame,
                frame_idx=self.total_frames_processed,
                now_ts=now_ts,
            )
            confirmed_tracks = []
            for track, (_, _, raw_cls), prof in zip(active_tracks, batch_items, profiles):
                if getattr(prof, "worker_id", -1) <= 0:
                    continue
                track.canonical_id = prof.worker_id
                track.head_cls_name = raw_cls
                track.historical_helmet_color = prof.last_known_helmet_color
                track.custom_label = prof.custom_label
                track.torso_roi = prof.last_torso_roi
                confirmed_tracks.append(track)

            worker_reports, alert_events = self.safety_engine.evaluate_frame(
                confirmed_tracks,
                detections,
                frame,
                camera_id=camera_id,
                is_helmet_only=True,
                role_policy=self.role_policy,
                class_names=self.detector.class_names,
            )

            for r in worker_reports:
                wid = r["track_id"]
                if isinstance(wid, int) and wid in self.gdut_reid.profiles:
                    prof = self.gdut_reid.profiles[wid]
                    prof.status = r["status"]
                    prof.violations = list(r["violations"])
            active_cids = [int(r["track_id"]) for r in worker_reports if isinstance(r["track_id"], int)]
            self.latest_active_track_ids = active_cids
            self.gdut_reid.update_frame_end(active_cids, self.total_frames_processed, now_ts=now_ts)
        else:
            if len(person_dets) > 0:
                valid_mask = []
                for p in person_dets:
                    pw = p[2] - p[0]
                    ph = p[3] - p[1]
                    touches_border = (p[0] <= 60 or p[1] <= 60 or p[2] >= w_frame - 60 or p[3] >= h_frame - 60)
                    max_aspect = 3.5 if touches_border else 2.6
                    valid_mask.append(ph >= 35 and pw >= 15 and (pw / max(1.0, ph)) <= max_aspect)
                person_dets = person_dets[np.array(valid_mask, dtype=bool)] if any(valid_mask) else np.empty((0, 6), dtype=np.float32)

            active_tracks = self.tracker.update(person_dets, frame=frame)
            for track in active_tracks:
                tb = track.tlbr
                matched_color = None
                th = max(1.0, tb[3] - tb[1])
                tw = max(1.0, tb[2] - tb[0])
                tcx = (tb[0] + tb[2]) / 2.0
                for d in detections:
                    dn = str(d.get("name", "")).lower()
                    is_h = ("helmet" in dn or dn in ("blue", "white", "yellow", "red", "green")) and "vest" not in dn
                    if is_h:
                        hb = d["box"]
                        hcx = (hb[0] + hb[2]) / 2.0
                        hw = max(0.0, hb[2] - hb[0])
                        hh = max(0.0, hb[3] - hb[1])
                        if hb[1] > (tb[1] + 0.22 * th) or hb[3] < (tb[1] - 0.15 * th):
                            continue
                        if not (tb[0] - 0.05 * tw <= hcx <= tb[2] + 0.05 * tw) and abs(hcx - tcx) > 0.45 * tw:
                            continue
                        if hw > 1.05 * tw or (hw / max(1.0, hh)) > 2.20:
                            continue
                        if not (hb[2] < tb[0] or hb[0] > tb[2] or hb[3] < tb[1] or hb[1] > tb[1] + 0.42 * th):
                            matched_color = d.get("color") or dn.replace("_helmet", "").replace("-helmet", "")
                            break
                canonical_id = reid_registry.get_or_create_canonical_id(
                    raw_track_id=track.track_id,
                    box=tb,
                    frame=frame,
                    helmet_color=matched_color,
                    frame_idx=self.total_frames_processed,
                )
                track.canonical_id = canonical_id

            worker_reports, alert_events = self.safety_engine.evaluate_frame(
                active_tracks, detections, frame, camera_id=camera_id, is_helmet_only=False
            )
            for r in worker_reports:
                wid = r["track_id"]
                if wid in reid_registry.profiles:
                    reid_registry.profiles[wid].status = r["status"]
                    reid_registry.profiles[wid].violations = r["violations"]
                    if r.get("helmet_color") and r["helmet_color"] != "none":
                        reid_registry.profiles[wid].helmet_color = r["helmet_color"]
            active_ids = [r["track_id"] for r in worker_reports]
            self.latest_active_track_ids = active_ids
            reid_registry.update_frame_end(active_ids, self.total_frames_processed)

        for alert in alert_events:
            alert_manager.enqueue_alert(alert)

        # Build lightweight normalized metadata (also powers vector overlay in C960 WebSocket mode)
        norm_tracks = []
        inv_w = 1.0 / max(1.0, float(w_frame))
        inv_h = 1.0 / max(1.0, float(h_frame))
        for r in worker_reports:
            bx1, by1, bx2, by2 = r["bbox"]
            tid = r["track_id"]
            traj_pts = []
            if isinstance(tid, int):
                raw_pts = self._get_trajectory_points(tid)[-30:]
                traj_pts = [[round(px * inv_w, 4), round(py * inv_h, 4)] for px, py in raw_pts]
            norm_tracks.append({
                "track_id": tid,
                "bbox": r["bbox"],
                "norm_bbox": [
                    round(bx1 * inv_w, 4),
                    round(by1 * inv_h, 4),
                    round(bx2 * inv_w, 4),
                    round(by2 * inv_h, 4),
                ],
                "trajectory": traj_pts,
                "helmet_color": r.get("helmet_color"),
                "current_class": r.get("current_class", r.get("helmet_color") or "none"),
                "historical_helmet_color": r.get("historical_helmet_color"),
                "role_title": r.get("role_title"),
                "role_short_code": r.get("historical_role_short_code") or r.get("role_short_code"),
                "role_hex_color": r.get("role_hex_color", "#15803D"),
                "custom_label": r.get("custom_label"),
                "status": r.get("status"),
                "violation_streak": r.get("violation_streak", 0),
                "confidence": r.get("confidence", 0.0),
            })
        self.latest_active_tracks = norm_tracks

        instant_workers = len(worker_reports)
        instant_compliant = sum(1 for r in worker_reports if r["status"] == "COMPLIANT")
        instant_violations = instant_workers - instant_compliant
        instant_compliance_rate = (instant_compliant / instant_workers * 100.0) if instant_workers > 0 else 100.0

        alpha = 0.25
        if self.total_frames_processed == 0:
            self.workers_count_ema = float(instant_workers)
            self.compliance_rate_ema = float(instant_compliance_rate)
        else:
            self.workers_count_ema = alpha * instant_workers + (1.0 - alpha) * self.workers_count_ema
            self.compliance_rate_ema = alpha * instant_compliance_rate + (1.0 - alpha) * self.compliance_rate_ema

        self.active_workers_count = int(round(self.workers_count_ema)) if instant_workers > 0 else 0
        self.compliant_workers_count = instant_compliant
        self.active_violations_count = instant_violations
        self.compliance_rate = round(self.compliance_rate_ema, 1)

        color_counts = {"blue": 0, "white": 0, "yellow": 0, "red": 0, "none": 0, "green": 0}
        role_counts: Dict[str, int] = {}
        for r in worker_reports:
            col = r.get("current_class") or r.get("helmet_color") or ("none" if r["status"] == "VIOLATION" else "yellow")
            if col in color_counts:
                color_counts[col] += 1
            rs = r.get("role_short_code") or "WORKER"
            role_counts[rs] = role_counts.get(rs, 0) + 1
        self.helmet_color_distribution = color_counts
        self.role_distribution = role_counts

        self._last_raw_frame = frame.copy()
        self._last_detections = detections
        self._last_worker_reports = worker_reports

        annotated = self._render_annotations(frame, detections, worker_reports) if render_image else None

        t_total = (time.perf_counter() - t_start) * 1000.0
        self.total_latency_ms = t_total
        fps_instant = 1000.0 / t_total if t_total > 0 else 30.0
        self.fps_rolling = 0.9 * self.fps_rolling + 0.1 * fps_instant if self.fps_rolling > 0 else fps_instant
        self.total_frames_processed += 1
        if annotated is not None:
            self.latest_annotated_frame = annotated

        self._refresh_read_caches_unlocked()
        return annotated

    def _report_matches_filter(self, report: Dict[str, Any]) -> bool:
        f_colors = [str(c).lower() for c in (getattr(self, "filter_colors", []) or [])]
        f_roles = [str(r).lower() for r in (getattr(self, "filter_roles", []) or [])]
        v_only = bool(getattr(self, "violators_only", False))

        if v_only and report.get("status") != "VIOLATION":
            return False

        if f_colors:
            cur_col = str(report.get("current_class") or report.get("helmet_color") or "none").lower()
            if cur_col not in f_colors:
                return False

        if f_roles:
            r_title = str(report.get("role_title", "")).lower()
            r_short = str(report.get("role_short_code", "")).lower()
            h_title = str(report.get("historical_role_title", "")).lower()
            h_short = str(report.get("historical_role_short_code", "")).lower()
            matched = any(
                fr in (r_title, r_short, h_title, h_short) or fr in r_title or fr in h_title
                for fr in f_roles
            )
            if not matched:
                return False

        return True

    def _get_trajectory_points(self, track_id: int) -> List[Tuple[int, int]]:
        gdut_reid = getattr(self, "gdut_reid", None)
        if gdut_reid is not None and track_id in gdut_reid.profiles:
            return [(int(pt[0]), int(pt[1])) for pt in gdut_reid.profiles[track_id].trajectory]
        if track_id in reid_registry.profiles:
            return [(int(pt[0]), int(pt[1])) for pt in reid_registry.profiles[track_id].trajectory]
        return []

    def _render_annotations(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        worker_reports: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Renders Milestone/Siemens industrial OSD annotations onto the frame.
        Safely supports minimal MockPipeline objects used in regression scripts.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        focus_id = getattr(self, "focus_track_id", None)
        f_colors = list(getattr(self, "filter_colors", []) or [])
        f_roles = list(getattr(self, "filter_roles", []) or [])
        v_only = bool(getattr(self, "violators_only", False))
        has_filter = bool(f_colors or f_roles or v_only)

        if focus_id is not None:
            focused = next((r for r in worker_reports if r.get("track_id") == focus_id), None)

            hud_h = 34
            cv2.rectangle(vis, (0, 0), (w, hud_h), (24, 20, 17), -1)
            cv2.line(vis, (0, hud_h), (w, hud_h), (129, 76, 15), 1)
            is_locked = focused is not None
            dot_bgr = (61, 128, 21) if is_locked else (9, 83, 180)
            cv2.circle(vis, (16, 17), 5, dot_bgr, -1)

            status_note = "LOCKED" if is_locked else "COASTING / MEMORY"
            role_note = f" | {focused.get('historical_role_short_code') or focused.get('role_short_code', '')}" if focused else ""
            hud_text = (
                f"SOLO FOCUS: ID #{focus_id}{role_note} | {status_note} | "
                f"{getattr(self, 'fps_rolling', 0.0):.1f} FPS | {getattr(self, 'inference_latency_ms', 0.0):.1f}ms"
            )
            cv2.putText(vis, hud_text, (30, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (241, 245, 249), 1, cv2.LINE_AA)

            if focused is not None:
                r = focused
                x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
                is_compliant = (r["status"] == "COMPLIANT")
                box_color = tuple(r.get("role_bgr_color", [61, 128, 21])) if is_compliant else (38, 38, 220)

                pts = self._get_trajectory_points(int(focus_id)) if isinstance(focus_id, int) else []
                for idx in range(1, len(pts)):
                    cv2.line(vis, pts[idx - 1], pts[idx], (9, 130, 217), 2, cv2.LINE_AA)
                if pts:
                    cv2.circle(vis, pts[-1], 4, (9, 130, 217), -1)

                t_roi = r.get("torso_roi")
                if t_roi and len(t_roi) == 4:
                    tx1, ty1, tx2, ty2 = [int(v) for v in t_roi]
                    cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), (100, 116, 139), 1)

                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 3)
                d = max(6, min(20, (x2 - x1) // 3, (y2 - y1) // 3))
                bracket_bgr = (241, 245, 249) if is_compliant else (38, 38, 220)
                for cx_b, cy_b, dx, dy in (
                    (x1, y1, d, d),
                    (x2, y1, -d, d),
                    (x1, y2, d, -d),
                    (x2, y2, -d, -d),
                ):
                    cv2.line(vis, (cx_b, cy_b), (cx_b + dx, cy_b), bracket_bgr, 2)
                    cv2.line(vis, (cx_b, cy_b), (cx_b, cy_b + dy), bracket_bgr, 2)

                m_helmet = r.get("matched_helmet_box")
                if m_helmet is not None and not r.get("is_helmet_only", False):
                    hx1, hy1, hx2, hy2 = [int(v) for v in m_helmet[:4]]
                    cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), (129, 76, 15), 2)

                lbl_name = f" {r['custom_label']}" if r.get("custom_label") else ""
                role_tag = r.get("historical_role_short_code") or r.get("role_short_code") or "ID"
                col_tag = (r.get("helmet_color") or "BARE-HEAD").upper()
                label_top = f"ID #{r['track_id']}{lbl_name} | {role_tag} | {col_tag}"
                (lw, _), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                top_y = max(36, y1 - 24)
                cv2.rectangle(vis, (x1, top_y), (x1 + lw + 10, top_y + 22), (23, 23, 17), -1)
                cv2.rectangle(vis, (x1, top_y), (x1 + lw + 10, top_y + 22), box_color, 1)
                cv2.putText(vis, label_top, (x1 + 5, top_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (248, 250, 252), 1, cv2.LINE_AA)

                if not is_compliant:
                    v_text = "VIOLATION: BARE HEAD"
                    (vw, _), _ = cv2.getTextSize(v_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(vis, (x1, y2 + 2), (x1 + vw + 8, y2 + 22), (28, 28, 185), -1)
                    cv2.putText(vis, v_text, (x1 + 4, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            return vis

        has_person_class = getattr(self.detector, "has_person_class", False)
        is_helmet_only = getattr(self.detector, "is_helmet_only", True)

        if not (is_helmet_only and len(worker_reports) > 0) and not has_filter:
            for d in detections:
                is_worker_body = "person" in d["name"] or "worker" in d["name"] or (has_person_class and d["cls"] == 0)
                if not is_worker_body:
                    c_name = d["name"]
                    color = CLASS_COLORS.get(c_name, (129, 76, 15))
                    x1, y1, x2, y2 = [int(v) for v in d["box"]]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    label = f"{c_name.replace('_', ' ').upper()} {d['conf']*100:.0f}%"
                    (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    cv2.rectangle(vis, (x1, max(0, y1 - 19)), (x1 + lw + 6, max(0, y1)), (23, 23, 17), -1)
                    cv2.putText(vis, label, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (248, 250, 252), 1, cv2.LINE_AA)

        visible_reports = [r for r in worker_reports if self._report_matches_filter(r)] if has_filter else worker_reports

        for r in visible_reports:
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            is_compliant = (r["status"] == "COMPLIANT")
            box_color = tuple(r.get("role_bgr_color", [61, 128, 21])) if is_compliant else (38, 38, 220)

            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
            if not is_compliant:
                d = max(5, min(14, (x2 - x1) // 4, (y2 - y1) // 4))
                for cx_b, cy_b, dx, dy in ((x1, y1, d, d), (x2, y1, -d, d)):
                    cv2.line(vis, (cx_b, cy_b), (cx_b + dx, cy_b), (38, 38, 220), 2)
                    cv2.line(vis, (cx_b, cy_b), (cx_b, cy_b + dy), (38, 38, 220), 2)

            lbl_name = f" {r['custom_label']}" if r.get("custom_label") else ""
            role_short = r.get("historical_role_short_code") or r.get("role_short_code") or "ID"
            col_tag = (r.get("helmet_color") or ("BARE" if not is_compliant else "HELMET")).upper()
            label_top = f"#{r['track_id']}{lbl_name} | {role_short} | {col_tag}"

            (lw, _), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
            tag_y1 = max(36, y1 - 20)
            cv2.rectangle(vis, (x1, tag_y1), (x1 + lw + 8, tag_y1 + 19), (23, 23, 17), -1)
            cv2.rectangle(vis, (x1, tag_y1), (x1 + lw + 8, tag_y1 + 19), box_color, 1)
            cv2.putText(vis, label_top, (x1 + 4, tag_y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (248, 250, 252), 1, cv2.LINE_AA)

            if not is_compliant and r.get("violations"):
                v_text = f"VIOLATION ({r.get('violation_streak', 0)}f)"
                (vw, _), _ = cv2.getTextSize(v_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.rectangle(vis, (x1, y2 + 1), (x1 + vw + 8, y2 + 18), (28, 28, 185), -1)
                cv2.putText(vis, v_text, (x1 + 4, y2 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        if getattr(config, "draw_video_hud", False):
            hud_h = 34
            cv2.rectangle(vis, (0, 0), (w, hud_h), (24, 20, 17), -1)
            cv2.line(vis, (0, hud_h), (w, hud_h), (75, 85, 99), 1)
            cv2.circle(vis, (16, 17), 5, (28, 28, 185), -1)

            if has_filter:
                f_parts = []
                if f_colors:
                    f_parts.append("COLOR=" + "+".join(c.upper() for c in f_colors))
                if f_roles:
                    f_parts.append("ROLE=" + "+".join(r.upper() for r in f_roles))
                if v_only:
                    f_parts.append("VIOLATORS")
                filter_note = f" | FILTER[{' '.join(f_parts)}] ({len(visible_reports)}/{len(worker_reports)})"
            else:
                filter_note = ""

            hud_text = (
                f"AEGIS-VMS // {config.camera_id} | {getattr(self, 'fps_rolling', 0.0):.1f} FPS | "
                f"{getattr(self, 'inference_latency_ms', 0.0):.1f}ms | ON-SITE: {len(worker_reports)} | "
                f"VIOL: {getattr(self, 'active_violations_count', 0)} | COMP: {getattr(self, 'compliance_rate', 100.0):.1f}%{filter_note}"
            )
            cv2.putText(vis, hud_text, (30, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (226, 232, 240), 1, cv2.LINE_AA)
        return vis

    def stop(self):
        self.is_running = False
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def get_registered_workers(self, filter_ghosts: bool = True) -> List[Dict[str, Any]]:
        if filter_ghosts and hasattr(self, "_read_lock"):
            with self._read_lock:
                if self._cached_workers is not None:
                    return copy.deepcopy(self._cached_workers)
        with self.lock:
            if getattr(self.detector, "is_helmet_only", False):
                return self.gdut_reid.get_registered_workers(self.role_policy, filter_ghosts=filter_ghosts)
            return reid_registry.get_registered_workers(filter_ghosts=filter_ghosts)

    def _build_telemetry_unlocked(self) -> Dict[str, Any]:
        is_helmet_only = bool(getattr(self.detector, "is_helmet_only", False))
        if is_helmet_only and getattr(self, "gdut_reid", None) is not None:
            reg_workers = self.gdut_reid.get_registered_workers(self.role_policy, filter_ghosts=True)
        else:
            reg_workers = reid_registry.get_registered_workers(filter_ghosts=True)

        focus_status = None
        if self.focus_track_id is not None:
            focus_status = "ACTIVE" if (self.focus_track_id in self.latest_active_track_ids) else "MEMORY"

        active_reg_workers = [w for w in reg_workers if w.get("is_active")]
        active_reg_count = len(active_reg_workers)
        if is_helmet_only and getattr(self, "gdut_reid", None) is not None:
            kpi_workers_count = active_reg_count
            kpi_violations_count = sum(1 for w in active_reg_workers if w.get("status") == "VIOLATION")
            kpi_compliant_count = kpi_workers_count - kpi_violations_count
        else:
            kpi_workers_count = self.active_workers_count
            kpi_compliant_count = self.compliant_workers_count
            kpi_violations_count = self.active_violations_count
        bitrate_mbps = round(max(1.2, min(15.0, (self.fps_rolling * 95.0 * 1024 * 8) / 1_000_000)), 1) if self.fps_rolling > 0 else 0.0

        return {
            "timestamp": time.time(),
            "fps": round(self.fps_rolling, 1),
            "inference_ms": round(self.inference_latency_ms, 1),
            "total_ms": round(self.total_latency_ms, 1),
            "workers_count": kpi_workers_count,
            "compliant_count": kpi_compliant_count,
            "violations_count": kpi_violations_count,
            "compliance_rate": round(self.compliance_rate, 1),
            "helmet_colors": dict(self.helmet_color_distribution),
            "role_distribution": dict(self.role_distribution),
            "device": self.detector.device.upper(),
            "use_selective_amp": bool(getattr(self.detector, "use_selective_amp", False)),
            "model_path": Path(self.detector.model_path).name,
            "video_source": Path(str(self.source_path)).name,
            "resolution": f"{self.frame_width}x{self.frame_height}",
            "stream_fps": round(self.stream_fps, 1),
            "bitrate_mbps": bitrate_mbps,
            "total_frames": self.total_frames_processed,
            "focus_track_id": self.focus_track_id,
            "focus_status": focus_status,
            "active_registered_count": active_reg_count,
            "registered_workers_count": len(reg_workers),
            "gallery_total_count": len(reg_workers),
            "workers": reg_workers,
            "is_helmet_only": is_helmet_only,
            "active_tracks": list(self.latest_active_tracks),
            "filter_state": self._get_filter_state_unlocked(),
            "playback_state": self._build_playback_state_unlocked(),
            "role_policy": copy.deepcopy(self.role_policy),
        }

    def get_telemetry(self) -> Dict[str, Any]:
        if hasattr(self, "_read_lock"):
            with self._read_lock:
                if self._cached_telemetry:
                    return copy.deepcopy(self._cached_telemetry)
        with self.lock:
            return self._build_telemetry_unlocked()


# Lazy singleton pipeline proxy
class _LazyPipelineProxy:
    _instance: Optional[CCTVSurveillancePipeline] = None
    _lock: threading.Lock = threading.Lock()

    def _get_instance(self) -> CCTVSurveillancePipeline:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = CCTVSurveillancePipeline()
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_instance(), name)


cctv_pipeline = _LazyPipelineProxy()
