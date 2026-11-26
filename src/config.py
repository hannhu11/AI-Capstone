"""
src/config.py
=============
Centralized configuration for the Industrial CCTV Safety Surveillance & Alerting System.
Decouples model weights, role-by-color policy, Re-ID parameters, video sources, thresholds, and bot tokens.
"""

from __future__ import annotations
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=False)
    else:
        load_dotenv(override=False)
except ImportError:
    pass

# Decoupled Model Weights (Default to GDUT-HWD 5-Class A6 Deploy Checkpoint)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "Output"
    / "GDUT_Bộ số 1_stage2-full-train-run-a6"
    / "deploy_weights"
    / "rep_yolo11s_gdut_a6_deploy.pt"
)
FALLBACK_MODEL_PATH = PROJECT_ROOT / "yolo11s.pt"

# Output & persistent policy paths
OUTPUT_DIR = PROJECT_ROOT / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROLE_POLICY_PATH = Path(os.getenv("ROLE_POLICY_PATH", str(OUTPUT_DIR / "role_policy.json")))

# Video simulation source
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "video_test" / "construction_site_workers_1080p.mp4"

# Snapshots directory
SNAPSHOTS_DIR = PROJECT_ROOT / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Alerts log directory
ALERTS_LOG_DIR = OUTPUT_DIR / "alerts"
ALERTS_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configurable Role-by-Color Hierarchy (GDUT-HWD 5-class standard)
DEFAULT_ROLE_HIERARCHY: Dict[str, Dict[str, Any]] = {
    "white": {
        "role_title": "Project Manager / Site Engineer",
        "short_code": "PM/ENG",
        "hex_color": "#475569",
        "bgr_color": [105, 85, 71],
        "severity": "compliant",
    },
    "yellow": {
        "role_title": "General Construction Worker",
        "short_code": "WORKER",
        "hex_color": "#B45309",
        "bgr_color": [9, 83, 180],
        "severity": "compliant",
    },
    "blue": {
        "role_title": "Technical Supervisor / MEP",
        "short_code": "SUP/MEP",
        "hex_color": "#0F4C81",
        "bgr_color": [129, 76, 15],
        "severity": "compliant",
    },
    "red": {
        "role_title": "HSE Safety Officer",
        "short_code": "HSE",
        "hex_color": "#991B1B",
        "bgr_color": [27, 27, 153],
        "severity": "compliant",
    },
    "none": {
        "role_title": "UNPROTECTED — VIOLATION",
        "short_code": "BARE-HEAD",
        "hex_color": "#DC2626",
        "bgr_color": [38, 38, 220],
        "severity": "critical",
    },
}


def hex_to_bgr(hex_str: str, fallback: Optional[List[int]] = None) -> List[int]:
    """Converts a hex RGB string like '#0F4C81' to OpenCV [B, G, R]."""
    if not isinstance(hex_str, str):
        return list(fallback or [129, 76, 15])
    cleaned = hex_str.strip().lstrip("#")
    if len(cleaned) != 6:
        return list(fallback or [129, 76, 15])
    try:
        r = int(cleaned[0:2], 16)
        g = int(cleaned[2:4], 16)
        b = int(cleaned[4:6], 16)
        return [b, g, r]
    except ValueError:
        return list(fallback or [129, 76, 15])


def _normalize_role_policy(raw_policy: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Merges user overrides onto DEFAULT_ROLE_HIERARCHY and normalizes fields."""
    merged = copy.deepcopy(DEFAULT_ROLE_HIERARCHY)
    if isinstance(raw_policy, dict):
        for color_key, entry in raw_policy.items():
            ck = str(color_key).strip().lower()
            if ck not in merged or not isinstance(entry, dict):
                continue
            base = merged[ck]
            title = str(entry.get("role_title", base["role_title"])).strip() or base["role_title"]
            short_code = str(entry.get("short_code", base["short_code"])).strip()[:16] or base["short_code"]
            hex_val = str(entry.get("hex_color", base["hex_color"])).strip()
            if not hex_val.startswith("#") or len(hex_val) != 7:
                hex_val = base["hex_color"]
            severity = str(entry.get("severity", base["severity"])).strip().lower()
            if severity not in ("compliant", "caution", "critical"):
                severity = base["severity"]
            if ck == "none":
                severity = "critical"
            merged[ck] = {
                "role_title": title,
                "short_code": short_code,
                "hex_color": hex_val,
                "bgr_color": hex_to_bgr(hex_val, fallback=base["bgr_color"]),
                "severity": severity,
            }
    return merged


def load_role_policy(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Loads persisted Role-by-Color hierarchy from disk, falling back to defaults."""
    target = Path(path) if path is not None else ROLE_POLICY_PATH
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return _normalize_role_policy(data)
        except Exception as e:
            print(f"⚠️ Failed to parse role policy at {target}: {e}")
    return _normalize_role_policy(None)


def save_role_policy(policy_updates: Dict[str, Any], path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Validates, merges, and persists the Role-by-Color hierarchy to disk."""
    target = Path(path) if path is not None else ROLE_POLICY_PATH
    current = load_role_policy(target)
    if isinstance(policy_updates, dict):
        for k, v in policy_updates.items():
            ck = str(k).strip().lower()
            if ck in current and isinstance(v, dict):
                updated_entry = dict(current[ck])
                updated_entry.update(v)
                current[ck] = updated_entry
    normalized = _normalize_role_policy(current)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(target)
    return normalized


def discover_available_models() -> List[Dict[str, Any]]:
    """Returns curated active GDUT 5-class weights and optional fallback models present on disk."""
    candidates = [
        (
            "Rep-YOLO11s GDUT A6 (Primary 5-Class Deploy)",
            "Output/GDUT_Bộ số 1_stage2-full-train-run-a6/deploy_weights/rep_yolo11s_gdut_a6_deploy.pt",
            "gdut_5class",
        ),
        (
            "YOLOv8s GDUT A6 (Comparative 5-Class Deploy)",
            "Output/GDUT_Bộ số 1_stage2-full-train-run-a6/deploy_weights/rep_yolov8s_gdut_a6_deploy.pt",
            "gdut_5class",
        ),
        (
            "YOLO11s GDUT A6 (Training Best)",
            "Output/GDUT_Bộ số 1_stage2-full-train-run-a6/GDUT_A6_CLEAN_VAL_RESULTS/YOLO11s/weights/GDUT_A6_YOLO11s_best.pt",
            "gdut_5class",
        ),
    ]
    found: List[Dict[str, Any]] = []
    for label, rel_path, taxonomy in candidates:
        p = PROJECT_ROOT / rel_path
        if p.exists():
            found.append({
                "label": label,
                "path": rel_path,
                "taxonomy": taxonomy,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            })
    return found


class SystemConfig:
    """Runtime Configuration with live mutable parameters."""

    def __init__(self):
        # 1. Model & Inference Settings
        if "MODEL_PATH" in os.environ and os.environ["MODEL_PATH"].strip():
            self.model_path: str = os.environ["MODEL_PATH"].strip()
        elif DEFAULT_MODEL_PATH.exists():
            self.model_path = str(DEFAULT_MODEL_PATH)
        elif FALLBACK_MODEL_PATH.exists():
            self.model_path = str(FALLBACK_MODEL_PATH)
        else:
            self.model_path = str(DEFAULT_MODEL_PATH)

        self.conf_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.25"))
        self.none_conf_threshold: float = float(os.getenv("NONE_CONF_THRESHOLD", "0.25"))
        self.iou_threshold: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
        self.imgsz: int = int(os.getenv("IMGSZ", "640"))
        self.device: str = os.getenv("DEVICE", "cuda:0")
        self.use_half: bool = os.getenv("USE_HALF", "False").lower() in ("true", "1", "yes")
        # Isolated RTX 3050 benchmark verified TF32 FP32 (allow_tf32=True, use_selective_amp=False)
        # is the fastest & most stable B=1 mode (11.41 ms / 87.7 FPS vs Selective AMP 11.64 ms & Strict FP32 15.20 ms).
        self.use_selective_amp: bool = os.getenv("USE_SELECTIVE_AMP", "False").lower() in ("true", "1", "yes")
        self.allow_tf32: bool = os.getenv("ALLOW_TF32", "True").lower() in ("true", "1", "yes")
        self.draw_video_hud: bool = os.getenv("DRAW_VIDEO_HUD", "False").lower() in ("true", "1", "yes")

        # 2. Tracking Settings (ByteTrack)
        self.track_high_thresh: float = float(os.getenv("TRACK_HIGH_THRESH", "0.30"))
        self.track_low_thresh: float = float(os.getenv("TRACK_LOW_THRESH", "0.15"))
        self.new_track_thresh: float = float(os.getenv("NEW_TRACK_THRESH", "0.30"))
        self.track_buffer: int = int(os.getenv("TRACK_BUFFER", "60"))
        self.match_thresh: float = float(os.getenv("MATCH_THRESH", "0.70"))
        self.min_tracklet_len: int = int(os.getenv("MIN_TRACKLET_LEN", "15"))
        self.track_grace_period: int = int(os.getenv("TRACK_GRACE_PERIOD", "2"))

        # 3. GDUT Upper-Torso Persistent Re-ID Settings
        self.gdut_reid_ttl_sec: float = float(os.getenv("GDUT_REID_TTL_SEC", "600.0"))
        self.gdut_reid_sim_thresh: float = float(os.getenv("GDUT_REID_SIM_THRESH", "0.70"))
        self.gdut_reid_use_deep: bool = os.getenv("GDUT_REID_USE_DEEP", "True").lower() in ("true", "1", "yes")

        # 4. Spatial Relational Safety Rules
        self.head_box_ratio: float = float(os.getenv("HEAD_BOX_RATIO", "0.35"))
        self.torso_box_min_ratio: float = float(os.getenv("TORSO_MIN_RATIO", "0.15"))
        self.torso_box_max_ratio: float = float(os.getenv("TORSO_MAX_RATIO", "0.75"))
        self.torso_min_ratio: float = self.torso_box_min_ratio
        self.torso_max_ratio: float = self.torso_box_max_ratio
        self.helmet_overlap_thresh: float = float(os.getenv("HELMET_OVERLAP_THRESH", "0.15"))
        self.vest_overlap_thresh: float = float(os.getenv("VEST_OVERLAP_THRESH", "0.20"))

        # 5. Debounce & Anti-Spam Cooldown Settings
        self.consecutive_violation_frames: int = int(os.getenv("CONSECUTIVE_VIOLATION_FRAMES", "8"))
        self.alert_cooldown_seconds: float = float(os.getenv("ALERT_COOLDOWN_SECONDS", "30.0"))

        # 6. Telegram Bot Credentials (loaded exclusively from env/.env)
        self.telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip().strip("'\"")
        self.telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "").strip().strip("'\"")
        self.telegram_enabled: bool = os.getenv("TELEGRAM_ENABLED", "True").lower() in ("true", "1", "yes")

        # 7. Zalo Bot Credentials (loaded exclusively from env/.env)
        self.zalo_bot_token: str = os.getenv("ZALO_BOT_TOKEN", "").strip().strip("'\"")
        self.zalo_api_base: str = os.getenv("ZALO_API_BASE", "https://bot-api.zaloplatforms.com")
        self.zalo_chat_id: str = os.getenv("ZALO_CHAT_ID", "")
        self.zalo_webhook_url: str = os.getenv("ZALO_WEBHOOK_URL", "")
        self.zalo_enabled: bool = os.getenv("ZALO_ENABLED", "True").lower() in ("true", "1", "yes")

        # 8. Video Simulation Source & Web Service
        self.video_source: str = os.getenv("VIDEO_SOURCE", str(DEFAULT_VIDEO_PATH))
        self.camera_id: str = os.getenv("CAMERA_ID", "CCTV_GATE_A1_1080P")
        self.web_host: str = os.getenv("WEB_HOST", "0.0.0.0")
        self.web_port: int = int(os.getenv("WEB_PORT", "8000"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "conf_threshold": self.conf_threshold,
            "none_conf_threshold": self.none_conf_threshold,
            "iou_threshold": self.iou_threshold,
            "imgsz": self.imgsz,
            "device": self.device,
            "use_half": self.use_half,
            "consecutive_violation_frames": self.consecutive_violation_frames,
            "alert_cooldown_seconds": self.alert_cooldown_seconds,
            "telegram_configured": bool(self.telegram_bot_token and self.telegram_chat_id),
            "zalo_configured": bool(self.zalo_bot_token and (self.zalo_chat_id or self.zalo_webhook_url)),
            "video_source": self.video_source,
            "camera_id": self.camera_id,
            "min_tracklet_len": self.min_tracklet_len,
            "track_grace_period": self.track_grace_period,
            "gdut_reid_ttl_sec": self.gdut_reid_ttl_sec,
            "gdut_reid_sim_thresh": self.gdut_reid_sim_thresh,
        }


def update_env_file(updates: Dict[str, str]) -> bool:
    """Updates key-value pairs in the .env file while preserving comments and structure."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        example_path = PROJECT_ROOT / ".env.example"
        if example_path.exists():
            import shutil
            shutil.copy(example_path, env_path)
        else:
            env_path.touch()

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
        keys_found = set()
        new_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                new_lines.append(line)
                continue
            k, _ = stripped.split("=", 1)
            k = k.strip()
            if k in updates:
                new_lines.append(f"{k}={updates[k]}")
                keys_found.add(k)
            else:
                new_lines.append(line)

        for k, v in updates.items():
            if k not in keys_found:
                new_lines.append(f"{k}={v}")

        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return True
    except Exception as e:
        print(f"⚠️ Failed to update .env: {e}")
        return False


# Global singleton configuration
config = SystemConfig()
