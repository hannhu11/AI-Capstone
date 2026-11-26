"""
src/web/app.py
==============
Dual-Mode FastAPI Enterprise Industrial VMS Server (`SERVICE_MODE=all | gateway | inference`).
- `all` (default): Unified local/edge runtime (GPU detection + Re-ID + DB + Bot Dispatch).
- `gateway`: Lightweight Control Plane (no PyTorch) with PostgreSQL DB & Circuit Breaker
  proxying to `ai-inference`.
- `inference`: Dedicated GPU Inference Server (`Rep-YOLO11s` Selective AMP + Tracking + Re-ID).
"""

from __future__ import annotations
import asyncio
import base64
from contextlib import asynccontextmanager
import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from fastapi import (
    Body,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import (
    config,
    SNAPSHOTS_DIR,
    PROJECT_ROOT,
    DEFAULT_ROLE_HIERARCHY,
    discover_available_models,
    load_role_policy,
    save_role_policy,
    update_env_file,
)
from src.db.database import db_manager

SERVICE_MODE = os.getenv("SERVICE_MODE", "all").strip().lower()
if SERVICE_MODE not in ("all", "gateway", "inference"):
    SERVICE_MODE = "all"

if SERVICE_MODE != "gateway":
    import cv2
    import numpy as np
    import torch
    from src.engine.pipeline import cctv_pipeline
else:
    cv2 = None  # type: ignore
    np = None  # type: ignore
    torch = None  # type: ignore
    cctv_pipeline = None  # type: ignore

if SERVICE_MODE != "inference":
    from src.alerting.alert_manager import alert_manager
else:
    from src.alerting.alert_manager import alert_manager

if SERVICE_MODE == "gateway":
    from src.web.gateway_client import InferenceCircuitBreakerClient

    ai_client = InferenceCircuitBreakerClient(
        base_url=os.getenv("AI_INFERENCE_URL", "http://ai-inference:8001"),
        failure_threshold=int(os.getenv("CB_FAILURE_THRESHOLD", "3")),
        recovery_timeout=float(os.getenv("CB_RECOVERY_TIMEOUT", "8.0")),
        timeout=float(os.getenv("AI_REQUEST_TIMEOUT", "4.0")),
    )
else:
    ai_client = None


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    try:
        import anyio.to_thread
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 120
    except Exception:
        pass
    yield


app = FastAPI(
    title="AEGIS-CCTV Enterprise VMS // GDUT-HWD 5-Class Surveillance",
    description=f"Rep-YOLO11s Head-Level ByteTrack, Upper-Torso Re-ID & Microservices VMS (mode={SERVICE_MODE})",
    version="2.3.0",
    lifespan=_app_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")

_METRICS_LOCK = threading.Lock()
_HTTP_REQUESTS_TOTAL = 0
_HTTP_ERRORS_TOTAL = 0
_PROCESS = psutil.Process()
_DB_SNAP_LOCK = threading.Lock()
_LAST_PERSONNEL_DB_SNAP_TS = 0.0
_BENCHMARK_CACHE_LOCK = threading.Lock()
_BENCHMARK_SUMMARY_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}


class _TelemetryASGIMiddleware:
    """Zero-copy pure ASGI middleware tracking request and 5xx error counts."""

    def __init__(self, asgi_app):
        self.app = asgi_app

    async def __call__(self, scope, receive, send):
        global _HTTP_REQUESTS_TOTAL, _HTTP_ERRORS_TOTAL
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def _send_wrapper(message):
            global _HTTP_REQUESTS_TOTAL, _HTTP_ERRORS_TOTAL
            if message["type"] == "http.response.start":
                with _METRICS_LOCK:
                    _HTTP_REQUESTS_TOTAL += 1
                    if int(message.get("status", 200)) >= 500:
                        _HTTP_ERRORS_TOTAL += 1
            await send(message)

        await self.app(scope, receive, _send_wrapper)


app.add_middleware(_TelemetryASGIMiddleware)


def _get_container_working_set_mb() -> float:
    """
    Returns container working-set memory in MB matching `docker stats` inside Linux containers
    (cgroup v2 `memory.current - inactive_file`, or cgroup v1 fallback), falling back to process RSS.
    """
    try:
        cg2_cur = Path("/sys/fs/cgroup/memory.current")
        if cg2_cur.exists():
            cur = int(cg2_cur.read_text(encoding="utf-8").strip())
            inactive_file = 0
            cg2_stat = Path("/sys/fs/cgroup/memory.stat")
            if cg2_stat.exists():
                for line in cg2_stat.read_text(encoding="utf-8").splitlines():
                    if line.startswith("inactive_file "):
                        inactive_file = int(line.split()[1])
                        break
            return round(max(0, cur - inactive_file) / (1024 * 1024), 1)
        cg1_cur = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if cg1_cur.exists():
            cur = int(cg1_cur.read_text(encoding="utf-8").strip())
            inactive_file = 0
            cg1_stat = Path("/sys/fs/cgroup/memory/memory.stat")
            if cg1_stat.exists():
                for line in cg1_stat.read_text(encoding="utf-8").splitlines():
                    if line.startswith("total_inactive_file "):
                        inactive_file = int(line.split()[1])
                        break
            return round(max(0, cur - inactive_file) / (1024 * 1024), 1)
    except Exception:
        pass
    return round(_PROCESS.memory_info().rss / (1024 * 1024), 1)


def _collect_host_telemetry() -> Dict[str, Any]:
    mem_mb = _get_container_working_set_mb()
    cpu_pct = round(psutil.cpu_percent(interval=None), 1)
    vram_mb = (
        round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
        if (torch is not None and torch.cuda.is_available())
        else 0.0
    )
    return {"cpu_percent": cpu_pct, "ram_used_mb": mem_mb, "cuda_vram_mb": vram_mb}


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


# ------------------------------------------------------------------
# Pydantic Payloads
# ------------------------------------------------------------------
class SettingsPayload(BaseModel):
    conf_threshold: Optional[float] = None
    none_conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    consecutive_violation_frames: Optional[int] = None
    alert_cooldown_seconds: Optional[float] = None
    telegram_chat_id: Optional[str] = None
    zalo_chat_id: Optional[str] = None
    zalo_webhook_url: Optional[str] = None


class SwitchSourcePayload(BaseModel):
    video_source: Optional[str] = None
    source: Optional[str] = None

    @property
    def target_source(self) -> str:
        return self.video_source or self.source or ""


class SwitchModelPayload(BaseModel):
    model_path: str


class TelegramTestPayload(BaseModel):
    chat_id: Optional[str] = None


class ZaloTestPayload(BaseModel):
    webhook_url: Optional[str] = None
    chat_id: Optional[str] = None


class FocusPayload(BaseModel):
    track_id: Optional[int] = None


class FilterPayload(BaseModel):
    colors: Optional[List[str]] = None
    roles: Optional[List[str]] = None
    violators_only: Optional[bool] = None
    clear: bool = False


class LabelPayload(BaseModel):
    track_id: int
    label: str = ""


class EnrollPayload(BaseModel):
    track_id: int
    label: Optional[str] = None


class PlayPausePayload(BaseModel):
    paused: Optional[bool] = None


class SeekPayload(BaseModel):
    frame: Optional[int] = None
    seconds: Optional[float] = None
    delta_seconds: Optional[float] = None
    percent: Optional[float] = None


class SpeedPayload(BaseModel):
    speed: float = 1.0


class ResolutionPayload(BaseModel):
    resolution: str = "original"


class FeedbackPayload(BaseModel):
    item_id: str
    worker_id: Optional[int] = None
    verdict: str  # "confirm" | "false_positive" | "edit_label"
    corrected_label: Optional[str] = None
    note: Optional[str] = None
    snapshot_filename: Optional[str] = None
    source_type: str = "incident"


def _proxy_or_degraded(
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    fallback: Optional[Any] = None,
    files: Optional[Dict[str, Any]] = None,
) -> Any:
    if ai_client is None:
        raise HTTPException(status_code=500, detail="Gateway client uninitialized")
    try:
        res = ai_client.request_json(method, path, json_body=body, files=files)
        if isinstance(res, dict) and "_http_error" in res:
            raise HTTPException(status_code=int(res["_http_error"]), detail=res.get("detail", "Upstream error"))
        return res
    except HTTPException:
        raise
    except Exception as exc:
        if fallback is not None:
            if isinstance(fallback, list):
                return list(fallback)
            out = dict(fallback)
            out["degraded"] = True
            out["degraded_reason"] = str(exc)
            out["circuit_breaker"] = ai_client.status_dict()
            return out
        raise HTTPException(status_code=503, detail=str(exc))


# ------------------------------------------------------------------
# 1. Dashboard UI & MJPEG Video Feed
# ------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = TEMPLATES_DIR / "dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard template not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


def mjpeg_frame_generator():
    if cctv_pipeline is None:
        return
    last_idx = -1
    while cctv_pipeline.is_running:
        jpeg_bytes, current_idx = cctv_pipeline.get_latest_frame(last_frame_idx=last_idx, timeout=0.5)
        if jpeg_bytes is None or current_idx == last_idx:
            time.sleep(0.01)
            continue
        last_idx = current_idx
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"


@app.get("/video_feed")
async def video_feed():
    if SERVICE_MODE == "gateway":
        raise HTTPException(status_code=307, headers={"Location": os.getenv("AI_INFERENCE_PUBLIC_FEED", "/video_feed")})
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ------------------------------------------------------------------
# 2. Telemetry, Status & Models API
# ------------------------------------------------------------------
@app.get("/api/metrics")
def get_metrics():
    if SERVICE_MODE == "gateway":
        fallback = {
            "status": "degraded",
            "fps": 0.0,
            "inference_ms": 0.0,
            "workers_count": 0,
            "compliant_count": 0,
            "violations_count": 0,
            "compliance_rate": 100.0,
            "gallery_total_count": 0,
            "workers": [],
            "helmet_colors": {"blue": 0, "white": 0, "yellow": 0, "red": 0, "none": 0},
            "role_distribution": {},
            "active_tracks": [],
            "role_policy": load_role_policy(),
        }
        return JSONResponse(content=_proxy_or_degraded("GET", "/api/metrics", fallback=fallback))
    return JSONResponse(content=cctv_pipeline.get_telemetry())


@app.get("/api/status")
def get_status():
    db_ok = db_manager.is_healthy()
    if SERVICE_MODE == "gateway":
        fallback = {
            "status": "degraded",
            "service_mode": SERVICE_MODE,
            "device": "CPU/OFFLINE",
            "cuda_available": False,
            "gpu_name": "Unavailable (AI Inference Degraded)",
            "model_path": config.model_path,
            "video_source": config.video_source,
            "classes": {0: "blue", 1: "white", 2: "yellow", 3: "red", 4: "none"},
            "is_helmet_only": True,
            "available_models": discover_available_models(),
            "role_policy": db_manager.get_role_policies() or load_role_policy(),
            "db_backend": db_manager.backend_name,
            "db_healthy": db_ok,
            "config": config.to_dict(),
        }
        data = _proxy_or_degraded("GET", "/api/status", fallback=fallback)
        data["service_mode"] = SERVICE_MODE
        data["db_backend"] = db_manager.backend_name
        data["db_healthy"] = db_ok
        if ai_client is not None:
            data["circuit_breaker"] = ai_client.status_dict()
        return data

    return {
        "status": "online",
        "service_mode": SERVICE_MODE,
        "device": cctv_pipeline.detector.device.upper(),
        "use_selective_amp": bool(getattr(cctv_pipeline.detector, "use_selective_amp", False)),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "model_path": cctv_pipeline.detector.model_path,
        "video_source": str(cctv_pipeline.source_path),
        "classes": cctv_pipeline.detector.class_names,
        "is_helmet_only": getattr(cctv_pipeline.detector, "is_helmet_only", False),
        "available_models": discover_available_models(),
        "role_policy": cctv_pipeline.get_role_policy(),
        "db_backend": db_manager.backend_name,
        "db_healthy": db_ok,
        "config": config.to_dict(),
    }


@app.get("/api/models")
def list_models():
    active = config.model_path if SERVICE_MODE == "gateway" else cctv_pipeline.detector.model_path
    return {
        "status": "ok",
        "active_model": active,
        "models": discover_available_models(),
    }


# ------------------------------------------------------------------
# 3. Alerts API
# ------------------------------------------------------------------
@app.get("/api/alerts")
async def get_alerts():
    if SERVICE_MODE == "gateway":
        mem_alerts = alert_manager.get_recent_alerts(limit=50)
        db_alerts = db_manager.list_incidents(limit=50)
        combined = mem_alerts if mem_alerts else db_alerts
        try:
            ai_alerts = _proxy_or_degraded("GET", "/api/alerts", fallback=[])
            if isinstance(ai_alerts, list) and ai_alerts:
                for ev in ai_alerts[:20]:
                    if isinstance(ev, dict) and ev.get("alert_id"):
                        db_manager.save_incident(ev)
                combined = ai_alerts
        except Exception:
            pass
        return JSONResponse(content=combined)
    return JSONResponse(content=alert_manager.get_recent_alerts(limit=50))


@app.post("/api/alerts/clear")
async def clear_alerts():
    alert_manager.clear_alerts()
    db_manager.clear_incidents()
    if SERVICE_MODE == "gateway":
        _ = _proxy_or_degraded("POST", "/api/alerts/clear", fallback={"message": "Cleared gateway buffer"})
    return {"message": "Alert history cleared"}


@app.post("/internal/incidents")
def ingest_internal_incident(payload: Dict[str, Any] = Body(...)):
    """Internal ingestion endpoint from `ai-inference` -> `api-gateway` -> PostgreSQL + Telegram/Zalo Bots."""
    rec = dict(payload)
    saved = db_manager.save_incident(rec)
    rec["_from_internal"] = True
    alert_manager.enqueue_alert(rec)
    return {"status": "ok", "saved": saved, "alert_id": rec.get("alert_id")}


# ------------------------------------------------------------------
# 4. Settings & Role-by-Color Hierarchy API
# ------------------------------------------------------------------
@app.post("/api/settings")
async def update_settings(payload: SettingsPayload):
    env_updates = {}
    if payload.conf_threshold is not None:
        config.conf_threshold = payload.conf_threshold
        env_updates["CONF_THRESHOLD"] = str(payload.conf_threshold)
    if payload.none_conf_threshold is not None:
        config.none_conf_threshold = payload.none_conf_threshold
        env_updates["NONE_CONF_THRESHOLD"] = str(payload.none_conf_threshold)
    if payload.iou_threshold is not None:
        config.iou_threshold = payload.iou_threshold
        env_updates["IOU_THRESHOLD"] = str(payload.iou_threshold)
    if payload.consecutive_violation_frames is not None:
        config.consecutive_violation_frames = payload.consecutive_violation_frames
        env_updates["CONSECUTIVE_VIOLATION_FRAMES"] = str(payload.consecutive_violation_frames)
    if payload.alert_cooldown_seconds is not None:
        config.alert_cooldown_seconds = payload.alert_cooldown_seconds
        env_updates["ALERT_COOLDOWN_SECONDS"] = str(payload.alert_cooldown_seconds)
    if payload.telegram_chat_id is not None:
        config.telegram_chat_id = payload.telegram_chat_id
        alert_manager.telegram.chat_id = payload.telegram_chat_id
        env_updates["TELEGRAM_CHAT_ID"] = payload.telegram_chat_id
    if payload.zalo_chat_id is not None:
        config.zalo_chat_id = payload.zalo_chat_id
        alert_manager.zalo.chat_id = payload.zalo_chat_id
        env_updates["ZALO_CHAT_ID"] = payload.zalo_chat_id
    if payload.zalo_webhook_url is not None:
        val = payload.zalo_webhook_url.strip()
        if val.startswith("http://") or val.startswith("https://"):
            config.zalo_webhook_url = val
            alert_manager.zalo.webhook_url = val
            env_updates["ZALO_WEBHOOK_URL"] = val
        elif val:
            config.zalo_chat_id = val
            alert_manager.zalo.chat_id = val
            env_updates["ZALO_CHAT_ID"] = val

    if env_updates:
        update_env_file(env_updates)

    if SERVICE_MODE == "gateway":
        _proxy_or_degraded("POST", "/api/settings", body=payload.model_dump(), fallback={"ok": True})
    else:
        if payload.conf_threshold is not None:
            cctv_pipeline.detector.conf_thresh = payload.conf_threshold
        if payload.none_conf_threshold is not None:
            cctv_pipeline.detector.none_conf_thresh = payload.none_conf_threshold
            eff_thresh = min(float(config.track_high_thresh), float(payload.none_conf_threshold))
            cctv_pipeline.tracker.track_thresh = eff_thresh
            cctv_pipeline.tracker.new_track_thresh = min(
                float(config.new_track_thresh), float(payload.none_conf_threshold)
            )
        if payload.iou_threshold is not None:
            cctv_pipeline.detector.iou_thresh = payload.iou_threshold
        if payload.consecutive_violation_frames is not None:
            cctv_pipeline.safety_engine.consecutive_frames_thresh = payload.consecutive_violation_frames
        if payload.alert_cooldown_seconds is not None:
            cctv_pipeline.safety_engine.cooldown_seconds = payload.alert_cooldown_seconds

    return {"message": "Settings updated and saved to .env successfully", "config": config.to_dict()}


@app.get("/api/roles")
def get_role_policy():
    if SERVICE_MODE == "gateway":
        roles = db_manager.get_role_policies() or load_role_policy()
        return {"status": "ok", "roles": roles, "defaults": DEFAULT_ROLE_HIERARCHY}
    return {
        "status": "ok",
        "roles": cctv_pipeline.get_role_policy(),
        "defaults": DEFAULT_ROLE_HIERARCHY,
    }


@app.post("/api/roles")
def update_role_policy(payload: Dict[str, Any] = Body(...)):
    updates = payload.get("roles", payload) if isinstance(payload, dict) else {}
    if not isinstance(updates, dict) or not updates:
        raise HTTPException(status_code=400, detail="Expected role policy mapping dictionary")
    if SERVICE_MODE == "gateway":
        updated = save_role_policy(updates)
        db_manager.upsert_role_policies(updated)
        _proxy_or_degraded("POST", "/api/roles", body=updated, fallback={"status": "degraded"})
        return {"status": "ok", "roles": updated}

    updated = cctv_pipeline.set_role_policy(updates)
    db_manager.upsert_role_policies(updated)
    return {"status": "ok", "roles": updated}


# ------------------------------------------------------------------
# 5. Video Sources, Upload & Transport Controls
# ------------------------------------------------------------------
@app.get("/api/videos")
@app.get("/api/sources")
async def list_available_videos():
    video_dir = PROJECT_ROOT / "video_test"
    videos = []
    if video_dir.exists():
        for vp in sorted(video_dir.glob("*.mp4")):
            videos.append({
                "filename": vp.name,
                "path": f"video_test/{vp.name}",
                "size_mb": round(vp.stat().st_size / (1024 * 1024), 2),
            })
    active_src = config.video_source if SERVICE_MODE == "gateway" else str(cctv_pipeline.source_path)
    return {"videos": videos, "active_source": active_src}


@app.post("/api/switch_source")
@app.post("/api/controls/source")
def switch_video_source(payload: SwitchSourcePayload):
    target = payload.target_source
    if not target:
        raise HTTPException(status_code=400, detail="Missing 'video_source' or 'source' field in request body")
    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded("POST", "/api/switch_source", body={"video_source": target})
    cctv_pipeline.switch_source(target)
    alert_manager.clear_alerts()
    return {
        "message": f"Switched video source to {target}",
        "alerts_cleared": True,
        "playback_state": cctv_pipeline.get_playback_state(),
    }


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    save_dir = PROJECT_ROOT / "video_test"
    save_dir.mkdir(exist_ok=True)
    safe_name = Path(file.filename or "uploaded_clip.mp4").name
    dest_path = save_dir / safe_name

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if SERVICE_MODE == "gateway":
        _proxy_or_degraded("POST", "/api/switch_source", body={"video_source": f"video_test/{safe_name}"})
    else:
        cctv_pipeline.switch_source(str(dest_path))
        alert_manager.clear_alerts()
    return {"message": "Uploaded and loaded video successfully", "filename": safe_name, "path": f"video_test/{safe_name}"}


@app.post("/api/video/play_pause")
def video_play_pause(payload: Optional[PlayPausePayload] = Body(default=None)):
    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded("POST", "/api/video/play_pause", body=payload.model_dump() if payload else {})
    paused_arg = payload.paused if payload is not None else None
    cctv_pipeline.set_paused(paused_arg)
    return {"status": "ok", **cctv_pipeline.get_playback_state()}


@app.post("/api/video/seek")
def video_seek(payload: SeekPayload):
    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded("POST", "/api/video/seek", body=payload.model_dump())
    state = cctv_pipeline.seek_to(
        frame=payload.frame,
        seconds=payload.seconds,
        delta_seconds=payload.delta_seconds,
        percent=payload.percent,
    )
    return {"status": "ok", **state}


@app.post("/api/video/speed")
def video_speed(payload: SpeedPayload):
    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded("POST", "/api/video/speed", body=payload.model_dump())
    cctv_pipeline.set_speed(payload.speed)
    return {"status": "ok", **cctv_pipeline.get_playback_state()}


@app.post("/api/video/resolution")
def video_resolution(payload: ResolutionPayload):
    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded("POST", "/api/video/resolution", body=payload.model_dump())
    cctv_pipeline.set_resolution(payload.resolution)
    return {"status": "ok", **cctv_pipeline.get_playback_state()}


@app.get("/api/video/playback_state")
def get_video_playback_state():
    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded(
            "GET",
            "/api/video/playback_state",
            fallback={"status": "degraded", "is_paused": True, "current_frame": 0, "total_frames": 0},
        )
    return {"status": "ok", **cctv_pipeline.get_playback_state()}


# ------------------------------------------------------------------
# 6. Tracking Registry, Solo Focus, Multi-Dimensional Filter, Enroll & CSV
# ------------------------------------------------------------------
@app.post("/api/tracking/focus")
def set_tracking_focus(payload: FocusPayload):
    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded("POST", "/api/tracking/focus", body=payload.model_dump())
    active_id = cctv_pipeline.set_focus_track(payload.track_id)
    return {"status": "ok", "focus_track_id": active_id}


@app.get("/api/tracking/filter")
def get_tracking_filter():
    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded("GET", "/api/tracking/filter", fallback={"status": "degraded", "colors": [], "roles": [], "violators_only": False})
    return {"status": "ok", **cctv_pipeline.get_filter_state()}


@app.post("/api/tracking/filter")
def set_tracking_filter(payload: FilterPayload):
    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded("POST", "/api/tracking/filter", body=payload.model_dump())
    state = cctv_pipeline.set_filter(
        colors=payload.colors,
        roles=payload.roles,
        violators_only=payload.violators_only,
        clear=payload.clear,
    )
    return {"status": "ok", **state}


@app.post("/api/tracking/label")
def set_tracking_label(payload: LabelPayload):
    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded("POST", "/api/tracking/label", body=payload.model_dump())
    ok = cctv_pipeline.set_track_label(payload.track_id, payload.label)
    return {"status": "ok", "updated": ok, "track_id": payload.track_id, "label": payload.label.strip()}


@app.post("/api/tracking/enroll")
def enroll_tracking_identity(payload: EnrollPayload):
    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded("POST", "/api/tracking/enroll", body=payload.model_dump())
    ok = cctv_pipeline.enroll_track(payload.track_id, label=payload.label)
    return {"status": "ok", "enrolled": ok, "track_id": payload.track_id}


@app.get("/api/tracking/workers")
def get_tracked_workers():
    global _LAST_PERSONNEL_DB_SNAP_TS
    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded(
            "GET",
            "/api/tracking/workers",
            fallback={"status": "degraded", "workers": [], "total": 0, "focus_track_id": None},
        )
    workers = cctv_pipeline.get_registered_workers()
    now_ts = time.time()
    if now_ts - _LAST_PERSONNEL_DB_SNAP_TS >= 10.0:
        with _DB_SNAP_LOCK:
            if now_ts - _LAST_PERSONNEL_DB_SNAP_TS >= 10.0:
                _LAST_PERSONNEL_DB_SNAP_TS = now_ts
                src_str = str(getattr(cctv_pipeline, "source_path", "") or "")
                sess_id = f"{getattr(cctv_pipeline, 'source_type', 'video')}:{Path(src_str).name}" if src_str else "default"
                db_manager.save_personnel_snapshot(workers, session_id=sess_id, source_name=src_str)
    return {
        "status": "ok",
        "workers": workers,
        "total": len(workers),
        "focus_track_id": cctv_pipeline.focus_track_id,
        "filter_state": cctv_pipeline.get_filter_state(),
    }


@app.get("/api/tracking/export_csv")
def export_tracked_workers_csv():
    if SERVICE_MODE == "gateway":
        res = _proxy_or_degraded("GET", "/api/tracking/workers", fallback={"workers": []})
        import csv as _csv, io as _io
        buf = _io.StringIO()
        wr = _csv.writer(buf)
        wr.writerow(["track_id", "custom_label", "current_class", "helmet_color", "historical_helmet_color", "role_title", "status", "total_frames_seen"])
        for w in res.get("workers", []):
            wr.writerow([
                w.get("track_id"), w.get("custom_label") or "", w.get("current_class") or "",
                w.get("helmet_color") or "", w.get("historical_helmet_color") or "",
                w.get("historical_role_title") or w.get("role_title") or "",
                w.get("status") or "", w.get("total_frames_seen", 0),
            ])
        csv_text = buf.getvalue()
    else:
        csv_text = cctv_pipeline.export_personnel_csv()

    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="aegis_personnel_registry.csv"'},
    )


@app.post("/api/tracking/reset")
def reset_tracking_registry():
    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded("POST", "/api/tracking/reset")
    cctv_pipeline.reset_tracking_registry()
    return {"status": "ok", "message": "Worker tracking registry reset"}


@app.post("/api/webhook/zalo")
async def zalo_webhook_receiver(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    return {"status": "received", "data": data}


# ------------------------------------------------------------------
# 7. High-FPS Browser Webcam WebSocket `/ws/webcam` (Offloaded Threadpool)
# ------------------------------------------------------------------
def _process_ws_frame_sync(
    frame_bytes: bytes,
    device_label: str,
    resolution: Optional[str],
    return_image: bool,
) -> Optional[Dict[str, Any]]:
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return None

    res = cctv_pipeline.process_external_frame(
        frame_bgr,
        camera_id=device_label,
        resolution=resolution,
        return_annotated=return_image,
    )
    tele = res["telemetry"]
    payload: Dict[str, Any] = {
        "status": "ok",
        "fps": tele["fps"],
        "inference_ms": tele["inference_ms"],
        "active_tracks": res["active_tracks"],
        "personnel": res["personnel"],
        "focus_track_id": tele["focus_track_id"],
        "filter_state": tele["filter_state"],
        "kpis": {
            "workers_count": tele["workers_count"],
            "compliant_count": tele["compliant_count"],
            "violations_count": tele["violations_count"],
            "compliance_rate": tele["compliance_rate"],
            "gallery_total_count": tele["gallery_total_count"],
            "helmet_colors": tele["helmet_colors"],
            "role_distribution": tele["role_distribution"],
        },
        "new_alerts": alert_manager.get_recent_alerts(limit=10),
    }

    if return_image and res.get("annotated_frame") is not None:
        ok_enc, buf = cv2.imencode(".jpg", res["annotated_frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok_enc:
            payload["image"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    return payload


@app.websocket("/ws/webcam")
async def webcam_websocket(websocket: WebSocket):
    if cctv_pipeline is None:
        await websocket.close(code=1013)
        return
    await websocket.accept()
    query_params = websocket.query_params
    q_res = query_params.get("resolution")
    q_dev = str(query_params.get("device_label") or "C960 Ultra")[:48]
    q_ret_img = query_params.get("return_image", "false").lower() in ("1", "true", "yes")

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            resolution = q_res
            device_label = q_dev
            return_image = q_ret_img
            frame_bytes: Optional[bytes] = None

            if msg.get("bytes") is not None:
                frame_bytes = msg["bytes"]
            elif msg.get("text") is not None:
                raw_txt = msg["text"]
                return_image = True  # Default JSON Text payload behavior for backward compatibility
                try:
                    payload = json.loads(raw_txt)
                    # Allow updating metadata without sending image
                    if "resolution" in payload:
                        q_res = payload.get("resolution")
                        resolution = q_res
                    if "device_label" in payload:
                        q_dev = str(payload.get("device_label") or "C960 Ultra")[:48]
                        device_label = q_dev
                    if "return_image" in payload:
                        q_ret_img = bool(payload["return_image"])
                        return_image = q_ret_img
                    img_data = str(payload.get("image") or "")
                except Exception:
                    img_data = raw_txt

                if img_data:
                    if "," in img_data and img_data.startswith("data:image"):
                        img_data = img_data.split(",", 1)[1]
                    try:
                        frame_bytes = base64.b64decode(img_data)
                    except Exception:
                        frame_bytes = None
                elif msg.get("text") and "image" not in msg["text"]:
                    await websocket.send_json({"status": "meta_updated"})
                    continue

            if not frame_bytes:
                await websocket.send_json({"status": "error", "detail": "Empty or malformed frame payload"})
                continue

            # Offload heavy OpenCV decode + GPU inference + Re-ID off asyncio event loop
            out_payload = await asyncio.to_thread(
                _process_ws_frame_sync, frame_bytes, device_label, resolution, return_image
            )
            if out_payload is None:
                await websocket.send_json({"status": "error", "detail": "Failed to decode JPEG frame"})
                continue

            await websocket.send_json(out_payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


# ------------------------------------------------------------------
# 8. Model Checkpoint Hot-Swap API
# ------------------------------------------------------------------
@app.post("/api/switch_model")
def switch_model(payload: SwitchModelPayload):
    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded("POST", "/api/switch_model", body=payload.model_dump())
    try:
        cctv_pipeline.switch_model(payload.model_path)
        return {
            "status": "success",
            "message": f"Successfully loaded model: {payload.model_path}",
            "model_path": payload.model_path,
            "classes": cctv_pipeline.detector.class_names,
            "is_helmet_only": cctv_pipeline.detector.is_helmet_only,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")


# ------------------------------------------------------------------
# 9. Bot Integration Test Endpoints
# ------------------------------------------------------------------
@app.get("/api/telegram/discover_chat_id")
async def discover_tg_chat_id():
    chat_id = alert_manager.telegram.discover_latest_chat_id()
    return {"chat_id": chat_id, "found": bool(chat_id)}


@app.get("/api/zalo/discover_chat_id")
async def discover_zalo_chat_id():
    chat_id = alert_manager.zalo.discover_latest_chat_id()
    return {"chat_id": chat_id, "found": bool(chat_id)}


@app.post("/api/alerts/test-telegram")
@app.post("/api/test_alert/telegram")
async def test_telegram_alert(payload: Optional[TelegramTestPayload] = None):
    chat_id = payload.chat_id if payload and payload.chat_id else None
    if not chat_id:
        chat_id = config.telegram_chat_id or alert_manager.telegram.discover_latest_chat_id()

    test_msg = (
        "🔔 *[TEST ALERT] AEGIS-CCTV Enterprise VMS*\n\n"
        "Automated health ping from GDUT-HWD 5-Class Surveillance Engine.\n"
        "Connection to Telegram Bot verified! ✅"
    )
    ok, msg = alert_manager.telegram.send_message(test_msg, chat_id=chat_id)
    return {"success": ok, "message": msg, "chat_id": chat_id}


@app.post("/api/alerts/test-zalo")
@app.post("/api/test_alert/zalo")
async def test_zalo_alert(payload: Optional[ZaloTestPayload] = None):
    if payload:
        if payload.webhook_url:
            val = payload.webhook_url.strip()
            if val.startswith("http://") or val.startswith("https://"):
                alert_manager.zalo.webhook_url = val
            else:
                alert_manager.zalo.chat_id = val
        if payload.chat_id:
            alert_manager.zalo.chat_id = payload.chat_id

    test_msg = (
        "[TEST ALERT - AEGIS-CCTV ENTERPRISE VMS]\n"
        "He thong giam sat mu bao ho GDUT-HWD 5 lop da ket noi thanh cong den Zalo Bot Platform!"
    )
    ok, msg = alert_manager.zalo.send_message(test_msg)
    return {"success": ok, "message": msg}


# ------------------------------------------------------------------
# 10. Single-Image Audit Endpoint
# ------------------------------------------------------------------
@app.post("/api/predict_image")
def predict_single_image(file: UploadFile = File(...)):
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload")

    if SERVICE_MODE == "gateway":
        return _proxy_or_degraded(
            "POST",
            "/api/predict_image",
            files={"file": (file.filename or "audit.jpg", raw, file.content_type or "image/jpeg")},
        )

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="Unsupported or corrupted image file")

    detections, _, infer_ms = cctv_pipeline.detector.predict(frame_bgr)
    role_policy = cctv_pipeline.get_role_policy()
    annotated = frame_bgr.copy()
    enriched_dets = []

    for d in detections:
        box = d.get("bbox") or d.get("box") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cls_name = str(d.get("name", d.get("cls", "obj"))).lower()
        conf = float(d.get("conf", 0.0))
        is_viol = any(k in cls_name for k in ("none", "no_helmet", "no_vest"))
        role_info = role_policy.get(cls_name, {})
        color = tuple(role_info.get("bgr_color", [38, 38, 220] if is_viol else [21, 128, 61]))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        role_tag = role_info.get("short_code", cls_name.upper())
        label = f"{cls_name.upper()} ({role_tag}) {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        y_top = max(0, y1 - th - 8)
        cv2.rectangle(annotated, (x1, y_top), (x1 + tw + 8, max(y1, y_top + th + 6)), (23, 23, 17), -1)
        cv2.rectangle(annotated, (x1, y_top), (x1 + tw + 8, max(y1, y_top + th + 6)), color, 1)
        cv2.putText(annotated, label, (x1 + 4, max(th + 2, y_top + th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (248, 250, 252), 1, cv2.LINE_AA)

        enriched = dict(d)
        enriched["role_title"] = role_info.get("role_title", cls_name)
        enriched["role_short_code"] = role_tag
        enriched_dets.append(enriched)

    ok_enc, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok_enc:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image")
    b64_img = base64.b64encode(buf.tobytes()).decode("ascii")
    return {
        "status": "ok",
        "filename": file.filename,
        "count": len(enriched_dets),
        "inference_ms": round(float(infer_ms), 2),
        "model_path": cctv_pipeline.detector.model_path,
        "detections": enriched_dets,
        "annotated_image_b64": f"data:image/jpeg;base64,{b64_img}",
    }


# ------------------------------------------------------------------
# 11. Human-in-the-Loop Operator Feedback
# ------------------------------------------------------------------
FEEDBACK_LOG_PATH = PROJECT_ROOT / "Output" / "operator_feedback.jsonl"


@app.post("/api/feedback")
def submit_operator_feedback(payload: FeedbackPayload):
    allowed = {"confirm", "false_positive", "edit_label"}
    if payload.verdict not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid verdict; expected one of {sorted(allowed)}")

    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    active_model = config.model_path if SERVICE_MODE == "gateway" else cctv_pipeline.detector.model_path
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "item_id": payload.item_id,
        "worker_id": payload.worker_id,
        "verdict": payload.verdict,
        "corrected_label": payload.corrected_label,
        "note": payload.note,
        "snapshot_filename": payload.snapshot_filename,
        "source_type": payload.source_type,
        "model_path": active_model,
    }
    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    db_manager.save_feedback(record)
    return {"status": "saved", "record": record}


@app.get("/api/feedback")
def list_operator_feedback(limit: int = 50):
    records = []
    if FEEDBACK_LOG_PATH.exists():
        lines = FEEDBACK_LOG_PATH.read_text(encoding="utf-8").splitlines()
        for line in lines[-limit:]:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    if not records and SERVICE_MODE == "gateway":
        records = db_manager.list_feedback(limit=limit)
        records = list(reversed(records))

    summary = {
        "total": len(records),
        "confirmed": sum(1 for r in records if r.get("verdict") == "confirm"),
        "false_positive": sum(1 for r in records if r.get("verdict") == "false_positive"),
        "edited": sum(1 for r in records if r.get("verdict") == "edit_label"),
    }
    return {"status": "ok", "summary": summary, "records": list(reversed(records))}


# ------------------------------------------------------------------
# 12. Optimization & Runtime Benchmark Summary Endpoint
# ------------------------------------------------------------------
@app.get("/api/benchmark/summary")
def get_benchmark_summary():
    gdut_report = PROJECT_ROOT / "Output" / "GDUT_Bộ số 1_stage2-full-train-run-a6" / "deploy_weights"
    now_ts = time.time()
    cached = _BENCHMARK_SUMMARY_CACHE.get("data")
    if cached is not None and (now_ts - float(_BENCHMARK_SUMMARY_CACHE.get("ts", 0.0))) < 2.0:
        static_payload = cached
    else:
        with _BENCHMARK_CACHE_LOCK:
            cached = _BENCHMARK_SUMMARY_CACHE.get("data")
            if cached is not None and (time.time() - float(_BENCHMARK_SUMMARY_CACHE.get("ts", 0.0))) < 2.0:
                static_payload = cached
            else:
                rt_gdut_json = PROJECT_ROOT / "Output" / "operational_multistream" / "rtx3050_gdut_a6_batched_multistream_results.json"
                rt_fallback_json = PROJECT_ROOT / "Output" / "operational_multistream" / "rtx3050_batched_multistream_results.json"
                rt_json = rt_gdut_json if rt_gdut_json.exists() else rt_fallback_json

                load_json = PROJECT_ROOT / "Output" / "load_test_results.json"
                docker_load_json = PROJECT_ROOT / "Output" / "load_test_enterprise_docker.json"
                opt_json = PROJECT_ROOT / "Output" / "task3_eval_opt" / "optimization_benchmark_table.json"
                reparam_json = PROJECT_ROOT / "Output" / "task3_eval_opt" / "gdut_a6_switch_to_deploy_benchmark.json"
                e2e_json = PROJECT_ROOT / "Output" / "task3_eval_opt" / "gdut_vms_end2end_reid_benchmark.json"

                multistream = []
                if rt_json.exists():
                    try:
                        multistream = json.loads(rt_json.read_text(encoding="utf-8"))
                    except Exception:
                        multistream = []

                load_test = []
                if load_json.exists():
                    try:
                        load_test = json.loads(load_json.read_text(encoding="utf-8"))
                    except Exception:
                        load_test = []

                docker_load_test = []
                if docker_load_json.exists():
                    try:
                        docker_load_test = json.loads(docker_load_json.read_text(encoding="utf-8"))
                    except Exception:
                        docker_load_test = []

                opt_records = []
                parity_info: Dict[str, Any] = {}
                if opt_json.exists():
                    try:
                        parsed_opt = json.loads(opt_json.read_text(encoding="utf-8"))
                        all_opt = parsed_opt.get("records", [])
                        gdut_only = [r for r in all_opt if "GDUT" in str(r.get("model", ""))]
                        opt_records = gdut_only if gdut_only else all_opt
                        parity_info = parsed_opt.get("video_parity_fp32_vs_amp_fp16", {})
                    except Exception:
                        opt_records = []

                reparam_gdut = {}
                if reparam_json.exists():
                    try:
                        reparam_gdut = json.loads(reparam_json.read_text(encoding="utf-8"))
                    except Exception:
                        reparam_gdut = {}

                e2e_reid = {}
                if e2e_json.exists():
                    try:
                        e2e_reid = json.loads(e2e_json.read_text(encoding="utf-8"))
                    except Exception:
                        e2e_reid = {}

                static_payload = {
                    "optimization_before_after": opt_records,
                    "video_parity": parity_info,
                    "gdut_switch_to_deploy": reparam_gdut,
                    "gdut_end2end_reid": e2e_reid,
                    "multistream_rtx3050": multistream,
                    "api_load_test": load_test,
                    "enterprise_docker_load_test": docker_load_test,
                    "capstone_gdut_deploy_ready": gdut_report.exists(),
                }
                _BENCHMARK_SUMMARY_CACHE["ts"] = time.time()
                _BENCHMARK_SUMMARY_CACHE["data"] = static_payload

    if SERVICE_MODE == "gateway":
        ai_status = _proxy_or_degraded("GET", "/api/status", fallback={})
        active_model_info = {
            "model_path": ai_status.get("model_path", config.model_path),
            "device": ai_status.get("device", "UNKNOWN"),
            "use_half": False,
            "use_selective_amp": ai_status.get("use_selective_amp", False),
            "is_helmet_only": ai_status.get("is_helmet_only", True),
            "classes": ai_status.get("classes", {}),
        }
    else:
        active_model_info = {
            "model_path": cctv_pipeline.detector.model_path,
            "device": cctv_pipeline.detector.device.upper(),
            "use_half": bool(getattr(cctv_pipeline.detector, "use_half", False)),
            "use_selective_amp": bool(getattr(cctv_pipeline.detector, "use_selective_amp", False)),
            "is_helmet_only": bool(getattr(cctv_pipeline.detector, "is_helmet_only", False)),
            "classes": cctv_pipeline.detector.class_names,
        }

    return {
        "status": "ok",
        "active_model": active_model_info,
        "host_telemetry": _collect_host_telemetry(),
        **static_payload,
    }


# ------------------------------------------------------------------
# 13. Prometheus Monitoring Metrics Endpoint (100% Aligned with Grafana)
# ------------------------------------------------------------------
@app.get("/metrics")
def prometheus_metrics():
    host = _collect_host_telemetry()
    db_ok = 1 if db_manager.is_healthy() else 0
    cb_info = ai_client.status_dict() if ai_client is not None else {"is_open": False, "last_latency_ms": 0.0}

    if SERVICE_MODE == "gateway":
        lines = [
            "# HELP aegis_gateway_http_requests_total Total HTTP requests handled by API Gateway",
            "# TYPE aegis_gateway_http_requests_total counter",
            f"aegis_gateway_http_requests_total {_HTTP_REQUESTS_TOTAL}",
            "# HELP aegis_gateway_http_errors_total Total HTTP 5xx responses handled by API Gateway",
            "# TYPE aegis_gateway_http_errors_total counter",
            f"aegis_gateway_http_errors_total {_HTTP_ERRORS_TOTAL}",
            "# HELP aegis_gateway_circuit_breaker_open Whether AI Inference circuit breaker is OPEN (1=open, 0=closed)",
            "# TYPE aegis_gateway_circuit_breaker_open gauge",
            f"aegis_gateway_circuit_breaker_open {1 if cb_info['is_open'] else 0}",
            "# HELP aegis_gateway_db_connected Database health status (1=connected, 0=down)",
            "# TYPE aegis_gateway_db_connected gauge",
            f"aegis_gateway_db_connected {db_ok}",
            "# HELP aegis_gateway_ai_latency_ms Last Gateway -> AI Inference HTTP RTT in ms",
            "# TYPE aegis_gateway_ai_latency_ms gauge",
            f"aegis_gateway_ai_latency_ms {float(cb_info['last_latency_ms'])}",
            "# HELP aegis_gateway_cpu_percent API Gateway process CPU percentage",
            "# TYPE aegis_gateway_cpu_percent gauge",
            f"aegis_gateway_cpu_percent {host['cpu_percent']}",
            "# HELP aegis_gateway_ram_used_mb API Gateway resident RAM in MB",
            "# TYPE aegis_gateway_ram_used_mb gauge",
            f"aegis_gateway_ram_used_mb {host['ram_used_mb']}",
        ]
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4; charset=utf-8")

    m = cctv_pipeline.get_telemetry()
    lines = [
        "# HELP aegis_cctv_fps Real-time inference pipeline throughput in FPS",
        "# TYPE aegis_cctv_fps gauge",
        f"aegis_cctv_fps {float(m.get('fps', 0.0))}",
        "# HELP aegis_cctv_inference_latency_ms Inference latency in milliseconds",
        "# TYPE aegis_cctv_inference_latency_ms gauge",
        f"aegis_cctv_inference_latency_ms {float(m.get('inference_ms', 0.0))}",
        "# HELP aegis_cctv_workers_active Currently tracked active workers",
        "# TYPE aegis_cctv_workers_active gauge",
        f"aegis_cctv_workers_active {int(m.get('workers_count', 0))}",
        "# HELP aegis_cctv_active_workers Currently tracked active personnel on site",
        "# TYPE aegis_cctv_active_workers gauge",
        f"aegis_cctv_active_workers {int(m.get('workers_count', 0))}",
        "# HELP aegis_cctv_violations_active Active bare-head safety violations",
        "# TYPE aegis_cctv_violations_active gauge",
        f"aegis_cctv_violations_active {int(m.get('violations_count', 0))}",
        "# HELP aegis_cctv_gallery_personnel_total Total persistent identities remembered in gallery",
        "# TYPE aegis_cctv_gallery_personnel_total gauge",
        f"aegis_cctv_gallery_personnel_total {int(m.get('gallery_total_count', 0))}",
        "# HELP aegis_cctv_compliance_rate_pct PPE compliance percentage",
        "# TYPE aegis_cctv_compliance_rate_pct gauge",
        f"aegis_cctv_compliance_rate_pct {float(m.get('compliance_rate', 100.0))}",
        "# HELP aegis_cctv_total_alerts Total debounced safety alerts stored in audit buffer",
        "# TYPE aegis_cctv_total_alerts counter",
        f"aegis_cctv_total_alerts {len(alert_manager.recent_alerts)}",
        "# HELP aegis_cctv_cpu_percent Host process/system CPU utilization percentage",
        "# TYPE aegis_cctv_cpu_percent gauge",
        f"aegis_cctv_cpu_percent {host['cpu_percent']}",
        "# HELP aegis_cctv_ram_used_mb Application process resident memory in MB",
        "# TYPE aegis_cctv_ram_used_mb gauge",
        f"aegis_cctv_ram_used_mb {host['ram_used_mb']}",
        "# HELP aegis_cctv_cuda_vram_allocated_mb PyTorch allocated GPU VRAM in MB",
        "# TYPE aegis_cctv_cuda_vram_allocated_mb gauge",
        f"aegis_cctv_cuda_vram_allocated_mb {host['cuda_vram_mb']}",
        "# HELP aegis_cctv_http_requests_total Total HTTP requests handled by FastAPI",
        "# TYPE aegis_cctv_http_requests_total counter",
        f"aegis_cctv_http_requests_total {_HTTP_REQUESTS_TOTAL}",
        "# HELP aegis_cctv_http_errors_total Total HTTP 5xx responses handled by FastAPI",
        "# TYPE aegis_cctv_http_errors_total counter",
        f"aegis_cctv_http_errors_total {_HTTP_ERRORS_TOTAL}",
        "# HELP aegis_gateway_http_requests_total Total HTTP requests handled by unified FastAPI endpoint",
        "# TYPE aegis_gateway_http_requests_total counter",
        f"aegis_gateway_http_requests_total {_HTTP_REQUESTS_TOTAL}",
        "# HELP aegis_gateway_circuit_breaker_open Circuit breaker state gauge (0 in monolithic mode)",
        "# TYPE aegis_gateway_circuit_breaker_open gauge",
        "aegis_gateway_circuit_breaker_open 0",
        "# HELP aegis_gateway_db_connected Database health status (1=connected, 0=down)",
        "# TYPE aegis_gateway_db_connected gauge",
        f"aegis_gateway_db_connected {db_ok}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4; charset=utf-8")


def run_server():
    import uvicorn

    uvicorn.run(
        app,
        host=config.web_host,
        port=config.web_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
