"""
src/db/database.py
==================
Relational Persistence Layer for AEGIS-CCTV Enterprise VMS.
Uses PostgreSQL when `DATABASE_URL` is configured (Docker Enterprise Compose stack)
and falls back cleanly to local SQLite (`Output/aegis_vms.db`) for single-node/dev/CI runs.
Tables:
1. `incidents` — Confirmed bare-head safety violation events & snapshots.
2. `operator_feedback` — Human-in-the-loop verification verdicts (`confirm`, `false_positive`, `edit_label`).
3. `role_policies` — Persisted GDUT 5-class color-to-role hierarchy.
4. `personnel_sessions` — Persistent tracked personnel records & labels.
"""

from __future__ import annotations
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, Float, Integer, String, Text, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import OUTPUT_DIR

Base = declarative_base()


class IncidentRecord(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(96), unique=True, nullable=False, index=True)
    camera_id = Column(String(64), nullable=False)
    worker_id = Column(String(64), nullable=False)
    custom_label = Column(String(64), nullable=True)
    role_title = Column(String(128), nullable=True)
    historical_role_title = Column(String(128), nullable=True)
    historical_helmet_color = Column(String(32), nullable=True)
    helmet_color = Column(String(32), nullable=True)
    confidence = Column(Float, nullable=False, default=0.0)
    violation_duration_frames = Column(Integer, nullable=False, default=0)
    violations_json = Column(Text, nullable=False, default="[]")
    snapshot_filename = Column(String(256), nullable=True)
    crop_filename = Column(String(256), nullable=True)
    bbox_json = Column(Text, nullable=False, default="[]")
    timestamp_str = Column(String(64), nullable=False)
    created_at_utc = Column(String(64), nullable=False)


class OperatorFeedbackRecord(Base):
    __tablename__ = "operator_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(String(96), nullable=False, index=True)
    worker_id = Column(Integer, nullable=True)
    verdict = Column(String(32), nullable=False)
    corrected_label = Column(String(64), nullable=True)
    note = Column(Text, nullable=True)
    snapshot_filename = Column(String(256), nullable=True)
    source_type = Column(String(32), nullable=False, default="incident")
    model_path = Column(String(256), nullable=True)
    timestamp_utc = Column(String(64), nullable=False)


class RolePolicyRecord(Base):
    __tablename__ = "role_policies"

    color_key = Column(String(32), primary_key=True)
    role_title = Column(String(128), nullable=False)
    short_code = Column(String(32), nullable=False)
    hex_color = Column(String(16), nullable=False)
    bgr_json = Column(String(64), nullable=False)
    severity = Column(String(32), nullable=False, default="compliant")
    updated_at_utc = Column(String(64), nullable=False)


class PersonnelSessionRecord(Base):
    __tablename__ = "personnel_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, default="default", index=True)
    source_name = Column(String(256), nullable=True)
    worker_id = Column(Integer, nullable=False, index=True)
    custom_label = Column(String(64), nullable=True)
    current_class = Column(String(32), nullable=False)
    historical_helmet_color = Column(String(32), nullable=True)
    role_title = Column(String(128), nullable=True)
    status = Column(String(32), nullable=False)
    total_frames_seen = Column(Integer, nullable=False, default=1)
    updated_at_utc = Column(String(64), nullable=False)


class DatabaseManager:
    """Thread-safe SQLAlchemy repository supporting PostgreSQL and automatic SQLite fallback."""

    def __init__(self, database_url: Optional[str] = None):
        self._lock = threading.Lock()
        self._engine = None
        self._Session = None
        self._initialized = False
        self._url = (database_url or os.getenv("DATABASE_URL", "")).strip()
        if not self._url:
            db_file = OUTPUT_DIR / "aegis_vms.db"
            db_file.parent.mkdir(parents=True, exist_ok=True)
            self._url = f"sqlite:///{db_file.as_posix()}"

    @property
    def backend_name(self) -> str:
        return "postgresql" if self._url.startswith("postgresql") else "sqlite"

    def reconfigure(self, database_url: str):
        with self._lock:
            if self._engine is not None:
                try:
                    self._engine.dispose()
                except Exception:
                    pass
            self._url = database_url.strip()
            self._engine = None
            self._Session = None
            self._initialized = False

    def _ensure_init(self) -> bool:
        if self._initialized and self._Session is not None:
            return True
        with self._lock:
            if self._initialized and self._Session is not None:
                return True
            try:
                kwargs: Dict[str, Any] = {"future": True}
                if self._url.startswith("sqlite"):
                    kwargs["connect_args"] = {"check_same_thread": False}
                else:
                    kwargs["pool_pre_ping"] = True
                    kwargs["pool_size"] = 5
                    kwargs["max_overflow"] = 5
                self._engine = create_engine(self._url, **kwargs)
                Base.metadata.create_all(self._engine)
                # Idempotent schema evolution for existing persistent volumes
                with self._engine.begin() as conn:
                    for col_sql in (
                        "ALTER TABLE personnel_sessions ADD COLUMN session_id VARCHAR(64) NOT NULL DEFAULT 'default'",
                        "ALTER TABLE personnel_sessions ADD COLUMN source_name VARCHAR(256)",
                    ):
                        try:
                            conn.execute(text(col_sql))
                        except Exception:
                            pass
                self._Session = sessionmaker(bind=self._engine, autoflush=False, autocommit=False)
                self._initialized = True
                return True
            except Exception as e:
                print(f"⚠️ Database initialization deferred ({self.backend_name}): {e}")
                return False

    def is_healthy(self) -> bool:
        if not self._ensure_init() or self._engine is None:
            return False
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def save_incident(self, event: Dict[str, Any]) -> bool:
        if not self._ensure_init() or self._Session is None:
            return False
        try:
            with self._Session() as sess:
                aid = str(event.get("alert_id") or "")
                existing = sess.query(IncidentRecord).filter_by(alert_id=aid).first()
                if existing is not None:
                    return True
                rec = IncidentRecord(
                    alert_id=aid or f"ALERT-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
                    camera_id=str(event.get("camera_id") or "CAM_01")[:64],
                    worker_id=str(event.get("worker_id") or "")[:64],
                    custom_label=event.get("custom_label"),
                    role_title=event.get("role_title"),
                    historical_role_title=event.get("historical_role_title"),
                    historical_helmet_color=event.get("historical_helmet_color"),
                    helmet_color=event.get("helmet_color"),
                    confidence=float(event.get("confidence") or 0.0),
                    violation_duration_frames=int(event.get("violation_duration_frames") or 0),
                    violations_json=json.dumps(event.get("violations") or []),
                    snapshot_filename=event.get("snapshot_filename"),
                    crop_filename=event.get("crop_filename"),
                    bbox_json=json.dumps(event.get("bbox") or []),
                    timestamp_str=str(event.get("timestamp") or ""),
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                )
                sess.add(rec)
                sess.commit()
                return True
        except Exception:
            return False

    def list_incidents(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self._ensure_init() or self._Session is None:
            return []
        try:
            with self._Session() as sess:
                rows = sess.query(IncidentRecord).order_by(IncidentRecord.id.desc()).limit(limit).all()
                out = []
                for r in rows:
                    out.append({
                        "alert_id": r.alert_id,
                        "camera_id": r.camera_id,
                        "worker_id": int(r.worker_id) if str(r.worker_id).isdigit() else r.worker_id,
                        "custom_label": r.custom_label,
                        "role_title": r.role_title,
                        "historical_role_title": r.historical_role_title,
                        "historical_helmet_color": r.historical_helmet_color,
                        "helmet_color": r.helmet_color,
                        "has_helmet": r.helmet_color not in (None, "none", "no_helmet"),
                        "has_vest": True,
                        "confidence": r.confidence,
                        "violation_duration_frames": r.violation_duration_frames,
                        "violations": json.loads(r.violations_json or "[]"),
                        "snapshot_filename": r.snapshot_filename,
                        "crop_filename": r.crop_filename,
                        "bbox": json.loads(r.bbox_json or "[]"),
                        "timestamp": r.timestamp_str,
                    })
                return out
        except Exception:
            return []

    def clear_incidents(self) -> bool:
        if not self._ensure_init() or self._Session is None:
            return False
        try:
            with self._Session() as sess:
                sess.query(IncidentRecord).delete()
                sess.commit()
                return True
        except Exception:
            return False

    def save_feedback(self, record: Dict[str, Any]) -> bool:
        if not self._ensure_init() or self._Session is None:
            return False
        try:
            with self._Session() as sess:
                rec = OperatorFeedbackRecord(
                    item_id=str(record.get("item_id") or ""),
                    worker_id=record.get("worker_id"),
                    verdict=str(record.get("verdict") or ""),
                    corrected_label=record.get("corrected_label"),
                    note=record.get("note"),
                    snapshot_filename=record.get("snapshot_filename"),
                    source_type=str(record.get("source_type") or "incident"),
                    model_path=str(record.get("model_path") or "")[:256],
                    timestamp_utc=str(record.get("timestamp_utc") or datetime.now(timezone.utc).isoformat()),
                )
                sess.add(rec)
                sess.commit()
                return True
        except Exception:
            return False

    def list_feedback(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self._ensure_init() or self._Session is None:
            return []
        try:
            with self._Session() as sess:
                rows = sess.query(OperatorFeedbackRecord).order_by(OperatorFeedbackRecord.id.desc()).limit(limit).all()
                return [
                    {
                        "timestamp_utc": r.timestamp_utc,
                        "item_id": r.item_id,
                        "worker_id": r.worker_id,
                        "verdict": r.verdict,
                        "corrected_label": r.corrected_label,
                        "note": r.note,
                        "snapshot_filename": r.snapshot_filename,
                        "source_type": r.source_type,
                        "model_path": r.model_path,
                    }
                    for r in rows
                ]
        except Exception:
            return []

    def upsert_role_policies(self, policy: Dict[str, Dict[str, Any]]) -> bool:
        if not self._ensure_init() or self._Session is None:
            return False
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            with self._Session() as sess:
                for ck, info in policy.items():
                    row = sess.query(RolePolicyRecord).filter_by(color_key=ck).first()
                    if row is None:
                        row = RolePolicyRecord(
                            color_key=ck,
                            role_title=str(info.get("role_title", "")),
                            short_code=str(info.get("short_code", "")),
                            hex_color=str(info.get("hex_color", "#475569")),
                            bgr_json=json.dumps(info.get("bgr_color", [105, 85, 71])),
                            severity=str(info.get("severity", "compliant")),
                            updated_at_utc=now_iso,
                        )
                        sess.add(row)
                    else:
                        row.role_title = str(info.get("role_title", row.role_title))
                        row.short_code = str(info.get("short_code", row.short_code))
                        row.hex_color = str(info.get("hex_color", row.hex_color))
                        row.bgr_json = json.dumps(info.get("bgr_color", [105, 85, 71]))
                        row.severity = str(info.get("severity", row.severity))
                        row.updated_at_utc = now_iso
                sess.commit()
                return True
        except Exception:
            return False

    def get_role_policies(self) -> Optional[Dict[str, Dict[str, Any]]]:
        if not self._ensure_init() or self._Session is None:
            return None
        try:
            with self._Session() as sess:
                rows = sess.query(RolePolicyRecord).all()
                if not rows:
                    return None
                return {
                    r.color_key: {
                        "role_title": r.role_title,
                        "short_code": r.short_code,
                        "hex_color": r.hex_color,
                        "bgr_color": json.loads(r.bgr_json),
                        "severity": r.severity,
                    }
                    for r in rows
                }
        except Exception:
            return None

    def save_personnel_snapshot(
        self,
        workers: List[Dict[str, Any]],
        session_id: str = "default",
        source_name: Optional[str] = None,
    ) -> bool:
        if not workers or not self._ensure_init() or self._Session is None:
            return False
        sess_key = str(session_id or "default")[:64]
        src_val = str(source_name)[:256] if source_name else None
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            with self._Session() as sess:
                for w in workers[:200]:
                    wid = int(w.get("track_id", 0))
                    row = (
                        sess.query(PersonnelSessionRecord)
                        .filter_by(session_id=sess_key, worker_id=wid)
                        .first()
                    )
                    if row is None:
                        row = PersonnelSessionRecord(
                            session_id=sess_key,
                            source_name=src_val,
                            worker_id=wid,
                            custom_label=w.get("custom_label"),
                            current_class=str(w.get("current_class") or "none"),
                            historical_helmet_color=w.get("historical_helmet_color"),
                            role_title=w.get("historical_role_title") or w.get("role_title"),
                            status=str(w.get("status") or "COMPLIANT"),
                            total_frames_seen=int(w.get("total_frames_seen") or 1),
                            updated_at_utc=now_iso,
                        )
                        sess.add(row)
                    else:
                        if src_val:
                            row.source_name = src_val
                        row.custom_label = w.get("custom_label")
                        row.current_class = str(w.get("current_class") or row.current_class)
                        row.historical_helmet_color = w.get("historical_helmet_color")
                        row.role_title = w.get("historical_role_title") or w.get("role_title")
                        row.status = str(w.get("status") or row.status)
                        row.total_frames_seen = int(w.get("total_frames_seen") or row.total_frames_seen)
                        row.updated_at_utc = now_iso
                sess.commit()
                return True
        except Exception:
            return False

    def list_personnel_sessions(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not self._ensure_init() or self._Session is None:
            return []
        try:
            with self._Session() as sess:
                q = sess.query(PersonnelSessionRecord)
                if session_id:
                    q = q.filter_by(session_id=str(session_id)[:64])
                rows = q.order_by(PersonnelSessionRecord.id.desc()).limit(limit).all()
                return [
                    {
                        "session_id": r.session_id,
                        "source_name": r.source_name,
                        "worker_id": r.worker_id,
                        "custom_label": r.custom_label,
                        "current_class": r.current_class,
                        "historical_helmet_color": r.historical_helmet_color,
                        "role_title": r.role_title,
                        "status": r.status,
                        "total_frames_seen": r.total_frames_seen,
                        "updated_at_utc": r.updated_at_utc,
                    }
                    for r in rows
                ]
        except Exception:
            return []


db_manager = DatabaseManager()
