"""
tests/conftest.py
=================
Session-level isolation fixture for pytest so test runs NEVER pollute real runtime
artifacts (`snapshots/*.jpg`, `Output/aegis_vms.db`, `Output/role_policy.json`,
`Output/operator_feedback.jsonl`, or `Output/alerts/alerts_audit_log.json`).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_pytest_runtime_artifacts(tmp_path_factory):
    """Redirects all persistent snapshot, SQLite DB, policy, feedback, and alert paths to tmp_path."""
    isolated_root: Path = tmp_path_factory.mktemp("aegis_pytest_isolated")
    snap_tmp = isolated_root / "snapshots"
    alerts_tmp = isolated_root / "alerts"
    snap_tmp.mkdir(parents=True, exist_ok=True)
    alerts_tmp.mkdir(parents=True, exist_ok=True)

    db_file = isolated_root / "aegis_pytest.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    role_tmp = isolated_root / "role_policy.json"
    fb_tmp = isolated_root / "operator_feedback.jsonl"
    audit_tmp = alerts_tmp / "alerts_audit_log.json"

    os.environ["DATABASE_URL"] = db_url
    os.environ["ROLE_POLICY_PATH"] = str(role_tmp)

    import src.config as cfg_mod
    from src.db.database import db_manager

    cfg_mod.SNAPSHOTS_DIR = snap_tmp
    cfg_mod.ROLE_POLICY_PATH = role_tmp
    cfg_mod.ALERTS_LOG_DIR = alerts_tmp
    db_manager.reconfigure(db_url)

    try:
        import src.engine.spatial_safety as ss_mod

        ss_mod.SNAPSHOTS_DIR = snap_tmp
    except Exception:
        pass

    try:
        import src.alerting.alert_manager as am_mod

        am_mod.ALERTS_LOG_DIR = alerts_tmp
        am_mod.alert_manager.audit_log_path = audit_tmp
    except Exception:
        pass

    try:
        import src.web.app as web_app

        web_app.SNAPSHOTS_DIR = snap_tmp
        web_app.FEEDBACK_LOG_PATH = fb_tmp
    except Exception:
        pass

    yield isolated_root

    # Wait briefly for any queued snapshot I/O tasks to flush into isolated snap_tmp
    try:
        from src.engine.pipeline import cctv_pipeline

        if getattr(cctv_pipeline, "safety_engine", None) is not None:
            cctv_pipeline.safety_engine.io_executor.submit(lambda: None).result(timeout=2.0)
    except Exception:
        pass
