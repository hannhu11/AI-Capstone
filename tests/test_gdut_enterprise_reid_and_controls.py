"""
tests/test_gdut_enterprise_reid_and_controls.py
=============================================
Comprehensive test suite for GDUT-HWD 5-Class Enterprise VMS upgrades:
1. Head-Anchored Upper-Torso Persistent Re-ID (exit/re-entry & helmet-removal continuity).
2. Person A vs Person B identity separation & anti-hijack guard.
3. SpatialSafetyEngine Head-Tracked Role-by-Color enrichment & Debounce FSM.
4. Role Policy API (`GET/POST /api/roles`).
5. Multi-Dimensional Follow/Filter, Solo Focus & Custom Labels (`POST /api/tracking/*`).
6. Video Transport Controls (`play_pause`, `seek`, `speed`, `resolution`, `playback_state`).
7. Live Browser Webcam WebSocket (`/ws/webcam`).
"""

from __future__ import annotations
import base64
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.config import DEFAULT_ROLE_HIERARCHY, save_role_policy
from src.engine.bytetrack import STrack
from src.engine.gdut_reid import (
    UpperTorsoReIDGallery,
    compute_three_strip_torso_signature,
    project_upper_torso_roi,
)
from src.engine.spatial_safety import SpatialSafetyEngine
from src.web.app import app


def _make_person_scene(
    width: int = 960,
    height: int = 540,
    person_a: tuple[int, int, str] | None = (220, 90, "blue_slate"),
    person_b: tuple[int, int, str] | None = (640, 100, "orange_hivis"),
) -> np.ndarray:
    """Renders synthetic construction scene with distinct patterned upper torsos below head boxes."""
    frame = np.full((height, width, 3), 210, dtype=np.uint8)
    # Neutral site background bands
    frame[380:, :] = (165, 170, 175)

    def _draw_person(cx: int, hy1: int, outfit: str):
        # Head region [hy1 : hy1+56, cx-24 : cx+24]
        cv2.rectangle(frame, (cx - 24, hy1), (cx + 24, hy1 + 56), (170, 190, 215), -1)
        # Upper torso region below head: [hy1+64 : hy1+235, cx-60 : cx+60]
        tx1, tx2 = cx - 60, cx + 60
        ty1, ty2 = hy1 + 64, hy1 + 235
        if outfit == "blue_slate":
            frame[ty1:ty1 + 55, tx1:tx2] = (140, 80, 25)    # Dark navy/steel shoulder
            frame[ty1 + 55:ty1 + 115, tx1:tx2] = (195, 135, 55)  # Steel blue chest band
            frame[ty1 + 115:ty2, tx1:tx2] = (85, 65, 50)    # Charcoal lower torso
        else:
            frame[ty1:ty1 + 55, tx1:tx2] = (35, 145, 235)   # Hi-vis orange shoulder
            frame[ty1 + 55:ty1 + 115, tx1:tx2] = (60, 225, 225)  # Reflective yellow chest
            frame[ty1 + 115:ty2, tx1:tx2] = (40, 110, 50)   # Olive lower vest

    if person_a is not None:
        _draw_person(*person_a)
    if person_b is not None:
        _draw_person(*person_b)
    return frame


def test_upper_torso_roi_excludes_head_cranium():
    head_box = np.array([200.0, 90.0, 248.0, 146.0], dtype=np.float32)  # w=48, h=56
    roi = project_upper_torso_roi(head_box, (540, 960, 3))
    tx1, ty1, tx2, ty2 = roi
    # ROI starts strictly below head box y2 (146 + 0.15*56 = 154)
    assert ty1 > 146
    assert ty2 <= 540
    assert tx1 < 200 and tx2 > 248


def test_persistent_reid_exit_reentry_and_helmet_removal():
    gallery = UpperTorsoReIDGallery(ttl_seconds=600.0, sim_thresh=0.70, use_deep=False, min_confirm_hits=6)
    head_a = np.array([196.0, 90.0, 244.0, 146.0], dtype=np.float32)

    # 0. Transient 3-frame reflection noise MUST NOT increment gallery.next_id (Confirm-Before-Mint)
    f_noise = _make_person_scene(person_a=(800, 200, "blue_slate"), person_b=None)
    head_noise = np.array([776.0, 200.0, 824.0, 256.0], dtype=np.float32)
    for fidx in range(3):
        p_noise = gallery.associate(999, head_noise, "red", f_noise, frame_idx=fidx, now_ts=90.0 + fidx * 0.03)
        assert p_noise.worker_id < 0
        assert p_noise.is_confirmed is False
        gallery.update_frame_end([], fidx, now_ts=90.0 + fidx * 0.03)
    assert gallery.next_id == 1
    assert len(gallery.profiles) == 0
    for fidx in range(3, 7):
        gallery.update_frame_end([], fidx, now_ts=90.0 + fidx * 0.03)
    assert len(gallery._tentative_tracks) == 0
    assert gallery.next_id == 1

    # 1. Person A (white helmet = Site Engineer) confirms at 6th consecutive hit over 10 frames
    f_a = _make_person_scene(person_a=(220, 90, "blue_slate"), person_b=None)
    for fidx in range(10, 20):
        prof = gallery.associate(1, head_a, "white", f_a, frame_idx=fidx, now_ts=100.0 + (fidx - 10) * 0.03)
        if (fidx - 10) < 5:
            assert prof.worker_id < 0
            assert gallery.next_id == 1
        else:
            assert prof.worker_id == 1
            assert gallery.next_id == 2
        gallery.update_frame_end([prof.worker_id], fidx, now_ts=100.0 + (fidx - 10) * 0.03)

    assert gallery.profiles[1].is_confirmed is True
    assert gallery.profiles[1].last_known_helmet_color == "white"

    # 2. Person A exits FOV for 36 frames (5 seconds later)
    for fidx in range(20, 56):
        gallery.update_frame_end([], fidx, now_ts=105.0 + (fidx - 20) * 0.03)
    assert gallery.profiles[1].is_active is False

    # 3. Person A re-enters at new X=380 with new ByteTrack raw_id=42 and BARE HEAD ("none")
    f_a_reenter = _make_person_scene(person_a=(380, 96, "blue_slate"), person_b=None)
    head_a_new = np.array([356.0, 96.0, 404.0, 152.0], dtype=np.float32)
    prof_re = gallery.associate(42, head_a_new, "none", f_a_reenter, frame_idx=56, now_ts=110.0)

    # MUST recover ID #1, remember prior "white" role, and mark VIOLATION
    assert prof_re.worker_id == 1, f"Expected ID #1 on bare-head re-entry, got #{prof_re.worker_id}"
    assert prof_re.current_helmet_color == "none"
    assert prof_re.last_known_helmet_color == "white"
    assert prof_re.status == "VIOLATION"

    # 4. Person B (yellow helmet, orange hi-vis torso) enters at X=680 -> tentative <6 hits, mints ID #2 at hit #6
    f_both = _make_person_scene(person_a=(380, 96, "blue_slate"), person_b=(680, 100, "orange_hivis"))
    head_b = np.array([656.0, 100.0, 704.0, 156.0], dtype=np.float32)
    for step, fidx_b in enumerate(range(57, 63)):
        profs = gallery.associate_batch([(42, head_a_new, "none"), (77, head_b, "yellow")], f_both, frame_idx=fidx_b, now_ts=110.1 + step * 0.04)
        prof_b = profs[1]
        if step < 5:
            assert prof_b.worker_id < 0
            assert gallery.next_id == 2
            gallery.update_frame_end([1], fidx_b, now_ts=110.1 + step * 0.04)
            assert all(w["track_id"] != 2 for w in gallery.get_registered_workers(DEFAULT_ROLE_HIERARCHY))
        else:
            assert prof_b.worker_id == 2
            assert prof_b.current_helmet_color == "yellow"
            assert gallery.next_id == 3
            gallery.update_frame_end([1, 2], fidx_b, now_ts=110.1 + step * 0.04)

    # Verify enriched registry output
    workers = gallery.get_registered_workers(DEFAULT_ROLE_HIERARCHY)
    w1 = next(w for w in workers if w["track_id"] == 1)
    w2 = next(w for w in workers if w["track_id"] == 2)
    assert w1["helmet_removed"] is True
    assert w1["historical_helmet_color"] == "white"
    assert "Project Manager" in w1["historical_role_title"]
    assert "Worker" in w2["role_title"]


def test_spatial_safety_head_tracked_gdut_roles_and_debounce():
    engine = SpatialSafetyEngine(consecutive_frames_thresh=4, min_tracklet_len=0)
    frame = _make_person_scene()

    trk = STrack(STrack.tlbr_to_tlwh(np.array([200.0, 90.0, 240.0, 140.0], dtype=np.float32)), score=0.91, cls_id=4)
    trk.track_id = 11
    trk.canonical_id = 1
    trk.is_activated = True
    trk.head_cls_name = "none"
    trk.historical_helmet_color = "white"
    trk.custom_label = "Engineer-01"

    alerts_all = []
    for fidx in range(4):
        reports, alerts = engine.evaluate_frame(
            [trk],
            [],
            frame,
            camera_id="CAM_GATE",
            is_helmet_only=True,
            role_policy=DEFAULT_ROLE_HIERARCHY,
            class_names={0: "blue", 1: "white", 2: "yellow", 3: "red", 4: "none"},
        )
        assert len(reports) == 1
        assert reports[0]["track_id"] == 1
        assert reports[0]["status"] == "VIOLATION"
        assert "Project Manager" in reports[0]["historical_role_title"]
        alerts_all.extend(alerts)

    assert len(alerts_all) == 1
    assert alerts_all[0]["worker_id"] == 1
    assert alerts_all[0]["custom_label"] == "Engineer-01"


def test_role_policy_and_tracking_filter_and_video_controls_api(tmp_path, monkeypatch):
    import src.config as cfg_mod
    from src.engine.pipeline import cctv_pipeline

    monkeypatch.setattr(cfg_mod, "ROLE_POLICY_PATH", tmp_path / "role_policy.json")
    orig_policy = cctv_pipeline.get_role_policy()

    client = TestClient(app)
    try:
        # 1. Roles GET/POST
        r_roles = client.get("/api/roles")
        assert r_roles.status_code == 200
        assert "white" in r_roles.json()["roles"]

        r_upd = client.post(
            "/api/roles",
            json={"blue": {"role_title": "Lead MEP Inspector", "short_code": "MEP-LEAD", "hex_color": "#1D4ED8"}},
        )
        assert r_upd.status_code == 200
        assert r_upd.json()["roles"]["blue"]["short_code"] == "MEP-LEAD"

        # 2. Multi-dimensional filter & focus & label
        r_f = client.post(
            "/api/tracking/filter",
            json={"colors": ["white", "none"], "roles": ["PM/ENG"], "violators_only": True},
        )
        assert r_f.status_code == 200
        fj = r_f.json()
        assert fj["colors"] == ["white", "none"]
        assert fj["violators_only"] is True

        r_clr = client.post("/api/tracking/filter", json={"clear": True})
        assert r_clr.status_code == 200
        assert r_clr.json()["colors"] == []
        assert r_clr.json()["violators_only"] is False

        r_lbl = client.post("/api/tracking/label", json={"track_id": 999, "label": "Han Nhu"})
        assert r_lbl.status_code == 200
        assert r_lbl.json()["label"] == "Han Nhu"

        # 3. Video Transport Controls
        r_st = client.get("/api/video/playback_state")
        assert r_st.status_code == 200
        assert "is_paused" in r_st.json()

        r_pause = client.post("/api/video/play_pause", json={"paused": True})
        assert r_pause.status_code == 200
        assert r_pause.json()["is_paused"] is True

        r_seek = client.post("/api/video/seek", json={"percent": 25.0})
        assert r_seek.status_code == 200

        r_speed = client.post("/api/video/speed", json={"speed": 1.5})
        assert r_speed.status_code == 200
        assert r_speed.json()["speed"] == 1.5

        r_res = client.post("/api/video/resolution", json={"resolution": "720p"})
        assert r_res.status_code == 200
        assert r_res.json()["resolution"] == "720p"

        client.post("/api/video/play_pause", json={"paused": False})
        client.post("/api/video/speed", json={"speed": 1.0})
        client.post("/api/video/resolution", json={"resolution": "original"})
    finally:
        cctv_pipeline.set_role_policy(orig_policy)


def test_websocket_webcam_c960_roundtrip():
    client = TestClient(app)
    img = _make_person_scene(width=640, height=480)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    b64_frame = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

    with client.websocket_connect("/ws/webcam") as ws:
        ws.send_json({"image": b64_frame, "resolution": "480p", "device_label": "C960 Ultra Test"})
        resp = ws.receive_json()
        assert resp["status"] == "ok"
        assert resp["image"].startswith("data:image/jpeg;base64,")
        assert "fps" in resp
        assert "inference_ms" in resp
        assert "kpis" in resp
        assert "personnel" in resp
        assert "active_tracks" in resp


def _make_uniform_variant_scene(
    cx: int,
    hy1: int,
    variant: str,
    width: int = 960,
    height: int = 540,
) -> np.ndarray:
    """
    Renders same-color steel-blue base uniforms separated ONLY by reflective stripe geometry
    and inner strap pattern (adversarial same-uniform stress test).
    """
    frame = np.full((height, width, 3), 215, dtype=np.uint8)
    cv2.rectangle(frame, (cx - 24, hy1), (cx + 24, hy1 + 56), (175, 195, 218), -1)
    tx1, tx2 = cx - 64, cx + 64
    ty1, ty2 = hy1 + 64, hy1 + 238
    # Identical base steel-blue torso color
    frame[ty1:ty2, tx1:tx2] = (145, 92, 42)
    if variant == "horiz_yellow_bar":
        # Two thick bright yellow reflective horizontal bars + dark badge
        cv2.rectangle(frame, (tx1 + 4, ty1 + 28), (tx2 - 4, ty1 + 52), (40, 235, 245), -1)
        cv2.rectangle(frame, (tx1 + 4, ty1 + 98), (tx2 - 4, ty1 + 122), (40, 235, 245), -1)
    else:
        # Crisp white diagonal safety sash + vertical chest zipper contrast
        cv2.line(frame, (tx1 + 12, ty1 + 8), (tx2 - 12, ty2 - 12), (248, 248, 248), 16)
        cv2.rectangle(frame, (cx - 8, ty1 + 8), (cx + 8, ty2 - 10), (30, 30, 35), -1)
    return frame


def test_video_webcam_video_recovery_no_hang():
    from src.engine.pipeline import cctv_pipeline

    client = TestClient(app)
    html = client.get("/").text
    assert "STATE.wasWebcamActive" in html
    assert "switchSelectedVideo(activeClip" in html

    frame = _make_person_scene(width=640, height=480)
    out = cctv_pipeline.process_external_frame(
        frame, camera_id="WEBCAM-C960", resolution="480p", return_annotated=False
    )
    assert out["annotated_frame"] is None
    assert cctv_pipeline.source_type == "browser_webcam"

    r_sw = client.post(
        "/api/switch_source",
        json={"video_source": "video_test/construction_site_workers_1080p.mp4"},
    )
    assert r_sw.status_code == 200
    assert cctv_pipeline.source_type == "video"
    st = client.get("/api/video/playback_state")
    assert st.status_code == 200


def test_same_uniform_reid_stress_no_hijack():
    gallery = UpperTorsoReIDGallery(ttl_seconds=600.0, sim_thresh=0.66, use_deep=False, min_confirm_hits=6)
    head_box = np.array([296.0, 90.0, 344.0, 146.0], dtype=np.float32)

    # Both Person A and Person B wear IDENTICAL "yellow" helmets + steel-blue uniform base (zero helmet color penalty crutch)
    f_a = _make_uniform_variant_scene(320, 90, "horiz_yellow_bar")
    for fidx in range(10):
        prof = gallery.associate(10, head_box, "yellow", f_a, frame_idx=fidx, now_ts=100.0 + fidx * 0.04)
        if fidx >= 5:
            assert prof.worker_id == 1
        gallery.update_frame_end([prof.worker_id], fidx, now_ts=100.0 + fidx * 0.04)

    # Person A leaves scene
    for fidx in range(10, 20):
        gallery.update_frame_end([], fidx, now_ts=104.0 + fidx * 0.04)

    # Person B stands at exact same X/Y box wearing identical steel-blue base & SAME yellow helmet (diag white sash texture)
    f_b = _make_uniform_variant_scene(320, 90, "diag_white_sash")
    for fidx in range(20, 26):
        prof_b = gallery.associate(21, head_box, "yellow", f_b, frame_idx=fidx, now_ts=106.0 + (fidx - 20) * 0.04)
        if fidx < 25:
            assert prof_b.worker_id < 0
        else:
            assert prof_b.worker_id == 2, f"Expected new ID #2 for same-helmet same-uniform Person B, got #{prof_b.worker_id}"
        gallery.update_frame_end([prof_b.worker_id], fidx, now_ts=106.0 + (fidx - 20) * 0.04)

    # Person B leaves and Person A returns bare-headed at shifted X=520
    gallery.update_frame_end([], 26, now_ts=107.0)
    f_a_ret = _make_uniform_variant_scene(520, 92, "horiz_yellow_bar")
    head_a_ret = np.array([496.0, 92.0, 544.0, 148.0], dtype=np.float32)
    prof_a_ret = gallery.associate(33, head_a_ret, "none", f_a_ret, frame_idx=27, now_ts=108.0)
    assert prof_a_ret.worker_id == 1
    assert prof_a_ret.last_known_helmet_color == "yellow"


def test_deep_mobilenet_multilayer_reid_integration():
    gallery = UpperTorsoReIDGallery(
        ttl_seconds=600.0, sim_thresh=0.66, use_deep=True, deep_refresh_interval=1, min_confirm_hits=6, device="cpu"
    )
    if not gallery.prewarm():
        pytest.skip("MobileNetV3 weights unavailable offline")

    head_box = np.array([296.0, 90.0, 344.0, 146.0], dtype=np.float32)
    # Strict SAME helmet color ("yellow") for both Person A & Person B
    f_a = _make_uniform_variant_scene(320, 90, "horiz_yellow_bar")
    for fidx in range(10):
        p = gallery.associate(1, head_box, "yellow", f_a, frame_idx=fidx, now_ts=200.0 + fidx * 0.04)
        if fidx >= 5:
            assert p.worker_id == 1
        gallery.update_frame_end([p.worker_id], fidx, now_ts=200.0 + fidx * 0.04)

    for fidx in range(10, 18):
        gallery.update_frame_end([], fidx, now_ts=203.0 + fidx * 0.04)

    f_b = _make_uniform_variant_scene(320, 90, "diag_white_sash")
    for fidx in range(18, 24):
        p_b = gallery.associate(5, head_box, "yellow", f_b, frame_idx=fidx, now_ts=205.0 + (fidx - 18) * 0.04)
        if fidx >= 23:
            assert p_b.worker_id == 2, f"Expected new ID #2 under deep MobileNetV3 same-yellow-helmet check, got #{p_b.worker_id}"
        gallery.update_frame_end([p_b.worker_id], fidx, now_ts=205.0 + (fidx - 18) * 0.04)
    gallery.update_frame_end([], 24, now_ts=206.0)

    f_a_ret = _make_uniform_variant_scene(520, 92, "horiz_yellow_bar")
    head_ret = np.array([496.0, 92.0, 544.0, 148.0], dtype=np.float32)
    p_ret = gallery.associate(9, head_ret, "none", f_a_ret, frame_idx=25, now_ts=207.0)
    assert p_ret.worker_id == 1


def test_gdut_majority_voting_suppresses_single_frame_glare_flicker():
    engine = SpatialSafetyEngine(consecutive_frames_thresh=4, min_tracklet_len=0)
    frame = _make_person_scene()
    trk = STrack(STrack.tlbr_to_tlwh(np.array([200.0, 90.0, 240.0, 140.0], dtype=np.float32)), score=0.92, cls_id=1)
    trk.track_id = 7
    trk.canonical_id = 7
    trk.is_activated = True

    # 5 compliant white frames populate temporal majority window
    for _ in range(5):
        trk.head_cls_name = "white"
        reports, _ = engine.evaluate_frame(
            [trk], [], frame, camera_id="CAM_MV", is_helmet_only=True, role_policy=DEFAULT_ROLE_HIERARCHY
        )
        assert reports[0]["helmet_color"] == "white"

    # 1 single-frame glare misclassification as yellow must be suppressed by Majority Voting
    trk.head_cls_name = "yellow"
    reports_glare, _ = engine.evaluate_frame(
        [trk], [], frame, camera_id="CAM_MV", is_helmet_only=True, role_policy=DEFAULT_ROLE_HIERARCHY
    )
    assert reports_glare[0]["helmet_color"] == "white"
    assert reports_glare[0]["status"] == "COMPLIANT"

    # Bare head must still alert immediately
    trk.head_cls_name = "none"
    trk.historical_helmet_color = "white"
    reports_viol, _ = engine.evaluate_frame(
        [trk], [], frame, camera_id="CAM_MV", is_helmet_only=True, role_policy=DEFAULT_ROLE_HIERARCHY
    )
    assert reports_viol[0]["status"] == "VIOLATION"
    assert reports_viol[0]["historical_helmet_color"] == "white"


def test_websocket_binary_jpeg_metadata_mode():
    client = TestClient(app)
    img = _make_person_scene(width=640, height=480)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    with client.websocket_connect("/ws/webcam?resolution=480p&device_label=C960_BIN&return_image=false") as ws:
        ws.send_bytes(buf.tobytes())
        resp = ws.receive_json()
        assert resp["status"] == "ok"
        assert "image" not in resp
        assert "active_tracks" in resp
        assert "fps" in resp
        assert "kpis" in resp
        assert "personnel" in resp


def test_gateway_circuit_breaker_and_sqlite_db_persistence(tmp_path):
    import time as _t
    import httpx
    from src.db.database import DatabaseManager
    from src.web.gateway_client import CircuitBreakerOpenError, InferenceCircuitBreakerClient

    # 1. Circuit Breaker state transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
    mode = {"fail": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if mode["fail"]:
            return httpx.Response(503, json={"detail": "overloaded"})
        return httpx.Response(200, json={"status": "online"})

    cb = InferenceCircuitBreakerClient("http://ai-inference:8001", failure_threshold=2, recovery_timeout=0.15)
    cb._client = httpx.Client(base_url="http://ai-inference:8001", transport=httpx.MockTransport(handler))

    with pytest.raises(RuntimeError):
        cb.request_json("GET", "/api/status")
    with pytest.raises(RuntimeError):
        cb.request_json("GET", "/api/status")
    assert cb.state == "OPEN"

    with pytest.raises(CircuitBreakerOpenError):
        cb.request_json("GET", "/api/status")

    _t.sleep(0.20)
    mode["fail"] = False
    res = cb.request_json("GET", "/api/status")
    assert res["status"] == "online"
    assert cb.state == "CLOSED"

    # 2. SQLAlchemy SQLite persistence across all 4 tables
    db_url = f"sqlite:///{(tmp_path / 'aegis_unit.db').as_posix()}"
    db = DatabaseManager(database_url=db_url)
    assert db.is_healthy() is True

    assert db.save_incident({
        "alert_id": "alt-test-1",
        "worker_id": 3,
        "role_title": "Project Manager / Site Engineer",
        "historical_helmet_color": "white",
        "camera_id": "GATE-01",
        "timestamp": "2026-11-26 08:00:00",
    })
    incs = db.list_incidents(limit=10)
    assert len(incs) == 1 and incs[0]["alert_id"] == "alt-test-1"

    assert db.save_feedback({
        "item_id": "alt-test-1",
        "worker_id": 3,
        "verdict": "confirm",
        "timestamp_utc": "2026-11-26T08:00:00Z",
    })
    assert len(db.list_feedback(limit=10)) == 1

    assert db.upsert_role_policies(DEFAULT_ROLE_HIERARCHY)
    pol = db.get_role_policies()
    assert pol is not None and "white" in pol

    assert db.save_personnel_snapshot([{
        "track_id": 3,
        "custom_label": "Site-PM",
        "current_class": "white",
        "historical_helmet_color": "white",
        "role_title": "Project Manager / Site Engineer",
        "status": "COMPLIANT",
        "total_frames_seen": 12,
    }])
    assert db.clear_incidents() is True
    assert len(db.list_incidents()) == 0


def test_export_csv_and_enroll_endpoints():
    client = TestClient(app)
    r_enroll = client.post("/api/tracking/enroll", json={"track_id": 1, "label": "Lead-Tester"})
    assert r_enroll.status_code == 200
    ej = r_enroll.json()
    assert ej["status"] == "ok"
    assert "enrolled" in ej and ej["track_id"] == 1

    r_csv = client.get("/api/tracking/export_csv")
    assert r_csv.status_code == 200
    assert "text/csv" in r_csv.headers.get("content-type", "")
    assert "track_id,custom_label,current_class" in r_csv.text


def test_prometheus_grafana_metric_name_parity(monkeypatch):
    import json as _json
    import re as _re
    from pathlib import Path as _Path
    import src.web.app as web_app

    dash_path = _Path(__file__).resolve().parent.parent / "monitoring" / "grafana" / "dashboards" / "aegis_vms_overview.json"
    dash = _json.loads(dash_path.read_text(encoding="utf-8"))
    exprs = [t.get("expr", "") for p in dash.get("panels", []) for t in p.get("targets", [])]
    referenced = set(_re.findall(r"aegis_(?:cctv|gateway)_[a-z0-9_]+", " ".join(exprs)))
    assert referenced

    client = TestClient(app)
    inf_metrics = client.get("/metrics").text
    monkeypatch.setattr(web_app, "SERVICE_MODE", "gateway")
    gw_metrics = client.get("/metrics").text

    combined_scraped = inf_metrics + "\n" + gw_metrics
    missing = [m for m in sorted(referenced) if m not in combined_scraped]
    assert not missing, f"Grafana PromQL metrics missing from /metrics: {missing}"


def test_real_video_no_id_explosion_regression():
    import os as _os
    from pathlib import Path as _Path
    from src.config import DEFAULT_MODEL_PATH, DEFAULT_VIDEO_PATH
    from src.engine.bytetrack import BYTETracker
    from src.engine.detector import SafetyObjectDetector

    model_env = _os.getenv("MODEL_PATH", "")
    if "stub" in model_env.lower() or (model_env and not _Path(model_env).exists()):
        pytest.skip("Skipping real-video regression in weightless CI stub mode")
    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_VIDEO_PATH.exists():
        pytest.skip("GDUT deploy checkpoint or test clip unavailable")

    det = SafetyObjectDetector(str(DEFAULT_MODEL_PATH), device="cpu", use_half=False, use_selective_amp=False)
    cap = cv2.VideoCapture(str(DEFAULT_VIDEO_PATH))
    assert cap.isOpened()

    tracker = BYTETracker(
        track_thresh=0.30, track_low_thresh=0.15, new_track_thresh=0.30,
        track_buffer=60, match_thresh=0.70, grace_period=2,
    )
    gal = UpperTorsoReIDGallery(
        ttl_seconds=600.0, sim_thresh=0.70, use_deep=True,
        deep_refresh_interval=30, min_confirm_hits=6, min_confirm_frames=24, device="cpu",
    )

    for fidx in range(800):
        ret, fr = cap.read()
        if not ret:
            break
        det.predict(fr)
        heads = det.get_head_tracks()
        tracks = tracker.update(heads, frame=None)
        items = [(t.track_id, t.tlbr, str(det.class_names.get(int(t.cls_id), "none")).lower()) for t in tracks]
        profs = gal.associate_batch(items, frame_bgr=fr, frame_idx=fidx, now_ts=100.0 + fidx / 60.0)
        gal.update_frame_end([p.worker_id for p in profs if p.worker_id > 0], frame_idx=fidx, now_ts=100.0 + fidx / 60.0)
    cap.release()

    workers = gal.get_registered_workers(filter_ghosts=True, now_ts=120.0)
    assert 1 <= len(workers) <= 3, f"Expected <= 3 workers on real 800-frame clip, got {len(workers)}: {workers}"
    assert max(w["track_id"] for w in workers) <= 3, f"Canonical ID inflation regression: {workers}"
    assert gal.next_id <= 4, f"next_id leak regression: next_id={gal.next_id}"


def test_coexisted_ids_blocks_consolidation_and_short_term_spatial_relink():
    gal = UpperTorsoReIDGallery(ttl_seconds=600.0, sim_thresh=0.70, use_deep=False, min_confirm_hits=10)
    hb1 = np.array([196.0, 90.0, 244.0, 146.0], dtype=np.float32)
    hb2 = np.array([616.0, 90.0, 664.0, 146.0], dtype=np.float32)
    f_both = _make_uniform_variant_scene(220, 90, "horiz_yellow_bar")
    # Paint second identical-uniform worker at x=640 so both coexist in-frame
    f_both2 = _make_uniform_variant_scene(640, 90, "horiz_yellow_bar")
    f_both[:, 540:, :] = f_both2[:, 540:, :]

    for fidx in range(10):
        profs = gal.associate_batch([(1, hb1, "yellow"), (2, hb2, "yellow")], f_both, frame_idx=fidx, now_ts=50.0 + fidx * 0.04)
        if fidx == 9:
            assert [p.worker_id for p in profs] == [1, 2]
        gal.update_frame_end([p.worker_id for p in profs if p.worker_id > 0], frame_idx=fidx, now_ts=50.0 + fidx * 0.04)

    assert 2 in gal.profiles[1].coexisted_ids
    assert 1 in gal.profiles[2].coexisted_ids

    # Even with identical outfit and only #1 active (#2 inactive), consolidation MUST NOT merge coexisted IDs
    gal.update_frame_end([1], frame_idx=10, now_ts=51.0)
    merged = gal.consolidate_profiles()
    assert merged == 0
    assert 1 in gal.profiles and 2 in gal.profiles

    # Short-term spatial re-link (< 3.0s, <= 0.18 diag): darkened/shifted worker #2 re-appears near X=680
    gal.update_frame_end([1], frame_idx=11, now_ts=52.0)
    f_shifted = (f_both.astype(np.float32) * 0.72).astype(np.uint8)
    hb2_shift = np.array([656.0, 94.0, 704.0, 150.0], dtype=np.float32)
    p_re = gal.associate(99, hb2_shift, "yellow", f_shifted, frame_idx=12, now_ts=52.2)
    assert p_re.worker_id == 2


def test_gateway_half_open_single_probe_and_malformed_4xx():
    import concurrent.futures
    import threading
    import time as _t
    import httpx
    from src.web.gateway_client import CircuitBreakerOpenError, InferenceCircuitBreakerClient

    probe_started = threading.Event()
    allow_probe_finish = threading.Event()
    mode = {"behavior": "503"}

    def handler(request: httpx.Request) -> httpx.Response:
        b = mode["behavior"]
        if b == "503":
            return httpx.Response(503, json={"detail": "down"})
        if b == "slow_200":
            probe_started.set()
            allow_probe_finish.wait(timeout=3.0)
            return httpx.Response(200, json={"status": "online"})
        if b == "bad_422":
            return httpx.Response(
                422,
                content=b"<html>Validation Failed</html>",
                headers={"content-type": "application/json"},
            )
        return httpx.Response(200, json={"status": "online"})

    cb = InferenceCircuitBreakerClient("http://ai-inference:8001", failure_threshold=2, recovery_timeout=0.10)
    cb._client = httpx.Client(base_url="http://ai-inference:8001", transport=httpx.MockTransport(handler))

    # 1. Malformed 4xx payload with JSON content-type MUST return _http_error WITHOUT incrementing failures
    mode["behavior"] = "bad_422"
    err_res = cb.request_json("POST", "/api/predict_image")
    assert err_res["_http_error"] == 422
    assert cb.consecutive_failures == 0
    assert cb.state == "CLOSED"

    # 2. Trip circuit to OPEN, wait cooldown, verify ONLY 1 probe enters HALF_OPEN while parallel callers fail fast
    mode["behavior"] = "503"
    with pytest.raises(RuntimeError):
        cb.request_json("GET", "/api/status")
    with pytest.raises(RuntimeError):
        cb.request_json("GET", "/api/status")
    assert cb.state == "OPEN"

    _t.sleep(0.15)
    mode["behavior"] = "slow_200"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_probe = pool.submit(cb.request_json, "GET", "/api/status")
        assert probe_started.wait(timeout=2.0)
        with pytest.raises(CircuitBreakerOpenError):
            cb.request_json("GET", "/api/status")
        allow_probe_finish.set()
        res = f_probe.result(timeout=3.0)
        assert res["status"] == "online"
        assert cb.state == "CLOSED"


def test_internal_incidents_sync_and_personnel_session_isolation(tmp_path, monkeypatch):
    from src.db.database import DatabaseManager
    import src.web.app as web_app
    from src.alerting.alert_manager import alert_manager

    test_db = DatabaseManager(database_url=f"sqlite:///{(tmp_path / 'sync_test.db').as_posix()}")
    monkeypatch.setattr(web_app, "db_manager", test_db)
    import src.db.database as db_mod
    monkeypatch.setattr(db_mod, "db_manager", test_db)

    client = TestClient(app)
    r = client.post(
        "/internal/incidents",
        json={
            "alert_id": "int-sync-001",
            "worker_id": 4,
            "camera_id": "AI-NODE-1",
            "historical_helmet_color": "blue",
            "timestamp": "2026-11-26 10:00:00",
        },
    )
    assert r.status_code == 200
    assert r.json()["saved"] is True
    incs = test_db.list_incidents(limit=10)
    assert len(incs) == 1 and incs[0]["alert_id"] == "int-sync-001"
    alert_manager.clear_alerts()

    # Session isolation check for worker_id=1 across Video vs Webcam sources
    assert test_db.save_personnel_snapshot(
        [{"track_id": 1, "current_class": "yellow", "role_title": "Worker", "status": "COMPLIANT", "total_frames_seen": 20}],
        session_id="video:clip_a.mp4",
        source_name="video_test/clip_a.mp4",
    )
    assert test_db.save_personnel_snapshot(
        [{"track_id": 1, "current_class": "white", "role_title": "Engineer", "status": "COMPLIANT", "total_frames_seen": 15}],
        session_id="browser_webcam:c960",
        source_name="WEBCAM-C960",
    )
    s_a = test_db.list_personnel_sessions(session_id="video:clip_a.mp4")
    s_b = test_db.list_personnel_sessions(session_id="browser_webcam:c960")
    assert len(s_a) == 1 and s_a[0]["current_class"] == "yellow"
    assert len(s_b) == 1 and s_b[0]["current_class"] == "white"


def test_no_biometric_mislabeling_in_benchmark_or_ui():
    import re as _re
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent
    paths = [
        root / "scripts" / "benchmark_gdut_vms_end2end.py",
        root / "src" / "web" / "templates" / "dashboard.html",
        root / "src" / "engine" / "gdut_reid.py",
        root / "Output" / "task3_eval_opt" / "gdut_vms_end2end_reid_benchmark.json",
    ]
    for p in paths:
        if not p.exists():
            continue
        cleaned = _re.sub(r"non[-_ ]?biometric", "", p.read_text(encoding="utf-8"), flags=_re.IGNORECASE)
        assert "biometric" not in cleaned.lower(), f"Forbidden standalone 'biometric' label found in {p}"


def test_solo_focus_no_duplicate_black_opencv_hud_when_disabled(monkeypatch, tmp_path):
    """Solo Focus must NOT paint the black OpenCV HUD bar when draw_video_hud=False (HTML header renders badge)."""
    import cv2
    import numpy as np
    from src.config import config
    from src.engine.pipeline import CCTVSurveillancePipeline

    clip = tmp_path / "hud_test.avi"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    assert vw.isOpened()
    vw.write(np.full((240, 320, 3), 220, dtype=np.uint8))
    vw.release()

    pipe = CCTVSurveillancePipeline(video_source=str(clip))
    pipe.set_focus_track(1)
    bright_frame = np.full((240, 640, 3), 220, dtype=np.uint8)
    report = {
        "track_id": 1,
        "bbox": [200.0, 80.0, 260.0, 140.0],
        "status": "COMPLIANT",
        "helmet_color": "blue",
        "role_short_code": "SUP",
        "historical_role_short_code": "SUP",
        "role_bgr_color": [129, 76, 15],
        "is_helmet_only": True,
    }

    monkeypatch.setattr(config, "draw_video_hud", False)
    vis_off = pipe._render_annotations(bright_frame, [], [report])
    assert float(vis_off[0:32, 0:200].mean()) > 200.0, "Top HUD strip must stay transparent when draw_video_hud=False"

    monkeypatch.setattr(config, "draw_video_hud", True)
    vis_on = pipe._render_annotations(bright_frame, [], [report])
    assert float(vis_on[0:32, 0:200].mean()) < 60.0, "Top HUD strip must render when draw_video_hud=True"
    pipe.stop()


def test_detector_zero_ablation_factory_dependency():
    """src/engine/detector.py must import BiFormerSequentialWrapper from custom_ablation_modules, not scripts.ablation_factory."""
    from pathlib import Path as _Path

    src_text = (_Path(__file__).resolve().parent.parent / "src" / "engine" / "detector.py").read_text(encoding="utf-8")
    assert "ablation_factory" not in src_text
    assert "from custom_ablation_modules import BiFormerSequentialWrapper" in src_text


def test_reid_two_stage_gate_skips_deep_extract_for_far_low_sim_noise():
    """1-frame far low-similarity noise must NOT invoke deep MobileNetV3 extraction in Step 2."""
    import numpy as np
    from src.engine.gdut_reid import UpperTorsoReIDGallery

    frame = np.full((720, 1280, 3), 25, dtype=np.uint8)
    # Worker A wears vivid orange-yellow clothing on left side (x: 60..220, y: 140..420)
    frame[140:420, 60:220] = [20, 165, 240]
    # Distant corner noise region with contrasting blue jacket (x: 1080..1240, y: 450..700)
    frame[450:700, 1080:1240] = [210, 40, 20]

    gal = UpperTorsoReIDGallery(ttl_seconds=600.0, sim_thresh=0.70, use_deep=False, min_confirm_hits=3, min_confirm_frames=6)
    fake_calls = []

    class _MockDeep:
        available = True

        def extract(self, _frame, _roi):
            fake_calls.append(1)
            vec = np.ones(712, dtype=np.float32)
            return vec / float(np.linalg.norm(vec))

    gal.deep_extractor = _MockDeep()

    # Confirm Worker A across 3 frames -> minted as canonical ID #1
    box_a = [100.0, 60.0, 180.0, 140.0]
    for fidx in range(3):
        profs = gal.associate_batch([(1, box_a, "yellow")], frame_bgr=frame, frame_idx=fidx, now_ts=100.0 + fidx * 0.04)
        gal.update_frame_end([p.worker_id for p in profs if p.worker_id > 0], frame_idx=fidx, now_ts=100.0 + fidx * 0.04)
    assert 1 in gal.profiles and len(fake_calls) == 1

    # Worker A exits -> becomes inactive candidate
    gal.update_frame_end([], frame_idx=5, now_ts=100.2)
    assert gal.profiles[1].is_active is False

    # Inject 1-frame far low-similarity noise on opposite corner (dist > 0.25 diag, distinct color)
    calls_before_noise = len(fake_calls)
    noise_box = [1120.0, 380.0, 1190.0, 450.0]
    profs_n = gal.associate_batch([(99, noise_box, "blue")], frame_bgr=frame, frame_idx=6, now_ts=100.24)
    assert len(profs_n) == 1 and profs_n[0].worker_id < 0
    assert len(fake_calls) == calls_before_noise, f"Expected 0 deep calls on 1-frame noise, got {len(fake_calls) - calls_before_noise}"


def test_hijack_excluded_ids_expires_after_ttl():
    """hijack_excluded_ids must store (cid, expire_frame) with 15-frame TTL and prune expired locks."""
    import numpy as np
    from src.engine.gdut_reid import UpperTorsoReIDGallery

    frame_a = np.full((720, 1280, 3), 20, dtype=np.uint8)
    for y in range(160, 460, 8):
        frame_a[y : y + 4, 180:360] = [30, 220, 250]  # bright yellow/orange horizontal stripes
    frame_b = np.full((720, 1280, 3), 20, dtype=np.uint8)
    for x in range(180, 360, 12):
        frame_b[160:460, x : x + 4] = [230, 30, 20]  # contrasting blue vertical stripes -> sim < 0.22

    gal = UpperTorsoReIDGallery(ttl_seconds=600.0, sim_thresh=0.70, use_deep=False, min_confirm_hits=3, min_confirm_frames=6)
    box = [220.0, 80.0, 310.0, 160.0]
    for fidx in range(4):
        profs = gal.associate_batch([(7, box, "yellow")], frame_bgr=frame_a, frame_idx=fidx, now_ts=10.0 + fidx * 0.04)
        gal.update_frame_end([p.worker_id for p in profs if p.worker_id > 0], frame_idx=fidx, now_ts=10.0 + fidx * 0.04)
    assert gal.raw_to_canonical.get(7) == 1

    # Sudden clothing divergence at frame 10 triggers anti-hijack break with TTL = 10 + 15 = 25
    gal.associate_batch([(7, box, "blue")], frame_bgr=frame_b, frame_idx=10, now_ts=10.4)
    assert 7 in gal.hijack_excluded_ids
    cid_excl, exp_frame = gal.hijack_excluded_ids[7]
    assert cid_excl == 1 and exp_frame == 25

    # Still locked at frame 24, auto-pruned at frame 26
    gal.associate_batch([], frame_bgr=frame_b, frame_idx=24, now_ts=10.96)
    gal.update_frame_end([], frame_idx=24, now_ts=10.96)
    gal.associate_batch([(8, [900.0, 90.0, 960.0, 150.0], "blue")], frame_bgr=frame_b, frame_idx=24, now_ts=10.96)
    assert 7 in gal.hijack_excluded_ids
    gal.associate_batch([(8, [900.0, 90.0, 960.0, 150.0], "blue")], frame_bgr=frame_b, frame_idx=26, now_ts=11.04)
    assert 7 not in gal.hijack_excluded_ids


def test_optimization_json_parity_internal_consistency():
    """After-2 accuracy_delta string and video_parity_fp32_vs_amp_fp16.mean_conf_diff must match identically."""
    import json as _json
    from pathlib import Path as _Path

    opt_path = _Path(__file__).resolve().parent.parent / "Output" / "task3_eval_opt" / "optimization_benchmark_table.json"
    if not opt_path.exists():
        return
    data = _json.loads(opt_path.read_text(encoding="utf-8"))
    vp = data.get("video_parity_fp32_vs_amp_fp16", {}).get("Rep-YOLO11s-GDUT-A6 (Capstone 5-Cls)", {})
    after2 = next((r for r in data.get("records", []) if r.get("stage") == "After-2 (+ CUDA AMP FP16)"), None)
    assert vp and after2 is not None
    expected_token = f"conf diff {vp['mean_conf_diff']}"
    assert expected_token in str(after2.get("accuracy_delta", "")), (
        f"Mismatch between After-2 '{after2.get('accuracy_delta')}' and video_parity mean_conf_diff={vp['mean_conf_diff']}"
    )


def test_none_conf_threshold_class_aware_and_settings_api(monkeypatch):
    import types as _types
    import torch as _torch
    from src.config import config as sys_cfg
    from src.engine.detector import SafetyObjectDetector
    from src.engine.pipeline import cctv_pipeline
    import src.web.app as web_app

    class _FakeBox:
        def __init__(self, cls_id: int, conf_val: float, xyxy: list[float]):
            self.cls = _torch.tensor([cls_id], dtype=_torch.int64)
            self.conf = _torch.tensor([conf_val], dtype=_torch.float32)
            self.xyxy = _torch.tensor([xyxy], dtype=_torch.float32)

    det = SafetyObjectDetector(
        model_path="nonexistent_stub.pt",
        conf_thresh=0.35,
        iou_thresh=0.45,
        device="cpu",
        none_conf_thresh=0.25,
    )
    fake_res = _types.SimpleNamespace(
        boxes=[
            _FakeBox(0, 0.28, [50.0, 40.0, 120.0, 110.0]),   # blue helmet < 0.35 -> filtered
            _FakeBox(4, 0.28, [220.0, 50.0, 290.0, 125.0]),  # none bare head >= 0.25 -> kept
        ]
    )
    det.yolo = _types.SimpleNamespace(predict=lambda *_a, **_kw: [fake_res])

    frame = np.full((360, 640, 3), 180, dtype=np.uint8)
    all_dets, _persons, _lat = det.predict(frame)
    head_tracks = det.get_head_tracks()
    assert len(all_dets) == 1 and all_dets[0]["name"] == "none"
    assert len(head_tracks) == 1 and int(head_tracks[0][5]) == 4

    monkeypatch.setattr(web_app, "update_env_file", lambda _updates: True)
    orig_state = (
        sys_cfg.conf_threshold,
        sys_cfg.none_conf_threshold,
        cctv_pipeline.detector.conf_thresh,
        cctv_pipeline.detector.none_conf_thresh,
        cctv_pipeline.tracker.track_thresh,
        cctv_pipeline.tracker.new_track_thresh,
    )
    client = TestClient(app)
    try:
        resp = client.post(
            "/api/settings",
            json={"conf_threshold": 0.35, "none_conf_threshold": 0.22},
        )
        assert resp.status_code == 200
        cfg_json = resp.json()["config"]
        assert cfg_json["conf_threshold"] == pytest.approx(0.35)
        assert cfg_json["none_conf_threshold"] == pytest.approx(0.22)
        assert cctv_pipeline.detector.none_conf_thresh == pytest.approx(0.22)
        assert cctv_pipeline.tracker.new_track_thresh <= 0.22 + 1e-6
    finally:
        (
            sys_cfg.conf_threshold,
            sys_cfg.none_conf_threshold,
            cctv_pipeline.detector.conf_thresh,
            cctv_pipeline.detector.none_conf_thresh,
            cctv_pipeline.tracker.track_thresh,
            cctv_pipeline.tracker.new_track_thresh,
        ) = orig_state


def test_bare_head_debounce_progress_exposed_in_report_and_ws(tmp_path):
    from src.engine.pipeline import CCTVSurveillancePipeline

    engine = SpatialSafetyEngine(consecutive_frames_thresh=5, min_tracklet_len=0)
    frame = _make_person_scene()
    trk = STrack(STrack.tlbr_to_tlwh(np.array([200.0, 90.0, 240.0, 140.0], dtype=np.float32)), score=0.82, cls_id=4)
    trk.track_id = 12
    trk.canonical_id = 1
    trk.is_activated = True
    trk.head_cls_name = "none"

    for fidx in range(5):
        reports, _ = engine.evaluate_frame(
            [trk],
            [],
            frame,
            camera_id="CAM_DEB",
            is_helmet_only=True,
            role_policy=DEFAULT_ROLE_HIERARCHY,
            class_names={0: "blue", 1: "white", 2: "yellow", 3: "red", 4: "none"},
        )
        assert len(reports) == 1
        assert reports[0]["violation_streak"] == fidx + 1
        assert reports[0]["debounce_frames"] == 5

    clip = tmp_path / "debounce_render.avi"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    assert vw.isOpened()
    vw.write(np.full((240, 320, 3), 200, dtype=np.uint8))
    vw.release()
    pipe = CCTVSurveillancePipeline(video_source=str(clip))
    try:
        canvas = np.full((360, 640, 3), 200, dtype=np.uint8)
        for streak in (3, 5):
            rep = {
                "track_id": 1,
                "bbox": [180.0, 80.0, 250.0, 150.0],
                "confidence": 0.82,
                "status": "VIOLATION",
                "violations": ["MISSING_SAFETY_HELMET"],
                "violation_streak": streak,
                "debounce_frames": 5,
                "helmet_color": None,
                "role_short_code": "BARE-HEAD",
                "historical_role_short_code": "PM/ENG",
                "role_bgr_color": [38, 38, 220],
                "is_helmet_only": True,
            }
            pipe.set_focus_track(None)
            vis_multi = pipe._render_annotations(canvas, [], [rep])
            assert vis_multi.shape == canvas.shape
            pipe.set_focus_track(1)
            vis_focus = pipe._render_annotations(canvas, [], [rep])
            assert vis_focus.shape == canvas.shape
    finally:
        pipe.stop()

    html = TestClient(app).get("/").text
    assert 'id="setNoneConf"' in html
    assert "VIOLATION: BARE HEAD" in html


def test_reid_plain_shirt_long_term_keeps_strict_deep_veto():
    from src.engine.gdut_reid import UpperTorsoIdentityProfile

    gal = UpperTorsoReIDGallery(ttl_seconds=600.0, sim_thresh=0.70, use_deep=False)

    # Profile A normalized 520-D strip ([0:474] color, [474:520] HOG texture) + 712-D deep vector
    col_a = np.zeros(474, dtype=np.float32)
    col_a[0] = 1.0
    tex_a = np.zeros(46, dtype=np.float32)
    tex_a[0] = 1.0
    strip_a = np.concatenate([col_a, tex_a])
    deep_a = np.zeros(712, dtype=np.float32)
    deep_a[0] = 1.0

    prof = UpperTorsoIdentityProfile(
        worker_id=1,
        head_box=np.array([100.0, 60.0, 160.0, 120.0], dtype=np.float32),
        helmet_class="yellow",
        strip_sig=strip_a,
        deep_sig=deep_a,
        torso_roi=(80, 130, 180, 290),
        frame_idx=0,
        now_ts=100.0,
    )

    # Probe B: same plain shirt color (sim_color > 0.88), different texture (sim_tex ~ 0.30 < 0.65),
    # intermediate deep similarity (sim_deep = 0.55: between 0.50 and 0.62)
    col_b = np.zeros(474, dtype=np.float32)
    col_b[0] = 0.90
    col_b[1] = float(np.sqrt(1.0 - 0.90 ** 2))
    tex_b_low = np.zeros(46, dtype=np.float32)
    tex_b_low[0] = 0.30
    tex_b_low[1] = float(np.sqrt(1.0 - 0.30 ** 2))
    strip_b_plain = np.concatenate([col_b, tex_b_low])

    deep_mid = np.zeros(712, dtype=np.float32)
    deep_mid[0] = 0.55
    deep_mid[1] = float(np.sqrt(1.0 - 0.55 ** 2))

    # Long-term (> 8.0s) plain-shirt probe MUST keep strict 0.62 veto floor and reject
    _sim_long, veto_long = gal._compute_appear_subscores(
        prof, strip_b_plain, deep_mid, skip_hard_veto=False, dt_seconds=15.0
    )
    assert veto_long is True

    # Short-term (<= 8.0s) re-entry still allows 0.50 floor
    _sim_short, veto_short = gal._compute_appear_subscores(
        prof, strip_b_plain, deep_mid, skip_hard_veto=False, dt_seconds=4.0
    )
    assert veto_short is False

    # Patterned uniform (sim_color >= 0.78 and sim_tex >= 0.80) allows 0.50 floor even at dt > 8.0s
    tex_b_high = np.zeros(46, dtype=np.float32)
    tex_b_high[0] = 0.80
    tex_b_high[1] = float(np.sqrt(1.0 - 0.80 ** 2))
    strip_b_patterned = np.concatenate([col_b, tex_b_high])
    _sim_pat, veto_pat = gal._compute_appear_subscores(
        prof, strip_b_patterned, deep_mid, skip_hard_veto=False, dt_seconds=15.0
    )
    assert veto_pat is False


def test_pipeline_startup_syncs_bytetrack_for_none_conf_and_c960_auto_select():
    """Verify CCTVSurveillancePipeline syncs ByteTrack thresholds at startup and dashboard avoids V-Cam."""
    from pathlib import Path as _Path
    from src.config import config as global_cfg
    from src.engine.pipeline import CCTVSurveillancePipeline

    saved = (
        global_cfg.conf_threshold,
        global_cfg.none_conf_threshold,
        global_cfg.track_high_thresh,
        global_cfg.new_track_thresh,
        global_cfg.model_path,
        global_cfg.device,
    )
    try:
        global_cfg.conf_threshold = 0.35
        global_cfg.none_conf_threshold = 0.25
        global_cfg.track_high_thresh = 0.30
        global_cfg.new_track_thresh = 0.30
        global_cfg.model_path = "nonexistent_ci_stub.pt"
        global_cfg.device = "cpu"

        pipe = CCTVSurveillancePipeline(video_source="nonexistent_startup_test.mp4")
        assert pipe.tracker.track_thresh <= 0.25 + 1e-6
        assert pipe.tracker.new_track_thresh <= 0.25 + 1e-6

        none_det = np.array([[100.0, 80.0, 180.0, 170.0, 0.27, 4.0]], dtype=np.float32)
        tracks = []
        for _ in range(2):
            tracks = pipe.tracker.update(none_det, frame=None)
        assert len(tracks) == 1
        tracks[0].head_cls_name = "none"

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        reports, _ = pipe.safety_engine.evaluate_frame(
            tracks,
            [],
            dummy_frame,
            is_helmet_only=True,
            role_policy=pipe.role_policy,
            class_names={0: "blue", 1: "white", 2: "yellow", 3: "red", 4: "none"},
        )
        assert len(reports) == 1
        assert reports[0]["current_class"] == "none"
        assert reports[0]["violation_streak"] >= 1
        assert reports[0]["debounce_frames"] == global_cfg.consecutive_violation_frames

        dash_html = (_Path(__file__).resolve().parent.parent / "src" / "web" / "templates" / "dashboard.html").read_text(
            encoding="utf-8"
        )
        assert "/c960|emeet/i" in dash_html
        assert "/v-cam|virtual/i" in dash_html
    finally:
        (
            global_cfg.conf_threshold,
            global_cfg.none_conf_threshold,
            global_cfg.track_high_thresh,
            global_cfg.new_track_thresh,
            global_cfg.model_path,
            global_cfg.device,
        ) = saved


def test_v26_class_agnostic_nms_merges_overlapping_different_colors():
    import types as _types
    import torch as _torch
    from src.engine.detector import SafetyObjectDetector

    class _FakeBox:
        def __init__(self, cls_id: int, conf_val: float, xyxy: list[float]):
            self.cls = _torch.tensor([cls_id], dtype=_torch.int64)
            self.conf = _torch.tensor([conf_val], dtype=_torch.float32)
            self.xyxy = _torch.tensor([xyxy], dtype=_torch.float32)

    det = SafetyObjectDetector(
        model_path="nonexistent_stub.pt",
        conf_thresh=0.35,
        iou_thresh=0.45,
        device="cpu",
        none_conf_thresh=0.25,
    )
    fake_res = _types.SimpleNamespace(
        boxes=[
            _FakeBox(2, 0.61, [100.0, 60.0, 220.0, 180.0]),   # yellow (lower conf)
            _FakeBox(1, 0.78, [104.0, 63.0, 224.0, 182.0]),   # white (higher conf, IoU ~0.88)
        ]
    )
    seen_kwargs: dict[str, object] = {}

    def _fake_predict(_frame, **kwargs):
        seen_kwargs.update(kwargs)
        return [fake_res]

    det.yolo = _types.SimpleNamespace(predict=_fake_predict)
    frame = np.full((360, 640, 3), 180, dtype=np.uint8)
    all_dets, _persons, _lat = det.predict(frame)
    head_tracks = det.get_head_tracks()

    assert seen_kwargs.get("agnostic_nms") is True
    assert len(all_dets) == 1
    assert all_dets[0]["name"] == "white"
    assert all_dets[0]["conf"] == pytest.approx(0.78, abs=1e-4)
    assert len(head_tracks) == 1
    assert int(head_tracks[0][5]) == 1


def test_v26_head_tracking_empty_confirmed_never_mints_head_v1():
    engine = SpatialSafetyEngine(consecutive_frames_thresh=8, min_tracklet_len=0)
    frame = _make_person_scene()
    raw_none_dets = [
        {"box": [200.0, 90.0, 260.0, 150.0], "cls": 4, "conf": 0.85, "name": "none"}
    ]
    reports, alerts = engine.evaluate_frame(
        [],
        raw_none_dets,
        frame,
        camera_id="WEBCAM-C960",
        is_helmet_only=True,
        role_policy=DEFAULT_ROLE_HIERARCHY,
        use_head_tracking=True,
    )
    assert reports == []
    assert alerts == []


def test_v26_prior_helmet_requires_10_frames_and_single_source_of_truth():
    from src.engine.pipeline import _resolve_display_role_short_code

    gal = UpperTorsoReIDGallery(ttl_seconds=600.0, sim_thresh=0.70, use_deep=False, min_confirm_hits=6)
    f_scene = _make_person_scene(person_a=(220, 90, "blue_slate"), person_b=None)
    head_box = np.array([196.0, 90.0, 244.0, 146.0], dtype=np.float32)

    # 3 frames of yellow chair/glare flicker followed by bare head confirmation at hit 6
    for fidx in range(6):
        cls_in = "yellow" if fidx < 3 else "none"
        prof = gal.associate(5, head_box, cls_in, f_scene, frame_idx=fidx, now_ts=10.0 + fidx * 0.04)
        gal.update_frame_end([prof.worker_id], fidx, now_ts=10.0 + fidx * 0.04)

    assert 1 in gal.profiles
    assert gal.profiles[1].last_known_helmet_color is None
    workers_early = gal.get_registered_workers(DEFAULT_ROLE_HIERARCHY, filter_ghosts=True, now_ts=10.3)
    assert workers_early[0]["helmet_removed"] is False

    # 10 consecutive white helmet frames reliably establish PRIOR WHITE
    for fidx in range(6, 16):
        prof = gal.associate(5, head_box, "white", f_scene, frame_idx=fidx, now_ts=10.0 + fidx * 0.04)
        gal.update_frame_end([prof.worker_id], fidx, now_ts=10.0 + fidx * 0.04)
    assert gal.profiles[1].last_known_helmet_color == "white"

    # Single source of truth for overlay tag
    assert _resolve_display_role_short_code({
        "status": "COMPLIANT",
        "role_short_code": "PM/ENG",
        "historical_role_short_code": "WORKER",
    }) == "PM/ENG"
    assert _resolve_display_role_short_code({
        "status": "VIOLATION",
        "role_short_code": "BARE-HEAD",
        "historical_role_short_code": "PM/ENG",
    }) == "PM/ENG"


def test_v26_exact_instant_compliance_no_kalman_ghost_and_solo_focus_trajectory():
    import types as _types
    from pathlib import Path as _Path
    from src.config import config as global_cfg
    from src.engine.pipeline import CCTVSurveillancePipeline

    # 1. Kalman coasting/predicted frame must NOT render ghost boxes in GDUT head-tracked mode
    engine = SpatialSafetyEngine(consecutive_frames_thresh=8, min_tracklet_len=0)
    frame = _make_person_scene()
    trk = STrack(STrack.tlbr_to_tlwh(np.array([200.0, 90.0, 240.0, 140.0], dtype=np.float32)), score=0.88, cls_id=4)
    trk.track_id = 2
    trk.canonical_id = 2
    trk.is_activated = True
    trk.head_cls_name = "none"
    trk.is_predicted = True
    trk.time_since_update = 1
    ghost_reports, ghost_alerts = engine.evaluate_frame(
        [trk],
        [],
        frame,
        is_helmet_only=True,
        role_policy=DEFAULT_ROLE_HIERARCHY,
        use_head_tracking=True,
    )
    assert ghost_reports == []
    assert ghost_alerts == []

    # 2. Exact instantaneous compliance calculation (no EMA drag: 1/3 -> 33.3%, 0/2 -> 0.0%)
    saved = (global_cfg.model_path, global_cfg.device)
    try:
        global_cfg.model_path = "nonexistent_ci_stub.pt"
        global_cfg.device = "cpu"
        pipe = CCTVSurveillancePipeline(video_source="nonexistent_kpi_test.mp4")
        pipe.gdut_reid.deep_extractor.enabled = False
        pipe.tracker.min_hits = 1

        three_people = np.array([
            [180.0, 80.0, 240.0, 140.0, 0.90, 1.0],   # white (compliant)
            [420.0, 80.0, 480.0, 140.0, 0.88, 4.0],   # none (violation)
            [680.0, 80.0, 740.0, 140.0, 0.86, 4.0],   # none (violation)
        ], dtype=np.float32)
        two_violators = np.array([
            [420.0, 80.0, 480.0, 140.0, 0.88, 4.0],
            [680.0, 80.0, 740.0, 140.0, 0.86, 4.0],
        ], dtype=np.float32)

        state = {"arr": three_people}

        def _fake_det_predict(_fr):
            arr = state["arr"]
            dets = [
                {
                    "box": row[:4].tolist(),
                    "cls": int(row[5]),
                    "conf": float(row[4]),
                    "name": "white" if int(row[5]) == 1 else "none",
                }
                for row in arr
            ]
            return dets, np.empty((0, 6), dtype=np.float32), 2.0

        pipe.detector.predict = _fake_det_predict
        pipe.detector.get_head_tracks = lambda: state["arr"]

        # Distinct background torso patches for 3 separate regions
        kpi_frame = np.full((540, 960, 3), 180, dtype=np.uint8)
        kpi_frame[155:340, 150:270] = (140, 80, 25)
        kpi_frame[155:340, 390:510] = (40, 210, 230)
        kpi_frame[155:340, 650:770] = (40, 90, 190)

        for _ in range(7):
            pipe._process_frame_unlocked(kpi_frame, "WEBCAM-C960", render_image=False)
        assert pipe.active_workers_count == 3
        assert pipe.compliance_rate == pytest.approx(33.3, abs=0.1)

        state["arr"] = two_violators
        pipe._process_frame_unlocked(kpi_frame, "WEBCAM-C960", render_image=False)
        assert pipe.active_workers_count == 2
        assert pipe.compliance_rate == pytest.approx(0.0, abs=1e-6)
        pipe.stop()
    finally:
        global_cfg.model_path, global_cfg.device = saved

    # 3. Dashboard UI invariants: trajectory Solo-Focus gate, 640px capture downscale, strict seq lock, Clip hide
    dash_html = (_Path(__file__).resolve().parent.parent / "src" / "web" / "templates" / "dashboard.html").read_text(
        encoding="utf-8"
    )
    assert "isFocused && Array.isArray(t.trajectory)" in dash_html
    assert "const maxSide = 640;" in dash_html
    assert "STATE.webcamReqSeq" in dash_html
    assert "clipSelectorLabel" in dash_html


def test_websocket_webcam_two_round_no_degradation_and_clean_reset():
    import statistics as _stats
    import time as _t
    from pathlib import Path as _Path
    from src.engine.pipeline import cctv_pipeline

    client = TestClient(app)
    img_360 = _make_person_scene(width=640, height=360)
    ok, buf = cv2.imencode(".jpg", img_360, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    assert ok
    frame_bytes = buf.tobytes()

    orig_source = str(getattr(cctv_pipeline, "source_path", "video_test/construction_site_workers_1080p.mp4"))
    r1_rtts: list[float] = []
    r2_rtts: list[float] = []
    try:
        # Round 1: 20 binary JPEG frames at 640x360 with resolution=720p query tag
        with client.websocket_connect("/ws/webcam?resolution=720p&device_label=C960_R1&return_image=false") as ws1:
            for _ in range(20):
                t0 = _t.perf_counter()
                ws1.send_bytes(frame_bytes)
                resp1 = ws1.receive_json()
                r1_rtts.append((_t.perf_counter() - t0) * 1000.0)
                assert resp1["status"] == "ok"
                assert "image" not in resp1
            assert cctv_pipeline.source_type == "browser_webcam"
            assert cctv_pipeline._external_webcam_active is True
            assert cctv_pipeline.detector.is_helmet_only is True
            cctv_pipeline.gdut_reid.profiles[9999] = object()
            assert 9999 in cctv_pipeline.gdut_reid.profiles

        # After disconnecting Round 1, _stream_worker must stay paused (_external_webcam_active stays True)
        assert cctv_pipeline._external_webcam_active is True

        # Round 2: Reconnect and confirm clean state reset + zero latency degradation + zero upscaling
        with client.websocket_connect("/ws/webcam?resolution=720p&device_label=C960_R2&return_image=false") as ws2:
            assert len(cctv_pipeline.gdut_reid.profiles) == 0
            for _ in range(20):
                t0 = _t.perf_counter()
                ws2.send_bytes(frame_bytes)
                resp2 = ws2.receive_json()
                r2_rtts.append((_t.perf_counter() - t0) * 1000.0)
                assert resp2["status"] == "ok"
                assert "image" not in resp2
            assert cctv_pipeline.frame_height == 360
    finally:
        if orig_source and orig_source != "browser_webcam":
            try:
                cctv_pipeline.switch_source(orig_source)
            except Exception:
                pass

    mean_r1 = _stats.mean(r1_rtts)
    mean_r2 = _stats.mean(r2_rtts)
    assert mean_r2 <= max(mean_r1 * 1.15, mean_r1 + 5.0), (
        f"Run #2 webcam RTT regressed: R1={mean_r1:.2f}ms vs R2={mean_r2:.2f}ms"
    )

    root_dir = _Path(__file__).resolve().parent.parent
    dash_html = (root_dir / "src" / "web" / "templates" / "dashboard.html").read_text(encoding="utf-8")
    assert "oldWs.onclose = null" in dash_html
    assert "bufferedAmount" in dash_html
    assert "clearInterval(STATE.webcamLoopTimer)" in dash_html

    det_src = (root_dir / "src" / "engine" / "detector.py").read_text(encoding="utf-8")
    assert 'predict_kwargs["agnostic_nms"] = True' in det_src


def test_websocket_webcam_disconnect_switch_mp4_error_ack_and_blank_probe(monkeypatch):
    from scripts.check_c960_webcam_live import _read_non_blank_probe
    from src.engine.pipeline import cctv_pipeline

    client = TestClient(app)
    img_360 = _make_person_scene(width=640, height=360)
    ok, buf = cv2.imencode(".jpg", img_360, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    assert ok
    frame_bytes = buf.tobytes()

    orig_source = str(getattr(cctv_pipeline, "source_path", "video_test/construction_site_workers_1080p.mp4"))
    mp4_source = "video_test/construction_site_workers_1080p.mp4"
    try:
        # 1. Verify /ws/webcam returns structured JSON error ack on processing exception instead of dropping
        def _boom(*_a, **_kw):
            raise RuntimeError("simulated inference glitch")

        with client.websocket_connect("/ws/webcam?resolution=480p&device_label=C960_ERR&return_image=false") as ws:
            monkeypatch.setattr(cctv_pipeline, "process_external_frame", _boom)
            ws.send_bytes(frame_bytes)
            err_resp = ws.receive_json()
            assert err_resp["status"] == "error"
            assert "simulated inference glitch" in err_resp.get("detail", "")

        # 2. Verify switching back to MP4 resets _external_webcam_active=False and source_type='video'
        cctv_pipeline.switch_source(mp4_source)
        assert cctv_pipeline._external_webcam_active is False
        assert cctv_pipeline.source_type == "video"
        with cctv_pipeline.lock:
            if cctv_pipeline.cap is not None and cctv_pipeline.cap.isOpened():
                ok_read, fr = cctv_pipeline.cap.read()
                assert ok_read and fr is not None and fr.size > 0
    finally:
        monkeypatch.undo()
        if orig_source and orig_source != "browser_webcam":
            try:
                cctv_pipeline.switch_source(orig_source)
            except Exception:
                pass

    # 3. Verify _read_non_blank_probe rejects both black frames and uniform flat virtual-camera frames
    class _FakeCap:
        def __init__(self, frame):
            self._frame = frame

        def read(self):
            return True, self._frame

    black = np.zeros((480, 640, 3), dtype=np.uint8)
    flat_gray = np.full((480, 640, 3), 120, dtype=np.uint8)
    assert _read_non_blank_probe(_FakeCap(black), warmup=3) is None
    assert _read_non_blank_probe(_FakeCap(flat_gray), warmup=3) is None
    assert _read_non_blank_probe(_FakeCap(img_360), warmup=3) is not None
