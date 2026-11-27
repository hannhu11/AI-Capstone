# AEGIS-CCTV Enterprise VMS Changelog — GDUT-HWD 5-Class Track (`FA26AI16` / `GFA26AI17`)

> Scope: Clean engineering history strictly for the Capstone Enterprise VMS (`GDUT-HWD 5-Class`: `0: blue`, `1: white`, `2: yellow`, `3: red`, `4: none` using `Rep-YOLO11s` with `CoordConv` + 6× `RepConv` + `BiFormerBlockLite`). All numbers below were verified directly via local tests, containerized stacks, and isolated RTX 3050 benchmarks.

---

## v2.3 — Confirm-Before-Mint Re-ID, Gateway `HALF_OPEN` Probe Lock & `TF32 FP32` Runtime Defaults
- **Core Re-ID (`src/engine/gdut_reid.py`, `src/engine/pipeline.py`):**
  - Replaced premature ID minting with **Confirm-Before-Mint** (`_tentative_tracks` staging buffer): transient 1–5 frame background/floor detections receive negative temporary IDs, expire cleanly after `> 2` missed frames, and **never increment `self.next_id`**. Promotion at `min_confirm_hits=6` re-evaluates inactive profiles before assigning a persistent positive canonical ID.
  - Removed the inverted `coexisted_nearby` check inside `associate_batch()` so co-workers who coexisted earlier can recover their identities even when standing `< 0.10 diag` apart, while preserving strict per-frame `claimed_cids` and `consolidate_profiles()` `coexisted_ids` merge-blocking.
  - Filtered tentative tracks (`worker_id <= 0`) out of `safety_engine.evaluate_frame()` and OSD rendering until promoted.
  - Verified on live 6-container Docker (`construction_site_workers_1080p.mp4`): persistent IDs remained `[(1, 'yellow', 1040, False), (2, 'yellow', 463, True)]` at **`41.5–41.9 FPS`** (`17.8–18.9 ms` inference), eliminating previous `#273/#343` inflation.
- **API Gateway & Circuit Breaker (`src/web/gateway_client.py`, `src/web/app.py`):**
  - Added single-probe `_half_open_in_flight` guard preventing Thundering Herd during `OPEN -> HALF_OPEN` recovery.
  - Wrapped `resp.json()` in `try/except` on `>= 400` responses so malformed `4xx` payloads do not trip the Circuit Breaker.
  - Added `httpx.Limits(max_connections=200, max_keepalive_connections=60, keepalive_expiry=30.0)` and increased AnyIO thread limiter tokens to `120`, lifting 6-container Enterprise Docker Load Test throughput to **`225.2–260.8 RPS`** across `C=10..100` with **`0.0%` errors** (`Output/load_test_enterprise_docker.json`).
- **Precision Defaults, Isolated Benchmarks & UI/Grafana Polish:**
  - Switched GPU runtime default to fastest verified `B=1` precision **`ALLOW_TF32=True, USE_SELECTIVE_AMP=False`** (`src/config.py`, `.env.example`, `docker-compose.enterprise.yml`, `docker-compose.edge-simulation.yml`): isolated RTX 3050 benchmark (`Output/task3_eval_opt/gdut_vms_end2end_reid_benchmark.json`) achieved **`10.79 ms` (`92.6 FPS`)** for `TF32 FP32` and **`16.00 ms` (`62.5 FPS`)** for Full End-to-End VMS + Multi-Layer MobileNetV3 Re-ID.
  - Measured independent FP32-vs-AMP parity in `scripts/benchmark_task3_optimization.py` for both `B=1` (`10.05 ms`, `0` box delta, `0.00036` mean conf diff) and `Micro-Batch N=4` (`6.90 ms/frame`, `0` box delta, `0.00032` mean conf diff).
  - Updated `Runtime & SLA` modal to render both Table `4A` (Single-Process CPU) and Table `4B` (6-Container Enterprise Docker Load Test), renamed UI KPI `Unique IDs (Session)` and tab `Exited Memory`, fixed Grafana Panel 15 unit to `mbytes`, removed 5 obsolete v2.0 PNGs, and aligned `.github/workflows/ci.yml` dependencies.
- **Automated Verification (`tests/`):** `126 passed` (CPU full mode) and `125 passed, 1 skipped` (`MODEL_PATH=ci_stub_missing.pt` CI stub mode); strengthened `test_same_uniform_reid_stress_no_hijack` and `test_deep_mobilenet_multilayer_reid_integration` to test two workers wearing the **same** `"yellow"` helmet, plus an 800-frame real-video regression enforcing `max(track_id) <= 3` and `next_id <= 4`.

## v2.2 — Anti-Flicker Confirmation, DB Session Isolation & Internal Incident Sync
- Added `session_id` and `source_name` columns to `PersonnelSessionRecord` (`src/db/database.py`) with idempotent `ALTER TABLE` migration and `list_personnel_sessions()`.
- Added asynchronous `ai-inference -> POST /internal/incidents -> api-gateway PostgreSQL` forwarding (`src/alerting/alert_manager.py`).
- Moved OpenCV frame JPEG encoding off the inference thread (`_jpeg_encode_worker` in `src/engine/pipeline.py`), synchronized `STATE.workers` in `pollMetrics()`, eliminated duplicate OpenCV HUD (`DRAW_VIDEO_HUD=False`), sanitized browser virtual camera file-path labels, and corrected `"Non-Biometric Upper-Torso Re-ID"` terminology everywhere.

## v2.1 — Selective AMP FP16, Multi-Layer MobileNetV3 Re-ID & 6-Microservice Docker Stack
- Implemented 6-service Enterprise stack (`postgres:16-alpine`, `ai-inference`, `api-gateway`, `frontend-nginx:1.27-alpine`, `prometheus:v2.51.2`, `grafana:10.4.2`) with Circuit Breaker degradation tests and `/metrics` $\leftrightarrow$ Grafana PromQL parity.
- Added Hungarian global assignment (`linear_sum_assignment`), `520-D` upper-torso descriptor + `712-D` MobileNetV3-Small multi-layer features, Temporal Majority Voting against helmet glare flicker, binary JPEG WebSocket `/ws/webcam` with `<canvas>` vector HUD projection (`getRenderedMediaRect`), HITL feedback logging, and CSV registry export.

## v2.0 — Pure GDUT-HWD 5-Class Enterprise VMS Re-Architecture
- Migrated deployment default from legacy CHV/Unified pipelines to `rep_yolo11s_gdut_a6_deploy.pt` (`0: blue, 1: white, 2: yellow, 3: red, 4: none`), Role-by-Color hierarchy (`GET/POST /api/roles`), Solo Focus lock, Video Transport controls, and flat industrial Milestone/Siemens-inspired VMS UI.
