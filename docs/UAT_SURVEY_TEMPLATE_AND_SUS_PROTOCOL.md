# UAT (User Acceptance Test) & SUS Protocol — Task 3 Review 3 (v2.3)

> **Field Trial Status:** **Pending Field Trial** (UAT questionnaire & System Usability Scale scores have NOT been collected from external safety officers yet; this protocol defines the evaluation rubric without synthetic/fabricated SUS figures).

## 1. System Usability Scale (SUS) Questionnaire (10 Items, 5-Point Likert)
Score 1 (Strongly Disagree) to 5 (Strongly Agree).
1. I think that I would like to use the AEGIS-CCTV VMS Safety Suite frequently for site monitoring.
2. I found the system unnecessarily complex.
3. I thought the dashboard was easy to use.
4. I think that I would need the support of a technical person to be able to use this system.
5. I found the various functions (Live Stream, Image Audit, Solo Focus Lock, C960 Binary Webcam, Alert Feedback, Runtime & SLA) well integrated.
6. I thought there was too much inconsistency in this system.
7. I would imagine that most safety officers would learn to use this system very quickly.
8. I found the system very cumbersome to use.
9. I felt very confident using the system to verify helmet/color-role compliance.
10. I needed to learn a lot of things before I could get going with this system.

**SUS Calculation Formula**:
- Odd items ($1,3,5,7,9$): contribution $= \text{score} - 1$
- Even items ($2,4,6,8,10$): contribution $= 5 - \text{score}$
- Total SUS Score $= 2.5 \times \sum_{i=1}^{10} \text{contribution}_i \in [0, 100]$ (Industry target $\ge 68.0$; Excellent $\ge 80.3$; Status: **Pending Field Trial**).

## 2. CSAT & Operational Task Scenarios (Recommended $N \ge 10$ Evaluators)
- **Scenario A (GDUT-HWD 5-Class & Role Policy)**: Switch input video clip or start USB C960 Ultra Binary WebSocket stream, verify active personnel count, 4-color site role hierarchy (`white=PM/ENG`, `yellow=WORKER`, `blue=SUP/MEP`, `red=HSE`), and bare-head (`none`) violation alert within 10 seconds.
- **Scenario B (Head-Anchored Re-ID, Enroll & Solo Focus)**: Click a tracked person directly on the live monitor (`getRenderedMediaRect` box) or press `[Focus]` / `[Enroll]` in the Personnel Registry, verify Solo Focus lock & exit/re-entry ID preservation even after helmet removal (`Prior Helmet -> Bare Head`), and export records via `[Export CSV]`.
- **Scenario C (HITL Incident Audit & Evidence DB)**: Inspect an Incident Evidence Snapshot, click `Confirm` / `False Pos` / `Edit Label` (5-class modal selector), and confirm persistence across restarts in PostgreSQL / SQLite.
- **Scenario D (Single-Image Audit & Runtime SLA)**: Upload a site photo in `Image Audit` to inspect per-class detections & latency, and open `Runtime & SLA` to verify all 4 hardware/load benchmark tables.

## 3. Task 3 Self-Assessment Evidence Matrix (Review 3 Readiness)
| STT | Task 3 Deliverable | Status | Verified Artifact / Command Evidence |
|---|---|---|---|
| 1 | Enterprise Industrial VMS UI/UX (Video/C960 Webcam, Solo Focus, Multi-Filter, Enroll, CSV Export, HITL Modal) | Complete | `src/web/templates/dashboard.html`, `src/web/app.py` (`/api/predict_image`, `/api/feedback`, `/api/tracking/*`) |
| 2 | High-FPS Streaming & Transport Controls (MJPEG + Binary JPEG WebSocket `/ws/webcam` + `<canvas>` Vector HUD) | Complete | `process_external_frame(..., return_annotated=False)`, `GET /api/video/*`, seamless Video <-> Webcam recovery |
| 3 | Model Optimization (RepConv Fusion + Default `TF32 FP32` + Separate `B=1`/`N=4` Parity Checks) | Complete | `custom_ablation_modules.py`, `src/engine/detector.py` (0 box-count delta on `B=1` `0.00036` & `N=4` `0.00032` over 12 worker frames; `92.6 FPS` `TF32 FP32` / `61.2 FPS` 3-strip Re-ID / `62.5 FPS` full MobileNetV3 Re-ID on RTX 3050) |
| 4 | Hardware Latency, Memory & Accuracy Parity Telemetry (`gdut_vms_end2end_reid`, `multistream`, `optimization`) | Complete | `Output/task3_eval_opt/gdut_vms_end2end_reid_benchmark.json`, `GET /api/benchmark/summary` |
| 5 | Automated CI/CD Pipeline | Complete | `.github/workflows/ci.yml` (`compileall` + `pytest tests -q` in weightless CI stub mode) |
| 6 | True 6-Microservice Enterprise Architecture (Circuit Breaker + PostgreSQL) | Complete | `Dockerfile`, `Dockerfile.gateway`, `deploy/nginx/nginx.conf`, `docker-compose.enterprise.yml` (`frontend-nginx`, `api-gateway`, `ai-inference`, `postgres`, `prometheus`, `grafana`) |
| 7 | Observability, Metrics & Live Monitoring Dashboards | Complete | `/metrics` (`aegis_cctv_*` + `aegis_gateway_*`), `monitoring/prometheus.yml`, `monitoring/grafana/dashboards/aegis_vms_overview.json` |
| 8 | Unit & Integration Tests (Normal & Weightless CI Stub Mode) | Complete | `pytest tests -q` (`126 passed` full / `125 passed, 1 skipped` CI stub, 0 failures, covering Confirm-Before-Mint Re-ID, `coexisted_ids`, same-yellow-helmet MobileNetV3, 800-frame `max_id <= 3`, `HALF_OPEN` single-probe lock, `4xx` CB isolation & Prometheus parity) |
| 9 | FastAPI Concurrent Load & Stress Test (`C=10..100`) | Complete | `scripts/load_test_api.py` (`446.7–501.1 RPS` single-node CPU in `Output/load_test_results.json`; `225.2–260.8 RPS`, `p95=69.09–480.67 ms`, `0.0%` errors across `C=10..100` on 6-container Nginx -> API Gateway -> PostgreSQL stack in `Output/load_test_enterprise_docker.json`) |
| 10 | UAT Protocol & SUS/CSAT Evaluation Framework | Pending Field Trial | `docs/UAT_SURVEY_TEMPLATE_AND_SUS_PROTOCOL.md` (Questionnaire & rubric ready; field trial execution pending) |
| 11 | English Capstone Thesis Manuscript (GDUT-HWD 5-Class Aligned) | Complete (Separate Track) | `thesis_report/Capstone_Thesis_Report.pdf` |
