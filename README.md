# AEGIS-CCTV Edge VMS Prototype — Safety Helmet & Color Classification (`GDUT-HWD 5-Class`)

Capstone Project (`FA26AI16` | Group `GFA26AI17` | FPT University) implementing **Rep-YOLO11s (`CoordConv` + 6× `RepConv` + `BiFormerBlockLite`)** with **Head-Level ByteTrack**, **Confirm-Before-Mint Non-Biometric Upper-Torso Persistent Re-ID**, a **6-Microservice Docker Stack**, and live **Prometheus/Grafana** telemetry.

- **Task Scope:** Pure **GDUT-HWD 5-Class** (`0: blue`, `1: white`, `2: yellow`, `3: red`, `4: none`) with configurable site-specific Role-by-Color mappings (`white=PM/ENG`, `blue=SUP/MEP`, `red=HSE`, `yellow=WORKER`, `none=VIOLATION`).
- **Privacy & Network Boundary:** Zero facial recognition; persistent worker re-identification relies exclusively on below-head upper-torso geometry, 520-D appearance/HOG descriptors, and optional Multi-Layer `MobileNetV3-Small` features. Video inference executes on-premises inside the site LAN; when optional Telegram/Zalo bot alerting is configured, violation JPEG snapshots are transmitted to the external bot API.
- **Cryptographic Weight Provenance (`v3.4.0`):** Canonical Kaggle Run A6 `best.pt` (`21866da3396e36a408236b1c3717bff0ed5b99f2e59e9cbbbce96ea6bdb68233`) $\rightarrow$ deployed `rep_yolo11s_gdut_a6_deploy.pt` (`d422dd1d826c3bc2292d4074dba8bc52db2a7115baef0fd74310a25a3050b100`). All four RTX 3050 benchmark receipts below are measured on this exact deployed checkpoint.

---

## 1. 6-Microservice Architecture (`docker-compose.enterprise.yml`)

```text
Browser / Chrome UI (:8080)
   ├── /                     ──▶ frontend-nginx (Nginx 1.27 Alpine reverse proxy & static UI)
   ├── /api/*                ──▶ api-gateway (FastAPI no-torch Control Plane + Circuit Breaker)
   │                              ├── PostgreSQL 16 (`incidents`, `operator_feedback`, `role_policies`, `personnel_sessions`)
   │                              └── HTTP Pool / HALF_OPEN Probe ──▶ ai-inference (PyTorch CUDA GPU + Rep-YOLO11s + Re-ID)
   └── /video_feed, /ws/*    ──▶ ai-inference (:8001, Binary JPEG WebSocket + MJPEG)

Monitoring:
   Prometheus (:9090) scrapes `api-gateway:8000/metrics` + `ai-inference:8001/metrics` ──▶ Grafana (:3000)
```

---

## 2. Quick Start (Clean Machine Setup)

1. **Clone & Configure Environment:**
   ```bash
   git clone https://github.com/hannhu11/AI-Capstone.git
   cd AI-Capstone
   cp .env.example .env
   ```
2. **Download Deploy Checkpoint, Thesis PDF & Google-Docs DOCX (GitHub Releases):**
   Heavy binaries (`rep_yolo11s_gdut_a6_deploy.pt`, `Capstone_Thesis_Report.pdf`, and `Capstone_Thesis_Report.docx`) are distributed via the repository's **GitHub Releases** tab (`v3.4.0`) rather than git blobs:
   - Place `rep_yolo11s_gdut_a6_deploy.pt` (`SHA-256: d422dd1d826c3bc2292d4074dba8bc52db2a7115baef0fd74310a25a3050b100`) at `Output/GDUT_Bộ số 1_stage2-full-train-run-a6/deploy_weights/rep_yolo11s_gdut_a6_deploy.pt` (or set `MODEL_PATH` in `.env`).
   - Place test video clip (e.g., `construction_site_workers_1080p.mp4`) in `video_test/`.
3. **Launch Full 6-Container Stack:**
   ```bash
   docker compose -f docker-compose.enterprise.yml up -d --build
   ```
   - **VMS UI:** `http://localhost:8080`
   - **Swagger / OpenAPI Docs:** `http://localhost:8080/docs`
   - **Prometheus Metrics / Targets:** `http://localhost:9090`
   - **Grafana Telemetry Dashboard:** `http://localhost:3000` (credentials configured in `.env`)
4. **Weightless CI / Local Test Verification:**
   ```bash
   python -m compileall -q src tests scripts
   MODEL_PATH=ci_stub_missing.pt CUDA_VISIBLE_DEVICES=-1 python -m pytest tests -q
   ```

---

## 3. Verified Quantitative Benchmarks (RTX 3050 Laptop GPU, 1920×1080, Canonical `21866da3 -> d422dd1d`)

### A. 5-Tier End-to-End Inference & Re-ID Breakdown (`gdut_vms_end2end_reid_benchmark.json`)
- **Paired Re-ID Protocol (Tiers 4–5):** `Within-frame paired shared-detection (AB alternating, n=120)`

| Tier | Runtime Precision / Config | Mean Latency | Throughput | p95 Latency | Peak VRAM | Notes |
|---|---|---:|---:|---:|---:|---|
| 1. Pure GDUT-A6 Detector | Strict FP32 (`TF32=False`) | `16.10 ms` | `62.1 FPS` | `17.07 ms` | `106.1 MB` | Baseline Ref (12/12 boxes) |
| 2. Pure GDUT-A6 Detector **[Default B=1]** | `TF32 FP32` (`TF32=True`) | `11.85 ms` | `84.4 FPS` | `12.36 ms` | `215.0 MB` | TensorCore TF32 (-4.25ms vs Strict) [Fastest B=1] |
| 3. Pure GDUT-A6 Detector | Selective AMP FP16 (`TF32=False`) | `12.68 ms` | `78.8 FPS` | `13.62 ms` | `210.2 MB` | Delta=0 boxes, conf diff=0.00029 |
| 4. End-to-End + 3-Strip Non-Biometric Re-ID | `TF32 FP32`, `use_deep=False` | `15.03 ms` | `66.5 FPS` | `16.35 ms` | `215.0 MB` | `det=12.15ms + strip_reid=2.88ms` |
| 5. Full VMS + Multi-Layer MobileNetV3 Re-ID | `TF32 FP32`, `use_deep=True` (`refresh=30`) | `15.22 ms` | `65.7 FPS` | `18.88 ms` | `218.6 MB` | Total Re-ID `+3.07ms` over det; marginal `+0.19ms` over 3-Strip (4 calls: extract=`20.45ms`, steady=`15.04ms`) |

- **RepConv Structural Reparameterization (`gdut_a6_switch_to_deploy_benchmark.json`, Strict FP32 `TF32=False`):** 6 layers (`[1, 3, 5, 7, 17, 20]`), `9,788,575 -> 9,487,583` parameters (`-300,992`), followed by `-10,800` Conv-BN folding via `model.fuse()` (`9,476,783` deploy fused); strict FP32 mean diff `4.20e-06`, max diff `7.93e-04`; FP32 `20.206 -> 19.177 ms`, AMP FP16 `11.321 -> 9.754 ms`.
- **Detector Optimization Ladder (`optimization_benchmark_table.json` & `.csv`, 100% cell parity, 12 real frames):**
  - `Batch=1`: `0` box-count delta (`12/12`), mean confidence diff `0.00029` (`AMP FP16 = 9.24 ms / 108.3 FPS` vs `Vec+Rep FP32 = 11.67 ms / 85.7 FPS` vs Loop BiFormer `FP32 (cuDNN TF32=True) = 12.16 ms`).
  - `Detector Micro-Batch N=4 (tensor-only)`: `0` box-count delta (`12/12`), mean confidence diff `0.00022` (`6.96 ms/frame / 143.7 FPS`).
- **End-to-End 1080p Multi-Stream (`rtx3050_gdut_a6_batched_multistream_results.json`, decode + AMP FP16 + overlay):** `N=1`: `45.3 FPS` (**PASS** 25 FPS SLA); `N=2`: `29.0 FPS/stream` (**PASS**); `N=4`: `15.4 FPS/stream` (below 25 FPS SLA); `N=8`: `7.9 FPS/stream` (below 25 FPS SLA).

### B. Dual Concurrent Load Tests (`C=10..100`, `1,400` total HTTP requests per sweep)
| Concurrency | In-Process CPU FastAPI (`load_test_results.json`) | 6-Container Docker Stack Nginx→Gateway→PostgreSQL (`load_test_enterprise_docker.json`) | Errors |
|---|---:|---:|---:|
| `C=10` (200 reqs) | `446.7 RPS` (`p95=30.56 ms`) | `260.8 RPS` (`p95=69.09 ms`) | `0.00%` |
| `C=25` (300 reqs) | `487.9 RPS` (`p95=58.05 ms`) | `256.6 RPS` (`p95=133.69 ms`) | `0.00%` |
| `C=50` (400 reqs) | `454.1 RPS` (`p95=150.7 ms`) | `238.1 RPS` (`p95=250.91 ms`) | `0.00%` |
| `C=100` (500 reqs) | `501.1 RPS` (`p95=199.25 ms`, `p99=200.31 ms`) | `225.2 RPS` (`p95=480.67 ms`, `p99=500.78 ms`) | `0.00%` |

---

## 4. Kaggle Reproducibility, Unified SOTA Baselines & Isolated `SHWD_V1` Track
- **15 Official `nvthanh2004` Kaggle Notebooks (`notebooks/gdut_official_kaggle/`):** Verified read-only 1-to-1 via Kaggle API (`11` `GDUT_V2` Prep/S1/A0–A6/FinalTest + `4` `Baselines` Stage1-YOLO11n, `B1 YOLO11n-DCRCF`, `B2 YOLOv8s-DWR-ASPP-NWD`, and unified `ce2a883e` Stage 1 comparator, raw CSV/JSON artifacts, and `raw_kaggle_outputs_summary.json`).
  - Under unified `ce2a883e` Stage 1 protocol (`1,587` GDUT Official Test images), **Proposed V2 `A6 (YOLO11s)`** achieves **`88.94% mAP@50` / `58.08% mAP@50-95`** (`87.08%` macro F1, `9,476,783` fused params, `15.207 ms` T4 FP32), outperforming **`B1`** (`84.85% / 54.77%`, `+4.09 / +3.31 pp`) and **`B2`** (`87.51% / 56.59%`, `+1.43 / +1.49 pp`, `-41.8%` params, `-32.1%` T4 latency); internal `A1` peaks at **`89.76% / 59.08%`** (with `mixup=0.1` / `hsv_v=0.45` asymmetry disclosed).
- **Isolated `SHWD_V1/` 2-Class Auxiliary Benchmark ([Summary](SHWD_V1/V1_VERIFICATION_SUMMARY.md)):** Sanitized 3-seed (`42, 1337, 2026`, `ddof=1`) evaluation tracing **Proposed V1** (`SHWD` test + zero-shot `HHW` 1,766 test images): `V1 A2 (Edge)` hits `94.30 ± 0.36%` SHWD mAP@50 (`13.55 ± 0.23 ms` T4 FP16, `p = 0.0386`) and peak zero-shot `HHW` `95.78 ± 0.07%` mAP@50 (`56.48 ± 0.42%` mAP@50-95, `4.7x` lower SD).
- **Re-ID Architecture Note:** Native GDUT 5-class runs **`src/engine/gdut_reid.py`** (`Confirm-Before-Mint` `min_confirm_hits=6`, **Two-Stage Gate** skipping `MobileNetV3` `deep_extractor.extract()` for noise unless `dist_norm <= 0.25` or `max_strip_sim >= 0.45`, adaptive back-view retention `maxlen=7` / `0.50` floor when `sim_color >= 0.78`, and 15-frame TTL `hijack_excluded_ids`); `src/engine/reid_tracker.py` is retained strictly as a legacy compatibility fallback/histogram helper.
- **Verification & Figure Provenance (`75/75 MATCH, 0 MISMATCH`):** [Kaggle Verification Report](docs/KAGGLE_FORENSIC_VERIFICATION_REPORT.md) | [Consolidated Code & Metrics Audit Snapshot](docs/AUDIT_CODE_SNAPSHOT.md) | [21-Figure Provenance Matrix](docs/FIGURE_PROVENANCE.md) | [Data-Driven Figure Generator](thesis_report/generate_gdut_figures.py).

## 5. Visual Evidence & Operational Limitations
- UI & Monitoring Screenshots (`v3.4.0`):
  - [Dashboard 1920×1080](docs/ui_screenshots/01_1920x1080_main.png) | [Compact 1366×768](docs/ui_screenshots/02_1366x768_compact.png)
  - [Runtime & SLA Tables 1–3](docs/ui_screenshots/03_runtime_sla_4tables_modal.png) | [Runtime & SLA Dual Load Tests 4A/4B](docs/ui_screenshots/03b_runtime_sla_loadtest.png)
  - [C960 Ultra Binary WebSocket Webcam Mode](docs/ui_screenshots/04_webcam_c960_canvas_mode.png) | [Solo Focus Lock & CSV Export](docs/ui_screenshots/05_solo_focus_and_export.png)
  - [Live Prometheus + Grafana Telemetry](docs/ui_screenshots/06_grafana_live_monitoring.png)
- **Field Status (`docs/UAT_SURVEY_TEMPLATE_AND_SUS_PROTOCOL.md`):** UAT & SUS rubric is finalized and maintained strictly at **`Pending Field Trial`** (no synthetic SUS scores). Physical USB `C960 Ultra` room-light cosine threshold calibration awaits on-site camera testing.
