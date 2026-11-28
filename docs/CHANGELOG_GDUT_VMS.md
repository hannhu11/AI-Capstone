# AEGIS-CCTV Edge VMS Changelog — GDUT-HWD 5-Class Track (`FA26AI16` / `GFA26AI17`)

> Scope: Clean engineering history strictly for **AEGIS-CCTV Edge VMS v3.0** (`GDUT-HWD 5-Class`: `0: blue`, `1: white`, `2: yellow`, `3: red`, `4: none` using `Rep-YOLO11s` with `CoordConv` + 6× `RepConv` + `BiFormerBlockLite`). All numbers below were verified directly via local tests, containerized stacks, and isolated RTX 3050 benchmarks.

---

## v3.4.1 — Final-Polish Thesis, Zero DOCX Macro Leaks, MiB RAM & Harness Reconciliation, Verified BibTeX
- **Zero DOCX Macro & Accent Leaks (`scripts/export_thesis_to_docx.py`):**
  - Mapped `\Vert -> ‖`, `\vert -> |`, `\mid -> |`, cleaned nested `\text{...}`/`\mathrm{...}` wrappers inside math runs (`\mathrm{Macro\text{-}F1}` -> `Macro-F1`), and preserved LaTeX acute/grave accents (`Doll{\'a}r` -> `P. Dollár`).
  - Hardened `verify_docx()` and `verify_thesis_v3_consistency.py` with automated regex assertions banning `\bVert\b`, `\text{`, `\mathrm{`, `P. Dollar`, `within < 1 MB`, `memory.md`, `FPT Portal Rule 4`, `3490–3499`, and `+1.49 pp`.
- **RAM Unit (`MiB`), Strict FP32 Harness Reconciliation & Selective AMP TF32 Note:**
  - Corrected Linux cgroup v2 Prometheus working-set RAM reporting in `Ch. V` & `Ch. VI` to `MiB` (`bytes / 1024^2`), replacing `"within < 1 MB"` with actual scrape-timing comparison vs. `docker stats` (`86.7–91.2 MiB` vs. `86.48–88.91 MiB` Gateway; `3,009.3–3,020.6 MiB` vs. `2.939–2.944 GiB = 3,009.5–3,014.7 MiB` AI Inference, `~0.2%` / `~5.9 MiB` offset).
  - Added explicit *Strict FP32 Harness Reconciliation* below `Table VI.15` distinguishing pre-`model.fuse()` (`19.177 ms`, `9,487,583` params in `Table VI.14`) from post-`model.fuse()` (`16.10 ms`, `9,476,783` params in `Table VI.15`), and noted Selective AMP auto-disabling TF32 (`allow_tf32=False`) in `src/engine/detector.py`.
- **Unrounded Metric Delta, BibTeX Fixes & Scientific Precision:**
  - Updated unrounded `A6_yolo11s - B2` `mAP@50-95` delta from `+1.49 pp` to `+1.50 pp` (`+1.497082 pp` in `comparison_test.csv`) across Abstract, Tóm tắt, Ch. VI, Ch. VII, and README.
  - Fixed `references.bib`: `[15]` TOOD pages (`3510--3519`), `[18]` PLOS ONE author names (`Wu, Zhihui and Lei, Xiaojia and Kumar, Munish`), `[28]` NWD title protection (`{Gaussian} {Wasserstein}`), and `[10]/[11]` access dates (`Nov. 2026`).
  - Tightened claims across `frontmatter.tex`, `ch1`, `ch2`, `ch4`, `ch6`, `ch7`, `ch9` (single-seed configuration comparison wording, 12-frame AMP parity scope, `p = 0.0386 vs. V1 A1`, CI vs. local GPU Dockerfile scope, removed internal `memory.md` and `FPT Portal Rule 4` jargon from thesis body) while preserving the crisp `74`-page PDF (`47` pages `Ch. I–VII`, `0` Overfull, `0` undefined).

## v3.4.0 — Condensed Audited Thesis (74p), 100% Data-Driven Figures, Live SLA Modal Sync & Full Provenance
- **Live SLA Modal & Same-Split Official Test Sync:**
  - Re-captured `docs/ui_screenshots/03_runtime_sla_4tables_modal.png` (`Fig. V.6` / `ui_03_sla_modal.png`) and `03b_runtime_sla_loadtest.png` live via Playwright Chrome against `AEGIS-CCTV Edge VMS v3.0` on canonical `d422dd1d` weights (`143.7 FPS`, `12.16 -> 6.96 ms`, `20.206 -> 19.177 ms`, `15.22 ms`, `+0.19 ms`).
  - Synced `Table VI.13` and prose strictly to `gdut_a6_switch_to_deploy_benchmark.json` (`20.206 -> 19.177 ms` strict FP32 `-5.1%`, `11.321 -> 9.754 ms` AMP FP16 `-13.8%`), and replaced `Fig. VI.7(a)` with Official Test `a6_yolo11s_test_pred0.jpg` (`SHA-256 = eae8d6b0...`) so both `Fig. VI.6` and `Fig. VI.7` compare `A6` vs `A1` on the identical 1,587-image Official Test split.
- **100% Data-Driven Figure Reproducibility (`thesis_report/generate_gdut_figures.py` & `docs/FIGURE_PROVENANCE.md`):**
  - Refactored all 7 Matplotlib generators in `thesis_report/generate_gdut_figures.py` to read 100% dynamically from raw Kaggle CSV/JSON artifacts, `SHWD_V1/outputs/*.csv`, and RTX 3050 / FastAPI JSON benchmarks (zero hardcoded numeric arrays, zero personal paths, zero legacy `aaiml2027` paths), adding Clean-Room **Gate 4C** execution verification in `scripts/sync_public_audit_repo.py`.
  - Documented all 21 thesis figures in `docs/FIGURE_PROVENANCE.md` (5 untouched raw Kaggle outputs, 7 data-driven Matplotlib charts, 4 live UI/Grafana screenshots, 4 vector/TikZ diagrams, 1 published baseline diagram pair).
- **Thesis Condensation & De-AIGC Pass:**
  - Condensed main content `Ch. I–VII` from 73 pages to 47 crisp pages (`74` total PDF pages, `0` Overfull, `0` undefined refs, preserving 100% of all 21 figures, 30+ tables, 13 equations, 6 listings, and 31 IEEE DOIs), purged AI-tone buzzwords across all chapters + locked them in `verify_thesis_v3_consistency.py` and `export_thesis_to_docx.py`, fixed `2.88 ms/frame`, `"five objective limitations"`, and unified naming to `AEGIS-CCTV Edge VMS v3.0`.
- **Bundle & Clean-Room Hardening:**
  - Enforced `sync_public_audit_repo.py` before `package_research_evidence_bundle.py`, added verbose CPU `pytest` logging (`pytest_cpu_latest.txt > 500 B`), and enforced SHA-256 parity of `AUDIT_CODE_SNAPSHOT.md`, `.github/workflows/ci.yml`, `generate_gdut_figures.py`, and `FIGURE_PROVENANCE.md` in `verify_bundle()`.

## v3.3.0 — Canonical `21866da3 -> d422dd1d` Cryptographic Provenance, Clean-Room Reproducibility & Audited Thesis
- **Single Unbroken Weight Provenance Chain (`scripts/export_gdut_a6_deploy.py`):**
  - Exported `rep_yolo11s_gdut_a6_deploy.pt` (`38,330,403 B`, `SHA-256 = d422dd1d826c3bc2292d4074dba8bc52db2a7115baef0fd74310a25a3050b100`) and `rep_yolov8s_gdut_a6_deploy.pt` (`49,063,773 B`, `SHA-256 = 79236446df8585071029198fe048520ecae736e11b2aa7c3fcf34a1e5a85cf3a`) directly from canonical Kaggle Run A6 `best.pt` (`21866da3396e36a408236b1c3717bff0ed5b99f2e59e9cbbbce96ea6bdb68233` and `f474a15792a29459913b456141008d07e9571ee224af7802137459fee92eb340`).
  - Re-ran ALL 4 RTX 3050 benchmarks on `21866da3 -> d422dd1d` with full cryptographic provenance and 100% CSV == JSON cell parity: `switch_to_deploy()` (`-300,992` params before BN fold, `-10,800` Conv-BN fold -> `9,476,783` fused; `4.20e-6` mean / `7.93e-4` max FP32 diff; `-5.6%` FP32 / `-5.3%` AMP FP16), detector optimization ladder (`12.16 -> 6.96 ms`, `82.3 -> 143.7 FPS`, `1.75x`, `Delta=0` boxes), paired 1080p VMS + MobileNetV3 Re-ID (`15.22 ms`, `65.7 FPS`, `+3.07 ms` total Re-ID / `+0.19 ms` marginal MobileNetV3), and end-to-end 1080p multi-stream (`N=1: 45.3 FPS PASS`, `N=2: 29.0 FPS/stream PASS`, `N=4: 15.4 FPS`, `N=8: 7.9 FPS`).
- **Clean-Room Reproducibility & Hardening:**
  - Removed legacy `COPY rep_yolo11s*.yaml /app/` from `Dockerfile` and added automated Dockerfile `COPY`-target + AST script-import + CSV/JSON parity + 4-benchmark SHA-256 chain gates to `scripts/sync_public_audit_repo.py`.
  - Cleaned and whitelisted `scripts/benchmark_multistream_rtx3050.py` so `scripts/benchmark_multistream_batched_rtx3050.py` runs in Clean-Room clones without `ModuleNotFoundError`.
  - Hardened `scripts/verify_thesis_v3_consistency.py` to fail loudly (`exit != 0`) on missing workspace artifacts (`--clean-room` flag for release-distributed binaries).
  - Replaced Windows-only `Consolas` in `thesis_report/main.tex` with cross-platform `\IfFontExistsTF` fallbacks (`DejaVu Sans Mono` / `Latin Modern Mono`).
  - Hardened `/ws/webcam` error-ack JSON handling (`src/web/app.py`), `_read_non_blank_probe` flat-frame rejection (`mean > 2.0 and std > 1.0` in `scripts/check_c960_webcam_live.py`), and added webcam disconnect -> MP4 switch-back regression (`141/141` workspace CPU `pytest`; `140 passed, 1 skipped` local stub; `84 passed, 1 skipped` Clean-Room CI).
- **Thesis `v3.3.0` Audit Fixes:**
  - Upgraded Figure VI.6 to a controlled same-split comparison (`A6 YOLO11s Official Test` vs `A1 YOLO11s Official Test`, 1,587 images) with explicit axes, column normalization, and `conf=0.25` operating point.
  - Corrected Table VI.8 deltas to exact unrounded 4-decimal `summary.csv` calculations (`-0.82 / -1.00 pp` YOLO11s, `-0.36 / -0.39 pp` YOLOv8s), removed inaccurate Pareto claim vs nano-scale `B1`, added verified DOIs across `references.bib`, rewrote Vietnamese `Tóm tắt` in natural academic Vietnamese, and disclosed prototype authentication/RBAC and external Telegram/Zalo webhook boundaries candidly.

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
