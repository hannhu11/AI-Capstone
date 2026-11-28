# Figure & Chart Provenance Matrix (`Capstone_Thesis_Report.pdf`, v3.4.0)

Every one of the **21 figures** in `Capstone_Thesis_Report.pdf` is traced below to its exact file path, generation method, and underlying raw experimental artifact. All 7 Matplotlib analytical charts are generated deterministically by [`thesis_report/generate_gdut_figures.py`](../thesis_report/generate_gdut_figures.py) directly from raw CSV/JSON outputs with **zero hardcoded numeric arrays** and execute standalone in both the full workspace and the Clean-Room public repository (`python thesis_report/generate_gdut_figures.py`).

## Summary by Provenance Category

| Category | Count | Figures | Verification Method |
| --- | :---: | --- | --- |
| **A. Raw Kaggle Notebook Outputs (Untouched)** | 5 | `IV.2`, `IV.3`, `VI.5`, `VI.6`, `VI.7` | Direct download from public `nvthanh2004` Kaggle kernels (pixel-exact original Ultralytics exports). |
| **B. Data-Driven Matplotlib Scripts (`generate_gdut_figures.py`)** | 7 | `IV.1`, `VI.1`, `VI.2`, `VI.3`, `VI.4`, `VI.8`, `VI.9` | 100% parsed at runtime from raw Kaggle CSV/JSONs, `SHWD_V1/outputs/*.csv`, and RTX 3050 / FastAPI JSON benchmarks. |
| **C. Live UI & Grafana Screenshots (`1920x1080`)** | 4 | `V.2`, `V.3`, `V.6`, `V.7` | Captured directly from running `AEGIS-CCTV Edge VMS v3.0` (`d422dd1d` deploy weights) & 6-container Grafana stack. |
| **D. Vector Diagrams (LaTeX TikZ & Vector PDF)** | 4 | `IV.4`, `V.1`, `V.4`, `V.5` | Compiled directly from LaTeX TikZ source or vector architecture PDF (`GDUT_A6_YOLO11s_Overall_Simplified.pdf`). |
| **E. Literature Baseline Diagrams** | 1 | `III.1` | Original published architecture diagrams from Zhao et al. (2025) (`B1`) and Song et al. (2024) (`B2`). |
| **Total** | **21** | **All Chapters III–VI** | **100% traceable; 0 fabricated charts; 0 hardcoded plot arrays.** |

---

## Complete 21-Figure Audit Table

| Fig. | LaTeX Label | Asset Path (`thesis_report/`) | Category | Exact Generator / Upstream Raw Source |
| :--- | :--- | :--- | :--- | :--- |
| **III.1** | `fig:ch3_baselines_arch` | `figures/fig_baseline_b1_yolo11n_dcrcf.jpg`, `figures/fig_baseline_b2_yolov8s_dwr_aspp_nwd.png` | Literature | Zhao et al. (2025) `YOLO11n-DCRCF` and Song et al. (2024) `YOLOv8s-DWR-ASPP-NWD` architecture figures. |
| **IV.1** | `fig:ch4_class_dist` | `figures/fig_gdut_class_distribution.png` | Script (`make_class_distribution`) | `generate_gdut_figures.py` $\leftarrow$ `gdut_stage0_and_init_checks.json` (verbatim `dataset_audit` from `nvthanh2004/gdut-dataset-preparation` `dataset_manifest.json`, `SHA-256 = ce2a883e...`). |
| **IV.2** | `fig:ch4_labels_geom` | `figures/gdut_labels_dist.jpg` | Raw Kaggle | `nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a6` $\rightarrow$ `runs/gdut_A6_yolo11s/labels.jpg`. |
| **IV.3** | `fig:ch4_train_batch` | `figures/gdut_train_batch0.jpg` | Raw Kaggle | `nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a6` $\rightarrow$ `runs/gdut_A6_yolo11s/train_batch0.jpg`. |
| **IV.4** | `fig:ch4_overall_arch` | `figures/GDUT_A6_YOLO11s_Overall_Simplified.pdf` | Vector PDF | Exact 24-layer `GDUT A6 YOLO11s` computational topology (`CoordConv` L0, 6$\times$`RepConv` L1/3/5/7/17/20, `BiFormerBlockLite` L10). |
| **V.1** | `fig:ch5_vms_arch` | Inline TikZ (`chapters/ch5_system.tex`) | LaTeX TikZ | Seven-layer architecture of the 6-container `AEGIS-CCTV Edge VMS v3.0` (`Nginx`, `Gateway`, `AI Engine`, `PostgreSQL`, `Prometheus`, `Grafana`). |
| **V.2** | `fig:ch5_ui_main` | `figures/ui_01_main.png` | Live UI | Playwright Chromium (`1920x1080`) capture of `docs/ui_screenshots/01_main_dashboard_live.png`. |
| **V.3** | `fig:ch5_ui_focus` | `figures/ui_05_solo_focus.png` | Live UI | Playwright Chromium (`1920x1080`) capture of `docs/ui_screenshots/05_solo_focus_csv_registry.png`. |
| **V.4** | `fig:ch5_db_erd` | Inline TikZ (`chapters/ch5_system.tex`) | LaTeX TikZ | Four-table relational schema matching `src/db/database.py` (`alerts`, `workers`, `cameras`, `audit_logs`). |
| **V.5** | `fig:ch5_sequence` | Inline TikZ (`chapters/ch5_system.tex`) | LaTeX TikZ | End-to-end inference, symmetrical debouncing, Re-ID, alert persistence, and HITL audit sequence. |
| **V.6** | `fig:ch5_ui_sla` | `figures/ui_03_sla_modal.png` | Live UI | Live Playwright capture (`docs/ui_screenshots/03_runtime_sla_4tables_modal.png`) rendering canonical `d422dd1d` JSON telemetry (`143.7 FPS`, `20.206 -> 19.177 ms`, `15.22 ms`, `+0.19 ms`). |
| **V.7** | `fig:ch5_ui_grafana` | `figures/ui_06_grafana.png` | Live Grafana | Playwright Chromium capture (`docs/ui_screenshots/06_grafana_15panel_live.png`) of the provisioned 15-panel Prometheus/Grafana dashboard. |
| **VI.1** | `fig:ch6_v2_baselines_vs_a6` | `figures/fig_v2_baselines_vs_a6_comparison.png` | Script (`make_v2_baselines_vs_a6`) | `generate_gdut_figures.py` $\leftarrow$ `comparison_test.csv` + `{A6_yolo11s,A6_yolov8s,yolo11_dcrcf,yolov8_dwr_aspp_nwd}_details.json` (`nvthanh2004/gdut-baseline-and-a6-stage1-protocol-comparison`). |
| **VI.2** | `fig:ch6_ablation_val_test` | `figures/fig_gdut_ablation_val_test.png` | Script (`make_ablation_val_test`) | `generate_gdut_figures.py` $\leftarrow$ `final_test_validation_summary.csv` & `final_test_summary.csv` (`nvthanh2004/gdut-final-test-evaluation`). |
| **VI.3** | `fig:ch6_perclass_a6` | `figures/fig_gdut_perclass_a6.png` | Script (`make_perclass_a6`) | `generate_gdut_figures.py` $\leftarrow$ `A6_yolo11s_per_class.json`, `A6_yolov8s_per_class.json` & `dataset_audit` (`nvthanh2004/gdut-final-test-evaluation`). |
| **VI.4** | `fig:ch6_init_epochs` | `figures/fig_gdut_init_check_and_epochs.png` | Script (`make_init_and_convergence`) | `generate_gdut_figures.py` $\leftarrow$ `initialization_check.json` (`A2/A3/A5/A6`, both backbones), `final_test_validation_summary.csv` (`S1`), and raw Kaggle `a1_yolo11s_results.csv`, `a6_yolo11s_results.csv`, `a6_yolov8s_results.csv`. |
| **VI.5** | `fig:ch6_a6_curves` | `figures/a6_yolo11s_results.png` | Raw Kaggle | `nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a6` $\rightarrow$ `runs/gdut_A6_yolo11s/results.png`. |
| **VI.6** | `fig:ch6_test_curves` | `figures/a6_yolo11s_test_pr.png`, `a6_yolo11s_test_cm.png`, `a1_yolo11s_test_pr.png`, `a1_yolo11s_test_cm.png` | Raw Kaggle | `nvthanh2004/gdut-final-test-evaluation` $\rightarrow$ `test_evaluation/{A6_yolo11s,A1_yolo11s}/{BoxPR_curve.png,confusion_matrix_normalized.png}` (1,587 Official Test images). |
| **VI.7** | `fig:ch6_qual_preds` | `figures/a6_yolo11s_test_pred0.jpg`, `figures/a1_yolo11s_test_pred0.jpg` | Raw Kaggle | `nvthanh2004/gdut-final-test-evaluation` $\rightarrow$ `test_evaluation/{A6_yolo11s,A1_yolo11s}/val_batch0_pred.jpg` (100% Official Test split for both panels; `A6` `SHA-256 = eae8d6b0...`). |
| **VI.8** | `fig:ch6_v1_shwd_hhw` | `figures/fig_v1_shwd_hhw_ablation.png` | Script (`make_v1_shwd_hhw_ablation`) | `generate_gdut_figures.py` $\leftarrow$ `SHWD_V1/outputs/shwd_test_ablation_seed{42,1337,2026}.csv`, `t4_latency_seed{42,1337,2026}.csv`, `v1_hhw_all_3seeds.csv`. |
| **VI.9** | `fig:ch6_opt_load` | `figures/fig_gdut_optimization_and_multistream.png` | Script (`make_optimization_and_load`) | `generate_gdut_figures.py` $\leftarrow$ `Output/task3_eval_opt/optimization_benchmark_table.json` (`d422dd1d` RTX 3050), `Output/load_test_results.json`, `Output/load_test_enterprise_docker.json`. |

## Reproducibility Command

```bash
python thesis_report/generate_gdut_figures.py --out-dir ./tmp_reproduced_figures --no-copy-evidence
```

This command runs automatically as **Gate 4C** inside `scripts/sync_public_audit_repo.py --verify` against a fresh isolated clone of the Clean-Room repository.
