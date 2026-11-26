# AEGIS-CCTV Enterprise VMS (GDUT-HWD 5-Class) — Consolidated Code & Audit Snapshot

> Self-contained verifier snapshot bundling the Kaggle 1-to-1 forensic verification report, core Re-ID and API Gateway resilience code, CI workflow, and hardware/Docker benchmark JSONs.

## 1. Kaggle Forensic Verification Report (`docs/KAGGLE_FORENSIC_VERIFICATION_REPORT.md`)

```markdown
# Kaggle Forensic Verification Report — GDUT-HWD 5-Class (`nvthanh2004` + `tundng111`)

> **Generated (UTC):** `2026-11-26T10:12:35.932046+00:00`
> **Verification Policy:** Zero trust in secondary `.md` summaries until independently verified 1-to-1 against public Read-Only Kaggle notebooks (`.ipynb`), raw `run_manifest.json`, `results.csv`, `initialization_check.json`, `architecture_rank_0.json`, `test_evaluation/summary.csv`, `*_per_class.json`, and raw Kaggle `execution_log.txt`.
> **Audit Score:** **`69/70` checks MATCHED (`1` mismatches)** across all 11 official `nvthanh2004` notebooks and 5 `tundng111` baseline/control notebooks (16 total).

---

## 1. Downloaded Public Kaggle Inventory (Read-Only, Zero `.pt` / Heavy Binary Bloat)

| # | Owner / Kernel Slug | Ver | Cells | Light Files Downloaded | Kaggle URL |
|---:|---|---:|---:|---:|---|
| 1 | `nvthanh2004/gdut-dataset-preparation` | v1 | 19 | 16 | [gdut-dataset-preparation](https://www.kaggle.com/code/nvthanh2004/gdut-dataset-preparation) |
| 2 | `nvthanh2004/gdut-structural-re-parameterize-yolo-architecture1` | v1 | 19 | 16 | [gdut-structural-re-parameterize-yolo-architecture1](https://www.kaggle.com/code/nvthanh2004/gdut-structural-re-parameterize-yolo-architecture1) |
| 3 | `nvthanh2004/gdut-structural-re-parameterize-yolo-architecture2` | v1 | 19 | 16 | [gdut-structural-re-parameterize-yolo-architecture2](https://www.kaggle.com/code/nvthanh2004/gdut-structural-re-parameterize-yolo-architecture2) |
| 4 | `nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a0` | v1 | 19 | 15 | [gdut-stage2-ablation-setup-full-train-run-a0](https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a0) |
| 5 | `nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a1` | v1 | 19 | 25 | [gdut-stage2-ablation-setup-full-train-run-a1](https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a1) |
| 6 | `nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a2` | v1 | 19 | 27 | [gdut-stage2-ablation-setup-full-train-run-a2](https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a2) |
| 7 | `nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a3` | v1 | 19 | 27 | [gdut-stage2-ablation-setup-full-train-run-a3](https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a3) |
| 8 | `nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a4` | v1 | 19 | 25 | [gdut-stage2-ablation-setup-full-train-run-a4](https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a4) |
| 9 | `nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a5` | v1 | 19 | 27 | [gdut-stage2-ablation-setup-full-train-run-a5](https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a5) |
| 10 | `nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a6` | v1 | 19 | 27 | [gdut-stage2-ablation-setup-full-train-run-a6](https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a6) |
| 11 | `nvthanh2004/gdut-final-test-evaluation` | v1 | 17 | 30 | [gdut-final-test-evaluation](https://www.kaggle.com/code/nvthanh2004/gdut-final-test-evaluation) |
| 12 | `tundng111/gdut-yolo11n-dcrcf-baseline` | v1 | 29 | 74 | [gdut-yolo11n-dcrcf-baseline](https://www.kaggle.com/code/tundng111/gdut-yolo11n-dcrcf-baseline) |
| 13 | `tundng111/gdut-yolov8-v5-matched-v8f` | v1 | 29 | 78 | [gdut-yolov8-v5-matched-v8f](https://www.kaggle.com/code/tundng111/gdut-yolov8-v5-matched-v8f) |
| 14 | `tundng111/gdut-yolo11n-dcrcf-matched-ablation-c3` | v1 | 29 | 78 | [gdut-yolo11n-dcrcf-matched-ablation-c3](https://www.kaggle.com/code/tundng111/gdut-yolo11n-dcrcf-matched-ablation-c3) |
| 15 | `tundng111/gdut-yolo11n-dcrcf-matched-ablation-c0` | v7 | 29 | 78 | [gdut-yolo11n-dcrcf-matched-ablation-c0](https://www.kaggle.com/code/tundng111/gdut-yolo11n-dcrcf-matched-ablation-c0) |
| 16 | `tundng111/gdut-yolov8-v5-matched-v8s` | v1 | 29 | 78 | [gdut-yolov8-v5-matched-v8s](https://www.kaggle.com/code/tundng111/gdut-yolov8-v5-matched-v8s) |

---

## 2. Dataset Provenance & Cryptographic Fingerprint Audit

- **Dataset Manifest SHA-256 (`prepared_gdut/dataset_manifest.json`):** `ce2a883e0b5050edb96d494218e85a9d16ea04b9a19dc5723c572cb502edfe9c`
- **Class Ordering Locked:** `['blue', 'white', 'yellow', 'red', 'none']` (`0: blue, 1: white, 2: yellow, 3: red, 4: none`), `seed = 3407`
- **Cross-Split SHA-256 Pixel Duplicates Purged from `trainval`:** `6` (preserving all `1,587` official test images):
  - Removed `JPEGImages/00286.jpg` == kept Test `JPEGImages/02142.jpg` (`pixel_hash=8f7e4392ff350df8...`)
  - Removed `JPEGImages/00641.jpg` == kept Test `JPEGImages/00615.jpg` (`pixel_hash=02248ec06f1cee1d...`)
  - Removed `JPEGImages/00676.jpg` == kept Test `JPEGImages/00239.jpg` (`pixel_hash=a494deee06dc5b39...`)
  - Removed `JPEGImages/00801.jpg` == kept Test `JPEGImages/01007.jpg` (`pixel_hash=2eb32cda849553dc...`)
  - Removed `JPEGImages/03087.jpg` == kept Test `JPEGImages/01726.jpg` (`pixel_hash=14a89d489f5f1760...`)
  - Removed `JPEGImages/03365.jpg` == kept Test `JPEGImages/02224.jpg` (`pixel_hash=ca61ca89138d7ee3...`)
- **Clean Split Image Counts:** `train=1265`, `val=316`, `test=1587` (total clean = `3168`)
- **Offline Color-Preserving HardAug (`augmentation_manifest.json`):** `hue_saturation_transform_removed=True`, `copies_per_train_image=1` -> Stage 2 Train = `2,530` files

---

## 3. Exact Full-Precision Test CSV & T4 Inference Log (`gdut-final-test-evaluation`)

> Notice: `01_KET_QUA_A1_A6.md` only had rounded 1-decimal log percentages. Pulling `test_evaluation/summary.csv` directly recovers full 4-decimal percentages (e.g., breaking the apparent `88.7%` tie between `YOLOv8s A1 = 88.7328%` and `A3 = 88.7305%`).

| Backbone | Config | Exact P (%) | Exact R (%) | Exact mAP@50 (%) | Exact mAP@50-95 (%) | Log T4 Infer (ms) | `none` AP@50 (%) | `none` AP@50-95 (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `yolo11s` | **A0** | 90.98 | 82.83 | 88.8149 | 58.4641 | 8.0 ms | 83.3 | 52.5 |
| `yolo11s` | **A1** | 91.24 | 84.36 | 89.7616 | 59.0804 | 7.5 ms | 84.9 | 54.4 |
| `yolo11s` | **A2** | 91.06 | 83.51 | 89.4383 | 58.5829 | 8.3 ms | 83.9 | 53.2 |
| `yolo11s` | **A3** | 90.73 | 83.90 | 89.2501 | 58.3275 | 8.9 ms | 84.2 | 53.4 |
| `yolo11s` | **A4** | 90.15 | 84.30 | 89.1989 | 58.4888 | 8.0 ms | 83.4 | 53.1 |
| `yolo11s` | **A5** | 90.22 | 84.22 | 89.2391 | 58.6058 | 8.5 ms | 83.9 | 53.9 |
| `yolo11s` | **A6** | 90.19 | 84.28 | 88.9379 | 58.0825 | 9.2 ms | 84.6 | 53.4 |
| `yolov8s` | **A0** | 90.12 | 81.84 | 87.4565 | 57.1973 | 7.5 ms | 81.9 | 50.9 |
| `yolov8s` | **A1** | 90.52 | 83.35 | 88.7328 | 57.7618 | 7.4 ms | 83.7 | 52.2 |
| `yolov8s` | **A2** | 90.59 | 83.07 | 88.5582 | 57.2438 | 7.7 ms | 83.7 | 51.8 |
| `yolov8s` | **A3** | 89.71 | 83.56 | 88.7305 | 57.1614 | 8.5 ms | 83.3 | 51.5 |
| `yolov8s` | **A4** | 90.95 | 82.56 | 88.2910 | 57.2358 | 7.5 ms | 83.6 | 51.7 |
| `yolov8s` | **A5** | 90.86 | 82.42 | 88.3448 | 57.3292 | 8.4 ms | 83.3 | 51.9 |
| `yolov8s` | **A6** | 90.32 | 82.63 | 88.3692 | 57.3711 | 9.1 ms | 83.5 | 51.8 |

---

## 4. Key Forensic Findings & Honest Nuances Discovered

- **[HONEST_NUANCE] All 11 `nvthanh2004` + 5 `tundng111` (16 total) notebooks pulled with 100% HTTP 200; A0-A6, B1 Control, C3, C0 & V8S match raw Kaggle artifacts.**
  Direct 1-to-1 verification against Kaggle `run_manifest.json`, `evaluation.json`, `initialization_check.json`, `results.csv`, `test_evaluation/summary.csv`, `A6_*_per_class.json`, `execution_log.txt`, and `B1/B2/C3/C0/V8S` `metrics.json` confirms exact agreement across A0-A6, B1 Control, `C3` (57.283±0.173%), `C0` (58.812±0.082%), and `V8S` (61.414±0.074%) (1 honest precision/SD-convention mismatch flagged on B2 V8F).
- **[FINDING_1] Precision resolution gap in `01_KET_QUA_A1_A6.md`: Test CSV stores full precision (`A1_V8 = 88.7328%` vs `A3_V8 = 88.7305%`), whereas `.md` only showed 1 decimal (`88.7%` tie).**
  `01_KET_QUA_A1_A6.md` Section 5.2 noted that `A1` and `A3` tied at `88.7%` and could not be ranked without full floats; pulling `test_evaluation/summary.csv` from Kaggle recovers exact floats: `A1 YOLOv8s mAP50 = 88.7328%`, `A3 YOLOv8s mAP50 = 88.7305%` (A1 strictly higher by +0.0024 pp), and `A6 YOLO11s = 88.9379%` / `58.0825%`, `A6 YOLOv8s = 88.3692%` / `57.3711%`.
- **[FINDING_2] Dataset manifest fingerprint nuance between Thanh's A0-A6 (`ce2a883e...`) and Dung's B2 (`026a2afb...`) is real and confirmed in manifests.**
  Inspecting `V8F-3407/run_manifest.json` -> `inputs.manifest_sha256` directly confirms `026a2afb617254b9dd2b9461651bce6f74295d18f29d52a790bf755d32944f65`, matching `BASELINE_AUDIT_B1_B2_REVIEW3.md` and Chapter IV's Provenance Note.
- **[FINDING_3] A1 is legitimately #1 on GDUT mAP; A6 does NOT beat A1 on GDUT single-seed (already reported honestly, NOT inflated).**
  Raw `summary.csv` proves `A1_yolo11s` reaches `89.7616% mAP50 / 59.0804% mAP50-95` vs `A6_yolo11s` `88.9379% / 58.0825%`. `gdut-final-test-evaluation` `final_test()` also ran unfused (`switch_to_deploy_in_final_test=False`), explaining `9.2 ms` vs `7.5 ms` T4 latency (Cell 8 `preflight.py` unit test calls `switch_to_deploy()` only on synthetic tensors); post-fusion RTX 3050 equivalence (`-300,992` params, max diff `7.93e-4`, `-3.6%` Strict FP32 / `-6.3%` AMP latency) was measured locally on `GDUT_A6_YOLO11s_best.pt`.
- **[FINDING_4] Stage 2 `args.yaml` augmentation asymmetry (`mixup=0.1` in A1 vs `0.0` in A2..A6) verified verbatim in Kaggle outputs.**
  Verified directly inside Kaggle `args.yaml` files: `A1` uses `mixup=0.1, hsv_v=0.45`, while `A2..A6` use `mixup=0.0, hsv_v=0.40` (with `hsv_h=0.0, hsv_s=0.0` locked across all runs).
- **[FINDING_5 (MISMATCH / NUANCE)] B2 5-Seed V8F precision rounding (`90.25%` vs exact `90.2447%` -> `90.24%`) & `pstdev (ddof=0)` vs `sample stdev (ddof=1)`.**
  `BASELINE_AUDIT_B1_B2_REVIEW3.md` / `ch6_results.tex` list `V8F` 5-seed as `P=90.25±0.94, R=86.47±0.58, mAP50=91.82±0.17, mAP50-95=60.781±0.353`. Recomputing from the 5 Kaggle `metrics.json` files gives `mean P = 90.2447%` (rounds to `90.24%`, -0.0053 pp vs `90.25%`), and the reported `±0.94/±0.58/±0.17/±0.353` corresponds to population `pstdev (ddof=0)`; standard sample SD (`ddof=1`, n=5) is `±1.05/±0.65/±0.19/±0.394`.

---

## 5. Complete 1-to-1 Cross-Verification Matrix (`MATCH / MISMATCH`)

| Category | Checked Item | Secondary Doc Claim (`.md`/`.tex`) | Raw Kaggle Truth (`.ipynb`/CSV/JSON`) | Verdict |
|---|---|---|---|---|
| 1. Dataset & Provenance | Cryptographic SHA-256 of prepared_gdut/dataset_manifest.json across Prep, S1, A0..A6 | `ce2a883e0b5050edb96d494218e85a9d16ea04b9a19dc5723c572cb502edfe9c` | Prep=`ce2a883e0b5050ed...`, all 16 S1/A0..A6 runs match=True | **MATCH** |
| 1. Dataset & Provenance | Deduplication (3,174 -> 3,168), Split (1,265/316/1,587), Taxonomy & Box Counts | 3,168 clean (6 dupes removed), 1265/316/1587, boxes 7,537/1,738/9,574, names=[blue,white,yellow,red,none] | records=3168, excluded=6 [('JPEGImages/00286.jpg', 'JPEGImages/02142.jpg'), ('JPEGImages/00641.jpg', 'JPEGImages/00615.jpg'), ('JPEGImages/00676.jpg', 'JPEGImages/00239.jpg'), ('JPEGImages/00801.jpg', 'JPEGImages/01007.jpg'), ('JPEGImages/03087.jpg', 'JPEGImages/01726.jpg'), ('JPEGImages/03365.jpg', 'JPEGImages/02224.jpg')], splits={'test': 1587, 'val': 316, 'train': 1265}, boxes={'train': {'red': 1865, 'white': 1462, 'yellow': 1465, 'none': 1767, 'blue': 978}, 'val': {'red': 279, 'white': 349, 'yellow': 448, 'none': 404, 'blue': 258}, 'test': {'white': 1968, 'red': 2083, 'none': 2200, 'yellow': 1962, 'blue': 1361}} | **MATCH** |
| 1. Dataset & Provenance | Color-Safe Aug (`hsv_h=0, hsv_s=0`) & Online Asymmetry (`A1` vs `A2..A6`) | hsv_h=0, hsv_s=0; A1 mixup=0.1, hsv_v=0.45; A2..A6 mixup=0, hsv_v=0.4 | hsv_locked=True; A1=(mixup=0.1, hsv_v=0.45); A6=(mixup=0.0, hsv_v=0.4) | **MATCH** |
| 2. Architecture & Code | Compiled A6/A3 Layer Indices, Param Count (9,788,575) & BiFormer(r=8,k=4,h=4) | Y11 A6: Coord[0], Rep[1,3,5,7,17,20], BiF[10], 9.79M; V8 A6: Rep[1,3,5,7,16,19], BiF[9]; Y11 A3: 7xRep | Y11_A6={'CoordConv': [0], 'RepConv': [1, 3, 5, 7, 17, 20], 'BiFormerBlockLite': [10]} (9788575); V8_A6={'CoordConv': [0], 'RepConv': [1, 3, 5, 7, 16, 19], 'BiFormerBlockLite': [9]}; Y11_A3={'CoordConv': [], 'RepConv': [0, 1, 3, 5, 7, 17, 20], 'BiFormerBlockLite': []} | **MATCH** |
| 2. Architecture & Code | RepConv Identity Branch Policy & Focal-EIoU Loss Wiring in ResearchTrainer | 02_PHUONG_PHAP Section 9.1: stride-2 RepConv has NO identity branch; Section 13.3: init_criterion wires FocalEIoUBboxLoss | rbr_identity=(c1==c2 and s==1): True; FocalEIoUBboxLoss wired in research_trainer.py: True | **MATCH** |
| 3. Validation (yolo11s) | yolo11s A0 Val (P / R / mAP50 / mAP50-95 %) | 92.21 / 84.80 / 91.83 / 60.83 | 92.21 / 84.80 / 91.83 / 60.83 | **MATCH** |
| 3. Validation (yolo11s) | yolo11s A1 Val (P / R / mAP50 / mAP50-95 %) | 91.33 / 87.37 / 93.40 / 61.69 | 91.33 / 87.37 / 93.40 / 61.69 | **MATCH** |
| 3. Validation (yolo11s) | yolo11s A2 Val (P / R / mAP50 / mAP50-95 %) | 91.01 / 86.25 / 92.72 / 61.31 | 91.01 / 86.25 / 92.72 / 61.31 | **MATCH** |
| 3. Validation (yolo11s) | yolo11s A3 Val (P / R / mAP50 / mAP50-95 %) | 90.79 / 87.86 / 92.88 / 61.41 | 90.79 / 87.86 / 92.88 / 61.41 | **MATCH** |
| 3. Validation (yolo11s) | yolo11s A4 Val (P / R / mAP50 / mAP50-95 %) | 90.64 / 87.49 / 92.77 / 61.36 | 90.64 / 87.49 / 92.77 / 61.36 | **MATCH** |
| 3. Validation (yolo11s) | yolo11s A5 Val (P / R / mAP50 / mAP50-95 %) | 90.05 / 87.14 / 92.40 / 61.08 | 90.05 / 87.14 / 92.40 / 61.08 | **MATCH** |
| 3. Validation (yolo11s) | yolo11s A6 Val (P / R / mAP50 / mAP50-95 %) | 90.45 / 87.58 / 92.50 / 61.23 | 90.45 / 87.58 / 92.50 / 61.23 | **MATCH** |
| 3. Validation (yolov8s) | yolov8s A0 Val (P / R / mAP50 / mAP50-95 %) | 91.54 / 85.75 / 91.88 / 60.55 | 91.54 / 85.75 / 91.88 / 60.55 | **MATCH** |
| 3. Validation (yolov8s) | yolov8s A1 Val (P / R / mAP50 / mAP50-95 %) | 91.55 / 86.21 / 93.14 / 61.19 | 91.55 / 86.21 / 93.14 / 61.19 | **MATCH** |
| 3. Validation (yolov8s) | yolov8s A2 Val (P / R / mAP50 / mAP50-95 %) | 91.51 / 86.79 / 92.47 / 60.32 | 91.51 / 86.79 / 92.47 / 60.32 | **MATCH** |
| 3. Validation (yolov8s) | yolov8s A3 Val (P / R / mAP50 / mAP50-95 %) | 91.47 / 86.67 / 92.41 / 60.30 | 91.47 / 86.67 / 92.41 / 60.30 | **MATCH** |
| 3. Validation (yolov8s) | yolov8s A4 Val (P / R / mAP50 / mAP50-95 %) | 92.60 / 84.91 / 92.35 / 60.21 | 92.60 / 84.91 / 92.35 / 60.21 | **MATCH** |
| 3. Validation (yolov8s) | yolov8s A5 Val (P / R / mAP50 / mAP50-95 %) | 92.02 / 85.56 / 92.59 / 60.26 | 92.02 / 85.56 / 92.59 / 60.26 | **MATCH** |
| 3. Validation (yolov8s) | yolov8s A6 Val (P / R / mAP50 / mAP50-95 %) | 93.21 / 85.11 / 92.56 / 60.68 | 93.21 / 85.11 / 92.56 / 60.68 | **MATCH** |
| 4. Initialization & Epochs | yolo11s A2 Pre-Finetune Val mAP50 / mAP50-95 | 91.83 / 60.83 | 91.83 / 60.83 | **MATCH** |
| 4. Initialization & Epochs | yolo11s A3 Pre-Finetune Val mAP50 / mAP50-95 | 91.83 / 60.83 | 91.83 / 60.83 | **MATCH** |
| 4. Initialization & Epochs | yolo11s A5 Pre-Finetune Val mAP50 / mAP50-95 | 86.34 / 56.15 | 86.34 / 56.15 | **MATCH** |
| 4. Initialization & Epochs | yolo11s A6 Pre-Finetune Val mAP50 / mAP50-95 | 86.34 / 56.15 | 86.34 / 56.15 | **MATCH** |
| 4. Initialization & Epochs | yolov8s A2 Pre-Finetune Val mAP50 / mAP50-95 | 91.88 / 60.55 | 91.88 / 60.55 | **MATCH** |
| 4. Initialization & Epochs | yolov8s A3 Pre-Finetune Val mAP50 / mAP50-95 | 91.88 / 60.55 | 91.88 / 60.55 | **MATCH** |
| 4. Initialization & Epochs | yolov8s A5 Pre-Finetune Val mAP50 / mAP50-95 | 91.88 / 60.55 | 91.88 / 60.55 | **MATCH** |
| 4. Initialization & Epochs | yolov8s A6 Pre-Finetune Val mAP50 / mAP50-95 | 91.88 / 60.55 | 91.88 / 60.55 | **MATCH** |
| 4. Initialization & Epochs | yolo11s A1 Epochs (Stopped / Best) | 70 / 20 | 70 / 20 | **MATCH** |
| 4. Initialization & Epochs | yolo11s A2 Epochs (Stopped / Best) | 52 / 2 | 52 / 2 | **MATCH** |
| 4. Initialization & Epochs | yolo11s A3 Epochs (Stopped / Best) | 66 / 16 | 66 / 16 | **MATCH** |
| 4. Initialization & Epochs | yolo11s A4 Epochs (Stopped / Best) | 65 / 15 | 65 / 15 | **MATCH** |
| 4. Initialization & Epochs | yolo11s A5 Epochs (Stopped / Best) | 65 / 15 | 65 / 15 | **MATCH** |
| 4. Initialization & Epochs | yolo11s A6 Epochs (Stopped / Best) | 66 / 16 | 66 / 16 | **MATCH** |
| 4. Initialization & Epochs | yolov8s A1 Epochs (Stopped / Best) | 63 / 13 | 63 / 13 | **MATCH** |
| 4. Initialization & Epochs | yolov8s A2 Epochs (Stopped / Best) | 65 / 15 | 65 / 15 | **MATCH** |
| 4. Initialization & Epochs | yolov8s A3 Epochs (Stopped / Best) | 51 / 1 | 51 / 1 | **MATCH** |
| 4. Initialization & Epochs | yolov8s A4 Epochs (Stopped / Best) | 66 / 16 | 66 / 16 | **MATCH** |
| 4. Initialization & Epochs | yolov8s A5 Epochs (Stopped / Best) | 51 / 1 | 51 / 1 | **MATCH** |
| 4. Initialization & Epochs | yolov8s A6 Epochs (Stopped / Best) | 66 / 16 | 66 / 16 | **MATCH** |
| 5. Official Test (yolo11s) | yolo11s A0 Test 1,587 imgs (P / R / mAP50 / mAP50-95) | 91.0 / 82.8 / 88.8 / 58.5% | 91.0 / 82.8 / 88.8 / 58.5% (exact CSV: 90.9805 / 82.8330 / 88.8149 / 58.4641%) | **MATCH** |
| 5. Official Test (yolo11s) | yolo11s A1 Test 1,587 imgs (P / R / mAP50 / mAP50-95) | 91.2 / 84.4 / 89.8 / 59.1% | 91.2 / 84.4 / 89.8 / 59.1% (exact CSV: 91.2358 / 84.3641 / 89.7616 / 59.0804%) | **MATCH** |
| 5. Official Test (yolo11s) | yolo11s A2 Test 1,587 imgs (P / R / mAP50 / mAP50-95) | 91.1 / 83.5 / 89.4 / 58.6% | 91.1 / 83.5 / 89.4 / 58.6% (exact CSV: 91.0594 / 83.5099 / 89.4383 / 58.5829%) | **MATCH** |
| 5. Official Test (yolo11s) | yolo11s A3 Test 1,587 imgs (P / R / mAP50 / mAP50-95) | 90.7 / 83.9 / 89.3 / 58.3% | 90.7 / 83.9 / 89.3 / 58.3% (exact CSV: 90.7325 / 83.9047 / 89.2501 / 58.3275%) | **MATCH** |
| 5. Official Test (yolo11s) | yolo11s A4 Test 1,587 imgs (P / R / mAP50 / mAP50-95) | 90.2 / 84.3 / 89.2 / 58.5% | 90.2 / 84.3 / 89.2 / 58.5% (exact CSV: 90.1532 / 84.2964 / 89.1989 / 58.4888%) | **MATCH** |
| 5. Official Test (yolo11s) | yolo11s A5 Test 1,587 imgs (P / R / mAP50 / mAP50-95) | 90.2 / 84.2 / 89.2 / 58.6% | 90.2 / 84.2 / 89.2 / 58.6% (exact CSV: 90.2225 / 84.2201 / 89.2391 / 58.6058%) | **MATCH** |
| 5. Official Test (yolo11s) | yolo11s A6 Test 1,587 imgs (P / R / mAP50 / mAP50-95) | 90.2 / 84.3 / 88.9 / 58.1% | 90.2 / 84.3 / 88.9 / 58.1% (exact CSV: 90.1942 / 84.2807 / 88.9379 / 58.0825%) | **MATCH** |
| 5. Official Test (yolov8s) | yolov8s A0 Test 1,587 imgs (P / R / mAP50 / mAP50-95) | 90.1 / 81.8 / 87.5 / 57.2% | 90.1 / 81.8 / 87.5 / 57.2% (exact CSV: 90.1246 / 81.8363 / 87.4565 / 57.1973%) | **MATCH** |
| 5. Official Test (yolov8s) | yolov8s A1 Test 1,587 imgs (P / R / mAP50 / mAP50-95) | 90.5 / 83.4 / 88.7 / 57.8% | 90.5 / 83.4 / 88.7 / 57.8% (exact CSV: 90.5209 / 83.3533 / 88.7328 / 57.7618%) | **MATCH** |
| 5. Official Test (yolov8s) | yolov8s A2 Test 1,587 imgs (P / R / mAP50 / mAP50-95) | 90.6 / 83.1 / 88.6 / 57.2% | 90.6 / 83.1 / 88.6 / 57.2% (exact CSV: 90.5929 / 83.0716 / 88.5582 / 57.2438%) | **MATCH** |
| 5. Official Test (yolov8s) | yolov8s A3 Test 1,587 imgs (P / R / mAP50 / mAP50-95) | 89.7 / 83.6 / 88.7 / 57.2% | 89.7 / 83.6 / 88.7 / 57.2% (exact CSV: 89.7054 / 83.5577 / 88.7305 / 57.1614%) | **MATCH** |
| 5. Official Test (yolov8s) | yolov8s A4 Test 1,587 imgs (P / R / mAP50 / mAP50-95) | 90.9 / 82.6 / 88.3 / 57.2% | 90.9 / 82.6 / 88.3 / 57.2% (exact CSV: 90.9469 / 82.5619 / 88.2910 / 57.2358%) | **MATCH** |
| 5. Official Test (yolov8s) | yolov8s A5 Test 1,587 imgs (P / R / mAP50 / mAP50-95) | 90.9 / 82.4 / 88.3 / 57.3% | 90.9 / 82.4 / 88.3 / 57.3% (exact CSV: 90.8566 / 82.4194 / 88.3448 / 57.3292%) | **MATCH** |
| 5. Official Test (yolov8s) | yolov8s A6 Test 1,587 imgs (P / R / mAP50 / mAP50-95) | 90.3 / 82.6 / 88.4 / 57.4% | 90.3 / 82.6 / 88.4 / 57.4% (exact CSV: 90.3183 / 82.6271 / 88.3692 / 57.3711%) | **MATCH** |
| 6. Per-Class Test A6 (yolo11s) | yolo11s A6 `blue` (R / AP50 / AP50-95) | 86.6 / 89.8 / 59.4% | 86.6 / 89.8 / 59.4% (JSON exact AP50=89.75%, AP50-95=59.43%) | **MATCH** |
| 6. Per-Class Test A6 (yolo11s) | yolo11s A6 `white` (R / AP50 / AP50-95) | 86.6 / 89.1 / 59.5% | 86.6 / 89.1 / 59.5% (JSON exact AP50=89.13%, AP50-95=59.51%) | **MATCH** |
| 6. Per-Class Test A6 (yolo11s) | yolo11s A6 `yellow` (R / AP50 / AP50-95) | 88.2 / 92.7 / 60.7% | 88.2 / 92.7 / 60.7% (JSON exact AP50=92.66%, AP50-95=60.67%) | **MATCH** |
| 6. Per-Class Test A6 (yolo11s) | yolo11s A6 `red` (R / AP50 / AP50-95) | 82.7 / 88.6 / 57.4% | 82.7 / 88.6 / 57.4% (JSON exact AP50=88.58%, AP50-95=57.39%) | **MATCH** |
| 6. Per-Class Test A6 (yolo11s) | yolo11s A6 `none` (R / AP50 / AP50-95) | 77.3 / 84.6 / 53.4% | 77.3 / 84.6 / 53.4% (JSON exact AP50=84.58%, AP50-95=53.41%) | **MATCH** |
| 6. Per-Class Test A6 (yolov8s) | yolov8s A6 `blue` (R / AP50 / AP50-95) | 83.3 / 89.3 / 58.8% | 83.3 / 89.3 / 58.8% (JSON exact AP50=89.26%, AP50-95=58.85%) | **MATCH** |
| 6. Per-Class Test A6 (yolov8s) | yolov8s A6 `white` (R / AP50 / AP50-95) | 86.9 / 90.4 / 59.6% | 86.9 / 90.4 / 59.6% (JSON exact AP50=90.37%, AP50-95=59.59%) | **MATCH** |
| 6. Per-Class Test A6 (yolov8s) | yolov8s A6 `yellow` (R / AP50 / AP50-95) | 87.8 / 92.4 / 60.6% | 87.8 / 92.4 / 60.6% (JSON exact AP50=92.39%, AP50-95=60.61%) | **MATCH** |
| 6. Per-Class Test A6 (yolov8s) | yolov8s A6 `red` (R / AP50 / AP50-95) | 80.1 / 86.3 / 56.0% | 80.1 / 86.3 / 56.0% (JSON exact AP50=86.29%, AP50-95=56.01%) | **MATCH** |
| 6. Per-Class Test A6 (yolov8s) | yolov8s A6 `none` (R / AP50 / AP50-95) | 75.1 / 83.5 / 51.8% | 75.1 / 83.5 / 51.8% (JSON exact AP50=83.54%, AP50-95=51.79%) | **MATCH** |
| 7. Inference Speed & Fusion Honesty | gdut-final-test-evaluation T4 speed (ms) & switch_to_deploy() in final_test() | A1 Y11=7.5ms, V8=7.4ms; A6 Y11=9.2ms, V8=9.1ms (unfused; switch_to_deploy() only in preflight test, NOT in final_test) | Log T4: A1_Y11=7.5ms, A1_V8=7.4ms, A6_Y11=9.2ms, A6_V8=9.1ms; switch_to_deploy_in_final_test=False | **MATCH** |
| 8. Dung Baselines (tundng111) | B1 Pair-3407 Control Val (P / R / mAP50 / mAP50-95) | 91.65 / 78.77 / 88.86 / 57.78% | 91.65 / 78.77 / 88.86 / 57.78% (OP: th=0.287, TP=306, FP=60, FN=98) | **MATCH** |
| 8. Dung Baselines (tundng111) | B2 V8F run_manifest.inputs.manifest_sha256 Provenance | `026a2afb617254b9dd2b9461651bce6f74295d18f29d52a790bf755d32944f65` | `026a2afb617254b9dd2b9461651bce6f74295d18f29d52a790bf755d32944f65` (differs intentionally from Thanh A0-A6 `ce2a883e...` manifest hash) | **MATCH** |
| 8. Dung Baselines (tundng111) | B2 V8F 5-Seed Mean +/- SD (seeds 3407..3411; P rounding & ddof=0 vs ddof=1) | P=90.25±0.94, R=86.47±0.58, mAP50=91.82±0.17, mAP50-95=60.781±0.353 | pstdev(ddof=0): P=90.2447% (90.24±0.94), R=86.47±0.58, mAP50=91.82±0.17, mAP50-95=60.781±0.353; sample stdev(ddof=1): ±1.05/±0.65/±0.19/±0.394 | **MISMATCH** |
| 8. Dung Baselines (tundng111) | B1 C3 Matched DCRCF 5-Seed Val mAP50-95 & mAP50 (seeds 3407..3411) | mAP50-95=57.283±0.173% (fp=026a2afb6172...) | pstdev(ddof=0): mAP50-95=57.283±0.173%, mAP50=88.528±0.224% (fp=026a2afb6172...); sample stdev(ddof=1): ±0.193/±0.250% | **MATCH** |
| 8. Dung Baselines (tundng111) | C0 Stock YOLO11n 5-Seed Val mAP50-95 & mAP50 (seeds 3407..3411) | mAP50-95=58.812±0.082% (fp=026a2afb6172...) | pstdev(ddof=0): mAP50-95=58.812±0.082%, mAP50=90.274±0.071% (fp=026a2afb6172...); sample stdev(ddof=1): ±0.092/±0.079% | **MATCH** |
| 8. Dung Baselines (tundng111) | V8S Stock YOLOv8s 5-Seed Val mAP50-95 & mAP50 (seeds 3407..3411) | mAP50-95=61.414±0.074% (fp=026a2afb6172...) | pstdev(ddof=0): mAP50-95=61.414±0.074%, mAP50=92.308±0.224% (fp=026a2afb6172...); sample stdev(ddof=1): ±0.083/±0.250% | **MATCH** |
```

## 2. Non-Biometric Upper-Torso Persistent Re-ID Engine (`src/engine/gdut_reid.py`)

```python
"""
src/engine/gdut_reid.py
=======================
Privacy-Safe Head-Anchored Upper-Torso Persistent Re-Identification Engine v2.3
for GDUT-HWD 5-Class (`blue, white, yellow, red, none`) Surveillance.

Upgrades over v2.0–v2.2:
1. Anti-Wall-Bleed Anatomical ROI & Gaussian Spatial Weighting:
   - Narrowed torso projection + anatomical per-band horizontal crops (collar/chest/mid-torso).
   - Border background sampling & median-wall distance suppression so identical walls
     behind different people on webcam C960 Ultra do not inflate similarity.
2. Same-Uniform Discriminative Descriptors:
   - Weighted 3-strip HSV + Lab + BGR appearance + 4-quadrant local gradient texture
     (HOG-8 x 4 quadrants + luminance grids, 520-D total: 474-D appearance + 46-D texture).
   - Multi-layer MobileNetV3-Small feature extraction (`features[6]` + `features[11]` +
     final `avgpool` -> 712-D) capturing fine straps, seams, badges, and fabric texture
     while remaining 100% non-biometric (no facial recognition).
3. Confirm-Before-Mint & Global Hungarian Assignment:
   - Provisional `_tentative_tracks` buffer prevents transient 1–5 frame noise detections
     from ever minting canonical IDs or inflating `self.next_id`; promotion runs Hungarian +
     re-match at `min_confirm_hits >= 6` before minting `cid = self.next_id`.
   - `linear_sum_assignment`, Coexistence Exclusion Graph, Ambiguity Margin Guard
     (`>= 0.06` tie margin), Pre-Warm, and Reference Enrollment.
"""

from __future__ import annotations
import cv2
import time
import numpy as np
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from scipy.optimize import linear_sum_assignment

from src.config import DEFAULT_ROLE_HIERARCHY


def project_upper_torso_roi(
    head_box: np.ndarray | List[float],
    frame_shape: Tuple[int, ...],
) -> Tuple[int, int, int, int]:
    """
    Projects a tightened anatomical upper-torso ROI below a head/helmet bounding box
    while excluding outer wall background and head cranium.
    """
    h_img, w_img = int(frame_shape[0]), int(frame_shape[1])
    if h_img <= 1 or w_img <= 1:
        return 0, 0, 1, 1

    hx1, hy1, hx2, hy2 = [float(v) for v in head_box[:4]]
    hw = max(2.0, hx2 - hx1)
    hh = max(2.0, hy2 - hy1)
    cx = 0.5 * (hx1 + hx2)

    tx1 = int(max(0, min(w_img - 1, round(cx - 1.05 * hw))))
    tx2 = int(max(tx1 + 1, min(w_img, round(cx + 1.05 * hw))))
    ty1 = int(max(0, min(h_img - 1, round(hy2 + 0.15 * hh))))
    ty2 = int(max(ty1 + 1, min(h_img, round(hy2 + 2.8 * hh))))

    # Bottom-border fallback: capture visible shoulder/collar band when head sits at bottom edge
    if (ty2 - ty1) < 8:
        fb_y1 = int(max(0, min(h_img - 1, round(hy1 + 0.65 * hh))))
        fb_y2 = int(max(fb_y1 + 1, h_img))
        fb_x1 = int(max(0, min(w_img - 1, round(cx - 1.0 * hw))))
        fb_x2 = int(max(fb_x1 + 1, min(w_img, round(cx + 1.0 * hw))))
        if (fb_y2 - fb_y1) > (ty2 - ty1):
            return fb_x1, fb_y1, fb_x2, fb_y2

    return tx1, ty1, tx2, ty2


def _sample_border_wall_bgr(
    frame_bgr: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    """Samples left/right exterior margins adjacent to the torso ROI to estimate background wall color."""
    h_img, w_img = frame_bgr.shape[:2]
    tx1, ty1, tx2, ty2 = roi
    y_step = max(1, (ty2 - ty1) // 48)
    samples = []
    if tx1 >= 8:
        lx1 = max(0, tx1 - 18)
        lx2 = max(lx1 + 1, tx1 - 2)
        if lx2 > lx1 and ty2 > ty1:
            samples.append(frame_bgr[ty1:ty2:y_step, lx1:lx2].reshape(-1, 3))
    if (w_img - tx2) >= 8:
        rx1 = min(w_img - 1, tx2 + 2)
        rx2 = min(w_img, tx2 + 18)
        if rx2 > rx1 and ty2 > ty1:
            samples.append(frame_bgr[ty1:ty2:y_step, rx1:rx2].reshape(-1, 3))
    if not samples:
        return None
    cat = np.concatenate(samples, axis=0)
    if cat.shape[0] < 16:
        return None
    return np.median(cat, axis=0).astype(np.float32)


_GAUSS_MASK_64x96 = np.exp(
    -0.5
    * (
        (np.linspace(-1.0, 1.0, 64, dtype=np.float32)[None, :] / 0.55) ** 2
        + (np.linspace(-1.0, 1.0, 96, dtype=np.float32)[:, None] / 0.85) ** 2
    )
).astype(np.float32)


def _compute_quadrant_hog_texture(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Computes a 46-D structural & spatial layout texture descriptor:
    - 2x2 spatial quadrants x 8 HOG orientation bins + 8 edge/contrast stats (40-D)
    - 3x2 zero-mean normalized spatial luminance layout (6-D) to distinguish horizontal
      reflective bars from diagonal straps/vertical pockets on identical-color uniforms.
    """
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 8:
        return np.zeros(46, dtype=np.float32)

    gray = cv2.cvtColor(cv2.resize(crop_bgr, (48, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_smooth = cv2.GaussianBlur(gray, (3, 3), 0.7)
    gx = cv2.Sobel(gray_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi  # unsigned [0, pi)

    bins = np.floor((ang / np.pi) * 8.0).astype(np.int32)
    bins = np.clip(bins, 0, 7)

    feats: List[np.ndarray] = []
    stats: List[float] = []
    qh, qw = 32, 24
    for ry in (0, 1):
        for rx in (0, 1):
            y0, y1 = ry * qh, (ry + 1) * qh
            x0, x1 = rx * qw, (rx + 1) * qw
            q_mag = mag[y0:y1, x0:x1]
            q_bin = bins[y0:y1, x0:x1]
            q_gray = gray[y0:y1, x0:x1]

            hog = np.bincount(q_bin.ravel(), weights=q_mag.ravel(), minlength=8).astype(np.float32)
            hn = float(np.linalg.norm(hog))
            if hn > 1e-6:
                hog /= hn
            feats.append(hog)
            stats.append(float(q_mag.mean()) / 255.0)
            stats.append(float(q_gray.std()) / 128.0)

    # 3x2 spatial luminance pattern (centered & scaled so solid shirts produce ~0, textured seams separate cleanly)
    grid = cv2.resize(gray_smooth, (2, 3), interpolation=cv2.INTER_AREA).flatten()
    grid_centered = (grid - float(grid.mean())) / 64.0

    tex = np.concatenate([*feats, np.asarray(stats, dtype=np.float32), grid_centered.astype(np.float32)]).astype(np.float32)
    tn = float(np.linalg.norm(tex))
    if tn > 1e-6:
        tex /= tn
    return tex


def _weighted_hist_1d(vals: np.ndarray, weights: np.ndarray, bins: int, hi: float) -> np.ndarray:
    scaled = np.clip((vals.astype(np.float32) / hi) * bins, 0, bins - 1e-4).astype(np.int32)
    h = np.bincount(scaled.ravel(), weights=weights.ravel(), minlength=bins).astype(np.float32)
    n = float(np.linalg.norm(h))
    if n > 1e-6:
        h /= n
    return h


def compute_three_strip_torso_signature(
    frame_bgr: Optional[np.ndarray],
    roi: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Computes a 520-D descriptor packed as two independently normalized subspaces:
    - [0:474]: L2-normalized 3 anatomical horizontal strips (158-D each), weighted by
      2D Gaussian core mask + exterior wall-color suppression.
    - [474:520]: L2-normalized 46-D 4-quadrant HOG + spatial layout texture.
    """
    feat_dim = 520
    if frame_bgr is None or frame_bgr.size == 0:
        return np.zeros(feat_dim, dtype=np.float32)

    h_img, w_img = frame_bgr.shape[:2]
    tx1, ty1, tx2, ty2 = roi
    tx1 = max(0, min(w_img - 1, int(tx1)))
    ty1 = max(0, min(h_img - 1, int(ty1)))
    tx2 = max(tx1 + 1, min(w_img, int(tx2)))
    ty2 = max(ty1 + 1, min(h_img, int(ty2)))

    crop = frame_bgr[ty1:ty2, tx1:tx2]
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return np.zeros(feat_dim, dtype=np.float32)

    wall_bg = _sample_border_wall_bgr(frame_bgr, (tx1, ty1, tx2, ty2))

    # Bound torso crop to canonical (64x96) so 1080p full-frame Re-ID stays O(1) fast on CPU
    if crop.shape[0] > 96 or crop.shape[1] > 64:
        crop = cv2.resize(crop, (64, 96), interpolation=cv2.INTER_AREA)

    ch, cw = crop.shape[:2]

    # Build 2D Gaussian core mask favoring foreground body core over outer fringes
    if ch == 96 and cw == 64:
        gauss_mask = _GAUSS_MASK_64x96.copy()
    else:
        yy = np.linspace(-1.0, 1.0, ch, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, cw, dtype=np.float32)[None, :]
        gauss_mask = np.exp(-0.5 * ((xx / 0.55) ** 2 + (yy / 0.85) ** 2)).astype(np.float32)

    if wall_bg is not None:
        diff = np.linalg.norm(crop.astype(np.float32) - wall_bg[None, None, :], axis=2)
        fg_ratio = float(np.mean(diff >= 18.0))
        if fg_ratio >= 0.25:
            fg_weight = np.where(diff < 18.0, 0.15, 1.0).astype(np.float32)
            gauss_mask = gauss_mask * fg_weight

    if ch >= 6:
        b1 = max(1, ch // 3)
        b2 = max(b1 + 1, (2 * ch) // 3)
        band_defs = [
            (0, b1, 0.14, 0.86),     # Strip 0: collar & inner shoulders (narrower to avoid walls)
            (b1, b2, 0.02, 0.98),    # Strip 1: full upper chest
            (b2, ch, 0.06, 0.94),    # Strip 2: lower torso
        ]
    else:
        band_defs = [(0, ch, 0.0, 1.0), (0, ch, 0.0, 1.0), (0, ch, 0.0, 1.0)]

    strips: List[np.ndarray] = []
    for y0, y1, xr0, xr1 in band_defs:
        x0 = max(0, min(cw - 1, int(round(xr0 * cw))))
        x1 = max(x0 + 1, min(cw, int(round(xr1 * cw))))
        band = crop[y0:y1, x0:x1]
        b_mask = gauss_mask[y0:y1, x0:x1]
        if band.size == 0:
            strips.append(np.zeros(158, dtype=np.float32))
            continue

        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        h_idx = np.clip((hsv[:, :, 0].astype(np.float32) / 180.0) * 12.0, 0, 11.99).astype(np.int32)
        s_idx = np.clip((hsv[:, :, 1].astype(np.float32) / 256.0) * 8.0, 0, 7.99).astype(np.int32)
        hs_flat = (h_idx * 8 + s_idx).ravel()
        hist_hs = np.bincount(hs_flat, weights=b_mask.ravel(), minlength=96).astype(np.float32)
        n_hs = float(np.linalg.norm(hist_hs))
        if n_hs > 1e-6:
            hist_hs /= n_hs

        lab = cv2.cvtColor(band, cv2.COLOR_BGR2Lab)
        hist_a = _weighted_hist_1d(lab[:, :, 1], b_mask, 16, 256.0)
        hist_b = _weighted_hist_1d(lab[:, :, 2], b_mask, 16, 256.0)
        n_ab = float(np.linalg.norm(np.concatenate([hist_a, hist_b])))
        if n_ab > 1e-6:
            hist_a /= n_ab
            hist_b /= n_ab

        bgr_hists = [_weighted_hist_1d(band[:, :, c_i], b_mask, 8, 256.0) for c_i in range(3)]

        w_sum = max(1e-6, float(b_mask.sum()))
        w_3 = b_mask[:, :, None]
        bf = band.astype(np.float32)
        mean_bgr = (bf * w_3).sum(axis=(0, 1)) / (w_sum * 255.0)
        var_bgr = (((bf - (mean_bgr * 255.0)[None, None, :]) ** 2) * w_3).sum(axis=(0, 1)) / w_sum
        std_bgr = np.sqrt(np.maximum(0.0, var_bgr)) / 128.0
        moments = np.concatenate([mean_bgr, std_bgr]).astype(np.float32) * 0.5

        band_vec = np.concatenate([hist_hs, hist_a, hist_b, *bgr_hists, moments]).astype(np.float32)
        strips.append(band_vec)

    color_vec = np.concatenate(strips).astype(np.float32)
    cn = float(np.linalg.norm(color_vec))
    if cn > 1e-6:
        color_vec /= cn
    tex_vec = _compute_quadrant_hog_texture(crop).astype(np.float32)
    return np.concatenate([color_vec, tex_vec]).astype(np.float32)


def cosine_sim(v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> float:
    """Computes cosine similarity for L2-normalized vectors in [-1, 1]."""
    if v1 is None or v2 is None or v1.size == 0 or v2.size == 0:
        return 0.0
    if v1.size != v2.size:
        m = min(v1.size, v2.size)
        v1 = v1[:m]
        v2 = v2[:m]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


class MobileNetV3SmallExtractor:
    """
    Non-biometric Multi-Layer Upper-Torso Extractor (`features[6]` + `features[11]` + `avgpool`).
    Extracts 712-D L2-normalized descriptor capturing fine fabric texture, seams, and stripes
    for identical-uniform discrimination without any facial embedding.
    """

    def __init__(self, enabled: bool = True, device: Optional[str] = None):
        import threading

        self.enabled = bool(enabled)
        self.device = device
        self._init_lock = threading.Lock()
        self._initialized = False
        self._available = False
        self._features = None
        self._avgpool = None
        self._torch = None
        self._mean = None
        self._std = None
        self.extract_calls: int = 0

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        if not self._initialized:
            with self._init_lock:
                if not self._initialized:
                    self._lazy_init()
        return self._available

    def prewarm(self) -> bool:
        """Eagerly initializes weights & runs a dummy forward pass outside the frame lock."""
        if not self.available:
            return False
        try:
            dummy = self._torch.zeros((1, 3, 128, 128), device=self.device, dtype=self._torch.float32)
            with self._torch.no_grad():
                _ = self._forward_multiscale(dummy)
            return True
        except Exception:
            return False

    def _lazy_init(self):
        self._initialized = True
        if not self.enabled:
            self._available = False
            return
        try:
            import os
            import torch
            import torchvision.models as tv_models

            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
            has_cached = cache_dir.exists() and any(cache_dir.glob("mobilenet_v3_small*.pth"))
            allow_dl = os.getenv("GDUT_REID_ALLOW_DOWNLOAD", "1").lower() in ("1", "true", "yes")
            if not has_cached and not allow_dl:
                self._available = False
                return

            weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT
            model = tv_models.mobilenet_v3_small(weights=weights)
            dev = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            if dev.startswith("cuda") and not torch.cuda.is_available():
                dev = "cpu"
            self.device = dev
            model.eval().to(dev)
            self._features = model.features
            self._avgpool = model.avgpool
            self._torch = torch
            self._mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
            self._available = True
        except Exception:
            self._available = False
            self._features = None

    def _forward_multiscale(self, x_norm):
        import torch.nn.functional as F

        feat_mid6 = None
        feat_mid11 = None
        h = x_norm
        for idx, layer in enumerate(self._features):
            h = layer(h)
            if idx == 6:
                feat_mid6 = F.normalize(self._avgpool(h).flatten(1), dim=1)
            elif idx == 11:
                feat_mid11 = F.normalize(self._avgpool(h).flatten(1), dim=1)
        feat_high = F.normalize(self._avgpool(h).flatten(1), dim=1)

        parts = []
        if feat_mid6 is not None:
            parts.append(0.45 * feat_mid6)
        if feat_mid11 is not None:
            parts.append(0.55 * feat_mid11)
        parts.append(1.0 * feat_high)
        combined = self._torch.cat(parts, dim=1)
        return F.normalize(combined, dim=1).squeeze(0)

    def extract(self, frame_bgr: Optional[np.ndarray], roi: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if not self.available or frame_bgr is None or self._features is None or self._torch is None:
            return None
        tx1, ty1, tx2, ty2 = roi
        crop = frame_bgr[ty1:ty2, tx1:tx2]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        try:
            rgb = cv2.cvtColor(cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0
            tensor = (tensor - self._mean) / self._std
            with self._torch.no_grad():
                vec = self._forward_multiscale(tensor).detach().cpu().numpy().astype(np.float32)
            self.extract_calls += 1
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                vec /= norm
            return vec
        except Exception:
            return None


class UpperTorsoIdentityProfile:
    """Persistent Re-ID profile bound to a canonical personnel ID."""

    def __init__(
        self,
        worker_id: int,
        head_box: np.ndarray,
        helmet_class: str,
        strip_sig: np.ndarray,
        deep_sig: Optional[np.ndarray],
        torso_roi: Tuple[int, int, int, int],
        frame_idx: int,
        now_ts: float,
    ):
        self.worker_id = worker_id
        self.custom_label: Optional[str] = None
        self.current_helmet_color: str = helmet_class
        self.min_history_votes: int = 10
        self.helmet_color_votes: Dict[str, int] = {}
        self._color_streak_cls: Optional[str] = None
        self._color_streak_len: int = 0
        # Hysteresis guard: require >= 10 confirmed frames before recording PRIOR <COLOR>
        self.last_known_helmet_color: Optional[str] = None
        if helmet_class and helmet_class != "none":
            self.helmet_color_votes[helmet_class] = 1
            self._color_streak_cls = helmet_class
            self._color_streak_len = 1
        self.status: str = "VIOLATION" if helmet_class == "none" else "COMPLIANT"
        self.violations: List[str] = ["MISSING_SAFETY_HELMET"] if helmet_class == "none" else []

        self.last_box = np.asarray(head_box[:4], dtype=np.float32).copy()
        self.last_torso_roi = torso_roi
        self.first_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.first_seen_ts = now_ts
        self.last_seen_ts = now_ts
        self.total_frames_seen = 1
        self.consecutive_hits = 1
        self.is_active = True
        self.is_confirmed = False
        self.last_deep_frame = frame_idx
        self.coexisted_ids: Set[int] = set()

        hw = max(2.0, float(head_box[2] - head_box[0]))
        tw = max(2.0, float(torso_roi[2] - torso_roi[0]))
        self.shoulder_head_ratio: float = tw / hw

        self.strip_gallery: deque[np.ndarray] = deque(maxlen=5)
        self.strip_ema: Optional[np.ndarray] = None
        if strip_sig is not None and np.linalg.norm(strip_sig) > 1e-6:
            self.strip_gallery.append(strip_sig.copy())
            self.strip_ema = strip_sig.copy()

        self.deep_gallery: deque[np.ndarray] = deque(maxlen=7)
        self.deep_ema: Optional[np.ndarray] = None
        if deep_sig is not None and np.linalg.norm(deep_sig) > 1e-6:
            self.deep_gallery.append(deep_sig.copy())
            self.deep_ema = deep_sig.copy()

        cx = float(0.5 * (head_box[0] + head_box[2]))
        cy = float(0.5 * (head_box[1] + head_box[3]))
        self.trajectory: deque[Tuple[float, float]] = deque(maxlen=90)
        self.trajectory.append((cx, cy))

    def update(
        self,
        head_box: np.ndarray,
        helmet_class: str,
        strip_sig: np.ndarray,
        torso_roi: Tuple[int, int, int, int],
        frame_idx: int,
        now_ts: float,
        deep_sig: Optional[np.ndarray] = None,
        min_confirm_hits: int = 6,
        min_confirm_frames: int = 24,
    ):
        self.last_box = np.asarray(head_box[:4], dtype=np.float32).copy()
        self.last_torso_roi = torso_roi
        self.last_seen_frame = frame_idx
        self.last_seen_ts = now_ts
        self.total_frames_seen += 1
        self.consecutive_hits += 1
        if self.consecutive_hits >= min_confirm_hits or self.total_frames_seen >= min_confirm_frames:
            self.is_confirmed = True
        self.is_active = True

        hw = max(2.0, float(head_box[2] - head_box[0]))
        tw = max(2.0, float(torso_roi[2] - torso_roi[0]))
        self.shoulder_head_ratio = 0.8 * self.shoulder_head_ratio + 0.2 * (tw / hw)

        self.current_helmet_color = helmet_class
        if helmet_class and helmet_class != "none":
            self.helmet_color_votes[helmet_class] = self.helmet_color_votes.get(helmet_class, 0) + 1
            if helmet_class == self._color_streak_cls:
                self._color_streak_len += 1
            else:
                self._color_streak_cls = helmet_class
                self._color_streak_len = 1
            if self._color_streak_len >= self.min_history_votes or (
                self.last_known_helmet_color is None
                and self.helmet_color_votes[helmet_class] >= self.min_history_votes
            ):
                self.last_known_helmet_color = helmet_class
            self.status = "COMPLIANT"
            self.violations = []
        else:
            self._color_streak_cls = None
            self._color_streak_len = 0
            self.status = "VIOLATION"
            self.violations = ["MISSING_SAFETY_HELMET"]

        if strip_sig is not None and np.linalg.norm(strip_sig) > 1e-6:
            if self.strip_ema is None:
                self.strip_ema = strip_sig.copy()
                self.strip_gallery.append(strip_sig.copy())
            else:
                blended = 0.85 * self.strip_ema + 0.15 * strip_sig
                n = float(np.linalg.norm(blended))
                if n > 1e-6:
                    self.strip_ema = blended / n
                if self.max_strip_sim(strip_sig) < 0.90:
                    self.strip_gallery.append(strip_sig.copy())

        if deep_sig is not None and np.linalg.norm(deep_sig) > 1e-6:
            self.last_deep_frame = frame_idx
            if self.deep_ema is None:
                self.deep_ema = deep_sig.copy()
                self.deep_gallery.append(deep_sig.copy())
            else:
                blended = 0.85 * self.deep_ema + 0.15 * deep_sig
                n = float(np.linalg.norm(blended))
                if n > 1e-6:
                    self.deep_ema = blended / n
                if self.max_deep_sim(deep_sig) < 0.85:
                    self.deep_gallery.append(deep_sig.copy())

        cx = float(0.5 * (head_box[0] + head_box[2]))
        cy = float(0.5 * (head_box[1] + head_box[3]))
        self.trajectory.append((cx, cy))

    def max_strip_sim(self, cand_strip: Optional[np.ndarray]) -> float:
        color_sim, tex_sim, has_tex = self.max_subspace_sims(cand_strip)
        if not has_tex:
            return color_sim
        return 0.55 * color_sim + 0.45 * tex_sim

    def max_subspace_sims(self, cand_strip: Optional[np.ndarray]) -> Tuple[float, float, bool]:
        """Returns (max_color_sim, max_texture_sim, both_have_texture)."""
        if cand_strip is None or cand_strip.size < 474 or np.linalg.norm(cand_strip[:474]) < 1e-6:
            return 0.0, 0.0, False
        cand_col = cand_strip[:474]
        cand_tex = cand_strip[474:] if cand_strip.size > 474 else None
        cand_has_tex = cand_tex is not None and float(np.linalg.norm(cand_tex)) > 0.1

        color_sims = [cosine_sim(v[:474], cand_col) for v in self.strip_gallery if v.size >= 474]
        if self.strip_ema is not None and self.strip_ema.size >= 474:
            color_sims.append(cosine_sim(self.strip_ema[:474], cand_col))
        best_col = max(color_sims) if color_sims else 0.0

        best_tex = 0.0
        prof_has_tex = False
        if cand_has_tex:
            tex_sims = []
            for v in list(self.strip_gallery) + ([self.strip_ema] if self.strip_ema is not None else []):
                if v.size > 474 and float(np.linalg.norm(v[474:])) > 0.1:
                    prof_has_tex = True
                    tex_sims.append(cosine_sim(v[474:], cand_tex))
            if tex_sims:
                best_tex = max(tex_sims)
        return best_col, best_tex, (cand_has_tex and prof_has_tex)

    def max_deep_sim(self, cand_deep: Optional[np.ndarray]) -> float:
        if cand_deep is None or np.linalg.norm(cand_deep) < 1e-6:
            return 0.0
        sims = [cosine_sim(v, cand_deep) for v in self.deep_gallery]
        if self.deep_ema is not None:
            sims.append(cosine_sim(self.deep_ema, cand_deep))
        return max(sims) if sims else 0.0


class UpperTorsoReIDGallery:
    """
    Persistent Re-ID gallery for GDUT 5-class head tracking.
    Uses Hungarian optimal assignment (`linear_sum_assignment`), coexistence exclusion,
    ambiguity margin gating, anatomical wall-suppressed ROI, and multi-layer MobileNetV3 features.
    """

    def __init__(
        self,
        ttl_seconds: float = 600.0,
        sim_thresh: float = 0.70,
        use_deep: bool = True,
        deep_refresh_interval: int = 30,
        min_confirm_hits: int = 6,
        min_confirm_frames: int = 24,
        device: Optional[str] = None,
    ):
        self.ttl_seconds = float(ttl_seconds)
        self.sim_thresh = float(sim_thresh)
        self.deep_refresh_interval = max(1, int(deep_refresh_interval))
        self.min_confirm_hits = max(1, int(min_confirm_hits))
        self.min_confirm_frames = max(1, int(min_confirm_frames))
        self.deep_extractor = MobileNetV3SmallExtractor(enabled=use_deep, device=device)

        self.profiles: Dict[int, UpperTorsoIdentityProfile] = {}
        self._tentative_tracks: Dict[int, UpperTorsoIdentityProfile] = {}
        self.raw_to_canonical: Dict[int, int] = {}
        self.raw_conflict_streak: Dict[int, int] = {}
        self.hijack_excluded_ids: Dict[int, Tuple[int, int]] = {}
        self.next_id: int = 1

    def prewarm(self) -> bool:
        return self.deep_extractor.prewarm()

    def reset(self):
        self.profiles.clear()
        self._tentative_tracks.clear()
        self.raw_to_canonical.clear()
        self.raw_conflict_streak.clear()
        self.hijack_excluded_ids.clear()
        self.next_id = 1

    def on_video_loop(self):
        """Preserves persistent identity profiles on video loop while clearing raw tracker IDs."""
        self._tentative_tracks.clear()
        self.raw_to_canonical.clear()
        self.raw_conflict_streak.clear()
        self.hijack_excluded_ids.clear()
        for p in self.profiles.values():
            p.is_active = False
            p.consecutive_hits = 0

    def set_label(self, canonical_id: int, label: str) -> bool:
        p = self.profiles.get(int(canonical_id))
        if p is None:
            return False
        cleaned = str(label).strip()[:40]
        p.custom_label = cleaned if cleaned else None
        return True

    def enroll_reference(
        self,
        canonical_id: int,
        frame_bgr: np.ndarray,
        head_box: Optional[np.ndarray | List[float]] = None,
        label: Optional[str] = None,
    ) -> bool:
        """Captures an explicit multi-angle reference view for a tracked personnel ID during demo setup."""
        p = self.profiles.get(int(canonical_id))
        if p is None or frame_bgr is None or frame_bgr.size == 0:
            return False
        hbox = np.asarray(head_box[:4], dtype=np.float32) if head_box is not None else p.last_box
        roi = project_upper_torso_roi(hbox, frame_bgr.shape)
        strip_sig = compute_three_strip_torso_signature(frame_bgr, roi)
        deep_sig = self.deep_extractor.extract(frame_bgr, roi) if self.deep_extractor.available else None
        if np.linalg.norm(strip_sig) > 1e-6:
            p.strip_gallery.append(strip_sig)
        if deep_sig is not None and np.linalg.norm(deep_sig) > 1e-6:
            p.deep_gallery.append(deep_sig)
        if label is not None and str(label).strip():
            p.custom_label = str(label).strip()[:40]
        return True

    def _compute_appear_subscores(
        self,
        profile: UpperTorsoIdentityProfile,
        strip_sig: np.ndarray,
        deep_sig: Optional[np.ndarray],
        skip_hard_veto: bool = False,
        dt_seconds: Optional[float] = None,
    ) -> Tuple[float, bool]:
        """Returns (appearance_similarity, vetoed_by_structural_mismatch)."""
        sim_color, sim_tex, has_tex = profile.max_subspace_sims(strip_sig)
        sim_strip = (0.52 * sim_color + 0.48 * sim_tex) if has_tex else sim_color

        if deep_sig is not None and profile.deep_ema is not None:
            sim_deep = profile.max_deep_sim(deep_sig)
            if has_tex:
                base_sim = 0.50 * sim_deep + 0.30 * sim_color + 0.20 * sim_tex
            else:
                base_sim = 0.52 * sim_deep + 0.48 * sim_color
            if not skip_hard_veto:
                short_term_reentry = (dt_seconds is not None and dt_seconds <= 8.0)
                strong_pattern_color = (sim_color >= 0.78 and (not has_tex or sim_tex >= 0.65))
                deep_veto_floor = 0.50 if (strong_pattern_color or (short_term_reentry and sim_color >= 0.78)) else 0.62
                if sim_deep < deep_veto_floor:
                    return float(min(base_sim, sim_deep)), True
            return float(base_sim), False

        if not skip_hard_veto and has_tex and sim_tex < 0.58:
            return float(min(sim_strip, 0.60 * sim_tex + 0.25 * sim_color)), True
        return float(sim_strip), False

    def _compute_candidate_score(
        self,
        profile: UpperTorsoIdentityProfile,
        strip_sig: np.ndarray,
        deep_sig: Optional[np.ndarray],
        cls_norm: str,
        cand_ratio: float,
        skip_hard_veto: bool = False,
        dt_seconds: Optional[float] = None,
    ) -> float:
        base_sim, vetoed = self._compute_appear_subscores(
            profile, strip_sig, deep_sig, skip_hard_veto=skip_hard_veto, dt_seconds=dt_seconds
        )
        if vetoed:
            return float(base_sim)

        # Geometric shoulder/head proportion check
        ratio_diff = abs(cand_ratio - profile.shoulder_head_ratio)
        ratio_adj = 0.02 if ratio_diff <= 0.18 else (-0.04 if ratio_diff > 0.45 else 0.0)

        # Helmet role continuity:
        # - If bare-headed ("none"), apply 0 penalty so helmet removal preserves canonical ID.
        # - If two distinct positive helmet colors mismatch, strongly penalize cross-swap.
        color_adj = 0.0
        if cls_norm != "none" and profile.last_known_helmet_color and profile.last_known_helmet_color != "none":
            if cls_norm == profile.last_known_helmet_color:
                color_adj = 0.04
            else:
                color_adj = -0.24

        return float(base_sim + ratio_adj + color_adj)

    def associate_batch(
        self,
        items: List[Tuple[int, np.ndarray | List[float], str]],
        frame_bgr: np.ndarray,
        frame_idx: int,
        now_ts: Optional[float] = None,
    ) -> List[UpperTorsoIdentityProfile]:
        """
        Globally associates a batch of (raw_track_id, head_box, helmet_class) tuples via:
        1. Direct raw-to-canonical binding with anti-hijack clothing divergence guard.
        2. Short-Term Spatio-Temporal Re-Link (<=3.0s, <=0.18 diag, no deep hard veto)
           and Long-Term Hungarian matching against inactive profiles within TTL,
           enforcing coexistence exclusion & ambiguity margin guard.
        3. Novel identity minting for genuinely new arrivals.
        """
        if not items:
            return []

        ts = float(now_ts if now_ts is not None else time.time())
        fidx_int = int(frame_idx)
        self.hijack_excluded_ids = {
            k: v for k, v in self.hijack_excluded_ids.items() if fidx_int <= int(v[1])
        }
        n_items = len(items)
        results: List[Optional[UpperTorsoIdentityProfile]] = [None] * n_items
        unmatched_indices: List[int] = []
        claimed_cids: Set[int] = set()
        claimed_boxes: Dict[int, Tuple[float, float]] = {}

        prepared: List[Tuple[int, np.ndarray, str, Tuple[int, int, int, int], np.ndarray, float]] = []
        for raw_track_id, head_box, helmet_class in items:
            cls_norm = str(helmet_class or "none").strip().lower()
            hbox = np.asarray(head_box[:4], dtype=np.float32)
            roi = project_upper_torso_roi(hbox, frame_bgr.shape)
            strip_sig = compute_three_strip_torso_signature(frame_bgr, roi)
            hw = max(2.0, float(hbox[2] - hbox[0]))
            tw = max(2.0, float(roi[2] - roi[0]))
            cand_ratio = tw / hw
            prepared.append((raw_track_id, hbox, cls_norm, roi, strip_sig, cand_ratio))

        # 1. Existing ByteTrack raw_id bindings with clothing anti-hijack guard + tentative continuation
        promoted_candidates: Dict[int, UpperTorsoIdentityProfile] = {}
        for idx, (raw_track_id, hbox, cls_norm, roi, strip_sig, _) in enumerate(prepared):
            if raw_track_id in self.raw_to_canonical:
                cid = self.raw_to_canonical[raw_track_id]
                p = self.profiles.get(cid)
                if p is not None and cid not in claimed_cids:
                    sim_strip = p.max_strip_sim(strip_sig)
                    should_break = False
                    if p.is_confirmed and np.linalg.norm(strip_sig) > 1e-6 and p.strip_ema is not None:
                        if sim_strip < 0.22:
                            should_break = True
                        elif sim_strip < 0.34:
                            self.raw_conflict_streak[raw_track_id] = self.raw_conflict_streak.get(raw_track_id, 0) + 1
                            if self.raw_conflict_streak[raw_track_id] >= 4:
                                should_break = True
                        else:
                            self.raw_conflict_streak[raw_track_id] = 0

                    if not should_break:
                        deep_sig = None
                        if self.deep_extractor.available and (
                            p.deep_ema is None or (frame_idx - p.last_deep_frame) >= self.deep_refresh_interval
                        ):
                            deep_sig = self.deep_extractor.extract(frame_bgr, roi)
                        p.update(
                            hbox,
                            cls_norm,
                            strip_sig,
                            roi,
                            frame_idx,
                            ts,
                            deep_sig=deep_sig,
                            min_confirm_hits=self.min_confirm_hits,
                            min_confirm_frames=self.min_confirm_frames,
                        )
                        results[idx] = p
                        claimed_cids.add(cid)
                        claimed_boxes[cid] = (0.5 * float(hbox[0] + hbox[2]), 0.5 * float(hbox[1] + hbox[3]))
                        continue
                    else:
                        self.raw_to_canonical.pop(raw_track_id, None)
                        self.raw_conflict_streak.pop(raw_track_id, None)
                        self.hijack_excluded_ids[raw_track_id] = (cid, fidx_int + 15)
            elif raw_track_id in self._tentative_tracks:
                tent = self._tentative_tracks[raw_track_id]
                tent.update(
                    hbox,
                    cls_norm,
                    strip_sig,
                    roi,
                    frame_idx,
                    ts,
                    deep_sig=None,
                    min_confirm_hits=self.min_confirm_hits,
                    min_confirm_frames=self.min_confirm_frames,
                )
                if tent.consecutive_hits < self.min_confirm_hits:
                    tent.is_confirmed = False
                    results[idx] = tent
                    continue
                # Reached min_confirm_hits: queue for Step 2 re-check before minting canonical ID
                promoted_candidates[idx] = tent
            unmatched_indices.append(idx)

        # 2. Optimal Hungarian Matching against eligible INACTIVE confirmed profiles
        if unmatched_indices:
            inactive_candidates = sorted(
                [
                    p for p in self.profiles.values()
                    if (not p.is_active)
                    and (p.worker_id not in claimed_cids)
                    and (ts - p.last_seen_ts) <= self.ttl_seconds
                ],
                key=lambda prof: (not prof.is_confirmed, -prof.last_seen_ts),
            )

            diag = float(np.hypot(frame_bgr.shape[1], frame_bgr.shape[0])) + 1e-5
            cand_deeps: Dict[int, Optional[np.ndarray]] = {}
            if inactive_candidates and self.deep_extractor.available:
                for orig_idx in unmatched_indices:
                    raw_id, hbox, _, roi, strip_sig, _ = prepared[orig_idx]
                    is_promoted = orig_idx in promoted_candidates
                    if is_promoted:
                        cand_deeps[orig_idx] = self.deep_extractor.extract(frame_bgr, roi)
                        continue
                    c_cx = 0.5 * float(hbox[0] + hbox[2])
                    c_cy = 0.5 * float(hbox[1] + hbox[3])
                    excl_entry = self.hijack_excluded_ids.get(raw_id)
                    excluded_cid = (
                        excl_entry[0]
                        if (excl_entry is not None and fidx_int <= int(excl_entry[1]))
                        else None
                    )
                    coarse_eligible = False
                    for prof in inactive_candidates:
                        if excluded_cid is not None and prof.worker_id == excluded_cid:
                            continue
                        p_cx = 0.5 * float(prof.last_box[0] + prof.last_box[2])
                        p_cy = 0.5 * float(prof.last_box[1] + prof.last_box[3])
                        dist_norm = float(np.hypot(p_cx - c_cx, p_cy - c_cy)) / diag
                        if dist_norm <= 0.25 or prof.max_strip_sim(strip_sig) >= 0.45:
                            coarse_eligible = True
                            break
                    cand_deeps[orig_idx] = (
                        self.deep_extractor.extract(frame_bgr, roi) if coarse_eligible else None
                    )

            if inactive_candidates:
                num_u = len(unmatched_indices)
                num_p = len(inactive_candidates)
                score_mat = np.full((num_u, num_p), -1.0, dtype=np.float32)
                thresh_mat = np.full((num_u, num_p), self.sim_thresh, dtype=np.float32)
                cost_mat = np.full((num_u, num_p), 2.0, dtype=np.float32)

                for r_i, orig_idx in enumerate(unmatched_indices):
                    raw_id, hbox, cls_norm, _, strip_sig, cand_ratio = prepared[orig_idx]
                    tent = promoted_candidates.get(orig_idx)
                    match_strip = (
                        tent.strip_ema
                        if (tent is not None and tent.strip_ema is not None and np.linalg.norm(tent.strip_ema) > 1e-6)
                        else strip_sig
                    )
                    match_ratio = tent.shoulder_head_ratio if tent is not None else cand_ratio
                    match_cls = (
                        tent.last_known_helmet_color
                        if (tent is not None and tent.last_known_helmet_color)
                        else cls_norm
                    )
                    excl_entry = self.hijack_excluded_ids.get(raw_id)
                    excluded_cid = (
                        excl_entry[0]
                        if (excl_entry is not None and fidx_int <= int(excl_entry[1]))
                        else None
                    )
                    c_deep = cand_deeps.get(orig_idx)
                    c_cx = 0.5 * float(hbox[0] + hbox[2])
                    c_cy = 0.5 * float(hbox[1] + hbox[3])

                    for c_j, prof in enumerate(inactive_candidates):
                        if excluded_cid is not None and prof.worker_id == excluded_cid:
                            continue
                        if prof.worker_id in claimed_cids:
                            continue

                        dt = max(0.0, ts - prof.last_seen_ts)
                        p_cx = 0.5 * float(prof.last_box[0] + prof.last_box[2])
                        p_cy = 0.5 * float(prof.last_box[1] + prof.last_box[3])
                        dist_norm = float(np.hypot(p_cx - c_cx, p_cy - c_cy)) / diag

                        has_pos_color_conflict = (
                            match_cls != "none"
                            and prof.last_known_helmet_color is not None
                            and prof.last_known_helmet_color != "none"
                            and match_cls != prof.last_known_helmet_color
                        )

                        if dt <= 3.0 and dist_norm <= 0.18 and not has_pos_color_conflict:
                            appear_sim, _ = self._compute_appear_subscores(
                                prof, match_strip, c_deep, skip_hard_veto=True
                            )
                            spatial_sim = float(np.exp(-dist_norm / 0.12))
                            color_adj = 0.04 if (match_cls != "none" and match_cls == prof.last_known_helmet_color) else 0.0
                            score = float(0.45 * spatial_sim + 0.55 * appear_sim + color_adj)
                            pair_thresh = 0.52
                        else:
                            score = self._compute_candidate_score(
                                prof, match_strip, c_deep, match_cls, match_ratio, skip_hard_veto=False, dt_seconds=dt
                            )
                            pair_thresh = self.sim_thresh

                        score_mat[r_i, c_j] = score
                        thresh_mat[r_i, c_j] = pair_thresh
                        cost_mat[r_i, c_j] = -(score - pair_thresh)

                row_ind, col_ind = linear_sum_assignment(cost_mat)
                still_unmatched: List[int] = list(unmatched_indices)

                for r_i, c_j in zip(row_ind, col_ind):
                    orig_idx = unmatched_indices[r_i]
                    best_score = float(score_mat[r_i, c_j])
                    pair_thresh = float(thresh_mat[r_i, c_j])
                    if best_score < pair_thresh:
                        continue

                    prof = inactive_candidates[c_j]
                    # Ambiguity Margin Guard: inspect second-best competing candidate
                    other_scores = [float(score_mat[r_i, k]) for k in range(num_p) if k != c_j]
                    second_best = max(other_scores) if other_scores else -1.0
                    raw_id, hbox, cls_norm, roi, strip_sig, _ = prepared[orig_idx]

                    if second_best >= self.sim_thresh and (best_score - second_best) < 0.06:
                        p_cx = 0.5 * (prof.last_box[0] + prof.last_box[2])
                        p_cy = 0.5 * (prof.last_box[1] + prof.last_box[3])
                        c_cx = 0.5 * (hbox[0] + hbox[2])
                        c_cy = 0.5 * (hbox[1] + hbox[3])
                        dist_norm = float(np.hypot(p_cx - c_cx, p_cy - c_cy)) / diag
                        if (best_score - second_best) < 0.03 and dist_norm > 0.35:
                            continue

                    tent = promoted_candidates.pop(orig_idx, None)
                    if tent is not None:
                        prof.total_frames_seen += max(0, tent.total_frames_seen - 1)
                        self._tentative_tracks.pop(raw_id, None)
                    prof.update(
                        hbox,
                        cls_norm,
                        strip_sig,
                        roi,
                        frame_idx,
                        ts,
                        deep_sig=cand_deeps.get(orig_idx),
                        min_confirm_hits=self.min_confirm_hits,
                        min_confirm_frames=self.min_confirm_frames,
                    )
                    self.raw_to_canonical[raw_id] = prof.worker_id
                    claimed_cids.add(prof.worker_id)
                    claimed_boxes[prof.worker_id] = (0.5 * float(hbox[0] + hbox[2]), 0.5 * float(hbox[1] + hbox[3]))
                    results[orig_idx] = prof
                    if orig_idx in still_unmatched:
                        still_unmatched.remove(orig_idx)

                unmatched_indices = still_unmatched

            # 3. Confirm-Before-Mint: tentative tracks accumulate hits WITHOUT incrementing next_id;
            #    only tracks promoted at >= min_confirm_hits mint a canonical ID.
            for orig_idx in unmatched_indices:
                raw_id, hbox, cls_norm, roi, strip_sig, _ = prepared[orig_idx]
                tent = promoted_candidates.get(orig_idx)
                if tent is None and self.min_confirm_hits > 1:
                    temp_id = -(abs(int(raw_id)) + 1)
                    tent_prof = UpperTorsoIdentityProfile(
                        worker_id=temp_id,
                        head_box=hbox,
                        helmet_class=cls_norm,
                        strip_sig=strip_sig,
                        deep_sig=None,
                        torso_roi=roi,
                        frame_idx=frame_idx,
                        now_ts=ts,
                    )
                    tent_prof.is_confirmed = False
                    self._tentative_tracks[raw_id] = tent_prof
                    results[orig_idx] = tent_prof
                    continue

                cid = self.next_id
                self.next_id += 1
                c_deep = cand_deeps.get(orig_idx)
                if c_deep is None and self.deep_extractor.available:
                    c_deep = self.deep_extractor.extract(frame_bgr, roi)

                if tent is not None:
                    self._tentative_tracks.pop(raw_id, None)
                    tent.worker_id = cid
                    tent.is_confirmed = True
                    if c_deep is not None and np.linalg.norm(c_deep) > 1e-6:
                        tent.last_deep_frame = frame_idx
                        tent.deep_ema = c_deep.copy()
                        tent.deep_gallery.append(c_deep.copy())
                    new_profile = tent
                else:
                    new_profile = UpperTorsoIdentityProfile(
                        worker_id=cid,
                        head_box=hbox,
                        helmet_class=cls_norm,
                        strip_sig=strip_sig,
                        deep_sig=c_deep,
                        torso_roi=roi,
                        frame_idx=frame_idx,
                        now_ts=ts,
                    )
                    new_profile.is_confirmed = True

                self.profiles[cid] = new_profile
                self.raw_to_canonical[raw_id] = cid
                claimed_cids.add(cid)
                claimed_boxes[cid] = (0.5 * float(hbox[0] + hbox[2]), 0.5 * float(hbox[1] + hbox[3]))
                results[orig_idx] = new_profile

        return [r for r in results if r is not None]

    def consolidate_profiles(self) -> int:
        """
        Merges fragmented profiles that never coexisted in the same frame and share high
        appearance + helmet similarity (>= 0.78). Strictly enforces `coexisted_ids` exclusion.
        """
        merged_count = 0
        cids = sorted(self.profiles.keys())
        dropped: Set[int] = set()

        for i, wid_a in enumerate(cids):
            if wid_a in dropped:
                continue
            pa = self.profiles.get(wid_a)
            if pa is None or not pa.is_confirmed:
                continue

            for wid_b in cids[i + 1 :]:
                if wid_b in dropped:
                    continue
                pb = self.profiles.get(wid_b)
                if pb is None or not pb.is_confirmed:
                    continue
                if pa.is_active and pb.is_active:
                    continue
                # Hard exclusion: two identities seen together in the same frame MUST NEVER merge
                if wid_b in pa.coexisted_ids or wid_a in pb.coexisted_ids:
                    continue

                if (
                    pa.last_known_helmet_color
                    and pb.last_known_helmet_color
                    and pa.last_known_helmet_color != "none"
                    and pb.last_known_helmet_color != "none"
                    and pa.last_known_helmet_color != pb.last_known_helmet_color
                ):
                    continue

                if pb.strip_ema is None or pa.strip_ema is None:
                    continue
                strip_sim = pa.max_strip_sim(pb.strip_ema)
                if strip_sim < 0.78:
                    continue
                if pa.deep_ema is not None and pb.deep_ema is not None:
                    deep_sim = pa.max_deep_sim(pb.deep_ema)
                    if deep_sim < 0.78:
                        continue

                # Consolidate pb into older wid_a
                pa.total_frames_seen += pb.total_frames_seen
                for v in pb.strip_gallery:
                    if pa.max_strip_sim(v) < 0.94:
                        pa.strip_gallery.append(v.copy())
                for dv in pb.deep_gallery:
                    if pa.max_deep_sim(dv) < 0.94:
                        pa.deep_gallery.append(dv.copy())
                pa.coexisted_ids.update(pb.coexisted_ids)
                pa.coexisted_ids.discard(pa.worker_id)
                if not pa.custom_label and pb.custom_label:
                    pa.custom_label = pb.custom_label
                if pb.last_seen_frame > pa.last_seen_frame:
                    pa.last_box = pb.last_box.copy()
                    pa.last_torso_roi = pb.last_torso_roi
                    pa.last_seen_frame = pb.last_seen_frame
                    pa.last_seen_ts = pb.last_seen_ts
                    pa.is_active = pb.is_active
                    pa.current_helmet_color = pb.current_helmet_color
                    if pb.last_known_helmet_color and not pa.last_known_helmet_color:
                        pa.last_known_helmet_color = pb.last_known_helmet_color

                for raw_id, mapped_cid in list(self.raw_to_canonical.items()):
                    if mapped_cid == wid_b:
                        self.raw_to_canonical[raw_id] = wid_a
                self.profiles.pop(wid_b, None)
                dropped.add(wid_b)
                merged_count += 1

        return merged_count

    def associate(
        self,
        raw_track_id: int,
        head_box: np.ndarray | List[float],
        helmet_class: str,
        frame_bgr: np.ndarray,
        frame_idx: int,
        now_ts: Optional[float] = None,
    ) -> UpperTorsoIdentityProfile:
        """Single-detection compatibility wrapper delegating to associate_batch()."""
        return self.associate_batch([(raw_track_id, head_box, helmet_class)], frame_bgr, frame_idx, now_ts=now_ts)[0]

    def update_frame_end(
        self,
        active_canonical_ids: List[int],
        frame_idx: int,
        now_ts: Optional[float] = None,
    ):
        """Updates active states, coexistence exclusion edges, prunes transient tentative tracks, and expires stale profiles."""
        ts = float(now_ts if now_ts is not None else time.time())
        active_set = {int(wid) for wid in active_canonical_ids if isinstance(wid, (int, np.integer)) and int(wid) > 0}

        if len(active_set) >= 2:
            for wid in active_set:
                p = self.profiles.get(wid)
                if p is not None:
                    p.coexisted_ids.update(active_set - {wid})

        for wid, p in self.profiles.items():
            if wid in active_set:
                p.is_active = True
                p.last_seen_frame = frame_idx
                p.last_seen_ts = ts
            else:
                p.is_active = False
                p.consecutive_hits = 0

        # Prune stale tentative tracks (> 2 missed frames) without ever touching self.next_id
        dead_tent = [
            rid for rid, tp in self._tentative_tracks.items()
            if (frame_idx - tp.last_seen_frame) > 2
        ]
        for rid in dead_tent:
            self._tentative_tracks.pop(rid, None)

        if frame_idx % 10 == 0:
            for raw_id, cid in list(self.raw_to_canonical.items()):
                prof = self.profiles.get(cid)
                if prof is None or (not prof.is_active and (frame_idx - prof.last_seen_frame) > 60):
                    self.raw_to_canonical.pop(raw_id, None)
                    self.raw_conflict_streak.pop(raw_id, None)
                    self.hijack_excluded_ids.pop(raw_id, None)

            to_delete = [
                wid for wid, p in self.profiles.items()
                if (
                    (not p.is_active and not p.is_confirmed and (frame_idx - p.last_seen_frame) > 30)
                    or (not p.is_active and (ts - p.last_seen_ts) > self.ttl_seconds)
                )
            ]
            for wid in to_delete:
                self.profiles.pop(wid, None)

            if not self.profiles:
                self.next_id = 1

        if frame_idx % 30 == 0 and len(self.profiles) >= 2:
            self.consolidate_profiles()

    def get_registered_workers(
        self,
        role_policy: Optional[Dict[str, Dict[str, Any]]] = None,
        filter_ghosts: bool = True,
        now_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Returns active + confirmed gallery-remembered personnel enriched with Role-by-Color hierarchy."""
        policy = role_policy or DEFAULT_ROLE_HIERARCHY
        ts = float(now_ts if now_ts is not None else time.time())
        latest_frame = max([p.last_seen_frame for p in self.profiles.values()], default=0)

        workers: List[Dict[str, Any]] = []
        for wid in sorted(self.profiles.keys()):
            p = self.profiles[wid]
            if filter_ghosts and not (p.is_confirmed or (p.is_active and p.consecutive_hits >= 6)):
                continue

            cur_cls = p.current_helmet_color or "none"
            hist_col = p.last_known_helmet_color if p.last_known_helmet_color != "none" else None
            cur_role = policy.get(cur_cls, policy.get("none", {}))
            hist_role = policy.get(hist_col, cur_role) if hist_col else cur_role
            is_viol_state = (p.status == "VIOLATION" and cur_cls == "none")

            frames_since = max(0, latest_frame - p.last_seen_frame) if not p.is_active else 0
            secs_since = round(max(0.0, ts - p.last_seen_ts), 1) if not p.is_active else 0.0

            workers.append({
                "track_id": p.worker_id,
                "custom_label": p.custom_label,
                "helmet_color": cur_cls if cur_cls != "none" else None,
                "current_class": cur_cls,
                "historical_helmet_color": hist_col,
                "role_title": cur_role.get("role_title", "Unclassified"),
                "role_short_code": cur_role.get("short_code", "N/A"),
                "role_hex_color": cur_role.get("hex_color", "#64748B"),
                "historical_role_title": hist_role.get("role_title", "Unclassified"),
                "historical_role_short_code": hist_role.get("short_code", "N/A"),
                "helmet_removed": (is_viol_state and hist_col is not None),
                "status": p.status,
                "is_active": p.is_active,
                "is_confirmed": p.is_confirmed,
                "first_frame": p.first_frame,
                "last_seen_frame": p.last_seen_frame,
                "total_frames_seen": p.total_frames_seen,
                "frames_since_seen": frames_since,
                "seconds_since_seen": secs_since,
                "violations": list(p.violations),
                "trajectory_len": len(p.trajectory),
                "torso_roi": [int(v) for v in p.last_torso_roi],
                "bbox": [round(float(v), 1) for v in p.last_box],
            })
        return workers
```

## 3. API Gateway Circuit Breaker & Proxy Client (`src/web/gateway_client.py`)

```python
"""
src/web/gateway_client.py
=========================
Resilient HTTP Client with Circuit Breaker for API Gateway <-> AI Inference Server.
Guarantees graceful degradation when the AI Inference container restarts or is overloaded:
- CLOSED: Normal forwarding with bounded timeout.
- OPEN: Fails fast for `recovery_timeout` seconds so Gateway + PostgreSQL UI stay responsive.
- HALF_OPEN: Probes AI Inference once cooldown expires and auto-recovers cleanly.
"""

from __future__ import annotations
import threading
import time
from typing import Any, Dict, Optional

import httpx


class CircuitBreakerOpenError(RuntimeError):
    """Raised when the AI Inference circuit breaker is OPEN."""


class InferenceCircuitBreakerClient:
    """Thread-safe Circuit Breaker HTTP client targeting the `ai-inference` service."""

    def __init__(
        self,
        base_url: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 8.0,
        timeout: float = 3.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout = float(recovery_timeout)
        self.timeout = float(timeout)

        self._lock = threading.Lock()
        self.state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._half_open_in_flight = False
        self.consecutive_failures = 0
        self.opened_at = 0.0
        self.failures_total = 0
        self.requests_total = 0
        self.last_latency_ms = 0.0
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=60, keepalive_expiry=30.0)
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, limits=limits)

    def _before_request(self):
        with self._lock:
            self.requests_total += 1
            if self.state == "OPEN":
                if (time.monotonic() - self.opened_at) >= self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self._half_open_in_flight = True
                else:
                    raise CircuitBreakerOpenError("AI Inference circuit breaker is OPEN")
            elif self.state == "HALF_OPEN":
                if self._half_open_in_flight:
                    raise CircuitBreakerOpenError("AI Inference circuit breaker HALF_OPEN probe in flight")
                self._half_open_in_flight = True

    def _record_success(self, latency_ms: float):
        with self._lock:
            self.consecutive_failures = 0
            self.state = "CLOSED"
            self._half_open_in_flight = False
            self.last_latency_ms = round(latency_ms, 2)

    def _record_failure(self):
        with self._lock:
            self.failures_total += 1
            self.consecutive_failures += 1
            if self.state == "HALF_OPEN" or self.consecutive_failures >= self.failure_threshold:
                self.state = "OPEN"
                self.opened_at = time.monotonic()
            self._half_open_in_flight = False

    def request_json(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Any = None,
    ) -> Dict[str, Any]:
        self._before_request()
        t0 = time.perf_counter()
        try:
            resp = self._client.request(method, path, json=json_body, params=params, files=files)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code >= 500:
                self._record_failure()
                raise RuntimeError(f"AI Inference returned HTTP {resp.status_code}")
            if resp.status_code >= 400:
                self._record_success(latency_ms)
                try:
                    if resp.headers.get("content-type", "").startswith("application/json"):
                        detail = resp.json().get("detail", resp.text)
                    else:
                        detail = resp.text
                except Exception:
                    detail = resp.text
                return {"_http_error": resp.status_code, "detail": detail}
            payload = resp.json()
            self._record_success(latency_ms)
            return payload
        except CircuitBreakerOpenError:
            raise
        except Exception as exc:
            self._record_failure()
            raise RuntimeError(f"AI Inference unavailable: {exc}") from exc

    def close(self):
        self._client.close()

    def status_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "is_open": self.state == "OPEN",
                "consecutive_failures": self.consecutive_failures,
                "failures_total": self.failures_total,
                "requests_total": self.requests_total,
                "last_latency_ms": self.last_latency_ms,
                "target_url": self.base_url,
            }
```

## 4. Continuous Integration Workflow (`.github/workflows/ci.yml`)

```yaml
name: Capstone AI CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: "pip"
          cache-dependency-path: "requirements.txt"

      - name: Install CPU PyTorch & project dependencies
        run: |
          python -m pip install --upgrade pip
          pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
          pip install --no-cache-dir -r requirements.txt ruff

      - name: Syntax, lint, and static import check
        run: |
          ruff check --select E9,F63,F7,F82 src tests
          python -m compileall -q src tests scripts

      - name: Run Unit and Integration Tests
        env:
          PYTHONPATH: .
          MODEL_PATH: ci_stub_missing.pt
          CUDA_VISIBLE_DEVICES: "-1"
        run: |
          python -m pytest tests -v

  docker-build-check:
    needs: lint-and-test
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Validate Docker Compose configuration & build slim Gateway image
        run: |
          docker compose -f docker-compose.edge-simulation.yml config -q
          docker compose -f docker-compose.enterprise.yml config -q
          docker build -f Dockerfile.gateway -t aegis-gateway:ci-check .
```

## 5. End-to-End RTX 3050 Paired Benchmark (`Output/task3_eval_opt/gdut_vms_end2end_reid_benchmark.json`)

```json
{
  "gpu_model": "NVIDIA GeForce RTX 3050 Laptop GPU",
  "torch_version": "2.0.0+cu118",
  "default_runtime_mode": "TF32 FP32",
  "matmul_allow_tf32": true,
  "cudnn_allow_tf32": true,
  "allow_tf32": true,
  "fastest_b1_mode": "TF32 FP32",
  "fastest_b1_ms": 12.05,
  "frame_resolution": "1920x1080",
  "warmup_iters": 20,
  "timed_iters": 60,
  "timed_e2e_paired_iters": 120,
  "paired_protocol": "Within-frame paired shared-detection (AB alternating, n=120)",
  "parity_fp32_vs_amp_fp16": {
    "frames": 12,
    "frame_indices": [
      180,
      200,
      220,
      240,
      260,
      280,
      300,
      320,
      340,
      360,
      380,
      400
    ],
    "fp32_boxes": 12,
    "amp_fp16_boxes": 12,
    "max_abs_count_delta": 0,
    "mean_conf_diff": 0.00033
  },
  "stages": [
    {
      "stage": "1. Pure GDUT-A6 Detector (Strict FP32)",
      "config": "FP32 (matmul_tf32=False, cudnn_tf32=False)",
      "latency_ms": 17.11,
      "p95_ms": 17.92,
      "p99_ms": 18.58,
      "fps": 58.4,
      "vram_mb": 106.1,
      "parity_note": "Baseline Ref (12/12 boxes)"
    },
    {
      "stage": "2. Pure GDUT-A6 Detector (TF32 FP32)",
      "config": "TF32 FP32 (matmul_tf32=True, cudnn_tf32=True)",
      "latency_ms": 12.05,
      "p95_ms": 14.06,
      "p99_ms": 15.23,
      "fps": 83.0,
      "vram_mb": 215.0,
      "parity_note": "TensorCore TF32 (-5.06ms vs Strict) [Fastest B=1]"
    },
    {
      "stage": "3. Pure GDUT-A6 Detector (Selective AMP FP16)",
      "config": "AMP FP16 + CoordConv/BiFormer FP32 guard (TF32=False)",
      "latency_ms": 13.29,
      "p95_ms": 15.65,
      "p99_ms": 16.47,
      "fps": 75.2,
      "vram_mb": 210.2,
      "parity_note": "Delta=0 boxes, conf diff=0.00033"
    },
    {
      "stage": "4. GDUT-A6 TF32 + ByteTrack + 3-Strip Non-Biometric Re-ID",
      "config": "TF32 FP32 + HSV/Lab + Quadrant HOG + Hungarian",
      "latency_ms": 15.17,
      "p95_ms": 18.03,
      "p99_ms": 18.45,
      "det_mean_ms": 12.46,
      "reid_mean_ms": 2.7,
      "fps": 65.9,
      "vram_mb": 215.0,
      "parity_note": "Paired Shared-Det (n=120): det=12.46ms + strip_reid=2.7ms (use_deep=False)"
    },
    {
      "stage": "5. Full VMS + Multi-Layer MobileNetV3 Non-Biometric Re-ID",
      "config": "TF32 FP32 + Deep Re-ID=GPU (refresh=30)",
      "latency_ms": 15.33,
      "p95_ms": 18.57,
      "p99_ms": 19.42,
      "det_mean_ms": 12.46,
      "reid_mean_ms": 2.87,
      "paired_delta_mean_ms": 0.16,
      "fps": 65.2,
      "vram_mb": 218.6,
      "deep_extract_calls": 4,
      "deep_frame_mean_ms": 20.27,
      "deep_frame_p95_ms": 22.78,
      "non_deep_frame_mean_ms": 15.16,
      "parity_note": "Paired Shared-Det (n=120): +0.16ms avg (4 MobileNetV3 calls; extract=20.27ms vs steady=15.16ms; >=25 FPS SLA)"
    }
  ]
}
```

## 6. Inference Optimization & Parity Table (`Output/task3_eval_opt/optimization_benchmark_table.json`)

```json
{
  "device": "NVIDIA GeForce RTX 3050 Laptop GPU",
  "torch_version": "2.0.0+cu118",
  "records": [
    {
      "model": "Rep-YOLO11s-GDUT-A6 (Capstone 5-Cls)",
      "stage": "CPU 4-Thread FP32",
      "device": "CPU (4 threads)",
      "precision": "FP32 (CPU)",
      "batch": 1,
      "latency_ms": 297.69,
      "fps": 3.4,
      "vram_mb": 0.0,
      "ckpt_mb": 36.55,
      "speedup_vs_before": 1.0,
      "accuracy_delta": "CPU FP32 Ref"
    },
    {
      "model": "Rep-YOLO11s-GDUT-A6 (Capstone 5-Cls)",
      "stage": "Before (Loop BiFormer FP32)",
      "device": "NVIDIA GeForce RTX 3050 Laptop GPU",
      "precision": "FP32 (cuDNN TF32=True)",
      "batch": 1,
      "latency_ms": 13.62,
      "fps": 73.4,
      "vram_mb": 222.5,
      "ckpt_mb": 36.55,
      "speedup_vs_before": 1.0,
      "accuracy_delta": "GPU FP32 Ref"
    },
    {
      "model": "Rep-YOLO11s-GDUT-A6 (Capstone 5-Cls)",
      "stage": "After-1 (Vec BiFormer + RepConv FP32)",
      "device": "NVIDIA GeForce RTX 3050 Laptop GPU",
      "precision": "FP32 (cuDNN TF32=True)",
      "batch": 1,
      "latency_ms": 13.28,
      "fps": 75.3,
      "vram_mb": 222.5,
      "ckpt_mb": 36.55,
      "speedup_vs_before": 1.03,
      "accuracy_delta": "Exact Reparameterized FP32"
    },
    {
      "model": "Rep-YOLO11s-GDUT-A6 (Capstone 5-Cls)",
      "stage": "After-2 (+ CUDA AMP FP16)",
      "device": "NVIDIA GeForce RTX 3050 Laptop GPU",
      "precision": "FP16-AMP",
      "batch": 1,
      "latency_ms": 12.08,
      "fps": 82.8,
      "vram_mb": 217.5,
      "ckpt_mb": 36.55,
      "speedup_vs_before": 1.13,
      "accuracy_delta": "B=1: Delta 0 boxes (12/12, conf diff 0.00033)"
    },
    {
      "model": "Rep-YOLO11s-GDUT-A6 (Capstone 5-Cls)",
      "stage": "After-3 (Micro-Batch N=4 AMP FP16)",
      "device": "NVIDIA GeForce RTX 3050 Laptop GPU",
      "precision": "FP16-AMP",
      "batch": 4,
      "latency_ms": 8.46,
      "fps": 118.2,
      "vram_mb": 295.3,
      "ckpt_mb": 36.55,
      "speedup_vs_before": 1.61,
      "accuracy_delta": "N=4: Delta 0 boxes (12/12, conf diff 0.00032)"
    }
  ],
  "video_parity_fp32_vs_amp_fp16": {
    "Rep-YOLO11s-GDUT-A6 (Capstone 5-Cls)": {
      "frames": 12,
      "frame_indices": [
        180,
        200,
        220,
        240,
        260,
        280,
        300,
        320,
        340,
        360,
        380,
        400
      ],
      "fp32_boxes": 12,
      "fp16_amp_boxes": 12,
      "max_abs_count_delta": 0,
      "mean_conf_diff": 0.00033
    }
  },
  "video_parity_fp32_vs_amp_fp16_batch4": {
    "Rep-YOLO11s-GDUT-A6 (Capstone 5-Cls)": {
      "frames": 12,
      "frame_indices": [
        180,
        200,
        220,
        240,
        260,
        280,
        300,
        320,
        340,
        360,
        380,
        400
      ],
      "fp32_boxes": 12,
      "fp16_amp_boxes": 12,
      "max_abs_count_delta": 0,
      "mean_conf_diff": 0.00032,
      "batch": 4
    }
  }
}
```

## 7. Enterprise Docker Stack Load Test (`Output/load_test_enterprise_docker.json`)

```json
[
  {
    "concurrency": 10,
    "total_requests": 200,
    "wall_s": 0.767,
    "rps": 260.8,
    "avg_ms": 37.04,
    "p50_ms": 30.72,
    "p95_ms": 69.09,
    "p99_ms": 76.32,
    "errors": 0,
    "error_pct": 0.0
  },
  {
    "concurrency": 25,
    "total_requests": 300,
    "wall_s": 1.169,
    "rps": 256.6,
    "avg_ms": 92.65,
    "p50_ms": 89.83,
    "p95_ms": 133.69,
    "p99_ms": 146.9,
    "errors": 0,
    "error_pct": 0.0
  },
  {
    "concurrency": 50,
    "total_requests": 400,
    "wall_s": 1.68,
    "rps": 238.1,
    "avg_ms": 196.97,
    "p50_ms": 198.47,
    "p95_ms": 250.91,
    "p99_ms": 267.62,
    "errors": 0,
    "error_pct": 0.0
  },
  {
    "concurrency": 100,
    "total_requests": 500,
    "wall_s": 2.22,
    "rps": 225.2,
    "avg_ms": 394.8,
    "p50_ms": 422.74,
    "p95_ms": 480.67,
    "p99_ms": 500.78,
    "errors": 0,
    "error_pct": 0.0
  }
]
```
