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
