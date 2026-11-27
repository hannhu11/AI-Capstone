# Kaggle Forensic Verification Report — GDUT-HWD 5-Class (`nvthanh2004`, 15 Official Notebooks)

> **Generated (UTC):** `2026-11-27T10:57:15.133692+00:00`
> **Canonical Source:** `FULL CODE CLEAN( ban da chinh sua _ them link Baseline chuan.docx` (`11 GDUT_V2` + `4 Unified Baselines`, `nvthanh2004`).
> **Audit Score:** **`75/75` checks MATCHED (`0` mismatches)** across all **15** official `nvthanh2004` notebooks under the unified `ce2a883e` Stage 1 protocol.

---

## 1. Official Public Kaggle Inventory (15 Notebooks, Read-Only, Zero `.pt` Bloat)

| # | Owner / Kernel Slug | Ver | Cells | Light Files | Kaggle URL |
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
| 12 | `nvthanh2004/gdut-stage1-yolo11n-for-dcrcf` | v1 | 22 | 25 | [gdut-stage1-yolo11n-for-dcrcf](https://www.kaggle.com/code/nvthanh2004/gdut-stage1-yolo11n-for-dcrcf) |
| 13 | `nvthanh2004/gdut-baseline-yolov8-dwr-aspp-nwd-stage1-finetune` | v1 | 24 | 29 | [gdut-baseline-yolov8-dwr-aspp-nwd-stage1-finetune](https://www.kaggle.com/code/nvthanh2004/gdut-baseline-yolov8-dwr-aspp-nwd-stage1-finetune) |
| 14 | `nvthanh2004/gdut-baseline-yolo11-dcrcf-stage1-finetune` | v1 | 24 | 29 | [gdut-baseline-yolo11-dcrcf-stage1-finetune](https://www.kaggle.com/code/nvthanh2004/gdut-baseline-yolo11-dcrcf-stage1-finetune) |
| 15 | `nvthanh2004/gdut-baseline-and-a6-stage1-protocol-comparison` | v1 | 23 | 31 | [gdut-baseline-and-a6-stage1-protocol-comparison](https://www.kaggle.com/code/nvthanh2004/gdut-baseline-and-a6-stage1-protocol-comparison) |

---

## 2. Unified Protocol Comparison on Official GDUT Test (`1,587` Imgs / `9,574` Boxes)

| Model | Deploy Params (Fused) | T4 FP32 (ms) | P (%) | R (%) | F1 (%) | mAP@50 (%) | mAP@50-95 (%) | `none` AP@50 / AP@50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `A6_yolo11s` | 9,476,783 | 15.207 | 90.19 | 84.28 | 87.08 | 88.94 | 58.08 | 84.6 / 53.4% |
| `A6_yolov8s` | 12,177,727 | 18.641 | 90.32 | 82.63 | 86.20 | 88.37 | 57.37 | 83.5 / 51.8% |
| `yolo11_dcrcf` | 6,796,045 | 11.048 | 90.17 | 76.48 | 82.68 | 84.85 | 54.77 | 78.3 / 48.5% |
| `yolov8_dwr_aspp_nwd` | 16,275,615 | 22.396 | 89.58 | 81.85 | 85.50 | 87.51 | 56.59 | 82.7 / 51.3% |

---

## 3. Exact Full-Precision Ablation `A0–A6` CSV (`gdut-final-test-evaluation`)

| Backbone | Config | P (%) | R (%) | Exact mAP@50 (%) | Exact mAP@50-95 (%) | `none` AP@50 / AP@50-95 |
|---|---|---:|---:|---:|---:|---:|
| `yolo11s` | **A0** | 90.98 | 82.83 | 88.8149 | 58.4641 | 83.3 / 52.5% |
| `yolo11s` | **A1** | 91.24 | 84.36 | 89.7616 | 59.0804 | 84.9 / 54.4% |
| `yolo11s` | **A2** | 91.06 | 83.51 | 89.4383 | 58.5829 | 83.9 / 53.2% |
| `yolo11s` | **A3** | 90.73 | 83.90 | 89.2501 | 58.3275 | 84.2 / 53.4% |
| `yolo11s` | **A4** | 90.15 | 84.30 | 89.1989 | 58.4888 | 83.4 / 53.1% |
| `yolo11s` | **A5** | 90.22 | 84.22 | 89.2391 | 58.6058 | 83.9 / 53.9% |
| `yolo11s` | **A6** | 90.19 | 84.28 | 88.9379 | 58.0825 | 84.6 / 53.4% |
| `yolov8s` | **A0** | 90.12 | 81.84 | 87.4565 | 57.1973 | 81.9 / 50.9% |
| `yolov8s` | **A1** | 90.52 | 83.35 | 88.7328 | 57.7618 | 83.7 / 52.2% |
| `yolov8s` | **A2** | 90.59 | 83.07 | 88.5582 | 57.2438 | 83.7 / 51.8% |
| `yolov8s` | **A3** | 89.71 | 83.56 | 88.7305 | 57.1614 | 83.3 / 51.5% |
| `yolov8s` | **A4** | 90.95 | 82.56 | 88.2910 | 57.2358 | 83.6 / 51.7% |
| `yolov8s` | **A5** | 90.86 | 82.42 | 88.3448 | 57.3292 | 83.3 / 51.9% |
| `yolov8s` | **A6** | 90.32 | 82.63 | 88.3692 | 57.3711 | 83.5 / 51.8% |

---

## 4. Forensic Audit Findings

- **[VERIFIED_PROTOCOL] All 15 official `nvthanh2004` notebooks (`11 GDUT_V2 + 4 Baselines`) share the exact SHA-256 dataset digest (`ce2a883e0b50...`) and 48/48 matched Stage 2 hyperparameters.**
  Replaces all legacy experimental baselines. On identical 1,587-image Official Test (`ce2a883e` Stage 1 protocol), `A6_yolo11s` reaches `88.94% mAP50 / 58.08% mAP50-95` (`9,476,783` deploy fused params, `15.207 ms` T4 FP32), outperforming `B1 (YOLO11n-DCRCF)` (`84.85% / 54.77%`, `+4.09 / +3.31 pp`) and `B2 (YOLOv8s-DWR-ASPP-NWD)` (`87.51% / 56.59%`, `+1.43 / +1.50 pp`, `-41.8%` deploy params, `-32.1%` T4 latency). Even `A6_yolov8s` (`88.37% / 57.37%`) beats both `B1` and `B2`.
- **[FINDING_1] Full-precision `test_evaluation/summary.csv` resolves `A1` vs `A3` rounding (`A1_V8 = 88.7328%` > `A3_V8 = 88.7305%`).**
  `A1_yolo11s` leads `A0..A6` on single-seed GDUT at `89.7616% / 59.0804%` (`A6_yolo11s = 88.9379% / 58.0825%`).
- **[FINDING_2] Fused deploy parameter count (`9,476,783` vs `9,487,583`) reconciled mathematically (`10,800` Conv-BN folding params).**
  In `Baseline_A6_Comparison.ipynb`, `A6_yolo11s` calls BOTH `switch_to_deploy()` AND `model.fuse()` (folding Conv2d+BatchNorm2d = `-10,800` affine parameters), producing `9,476,783` deployed parameters (`15.207 ms` T4 FP32); calling `switch_to_deploy()` alone yields `9,487,583` parameters (`-300,992` from `RepConv`).
- **[FINDING_3] Paper-to-GDUT cross-domain gaps documented from `paper_vs_experiment.csv` (`B1 -7.85 / -5.83 pp`, `B2 -4.49 / -5.61 pp`).**
  Both peer-reviewed detectors suffer sharp drops moving from 2-class SHWD to deduplicated 5-class GDUT Test: B1 recalls drop `-12.02 pp` (Test `none` AP50 = `78.3%`, R = `68.0%`) due to `26.22%` neck transfer discontinuity (`78` newly initialized tensors), while B2 inflates parameters to `16,275,615` fused (`22.396 ms` T4 FP32).

---

## 5. Complete 1-to-1 Cross-Verification Matrix (`MATCH / MISMATCH`)

| Category | Checked Item | Expected Claim | Raw Kaggle Truth | Verdict |
|---|---|---|---|---|
| 1. Dataset & Provenance | Cryptographic SHA-256 of dataset_manifest.json across ALL 15 notebooks (Prep, S1, A0..A6, B1, B2) | `ce2a883e0b5050edb96d494218e85a9d16ea04b9a19dc5723c572cb502edfe9c` | Prep=`ce2a883e0b5050ed...`, all 20 runs identical=True | **MATCH** |
| 1. Dataset & Provenance | Deduplication (3,174 -> 3,168), Split (1,265/316/1,587), Taxonomy & Box Counts | 3,168 clean (6 dupes removed), 1265/316/1587, boxes 7,537/1,738/9,574, names=[blue,white,yellow,red,none] | records=3168, excluded=6 [('JPEGImages/00286.jpg', 'JPEGImages/02142.jpg'), ('JPEGImages/00641.jpg', 'JPEGImages/00615.jpg'), ('JPEGImages/00676.jpg', 'JPEGImages/00239.jpg'), ('JPEGImages/00801.jpg', 'JPEGImages/01007.jpg'), ('JPEGImages/03087.jpg', 'JPEGImages/01726.jpg'), ('JPEGImages/03365.jpg', 'JPEGImages/02224.jpg')], splits={'test': 1587, 'val': 316, 'train': 1265}, boxes={'train': {'red': 1865, 'white': 1462, 'yellow': 1465, 'none': 1767, 'blue': 978}, 'val': {'red': 279, 'white': 349, 'yellow': 448, 'none': 404, 'blue': 258}, 'test': {'white': 1968, 'red': 2083, 'none': 2200, 'yellow': 1962, 'blue': 1361}} | **MATCH** |
| 1. Dataset & Provenance | Color-Safe Aug (`hsv_h=0, hsv_s=0`) & Online Asymmetry (`A1` vs `A2..A6`) | hsv_h=0, hsv_s=0; A1 mixup=0.1, hsv_v=0.45; A2..A6 mixup=0, hsv_v=0.4 | hsv_locked=True; A1=(mixup=0.1, hsv_v=0.45); A6=(mixup=0.0, hsv_v=0.4) | **MATCH** |
| 1. Dataset & Provenance | Unified 48-Hyperparameter Stage 2 Recipe Matrix (`B1`, `B2`, `A6_yolo11s`, `A6_yolov8s`) | All 48 parameters matched (AdamW, lr0=1e-4, batch=16, seed=3407, hsv_h/s=0, mixup=0) | stage2_recipe_audit.csv rows=48, all_matched=True | **MATCH** |
| 2. Architecture & Code | Compiled A6/A3 Layer Indices, Param Count (9,788,575) & BiFormer(r=8,k=4,h=4) | Y11 A6: Coord[0], Rep[1,3,5,7,17,20], BiF[10], 9.79M; V8 A6: Rep[1,3,5,7,16,19], BiF[9]; Y11 A3: 7xRep | Y11_A6={'CoordConv': [0], 'RepConv': [1, 3, 5, 7, 17, 20], 'BiFormerBlockLite': [10]} (9788575); V8_A6={'CoordConv': [0], 'RepConv': [1, 3, 5, 7, 16, 19], 'BiFormerBlockLite': [9]}; Y11_A3={'CoordConv': [], 'RepConv': [0, 1, 3, 5, 7, 17, 20], 'BiFormerBlockLite': []} | **MATCH** |
| 2. Architecture & Code | RepConv Identity Branch Policy & Focal-EIoU Loss Wiring in ResearchTrainer | stride-2 RepConv has NO identity branch; init_criterion wires FocalEIoUBboxLoss | rbr_identity=(c1==c2 and s==1): True; FocalEIoUBboxLoss wired: True | **MATCH** |
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
| 6. Per-Class Test A6 (yolo11s) | yolo11s A6 `blue` (R / AP50 / AP50-95) | 86.6 / 89.8 / 59.4% | 86.6 / 89.8 / 59.4% | **MATCH** |
| 6. Per-Class Test A6 (yolo11s) | yolo11s A6 `white` (R / AP50 / AP50-95) | 86.6 / 89.1 / 59.5% | 86.6 / 89.1 / 59.5% | **MATCH** |
| 6. Per-Class Test A6 (yolo11s) | yolo11s A6 `yellow` (R / AP50 / AP50-95) | 88.2 / 92.7 / 60.7% | 88.2 / 92.7 / 60.7% | **MATCH** |
| 6. Per-Class Test A6 (yolo11s) | yolo11s A6 `red` (R / AP50 / AP50-95) | 82.7 / 88.6 / 57.4% | 82.7 / 88.6 / 57.4% | **MATCH** |
| 6. Per-Class Test A6 (yolo11s) | yolo11s A6 `none` (R / AP50 / AP50-95) | 77.3 / 84.6 / 53.4% | 77.3 / 84.6 / 53.4% | **MATCH** |
| 6. Per-Class Test A6 (yolov8s) | yolov8s A6 `blue` (R / AP50 / AP50-95) | 83.3 / 89.3 / 58.8% | 83.3 / 89.3 / 58.8% | **MATCH** |
| 6. Per-Class Test A6 (yolov8s) | yolov8s A6 `white` (R / AP50 / AP50-95) | 86.9 / 90.4 / 59.6% | 86.9 / 90.4 / 59.6% | **MATCH** |
| 6. Per-Class Test A6 (yolov8s) | yolov8s A6 `yellow` (R / AP50 / AP50-95) | 87.8 / 92.4 / 60.6% | 87.8 / 92.4 / 60.6% | **MATCH** |
| 6. Per-Class Test A6 (yolov8s) | yolov8s A6 `red` (R / AP50 / AP50-95) | 80.1 / 86.3 / 56.0% | 80.1 / 86.3 / 56.0% | **MATCH** |
| 6. Per-Class Test A6 (yolov8s) | yolov8s A6 `none` (R / AP50 / AP50-95) | 75.1 / 83.5 / 51.8% | 75.1 / 83.5 / 51.8% | **MATCH** |
| 7. Inference Speed & Fusion Honesty | gdut-final-test-evaluation T4 speed (ms) & switch_to_deploy() in final_test() | A1 Y11=7.5ms, V8=7.4ms; A6 Y11=9.2ms, V8=9.1ms (unfused; final_test does NOT call switch_to_deploy) | Log T4: A1_Y11=7.5ms, A1_V8=7.4ms, A6_Y11=9.2ms, A6_V8=9.1ms; switch_to_deploy_in_final_test=False | **MATCH** |
| 8. Unified SOTA Baselines (nvthanh2004) | B1 YOLO11n-DCRCF Transfer & Epochs (Stage 1 YOLO11n -> Finetune) | 1,783,391 / 6,802,005 (26.22%), S1 100 ep (best 70), S2 100 ep (best 54), ModDCNv2[6,8]+SBA[11,13,15,17] | transferred=1783391/6802005 (26.22%), S1=100, S2=100 best=54 | **MATCH** |
| 8. Unified SOTA Baselines (nvthanh2004) | B2 YOLOv8s-DWR-ASPP-NWD Transfer & Epochs (Stage 1 YOLOv8s -> Finetune) | 10,612,223 / 16,284,863 (65.17%), S1 97 ep, S2 59 ep (best 9), DWR[2,4,6]+ASPP[9] | transferred=10612223/16284863 (65.17%), S2=59 best=9 | **MATCH** |
| 8. Unified Comparator (val) | `A6_yolo11s` VAL (P / R / mAP50 / mAP50-95 / F1 | deploy | T4 ms) | 90.45 / 87.58 / 92.50 / 61.23 / 88.87 | 9,476,783 | 15.207ms | 90.45 / 87.58 / 92.50 / 61.23 / 88.87 | 9,476,783 | 15.207ms | **MATCH** |
| 8. Unified Comparator (test) | `A6_yolo11s` TEST (P / R / mAP50 / mAP50-95 / F1 | deploy | T4 ms) | 90.19 / 84.28 / 88.94 / 58.08 / 87.08 | 9,476,783 | 15.207ms | 90.19 / 84.28 / 88.94 / 58.08 / 87.08 | 9,476,783 | 15.207ms | **MATCH** |
| 8. Unified Comparator (val) | `A6_yolov8s` VAL (P / R / mAP50 / mAP50-95 / F1 | deploy | T4 ms) | 93.21 / 85.11 / 92.56 / 60.68 / 88.85 | 12,177,727 | 18.641ms | 93.21 / 85.11 / 92.56 / 60.68 / 88.85 | 12,177,727 | 18.641ms | **MATCH** |
| 8. Unified Comparator (test) | `A6_yolov8s` TEST (P / R / mAP50 / mAP50-95 / F1 | deploy | T4 ms) | 90.32 / 82.63 / 88.37 / 57.37 / 86.20 | 12,177,727 | 18.641ms | 90.32 / 82.63 / 88.37 / 57.37 / 86.20 | 12,177,727 | 18.641ms | **MATCH** |
| 8. Unified Comparator (val) | `yolo11_dcrcf` VAL (P / R / mAP50 / mAP50-95 / F1 | deploy | T4 ms) | 91.11 / 79.40 / 88.27 / 57.46 / 84.63 | 6,796,045 | 11.048ms | 91.11 / 79.40 / 88.27 / 57.46 / 84.63 | 6,796,045 | 11.048ms | **MATCH** |
| 8. Unified Comparator (test) | `yolo11_dcrcf` TEST (P / R / mAP50 / mAP50-95 / F1 | deploy | T4 ms) | 90.17 / 76.48 / 84.85 / 54.77 / 82.68 | 6,796,045 | 11.048ms | 90.17 / 76.48 / 84.85 / 54.77 / 82.68 | 6,796,045 | 11.048ms | **MATCH** |
| 8. Unified Comparator (val) | `yolov8_dwr_aspp_nwd` VAL (P / R / mAP50 / mAP50-95 / F1 | deploy | T4 ms) | 92.39 / 84.05 / 91.46 / 60.08 / 87.94 | 16,275,615 | 22.396ms | 92.39 / 84.05 / 91.46 / 60.08 / 87.94 | 16,275,615 | 22.396ms | **MATCH** |
| 8. Unified Comparator (test) | `yolov8_dwr_aspp_nwd` TEST (P / R / mAP50 / mAP50-95 / F1 | deploy | T4 ms) | 89.58 / 81.85 / 87.51 / 56.59 / 85.50 | 16,275,615 | 22.396ms | 89.58 / 81.85 / 87.51 / 56.59 / 85.50 | 16,275,615 | 22.396ms | **MATCH** |
