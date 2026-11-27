# Proposed V1 Auxiliary Cross-Domain Benchmark Summary (`SHWD` + Zero-Shot `HHW`)

> **Scope Isolation:** Strictly isolated 2-class binary auxiliary evaluation (`hat`, `person`) used in Capstone Project `FA26AI16` to trace the architectural evolution from **Proposed V1** (12 PAN `RepConv` blocks + dual `BiFormerBlockLite` at P4/P5, 10.74M deployed parameters) to **Proposed V2** (5-class `GDUT-HWD` primary benchmark).

## 1. Controlled SOTA YOLO Baselines on SHWD Test (`1,517` Images, `imgsz=640`, Seed 42)

| Model | Params | T4 FP16 (ms / FPS) | SHWD mAP@50 (%) | SHWD mAP@50-95 (%) | `hat` AP@50 (%) | `person` AP@50 (%) |
|---|---:|---:|---:|---:|---:|---:|
| `YOLOv8s` | 11,136,374 | 10.16 / 98.4 | 94.23 | 61.05 | 93.50 | 94.96 |
| `YOLOv9s` | 7,288,182 | 36.41 / 27.5 | 94.34 | 61.72 | 93.63 | 95.05 |
| `YOLOv10s` | 8,067,900 | 16.69 / 59.9 | 94.14 | 61.69 | 93.08 | 95.20 |
| `YOLO12s` | 9,253,910 | 21.95 / 45.5 | 94.56 | 61.70 | 93.90 | 95.21 |

## 2. Proposed V1 Three-Seed Ablation on SHWD Test (`Seeds 42, 1337, 2026`, Mean ± Sample SD `ddof=1`)

| Config | Deploy Params | SHWD mAP@50 (%) | SHWD mAP@50-95 (%) | `hat` P (%) | `hat` R (%) | `hat` AP@50 (%) | T4 FP16 Latency (ms) | T4 FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A0** (Stock YOLO11s (CIoU)) | 9,428,566 | 94.51 ± 0.06 | 61.67 ± 0.03 | 92.76 ± 0.13 | 89.63 ± 0.11 | 93.87 ± 0.13 | 14.15 ± 0.24 | 70.67 ± 1.19 |
| **A1** (+ CoordConv Stem) | 9,429,142 | 94.38 ± 0.26 | 61.61 ± 0.30 | 92.23 ± 0.50 | 89.61 ± 0.29 | 93.60 ± 0.53 | 14.59 ± 0.59 | 68.60 ± 2.75 |
| **A2** (+ Neck RepConv (V1 Edge)) | 9,427,766 | 94.30 ± 0.36 | 61.61 ± 0.05 | 92.28 ± 0.29 | 89.65 ± 0.63 | 93.51 ± 0.65 | 13.55 ± 0.23 | 73.83 ± 1.27 |
| **A3** (+ BiFormer P4/P5) | 10,740,024 | 94.44 ± 0.20 | 61.65 ± 0.10 | 92.87 ± 0.51 | 89.22 ± 0.14 | 93.78 ± 0.24 | 16.43 ± 0.11 | 60.83 ± 0.42 |
| **A4** (+ Focal-EIoU (Full V1)) | 10,740,024 | 94.18 ± 0.23 | 61.33 ± 0.25 | 92.75 ± 0.43 | 89.07 ± 0.93 | 93.52 ± 0.36 | 16.25 ± 0.09 | 61.53 ± 0.35 |

- **Paired two-tailed t-tests (`n=3` seeds):** `A2` vs `A0` SHWD mAP@50 parity `p = 0.3834`; `A2` vs `A1` T4 FP16 speedup `p = 0.0386`; `A4` vs `A0` `person` AP@50 `p = 0.0144` (focal down-weighting trade-off on tiny non-hardhat heads).

## 3. Zero-Shot Cross-Domain Generalization on HHW Test (`1,766` Images, Mean ± Sample SD `ddof=1`)

| Config | HHW 2-Class mAP@50 (%) | HHW 2-Class mAP@50-95 (%) | HHW Hat-Only mAP@50 (%) | HHW Hat-Only mAP@50-95 (%) |
|---|---:|---:|---:|---:|
| **A0** | 95.63 ± 0.33 | 55.90 ± 0.58 | 96.64 ± 0.23 | 54.87 ± 0.50 |
| **A1** | 95.74 ± 0.08 | 56.46 ± 0.51 | 96.60 ± 0.27 | 55.30 ± 0.53 |
| **A2** | 95.78 ± 0.07 | 56.48 ± 0.42 | 96.71 ± 0.31 | 55.54 ± 0.48 |
| **A3** | 95.48 ± 0.16 | 56.10 ± 0.24 | 96.50 ± 0.11 | 55.09 ± 0.02 |
| **A4** | 95.69 ± 0.05 | 56.25 ± 0.44 | 96.72 ± 0.09 | 55.17 ± 0.48 |

- **Zero-Shot HHW Takeaways:** `V1 A2 (Edge)` achieves peak cross-domain 2-Class `mAP@50 = 95.78 ± 0.07%` (`4.7x` lower SD than `A0`) and peak `mAP@50-95 = 56.48 ± 0.42%` (`Hat-Only mAP@50-95 = 55.54 ± 0.48%`). `V1 A4` achieves the lowest 2-Class `mAP@50` SD (`±0.05`) and reaches paired `mAP@50-95 p = 0.0527` vs `A0`.

### Zero-Shot HHW Reference Baselines (`hhw_zeroshot_baselines.csv`, Seed 42 Test Split)

| Model | Eval Split | mAP@50 (%) | mAP@50-95 (%) | Precision (%) | Recall (%) |
|---|---|---:|---:|---:|---:|
| `YOLOv8s` | `hhw_2class_test` | 94.45 | 54.96 | 92.01 | 91.64 |
| `YOLOv8s` | `hhw_hatonly_test` | 96.08 | 54.13 | 94.24 | 90.52 |
| `YOLOv9s` | `hhw_2class_test` | 95.70 | 55.91 | 93.36 | 93.02 |
| `YOLOv9s` | `hhw_hatonly_test` | 96.64 | 54.85 | 94.80 | 91.94 |
| `YOLOv10s` | `hhw_2class_test` | 95.90 | 56.70 | 94.33 | 92.58 |
| `YOLOv10s` | `hhw_hatonly_test` | 96.67 | 55.25 | 95.19 | 91.47 |
| `YOLO12s` | `hhw_2class_test` | 95.95 | 56.34 | 93.74 | 93.00 |
| `YOLO12s` | `hhw_hatonly_test` | 96.70 | 54.85 | 94.45 | 92.42 |
