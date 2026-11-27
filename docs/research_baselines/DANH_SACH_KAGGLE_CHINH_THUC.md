# Danh Sách 15 Notebook Kaggle Chính Thức (`nvthanh2004`)

> **Nguồn đối soát duy nhất:** `Research_Code_Results ( Của Thành chuẩn bị )\FULL CODE CLEAN( bản đã chỉnh sửa _ thêm link Baseline chuan.docx`. Toàn bộ 15 notebook đều chạy trên tài khoản chính thức **`nvthanh2004`**, sử dụng cùng giao thức đánh giá thống nhất **`Stage 1 clean` (`ce2a883e`, `imgsz=640`, `conf=0.25`, `iou=0.45`, FP32 Batch-1 trên Tesla T4)** và cố định hạt giống `seed=3407`. Toàn bộ notebook và số liệu baseline thử nghiệm cũ ngoài danh sách này đã bị loại bỏ hoàn toàn.

---

## 1. Nhóm `GDUT_V2` — Tiền xử lý, Stage 1, Ablation A0–A6 & Final Test (11 Notebooks)

| STT | Vai trò / Giai đoạn | Link Kaggle chính thức (`nvthanh2004`) |
| :---: | :--- | :--- |
| 1 | **Dataset Preparation & SHA-256 Deduplication** (`3,174` ảnh gốc $\to$ `3,171` sạch: `1,268` train / `316` val / `1,587` test) | https://www.kaggle.com/code/nvthanh2004/gdut-dataset-preparation |
| 2 | **Stage 1 Backbone/Neck Pretraining — Stock `YOLOv8s`** | https://www.kaggle.com/code/nvthanh2004/gdut-structural-re-parameterize-yolo-architecture1 |
| 3 | **Stage 1 Backbone/Neck Pretraining — Stock `YOLO11s`** | https://www.kaggle.com/code/nvthanh2004/gdut-structural-re-parameterize-yolo-architecture2 |
| 4 | **Stage 2 Ablation `A0`** (Eval-only Stage 1 baseline) | https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a0 |
| 5 | **Stage 2 Ablation `A1`** (`+ HardAug + Online Mixup/V`) | https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a1 |
| 6 | **Stage 2 Ablation `A2`** (`+ CoordConv Stem`) | https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a2 |
| 7 | **Stage 2 Ablation `A3`** (`+ 7x RepConv Neck`) | https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a3 |
| 8 | **Stage 2 Ablation `A4`** (`+ Focal-EIoU`) | https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a4 |
| 9 | **Stage 2 Ablation `A5`** (`+ BiFormerBlockLite`) | https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a5 |
| 10 | **Stage 2 Ablation `A6`** (Full Proposed `Rep-YOLO11s`) | https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a6 |
| 11 | **Official 1,587-Image Test Evaluation** (`14` Checkpoints `A0`–`A6`) | https://www.kaggle.com/code/nvthanh2004/gdut-final-test-evaluation |

---

## 2. Nhóm `Baselines` — Hai Mô Hình SOTA Đối Chứng Trực Tiếp Theo Cùng Giao Thức (4 Notebooks)

| STT | Vai trò / Mô hình | Tóm tắt số liệu chuẩn `ce2a883e` (Official Test `1,587` ảnh / `9,574` boxes) | Link Kaggle chính thức (`nvthanh2004`) |
| :---: | :--- | :--- | :--- |
| 12 | **Stage 1 Pretrain `YOLO11n` cho B1** (`SHWD+HHW`) | Pretrain Stage 1 (`5-cls`/`2-cls` transfer) phục vụ B1 | https://www.kaggle.com/code/nvthanh2004/gdut-stage1-yolo11n-for-dcrcf |
| 13 | **Baseline B2: `YOLOv8s–DWR–ASPP–NWD`** *(Song et al., 2024)* | Val `91.46% / 60.08%`; Test **`87.51%` mAP50 / `56.59%` mAP50-95** (`16.276M` deploy fused, `22.396 ms` T4 FP32) | https://www.kaggle.com/code/nvthanh2004/gdut-baseline-yolov8-dwr-aspp-nwd-stage1-finetune |
| 14 | **Baseline B1: `YOLO11n–DCRCF`** *(Zhao et al., 2025)* | Val `88.27% / 57.46%`; Test **`84.85%` mAP50 / `54.77%` mAP50-95** (`6.796M` deploy fused, `11.048 ms` T4 FP32) | https://www.kaggle.com/code/nvthanh2004/gdut-baseline-yolo11-dcrcf-stage1-finetune |
| 15 | **Unified Protocol Comparator (`A6` vs `B1` & `B2`)** | So sánh trực tiếp cùng giao thức: **`A6 (YOLO11s)` `88.94% / 58.08%`** (`9.477M` fused, `15.207 ms`) vượt **`B1` (`+4.09 / +3.31 pp`)** và **`B2` (`+1.43 / +1.49 pp`, ít hơn `-41.8%` tham số, nhanh hơn `-32.1%`)** | https://www.kaggle.com/code/nvthanh2004/gdut-baseline-and-a6-stage1-protocol-comparison |
