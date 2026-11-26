# Danh Sách Notebook Kaggle Chính Thức Đang Dùng (Khớp Báo Cáo `BASELINE_AUDIT_B1_B2_REVIEW3.md`)

> **Phạm vi lọc:** Chỉ giữ các bản Kaggle của **Thanh (`nvthanh2004`)** và **Dũng (`tundng111`)** **thực sự được trích dẫn và sử dụng** trong [BASELINE_AUDIT_B1_B2_REVIEW3.md](BASELINE_AUDIT_B1_B2_REVIEW3.md). Đã loại bỏ toàn bộ bản `smoke`, job `queued` và bản rerun cũ không dùng (`Canonical V3 v368412285`).

---

## 1. Các Bản Kaggle Baseline Của **Thanh (`nvthanh2004`)**

| Vai trò trong báo cáo | Mô hình | Dùng ở mục nào | Link Kaggle chính thức |
|---|---|---|---|
| **Baseline 1 (SOTA)** | `YOLO11n–DCRCF` (`seed=3407`) | **Bảng 1**, **Bảng 2**, Mục 3 *(Val `57,47%` · Test `54,37%`)* | [baseline-yolo11-dcrcf-stage1-finetune (v363134475)](https://www.kaggle.com/code/nvthanh2004/baseline-yolo11-dcrcf-stage1-finetune/output?scriptVersionId=363134475) |
| **Baseline 2 (SOTA)** | `YOLOv8s–DWR–ASPP–NWD` (`seed=3407`) | **Bảng 1**, **Bảng 2**, Mục 4 *(Val `61,06%` · Test `57,98%`)* | [baseline-yolov8-dwr-aspp-nwd-stage1-finetune (v363129318)](https://www.kaggle.com/code/nvthanh2004/baseline-yolov8-dwr-aspp-nwd-stage1-finetune/output?scriptVersionId=363129318) |
| **Evaluator Test chung** | Chấm B1 & B2 trên `1.587` ảnh GDUT Test | Nguồn số Test của **Bảng 1** | [baseline-and-a6-clean-stage1-protocol-comparison (v363151440)](https://www.kaggle.com/code/nvthanh2004/baseline-and-a6-clean-stage1-protocol-comparison?scriptVersionId=363151440) |

---

## 2. Các Bản Kaggle Chính Thức Của **Dũng (`tundng111`)** Đang Dùng Trong Báo Cáo

### A. Hai Bản Trình Bày Trực Tiếp Trên **Bảng 2** (Chốt Chính)

| Vai trò trong Bảng 2 | Mô hình / Nhánh | Dùng ở mục nào & Số liệu chốt | Link Kaggle chính thức |
|---|---|---|---|
| **B1 (Dũng chọn tốt nhất)** | `YOLO11n–DCRCF` (`Pair-3407 Control`, + `Treatment pos_weight=1.5` đối chứng) | **Bảng 2**, Mục 3.1, 3.3, 3.4 *(Control: **`57,78%`** Val, P `91,65%`, R `78,77%`; Treatment: `57,28%`)* | [gdut-yolo11n-dcrcf-baseline (v372678468)](https://www.kaggle.com/code/tundng111/gdut-yolo11n-dcrcf-baseline/output?scriptVersionId=372678468) |
| **B2 (Dũng 5-seed)** | `YOLOv8s–DWR–ASPP–NWD` (`V8F`, 5 seed `3407–3411`) | **Bảng 2**, Mục 4.1, 4.2 *(**`60,78 ± 0,35%`** Val; `AP none = 55,01%`, `R none = 83,93%`)* | [gdut-yolov8-v5-matched-v8f](https://www.kaggle.com/code/tundng111/gdut-yolov8-v5-matched-v8f) |

### B. Năm Bản Dùng Làm Bằng Chứng Chẩn Đoán & Đối Chứng Trong **Mục 3 & Mục 4**

| Mã run | Nhiệm vụ cụ thể | Dùng ở mục nào & Kết quả trích dẫn | Link Kaggle chính thức |
|---|---|---|---|
| **Lượt A** *(Eval-only)* | Chấm lại checkpoint Thanh B1 & replay phân rã `98 FN` lớp `none` | Mục 3.2, 3.4 *(`57,4731%` Val; `84/98` ca tụt score, `77/87` ở P3)* | [baseline-yolo11-dcrcf-stage1-finetune-test (v372212552)](https://www.kaggle.com/code/tundng111/baseline-yolo11-dcrcf-stage1-finetune-test/output?scriptVersionId=372212552) |
| **Lượt B** *(LR Pilot)* | Tách 2 nhóm LR (`1e-4` phần cũ, `3e-4` phần mới `SBA/DCNv2`) | Mục 3.1, 3.3, 3.4 *(Recall `none` tụt `-2,97%` → FAIL)* | [gdut-b1-lr-pilot-seed3407 (v372403468)](https://www.kaggle.com/code/tundng111/gdut-b1-lr-pilot-seed3407/output?scriptVersionId=372403468) |
| **Canonical `V5`** | Thử tăng `lr0 = 3e-4` toàn mạng (`Control` vs `Tuned`) | Chú thích Bảng 2, Mục 3.1, 3.3, 3.4 *(`57,75%` Val; `R none = 77,72%`, `+8 TP / +27 FP`)* | [baseline-yolo11-dcrcf-stage1-audited-rerun (v369749859)](https://www.kaggle.com/code/tundng111/baseline-yolo11-dcrcf-stage1-audited-rerun?scriptVersionId=369749859) |
| **`C3` (5-seed)** | Kiểm chứng 5 seed `3407–3411` cho `YOLO11n–DCRCF` | Chú thích Bảng 2, Mục 3.1, 3.4, 3.5 *(`57,283 ± 0,173%` Val)* | [gdut-yolo11n-dcrcf-matched-ablation-c3 (v370113435)](https://www.kaggle.com/code/tundng111/gdut-yolo11n-dcrcf-matched-ablation-c3/output?scriptVersionId=370113435) |
| **`C0` & `V8S`** *(Stock đối chứng)* | Stock `YOLO11n` & Stock `YOLOv8s` (5 seed `3407–3411`) dùng để chứng minh `C3` & `V8F` đều thấp hơn bản gốc | Mục 3.4, 3.5 *(`C0 = 58,812%` > `C3` `+1,53` đpt)* & Mục 4.2 *(`V8S = 61,414%` > `V8F` `+0,63` đpt)* | • [C0: gdut-yolo11n-dcrcf-matched-ablation-c0](https://www.kaggle.com/code/tundng111/gdut-yolo11n-dcrcf-matched-ablation-c0)<br>• [V8S: gdut-yolov8-v5-matched-v8s (v370372416)](https://www.kaggle.com/code/tundng111/gdut-yolov8-v5-matched-v8s?scriptVersionId=370372416) |

---

## 3. Các Bản Đã Loại Khỏi Danh Sách Dùng Chính Thức
* **Không dùng trong báo cáo Review 3:** `Canonical V3` (`v368412285` — bản chạy cũ trước `V5`), `gdut-yolo11n-dcrcf-v3-learning-rate-audit` (`v369558216` — `Queued`), và toàn bộ các bản `smoke` (`gdut-dcrcf-v4-smoke`, `notebook9fa5bce971`).
