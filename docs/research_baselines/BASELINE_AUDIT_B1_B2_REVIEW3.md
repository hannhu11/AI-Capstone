# Báo Cáo Kiểm Chứng & Chẩn Đoán 2 Baseline SOTA (`B1` & `B2`) — Review 3

> **Phạm vi:** Chỉ xét **2 Baseline SOTA tái dựng** (`B1: YOLO11n–DCRCF` và `B2: YOLOv8s–DWR–ASPP–NWD`) trên GDUT-HWD 5 lớp (`1.265` train gốc / `2.530` train hardaug / `316` val / `1.587` test; fingerprint `026a2afb...`). Không đưa A6 vào tài liệu này.

---

## 1. Bảng 1 (4 Hàng): So Sánh 2 Baseline SOTA Với Paper Gốc

*Lưu ý giao thức: Paper gốc đánh giá trên **SHWD 2 lớp**; bản tái dựng của Thanh đánh giá trên **GDUT-HWD 5 lớp** (Stage1 fine-tune, `seed=3407`, `imgsz=640`, `batch=16`, `AdamW lr0=1e-4`). So thứ hạng trong cùng dataset và split.*

| Mô hình | Nguồn / Dataset / Split | Params (M) | P (%) | R (%) | mAP50 (%) | mAP50–95 (%) | Ghi chú giao thức |
|---|---|---:|---:|---:|---:|---:|---|
| **B1 · YOLO-DCRCF** | Paper Zhao 2025 · SHWD (2 lớp) | — *(13,4G)* | 90,90 | 88,50 | 92,70 | **60,60** | Scratch 300 epoch, SGD |
| **B1 · YOLO11n–DCRCF (Thanh)** | GDUT Val (`316`) / **Test (`1.587`)** | 6,802 | 89,85 / **89,57** | 81,53 / **76,96** | 88,86 / **84,36** | 57,47 / **54,37** | Test `none`: AP50–95 **47,97%**, R **69,90%** |
| **B2 · YOLOv8s DWR–ASPP–NWD** | Paper Song 2024 · SHWD (2 lớp) | — | 91,80 | 86,60 | 92,00 | **62,20** | 100 epoch, SGD, batch 16 |
| **B2 · YOLOv8s–DWR–ASPP–NWD (Thanh)** | GDUT Val (`316`) / **Test (`1.587`)** | 16,285 | 90,35 / **90,43** | 86,90 / **81,81** | 92,13 / **88,19** | 61,06 / **57,98** | Test `none`: AP50–95 **52,62%**, R **76,80%** |

---

## 2. Bảng 2 (4 Hàng): So Sánh Thực Nghiệm & Cải Thiện Baseline (Trên GDUT Validation `316` Ảnh)

*Chọn **1 bản tốt nhất của Dũng cho B1** theo [YOLO11_USER_RUN_SELECTION.md](YOLO11_USER_RUN_SELECTION.md) (**Pair control seed 3407** — giữ nguyên khung DCRCF, đạt `mAP50–95` cao nhất trong các run DCRCF của Dũng), kèm **5-seed `V8F`** cho B2.*

| Vai trò | Mô hình / Nguồn | Seed | P (%) | R (%) | mAP50 (%) | mAP50–95 (%) | Chỉ số lớp `none` (Val) |
|---|---|---:|---:|---:|---:|---:|---|
| **Baseline 1 (SOTA)** | `YOLO11n–DCRCF` (**Thanh**) | 3407 | 89,85 | **81,53** | 88,86 | 57,47 | AP50–95: **51,40%** · R@0.25: **75,99%** |
| **B1 (Dũng chọn tốt nhất)** | `YOLO11n–DCRCF` (**Pair control**, giữ nguyên khung DCRCF) | 3407 | **91,65** | 78,77 | 88,86 | **57,78** | AP50–95: 51,08% · R@0.287: 75,74% *(P=83,61%)* |
| **Baseline 2 (SOTA)** | `YOLOv8s–DWR–ASPP–NWD` (**Thanh**) | 3407 | **90,35** | **86,90** | **92,13** | **61,06** | Mốc 1 seed của Thanh *(Test: 57,98%)* |
| **B2 (Dũng 5-seed)** | `YOLOv8s–DWR–ASPP–NWD` (**`V8F`**) | 3407–3411 | 90,25 ± 0,94 | 86,47 ± 0,58 | 91,82 ± 0,17 | 60,78 ± 0,35 | AP50–95: **55,01%** · R@0.25: **83,93%** *(P=84,18%)* |

> **Cảnh báo trung thực về cải thiện trên B1 (Dũng):** Dù `Pair control` đạt `mAP50–95` cao nhất trên khung DCRCF (**`57,78%`**, `+0,31` đpt so với Thanh) và kéo **Precision tổng tăng `+1,80` đpt (`89,85% → 91,65%`)**, nhưng **Recall tổng lại giảm `-2,76` đpt (`81,53% → 78,77%`)** và `AP50–95 none` / `Recall none` cũng giảm nhẹ (`51,40% → 51,08%`; `75,99% → 75,74%`). Trong bài toán an toàn (bắt lỗi người không đội mũ), giảm Recall đồng nghĩa tăng nguy cơ bỏ sót vi phạm.
>
> **Chốt phạm vi bổ sung cho B1:** Ngoài `Pair control (57,78%)`, Dũng còn chạy **`V5 tuned` (`lr0=3e-4`, `57,75%` Val)** kéo Recall `none @0.25` lên `77,72%` (`+8 TP`) nhưng tăng `27 FP` (P tụt xuống `78,11%`) và Recall tổng vẫn giảm `-1,18` đpt (`81,53% → 80,35%`), và **`C3` 5-seed (`57,28 ± 0,17%` Val)** để khóa bộ bằng chứng thống kê. **Pair control và V5 chưa chạy GDUT Test**, nên trên bảng Test chính thức vẫn dùng **Thanh B1 (`54,37%`)** và **Thanh B2 (`57,98%`)**.

---

## 3. Phân Tích Chi Tiết `B1 (YOLO11n–DCRCF)`: Bộ Bằng Chứng `C3 + V5 + Pair-3407`

### 3.1. Đối chiếu Code & Thông Số Thực Tế (Thanh vs. Dũng)
Toàn bộ 18 module lõi (`baseline_model.py` `e2dd3140`, `baseline_modules.py` `cf8d4639`, `warm_model.py` `a31f59c7`, `gdut_data.py` `620c14fb`, ...) **khớp SHA-256 100% từng byte** giữa Thanh và Dũng. Không có hiện tượng đổi tràn lan thông số:

* **Kiến trúc cố định:** `ModulatedDCNv2` ở backbone (`model.6, 8`, predictor chung 27 kênh = 18 offset + 9 mask), `4 SBA` ở neck (`model.11, 13, 15, 17`, gating `a*σ(conv(a)) + (1−gate_a)*b*σ(conv(b))`), `Detect` 3 mức ở `model.19` (`P3/P4/P5`). Tổng **`6.802.005` tham số**; chỉ kế thừa **`1.783.391` (`26,22%`)** từ Stage1 (`ea174f78...`).
* **Khác biệt thực tế ở 3 nhánh của Dũng (mỗi nhánh đổi đúng 1 biến kiểm soát trên bộ ~110 `training_args` giống hệt Thanh):**
  1. **`C3` (5 seed `3407–3411`):** Giữ **0 thông số train đổi** (chỉ khác 4 đường dẫn `data/name/project/save_dir`); bổ sung nhóm `v4_*.py` và `warm_launch.py` (`976e4396`) để chạy 5 seed; runtime dùng **CUDA 13.0** (`torch 2.11+cu130`) vs Thanh **CUDA 12.8** (`cu128`).
  2. **`V5` (`Canonical` — `Lượt B` & `tuned`):** Thay đúng 1 tham số optimizer `lr0: 0.0001 → 0.0003` (trên nhánh `Lượt B` tách 2 param-group bằng `b1_lr_trainer.py` để giữ `1e-4` cho phần cũ và `3e-4` cho phần mới; trên `V5 tuned` chạy `3e-4`).
  3. **`Pair-3407` (`Control` & `Treatment`):** Giữ **0 thông số `args.yaml` đổi**; thêm `b1_lr_trainer.py` đặt **`GradScaler(init_scale=16.0)`** chống tràn/bỏ qua bước cập nhật FP16; thêm `loss_trainer.py` + `positive_loss.py` đặt **`pos_weight = 1.5`** riêng cho lớp `none` (index 4) ở nhánh `Treatment` (`Control` giữ `1.0`), kèm `decision.py`, `evaluate.py`, `workflow.py` để khóa kiểm toán.

### 3.2. Đã Mổ Xẻ Được Gì Về Điểm Yếu Lớp `none`? (Replay `316` Ảnh Val, `404` Box `none`)
* **Phân rã `98` ca bỏ sót (`FN`) của Thanh tại `conf = 0.25`:**
  * **`84/98` ca (`85,7%`):** Đã bắt được ứng viên đúng vị trí (`IoU ≥ 0,50`) nhưng **score lớp `none` tụt dưới `0,25`** (trong đó `68` ca mọi lớp đều dưới `0,25`; `16` ca bị 4 lớp màu mũ thắng điểm).
  * **`11/98` ca (`11,2%`):** Hoàn toàn không có box nào đạt `IoU ≥ 0,50` (100% là box nhỏ `< 32 px`).
  * **`3/98` ca (`3,1%`):** Có ứng viên qua ngưỡng nhưng bị NMS loại.
* **Vị trí điểm nghẽn theo tầng & kích thước:**
  * **`358/393`** box `none` có ứng viên `IoU ≥ 0,50` (và **`77/87`** ca `FN` có ứng viên) xuất phát từ **tầng P3 (`80×80`, `model.14`)**.
  * **`68/230`** box nhỏ (`< 32 px`) bị bỏ sót, so với chỉ `30/174` ở box lớn.

### 3.3. Tại Sao Dũng Tinh Chỉnh Từng Tham Số Đó? (Lý Do & Hệ Quả Precision/Recall)
Mỗi tham số tinh chỉnh trên khung `DCRCF` đều xuất phát trực tiếp từ kết quả chẩn đoán mục 3.2, không phải thử mù:

1. **Tại sao đặt `GradScaler(init_scale=16.0)` (`Pair-3407`):**
   * *Lý do từ code:* Vì `73,78%` tham số neck (`DCNv2 + 4 SBA`) mới khởi tạo từ 0 đi chung với `Stage1`, dùng scale mặc định lớn của PyTorch AMP FP16 dễ gây tràn số (`inf/nan`) ở các bước đầu, làm optimizer bỏ qua bước cập nhật trọng số. Đặt `init_scale=16.0` khóa chắc chắn mọi bước đầu đều cập nhật gradient hợp lệ.
   * *Hệ quả thực tế (`Pair control`):* Giúp mô hình hội tụ chặt hơn ở Precision (**`89,85% → 91,65%`**, `mAP50–95` lên **`57,78%`**), **nhưng Recall tổng lại giảm mạnh `-2,76` đpt (`81,53% → 78,77%`)** — tức cải thiện được độ chắc chắn khi báo nhưng lại bỏ sót nhiều đối tượng hơn.
2. **Tại sao tăng `lr0: 1e-4 → 3e-4` (`Lượt B` tách 2 nhóm & `V5 tuned`):**
   * *Lý do từ thuật toán:* Mức `lr0=1e-4` vốn chỉ hợp để fine-tune nhẹ, không đủ lực cho `4,08M` tham số `SBA` mới học kịp trước khi mạng dừng sớm ở epoch `33`. Tăng lên `3e-4` nhằm đẩy tốc độ thích nghi của neck mới và kéo confidence score lớp `none` vượt ngưỡng `0,25`.
   * *Hệ quả thực tế:* `Lượt B` làm Recall `none` giảm `-2,97%` (`306→294 TP`). `V5 tuned` cứu thêm được `+8 TP` lớp `none` (`75,99% → 77,72%`), nhưng phải trả giá bằng **`+27 FP`** (Precision `none` rớt `83,42% → 78,11%`) và **Recall tổng 5 lớp vẫn giảm (`81,53% → 80,35%`)**.
3. **Tại sao đặt `pos_weight = 1.5` cho lớp `none` (`Pair-3407 Treatment`):**
   * *Lý do từ chẩn đoán `84/98` ca `FN`:* `85,7%` lỗi bỏ sót lớp `none` đã bắt đúng vị trí nhưng score tụt dưới `0,25` (và `16` ca bị 4 lớp màu mũ lấn át). Tăng trọng số `pos_weight=1.5` trong `BCEWithLogitsLoss` riêng cho class `none` (index `4`) nhằm phạt nặng khi gán score thấp cho đầu không đội mũ.
   * *Hệ quả thực tế:* Tại ngưỡng khóa, Precision `none` nhích nhẹ (`83,61% → 84,34%`, giảm `3 FP`), Recall `none` nhích `+1 TP` (`75,74% → 75,99%`), **nhưng `mAP50–95` toàn tập giảm `-0,50` đpt (`57,78% → 57,28%`) và `AP50–95 none` giảm `-0,67` đpt (`51,08% → 50,41%`)** → rớt tiêu chí nghiệm thu (`lock_validation` chỉ cho phép giảm tối đa `0,10` đpt).

### 3.4. Bảng Tổng Hợp Các Lượt Tinh Chỉnh Trên Khung `DCRCF`
| Lượt thử nghiệm (Giữ nguyên khung DCRCF) | Mục tiêu tinh chỉnh | Kết quả đo được | Kết luận |
|---|---|---|---|
| **Lượt A** (`Eval-only`) | Kiểm tra lại checkpoint Thanh | Val `57,4731%` (khớp `<0,000003%`); hạ conf `0,25→0,20` thêm `+10 TP` nhưng dính **`+15 FP`** | Không thể cứu bằng chỉnh ngưỡng |
| **Lượt B** (`LR = 3e-4` cho lớp mới) | Đẩy tốc độ học phần neck `SBA/DCNv2` mới | `mAP50–95` nhích `+0,13%`, nhưng **Recall `none` tụt `-2,97%`** (`306→294 TP`) | **FAIL** — tăng LR riêng hại lớp `none` |
| **`V5 tuned`** (`lr0 = 3e-4` toàn mạng) | Tăng tín hiệu score lớp `none` | Val `57,75%` (`+0,28` vs Thanh); `none @0.25` lên **`314 TP` (`77,72%`)** nhưng **`FP` vọt `61→88`**, **R tổng giảm `-1,18` đpt** | Đánh đổi: tăng 8 TP mất 27 FP, không thực sự tốt hơn |
| **`Pair-3407`** (`Control` vs `Treatment pos_weight=1.5`) | Ép `BCEWithLogitsLoss` học kỹ lớp `none` + AMP `init_scale=16.0` | `Control` đạt **`57,78%`** Val (**P tăng `91,65%`**, nhưng **R tổng tụt `78,77%`**); `Treatment` tụt `57,28%` (`AP none` giảm `51,08%→50,41%`) | `Control` chỉ tăng P giảm R; `pos_weight=1.5` **không qua cổng chọn** |
| **`C3`** (5 seed `3407–3411`) | Đo độ ổn định thống kê của toàn gói DCRCF | `57,283 ± 0,173%` mAP50–95 — **thua Stock `YOLO11n (C0)` (`58,812 ± 0,082%`) tới `-1,529` đpt** | **Kết luận dứt khoát:** Khung DCRCF kém hơn cả Stock trên GDUT |

### 3.5. Tại Sao Giữ Khung `DCRCF` Tinh Chỉnh Vẫn Không Cứu Được? (Lý Do Kỹ Thuật Để Bảo Vệ Trước Hội Đồng)
1. **Đứt gãy biểu diễn khi chuyển trọng số (`73,78%` tham số mới):** Neck cũ bị thay hoàn toàn bằng `4 SBA` (`4,08M` tham số = `60,03%` toàn mạng) khởi tạo từ 0, trong khi `Detect 19` vẫn kế thừa 100% từ Stage1 nhưng phải nhận feature map mới hoàn toàn từ `SBA → C3k2` (`RMS` đầu ra `SBA` thấp chỉ `0,079–0,107`).
2. **Xung đột giao thức Train vs. Paper:** Paper Zhao 2025 train **scratch 300 epoch SGD** trên **2 lớp**; GDUT fine-tune **100 epoch AdamW (`patience=50`)** trên **5 lớp** nên `best epoch` dừng sớm ở **epoch `33`** (sau epoch 33 `val box_loss` tăng `1,225 → 1,273` dấu hiệu overfit nhẹ).
3. **Bất định gradient `DCNv2`:** Toán tử `deform_conv2d` phát sinh `166` cảnh báo `compute_grad_input` non-deterministic trên CUDA.
4. **Giới hạn trần kiến trúc:** Đưa `C3 (6,80M, 14,8G FLOPs)` đối đầu trực diện `Stock YOLO11n (2,59M, 6,3G FLOPs)` trên cùng dữ liệu cho thấy chính nhánh `DCRCF` làm giảm `-1,53` đpt mAP50–95 và `-1,77` đpt AP50–95 lớp `none` (`52,93% → 51,16%`). Vì vậy tinh chỉnh siêu tham số/loss trên khung này chỉ dịch chuyển điểm trên đường PR (được Recall mất Precision), **không giải quyết được gốc rễ mất thông tin ở P3**.

---

## 4. Phân Tích Chi Tiết `B2 (YOLOv8s–DWR–ASPP–NWD)`: Mã Nguồn, 5-Seed `V8F` & Lý Do Giữ Nguyên

### 4.1. Thực Tế Code & Transfer (`baseline_model.py`, `V8F-3407/run_manifest.json`)
* **Kiến trúc & Loss:** Bọc `C2f` bằng **`DWR` (Dilation-wise Residual, dilation 1/3/5)** tại `model.2, 4, 6`; thay `SPPF` bằng **`ASPP` (dilation 6/12/18 + Global Avg Pool)** tại `model.9`; giữ nguyên toàn bộ Neck + Head ba mức của `YOLOv8s`; thay BboxLoss bằng **`CIoUNWDBboxLoss` = `0,5 * CIoU + 0,5 * NWD` (`constant = 12,8` trên tọa độ pixel, giữ nguyên DFL)**.
* **Khả năng kế thừa Stage1 vượt trội `B1`:** Tổng **`16.284.863` tham số**, giữ lại **`10.612.223` tham số (`65,16%`)** từ Stage1 (`af430179...`) — đặc biệt copy trực tiếp trọng số `cv1` vào nhánh `0` của `ASPP` và giữ nguyên toàn bộ Neck/Head → không bị sốc feature như `B1`.
* **Đối chứng code Thanh B2 vs. Dũng `V8F`:** 100% file lõi khớp SHA-256 (`baseline_model.py`, `baseline_modules.py`, `warm_model.py`, ...); `args.yaml` **giống 100%** (chỉ khác 4 đường dẫn `data/name/project/save_dir`), mở rộng chạy đủ **5 seed (`3407–3411`)** trên cùng cấu hình.

### 4.2. Điểm Mạnh, Điểm Yếu & Kết Luận Cho `B2`
* **Điểm mạnh (trên `V8F` 5-seed & Test Thanh):**
  * Test Thanh đạt **`57,98%` mAP50–95** (`+3,61` đpt so với `B1`), lớp `none` đạt `AP50–95 = 52,62%`, `Recall = 76,80%`.
  * Trên Val 5-seed (`V8F`), `Recall none @0.25` đạt **`83,93%`** (nhờ hàm `NWD` nhạy với khoảng cách tâm của vật thể nhỏ).
* **Điểm yếu đo được khi đối chiếu 5 seed với `Stock YOLOv8s (V8S)`:**
  * `V8F` (`60,781 ± 0,353%`) **thua `V8S` (`61,414 ± 0,074%`) tới `-0,633` đpt `mAP50–95`** (thua 4/5 seed; `mAP50` thua 5/5 seed `-0,488` đpt), riêng `AP50–95 none` tụt `56,02% → 55,01%` và `Precision none @0.25` tụt `87,06% → 84,18%`.
  * Nguyên nhân kỹ thuật: Trọng số cố định `0,5 * NWD (C=12,8)` trong `CIoUNWDBboxLoss` làm giảm độ chặt hộp bao ở ngưỡng IoU cao (`0,75–0,95`), trong khi `DWR + ASPP` làm phình mô hình lên **`16,29M` params (`+46,2%`)**, **`41,23 GFLOPs` (`+45,0%`)**, độ trễ forward nặng nhất (**`16,44 ms`**).
* **Quyết định đối với `B2`:** **Giữ nguyên hoàn toàn làm Baseline SOTA đối chứng thứ hai** (đã hoàn tất kiểm chứng 5 seed `V8F`), **không sửa đổi code/loss của `B2`** để bảo toàn tính nguyên bản đối chiếu với paper Song 2024.
