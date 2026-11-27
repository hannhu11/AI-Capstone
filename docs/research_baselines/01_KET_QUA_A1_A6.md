# Kết quả A1–A6 trên GDUT và cách hiểu kết quả

Ngày tổng hợp: 25/11/2026. Tài liệu dùng các run GDUT CLEAN đã chốt, trên hai backbone YOLO11s và YOLOv8s. A0 được đưa vào làm mốc đối chiếu. Tài liệu này không thay đổi code, checkpoint hay hình kiến trúc.

## 1. Kết luận trước khi xem các bảng

**A6 đã chạy với đầy đủ các thành phần của kiến trúc chốt, nhưng A6 không phải cấu hình có điểm cao nhất trong lần thực nghiệm này.** A1 có mAP50–95 cao nhất trong nhóm A1–A6 ở cả hai backbone, trên cả val và test.

A6 đạt:

- **YOLO11s:** val mAP50 **92,50%**, mAP50–95 **61,23%**; test mAP50 khoảng **88,9%**, mAP50–95 khoảng **58,1%**.
- **YOLOv8s:** val mAP50 **92,56%**, mAP50–95 **60,68%**; test mAP50 khoảng **88,4%**, mAP50–95 khoảng **57,4%**.

Kết quả này cho thấy mô hình học được bài toán phát hiện năm lớp trên GDUT. Tuy nhiên, nếu câu hỏi là “cứ thêm đủ CoordConv, RepConv, BiFormer và Focal-EIoU thì có tốt hơn cấu hình đơn giản không?”, thì **các số liệu hiện tại chưa chứng minh điều đó**.

Điểm có giá trị của thí nghiệm là đã kiểm tra được từng hướng can thiệp và tổ hợp của chúng. Kết luận có thể bảo vệ là: tổ hợp A6 hoạt động, nhưng lợi ích của các thành phần không cộng dồn thành cấu hình tốt nhất trên GDUT với thiết lập đang dùng.

## 2. Số liệu lấy từ đâu?

### 2.1. Validation và test là hai phép đánh giá khác nhau

| Nội dung | Validation | Test |
| --- | --- | --- |
| Dữ liệu | 316 ảnh, 1.738 box | 1.587 ảnh, 9.574 box |
| Vai trò | Chọn checkpoint và theo dõi quá trình học | Đánh giá checkpoint đã chọn trên tập test chính thức |
| Nguồn trong báo cáo | Trường validation của run_manifest.json; A0 lấy evaluation.json | Log final test do nhóm gửi |
| Độ chính xác hiển thị | Tính từ số thực trong JSON, làm tròn đến 2 chữ số phần trăm | Log đã làm tròn đến 3 chữ số thập phân; chuyển sang phần trăm với 1 chữ số |

Các giá trị val ở đây là kết quả **nạp lại best.pt và đánh giá trong val_check**. Đây không phải số tự chọn lớn nhất ở từng cột của results.csv, cũng không phải kết quả epoch cuối.

Ví dụ, không lấy Precision tốt nhất ở epoch 12, Recall tốt nhất ở epoch 25 rồi ghép với mAP tốt nhất ở epoch 16 thành một dòng. Một dòng phải mô tả cùng một checkpoint.

Trong log test, tiền tố “val:” là tên bộ đánh giá mà thư viện dùng. Đường dẫn thực tế là **base/labels/test**, code gọi **split=test**, và số ảnh là 1.587. Vì vậy các dòng đó thực sự thuộc test.

### 2.2. Đã kiểm tra đúng phiên bản run

Tất cả A0–A6 dùng cùng dấu vân tay dataset:

    ce2a883e0b5050edb96d494218e85a9d16ea04b9a19dc5723c572cb502edfe9c

Trong từng họ mô hình, A1–A6 cũng bắt đầu từ cùng checkpoint Stage1. Điều này loại trừ việc vô tình so sánh các lần chia dữ liệu khác nhau hoặc các checkpoint khởi đầu khác nhau trong nhóm run đang xét.

**Không dùng lại bảng comparison cũ của các lần chạy trước.** Chẳng hạn, các giá trị A6 quanh 0,925 ở val và 0,8908 ở test trong một bảng cũ không tự động thuộc phiên bản hiện tại.

Ở đây chưa có CSV test đầy đủ độ chính xác của đúng lần final test này. Do đó, các chênh lệch test rất nhỏ cần được đọc trong giới hạn làm tròn của log.

## 3. A1–A6 thực sự khác nhau ở đâu?

| Cấu hình | Kiến trúc và loss | Dữ liệu huấn luyện |
| --- | --- | --- |
| A0 | Mô hình stock sau Stage1; chỉ đánh giá, không train tiếp | Stage1 đã học trên ảnh train gốc |
| A1 | Giữ kiến trúc stock và CIoU | Train gốc + bản hard augmentation; online mixup 0,1 và hsv_v 0,45 |
| A2 | Thêm CoordConv ở stem; giữ CIoU | Train gốc + hard augmentation; mixup 0, hsv_v 0,4 |
| A3 | Thay các Conv 3×3 được chọn bằng RepConv, gồm stem; giữ CIoU | Giống A2 |
| A4 | Giữ kiến trúc stock; thay box loss bằng Focal-EIoU | Giống A2 |
| A5 | Thêm BiFormerBlockLite theo vị trí riêng của từng backbone; giữ CIoU | Giống A2 |
| A6 | CoordConv + RepConv + BiFormerBlockLite + Focal-EIoU | Giống A2 |

Tất cả vẫn dùng BCE cho phân loại và DFL cho phân bố khoảng cách box.

**A1–A6 là các nhánh độc lập từ Stage1.** A3 không lấy trọng số A2, A6 không lấy trọng số A5. Các số A chỉ đánh dấu cấu hình thử nghiệm, không phải sáu chặng train nối tiếp.

Có hai điểm cần nhớ khi suy luận:

1. A1 và A6 khác cả module lẫn một phần online augmentation. Vì thế không thể quy toàn bộ khoảng cách A1–A6 cho riêng BiFormer, RepConv hay tương tác giữa chúng.
2. A0 chưa trải qua lượt fine-tune Stage2, còn A1 có thêm huấn luyện và dữ liệu tăng cường. Khoảng tăng A1 so với A0 không phải phép đo riêng biệt tác dụng của augmentation.

## 4. Kết quả validation

Các số trong hai bảng dưới là phần trăm. mAP50 không phải “Accuracy”; đây là độ chính xác trung bình của bài toán detection tại ngưỡng IoU 0,50. mAP50–95 xét nhiều ngưỡng IoU nên yêu cầu box khớp chặt hơn.

### 4.1. YOLO11s

| Cấu hình | Precision (%) | Recall (%) | mAP50 (%) | mAP50–95 (%) |
| --- | --- | --- | --- | --- |
| A0 | 92.21 | 84.80 | 91.83 | 60.83 |
| A1 | 91.33 | 87.37 | 93.40 | 61.69 |
| A2 | 91.01 | 86.25 | 92.72 | 61.31 |
| A3 | 90.79 | 87.86 | 92.88 | 61.41 |
| A4 | 90.64 | 87.49 | 92.77 | 61.36 |
| A5 | 90.05 | 87.14 | 92.40 | 61.08 |
| A6 | 90.45 | 87.58 | 92.50 | 61.23 |

A1 đứng đầu về mAP50 và mAP50–95. A3 có Recall cao nhất, nhưng Precision thấp hơn A0. Điều này minh họa rằng “tốt hơn” phải gắn với chỉ số cụ thể: bắt được nhiều đối tượng hơn chưa chắc tạo ra AP cao nhất.

A6 tăng so với A0 trên hai chỉ số mAP của val, nhưng vẫn thấp hơn A1. Không nên chỉ dẫn một con số 92,50% rồi kết luận A6 vượt toàn bộ ablation.

### 4.2. YOLOv8s

| Cấu hình | Precision (%) | Recall (%) | mAP50 (%) | mAP50–95 (%) |
| --- | --- | --- | --- | --- |
| A0 | 91.54 | 85.75 | 91.88 | 60.55 |
| A1 | 91.55 | 86.21 | 93.14 | 61.19 |
| A2 | 91.51 | 86.79 | 92.47 | 60.32 |
| A3 | 91.47 | 86.67 | 92.41 | 60.30 |
| A4 | 92.60 | 84.91 | 92.35 | 60.21 |
| A5 | 92.02 | 85.56 | 92.59 | 60.26 |
| A6 | 93.21 | 85.11 | 92.56 | 60.68 |

A6 có Precision cao nhất trong bảng, nhưng Recall thấp hơn A0 và A1. Nghĩa là ở điểm hoạt động được bộ đánh giá báo cáo, các dự đoán giữ lại có tỷ lệ đúng cao hơn, đồng thời vẫn bỏ sót nhiều hơn một số cấu hình khác.

A1 tiếp tục đứng đầu về mAP50 và mAP50–95. A2–A5 tăng mAP50 so với A0 nhưng mAP50–95 lại thấp hơn A0. Đây là lý do cần báo cáo cả hai chỉ số: box có thể đạt ngưỡng IoU 0,50 nhiều hơn mà chưa khớp tốt ở các ngưỡng nghiêm ngặt.

### 4.3. A6 tăng hay giảm bao nhiêu trên val?

Đpt là **điểm phần trăm**. Ví dụ từ 91% lên 92% là tăng 1 đpt, không phải tăng tương đối 1%.

| So sánh | Backbone | Δ mAP50 (đpt) | Δ mAP50–95 (đpt) |
| --- | --- | --- | --- |
| A6 − A0 | YOLO11s | +0.67 | +0.39 |
| A6 − A0 | YOLOv8s | +0.68 | +0.13 |
| A6 − A1 | YOLO11s | -0.90 | -0.46 |
| A6 − A1 | YOLOv8s | -0.58 | -0.51 |

Mức tăng A6 so với A0 trên val là có thật trong các run này, nhưng tương đối nhỏ. Chưa có nhiều seed để biết mức tăng đó có ổn định hay không.

## 5. Kết quả test

Các bảng test giữ đúng độ chính xác mà log cho phép. A0 ở đây là checkpoint Stage1 tương ứng; trong thư mục kết quả test nó được đặt tên S1.

### 5.1. YOLO11s

| Cấu hình | Precision (%) | Recall (%) | mAP50 (%) | mAP50–95 (%) |
| --- | --- | --- | --- | --- |
| A0 | 91.0 | 82.8 | 88.8 | 58.5 |
| A1 | 91.2 | 84.4 | 89.8 | 59.1 |
| A2 | 91.1 | 83.5 | 89.4 | 58.6 |
| A3 | 90.7 | 83.9 | 89.3 | 58.3 |
| A4 | 90.2 | 84.3 | 89.2 | 58.5 |
| A5 | 90.2 | 84.2 | 89.2 | 58.6 |
| A6 | 90.2 | 84.3 | 88.9 | 58.1 |

A1 tiếp tục cao nhất. A6 chỉ nhỉnh hơn A0 khoảng 0,1 đpt mAP50, nhưng thấp hơn khoảng 0,4 đpt mAP50–95. Vì vậy không có cơ sở nói A6 YOLO11s cải thiện đồng thời phát hiện và định vị so với A0 trên test.

### 5.2. YOLOv8s

| Cấu hình | Precision (%) | Recall (%) | mAP50 (%) | mAP50–95 (%) |
| --- | --- | --- | --- | --- |
| A0 | 90.1 | 81.8 | 87.5 | 57.2 |
| A1 | 90.5 | 83.4 | 88.7 | 57.8 |
| A2 | 90.6 | 83.1 | 88.6 | 57.2 |
| A3 | 89.7 | 83.6 | 88.7 | 57.2 |
| A4 | 90.9 | 82.6 | 88.3 | 57.2 |
| A5 | 90.9 | 82.4 | 88.3 | 57.3 |
| A6 | 90.3 | 82.6 | 88.4 | 57.4 |

A6 tăng nhẹ so với A0 trên cả hai chỉ số mAP. Tuy nhiên, A1 vẫn cao hơn. A1 và A3 cùng hiển thị 88,7% mAP50 trong log đã làm tròn; không thể xác định thứ hạng tuyệt đối của hai giá trị này khi chưa có số gốc đầy đủ. Riêng mAP50–95, A1 cao hơn rõ trong độ chính xác đang có.

### 5.3. Chênh lệch test

| So sánh | Backbone | Δ mAP50 (đpt) | Δ mAP50–95 (đpt) |
| --- | --- | --- | --- |
| A6 − A0 | YOLO11s | +0.1 | -0.4 |
| A6 − A0 | YOLOv8s | +0.9 | +0.2 |
| A6 − A1 | YOLO11s | -0.9 | -1.0 |
| A6 − A1 | YOLOv8s | -0.3 | -0.4 |

A6 YOLO11s có mAP test cao hơn A6 YOLOv8s, nhưng không vượt trên mọi tiêu chí: Precision test của YOLOv8s là 90,3%, so với 90,2% của YOLO11s. Nên nói “YOLO11s có mAP tốt hơn trong phép đánh giá này”, thay vì “YOLO11s tốt hơn toàn diện”.

## 6. Vì sao lúc train thấy hơn 92%, sang test còn khoảng 89%?

Đó là hai tập ảnh khác nhau. Val được tách từ phần trainval, còn test là tập chính thức được giữ riêng. Nhóm đã dùng val để chọn best.pt; test không tham gia tính gradient hoặc chọn best.pt trong các run này.

Với A6:

| Backbone | mAP50 val → test | Giảm xấp xỉ | mAP50–95 val → test | Giảm xấp xỉ |
| --- | --- | --- | --- | --- |
| YOLO11s | 92,50% → 88,9% | 3,60 đpt | 61,23% → 58,1% | 3,13 đpt |
| YOLOv8s | 92,56% → 88,4% | 4,16 đpt | 60,68% → 57,4% | 3,28 đpt |

Khoảng giảm này không tự chứng minh code đánh giá bị sai. Dấu vân tay dữ liệu, đường dẫn test và số lượng ảnh đều khớp. Nhưng cũng không nên kết luận chắc chắn rằng “test khó hơn vì nhiều vật nhỏ hoặc che khuất”: hiện chưa có thống kê theo kích thước vật, mức che khuất hay bối cảnh để xác nhận riêng những nguyên nhân đó.

Diễn giải có căn cứ hơn là: **điểm val chưa phản ánh trọn khả năng tổng quát sang tập test chính thức**. Khác biệt phân bố ảnh và việc checkpoint được chọn theo val có thể góp phần tạo ra khoảng cách. Cần phân tích từng nhóm ảnh nếu muốn chỉ ra nguyên nhân chi tiết hơn.

## 7. Tại sao A6 đầy đủ module mà không cao nhất?

### 7.1. “Full” mô tả cấu hình, không phải bảo đảm về điểm số

Các thành phần tác động lên những việc khác nhau: CoordConv bổ sung vị trí, RepConv thay cấu trúc convolution, BiFormer thay cách trao đổi thông tin, Focal-EIoU thay tín hiệu học box. Khi ghép lại, chúng cùng làm thay đổi quá trình tối ưu.

Ví dụ, attention có thể giúp đưa ngữ cảnh vào đặc trưng nhưng cũng làm mất một phép biến đổi đã học nếu nó thay hẳn một block pretrained. Cấu trúc có nhiều khả năng biểu diễn hơn vẫn cần dữ liệu và cách tối ưu phù hợp để khai thác khả năng đó.

### 7.2. Có bằng chứng cụ thể ở bước khởi tạo BiFormer của YOLO11s

Bảng sau được lấy từ initialization_check.json, tức đánh giá **sau khi sửa cấu trúc nhưng trước fine-tune**:

| Cấu hình, trước fine-tune | YOLO11s: mAP50 / mAP50–95 (%) | YOLOv8s: mAP50 / mAP50–95 (%) |
| --- | --- | --- |
| A2 | 91.83 / 60.83 | 91.88 / 60.55 |
| A3 | 91.83 / 60.83 | 91.88 / 60.55 |
| A5 | 86.34 / 56.15 | 91.88 / 60.55 |
| A6 | 86.34 / 56.15 | 91.88 / 60.55 |

A0 YOLO11s ban đầu là 91,83% / 60,83%. Sau khi thay C2PSA bằng BiFormerBlockLite trong A5 hoặc A6, điểm trước fine-tune còn khoảng 86,34% / 56,15%.

Lý do nhìn thấy ngay từ code:

- CoordConv giữ trọng số RGB cũ, thêm trọng số X/Y bằng 0.
- RepConv giữ nhánh Conv 3×3 cũ, cho đóng góp ban đầu của nhánh mới bằng 0.
- BiFormerBlockLite khởi tạo như một phép identity. Nhưng **identity không phải C2PSA đã học ở Stage1**.

Vì vậy “khởi tạo bảo toàn pretrained” đúng cho các phép đổi Conv nói trên, nhưng không có nghĩa toàn bộ YOLO11s giữ nguyên hàm sau khi thay C2PSA. Mô hình phải học bù lại phần này. A6 sau fine-tune phục hồi lên 92,50% mAP50 val, nên không phải nó không học được.

Ở YOLOv8s, SPPF được giữ nguyên và BiFormer được đặt phía sau, khởi đầu như identity. Điểm trước fine-tune gần như giữ nguyên A0. Đây là khác biệt thực chất giữa hai cách tích hợp, không phải chỉ khác tên backbone.

**Giới hạn của suy luận:** bước giảm trước fine-tune đã được đo. Tuy nhiên, chưa có phép thử đối chứng đủ để nói nó là nguyên nhân duy nhất khiến A6 cuối cùng thua A1.

### 7.3. Train lâu hơn không tự giải quyết được

| Cấu hình | YOLO11s: đã chạy / epoch tốt nhất | YOLOv8s: đã chạy / epoch tốt nhất |
| --- | --- | --- |
| A1 | 70 / 20 | 63 / 13 |
| A2 | 52 / 2 | 65 / 15 |
| A3 | 66 / 16 | 51 / 1 |
| A4 | 65 / 15 | 66 / 16 |
| A5 | 65 / 15 | 51 / 1 |
| A6 | 66 / 16 | 66 / 16 |

“Epoch tốt nhất” trong bảng này lấy theo mAP50–95 của results.csv. Giá trị đánh giá lại best.pt có thể lệch nhẹ so với dòng CSV do điều kiện tính toán lúc đánh giá; báo cáo chính vẫn dùng manifest như đã giải thích.

Tất cả được đặt tối đa 100 epoch, patience 50. Chẳng hạn A6 YOLO11s tốt nhất ở epoch 16, dừng ở epoch 66. A6 YOLOv8s cũng là 16 và 66. Đây là early stopping đúng thiết lập.

Ở cuối quá trình, mAP50–95 CSV của A6 YOLO11s giảm từ mức tốt nhất khoảng 61,23% xuống 59,96%; YOLOv8s từ 60,69% xuống 59,31%. Điều này cho thấy các epoch tiếp theo không tiếp tục cải thiện val. Nó phù hợp với khả năng bắt đầu khớp quá mức hoặc quỹ đạo tối ưu đi xa nghiệm tốt, nhưng chỉ từ lịch sử này chưa tách riêng được hai nguyên nhân.

Vì thế không nên giải thích rằng “A6 thấp chỉ vì chưa chạy đủ 100 epoch”. Checkpoint tốt hơn trước đó đã được lưu và chính checkpoint đó được đem đánh giá.

### 7.4. Chưa tách sạch tác dụng của augmentation và từng thành phần

A1 dùng mixup 0,1 và hsv_v 0,45; A6 dùng mixup 0 và hsv_v 0,4. Ngoài ra A3 có RepConv ở stem, còn A6 dùng CoordConv ở stem nên chỉ có sáu RepConv.

Do đó, các bảng hữu ích để so sánh **các cấu hình đã định nghĩa**, nhưng chưa phải thí nghiệm nhân tố đầy đủ để xác định chính xác phần đóng góp hoặc phần gây giảm của từng module trong A6.

Nếu cần kết luận nhân quả mạnh hơn trong nghiên cứu tiếp theo, có thể giữ nguyên augmentation và so A6 với A6 bỏ từng thành phần, lặp nhiều seed. Đây là hướng bổ sung bằng chứng, không phải yêu cầu sửa hay chạy lại mô hình đã chốt để dùng tài liệu này.

## 8. Nhìn từng lớp: mô hình đang khó ở đâu?

| Lớp | Số box test | YOLO11s: Recall / AP50 / AP50–95 (%) | YOLOv8s: Recall / AP50 / AP50–95 (%) |
| --- | --- | --- | --- |
| blue | 1361 | 86.6 / 89.8 / 59.4 | 83.3 / 89.3 / 58.8 |
| white | 1968 | 86.6 / 89.1 / 59.5 | 86.9 / 90.4 / 59.6 |
| yellow | 1962 | 88.2 / 92.7 / 60.7 | 87.8 / 92.4 / 60.6 |
| red | 2083 | 82.7 / 88.6 / 57.4 | 80.1 / 86.3 / 56.0 |
| none | 2200 | 77.3 / 84.6 / 53.4 | 75.1 / 83.5 / 51.8 |

Lớp none có AP50 và AP50–95 thấp nhất trong A6 ở cả hai backbone. Recall cũng thấp nhất: khoảng 77,3% và 75,1%. Đây là điểm cần chú ý nếu ứng dụng ưu tiên phát hiện người không đội mũ.

Không thể giải thích đơn giản là “none ít dữ liệu”: tập test có 2.200 box none, nhiều nhất trong năm lớp, và train cũng có 1.767 box. Những khả năng như đặc trưng đầu không mũ đa dạng, nhầm với nền hay box khó định vị cần được kiểm tra trên ảnh lỗi; bảng tổng hợp chưa đủ để xác nhận.

Yellow đạt AP50 cao nhất ở cả hai backbone. YOLO11s tốt hơn YOLOv8s ở lớp red và none theo AP trong lần test này, nhưng YOLOv8s nhỉnh hơn ở white. Điều này giúp mô tả ưu nhược điểm cụ thể hơn một câu “mô hình tốt hơn baseline”.

## 9. Tốc độ và độ phức tạp có phải điểm mạnh đã chứng minh?

Log final test ghi thời gian inference trung bình:

| Cấu hình | YOLO11s (ms/ảnh) | YOLOv8s (ms/ảnh) |
| --- | --- | --- |
| A1 | 7,5 | 7,4 |
| A6 | 9,2 | 9,1 |

Đây là thời gian inference được thư viện báo trong một lượt validation trên Tesla T4; không bao gồm đầy đủ mọi chi phí của ứng dụng thực tế. Chưa có benchmark lặp nhiều lần với phân bố thời gian.

**Trong phép đo đang có, A6 chậm hơn A1.** Không nên dùng khả năng gộp nhánh của RepConv để khẳng định A6 đã chạy nhanh hơn.

Code có hàm switch_to_deploy() để gộp RepConv. Tuy nhiên, final test không gọi phép chuyển đó. Dòng “fused” trong model summary phản ánh thao tác fusion thông thường của thư viện, không đủ để chứng minh các RepConv đã được chuyển sang deploy.

Muốn tuyên bố lợi ích tốc độ sau re-parameterization cần một phép đo riêng: cùng checkpoint, kiểm tra sai số trước/sau chuyển đổi, rồi đo trên cùng thiết bị và điều kiện.

## 10. Nhìn riêng từng cấu hình: đã đạt được gì?

### A1 — Cấu hình đơn giản nhưng hiệu quả nhất về mAP

A1 không thêm block vào mạng. Nó tiếp tục học từ Stage1 với dữ liệu tăng cường và thiết lập online augmentation của A1. So với A0, mAP50 test tăng khoảng 1,0 đpt trên YOLO11s và 1,2 đpt trên YOLOv8s; mAP50–95 cùng tăng khoảng 0,6 đpt.

Đây là tín hiệu khá nhất quán giữa hai backbone trong lần chạy này: tiếp tục huấn luyện với chính sách dữ liệu của A1 đem lại hiệu quả tốt. Chưa tách được bao nhiêu phần đến từ thêm epoch, bao nhiêu từ hardaug, mixup hay độ sáng.

### A2 — Tọa độ giúp tăng mAP50, nhưng chưa cho lợi ích định vị nhất quán

Trên test, A2 tăng mAP50 so với A0 ở cả hai backbone: khoảng 0,6 đpt với YOLO11s và 1,1 đpt với YOLOv8s. mAP50–95 chỉ tăng khoảng 0,1 đpt trên YOLO11s, còn YOLOv8s giữ 57,2% ở độ chính xác của log.

Như vậy, cấu hình có CoordConv hoạt động và có kết quả mAP50 tốt hơn mốc Stage1. Nhưng vì A2 cũng có thêm fine-tune và hardaug, đây chưa phải bằng chứng tách riêng tác dụng của hai kênh tọa độ.

### A3 — RepConv không làm mất khả năng học, nhưng cũng chưa chứng minh ưu thế ở box khớp chặt

A3 có mAP50 test cao hơn A0 ở hai backbone. Riêng YOLO11s, mAP50–95 lại từ 58,5% xuống 58,3%; YOLOv8s vẫn hiển thị 57,2%. Trên val, A3 YOLO11s có Recall cao nhất nhóm.

Điều đó phù hợp với kết luận rằng cấu hình A3 có thể phát hiện thêm một số object nhưng chất lượng box ở các ngưỡng IoU cao chưa tăng đồng đều. Việc A3 cao hơn A6 ở một vài cột không phải tự thân là dấu hiệu nối sai module; A6 còn đổi stem, attention và box loss.

### A4 — Đã tích hợp Focal-EIoU, nhưng kết quả chưa xác nhận loss mới định vị tốt hơn

A4 test mAP50 cao hơn A0 khoảng 0,4 đpt trên YOLO11s và 0,8 đpt trên YOLOv8s. Tuy nhiên, mAP50–95 của A4 và A0 bằng nhau trong mức làm tròn của log ở cả hai backbone.

Focal-EIoU trực tiếp tác động mục tiêu học box, nên mAP50–95 là chỉ số cần xem kỹ. Với số liệu hiện tại, cách viết phù hợp là “đã thử nghiệm loss này”, chưa phải “loss này đã chứng minh cải thiện độ khít box so với CIoU”.

### A5 — Attention có hoạt động, nhưng cách gắn vào hai backbone tạo điều kiện học khác nhau

A5 YOLO11s bị giảm điểm rõ ngay lúc thay C2PSA, rồi phục hồi qua fine-tune. A5 YOLOv8s giữ gần nguyên điểm khởi tạo vì SPPF không bị thay.

Trên test, A5 cao hơn A0 nhẹ về mAP ở cả hai backbone, nhưng vẫn thấp hơn A1. Trên val YOLOv8s, A5 có mAP50 cao hơn A6 một chút, còn mAP50–95 thấp hơn A6. Vì vậy không thể chỉ chọn cột mAP50 để kết luận attention riêng luôn hơn tổ hợp đầy đủ.

### A6 — Tổ hợp triển khai hoàn chỉnh, hiệu quả chưa cộng dồn

A6 khôi phục được mức mAP50 val khoảng 92,5% sau fine-tune. Trên test, YOLOv8s tăng nhẹ so với A0; YOLO11s tăng rất nhẹ mAP50 nhưng giảm mAP50–95.

Điểm đạt được là đã triển khai và kiểm chứng được tổ hợp đầy đủ trên hai backbone. Điểm chưa đạt là tổ hợp này chưa vượt A1, đồng thời tốn thời gian inference hơn trong log đang có. Đây là kết quả về hiệu quả của thiết kế trong điều kiện thí nghiệm hiện tại, không phải lý do để sửa hoặc giấu các hàng ablation.

## 11. Đánh giá chung và cách trình bày trước hội đồng

### Những điều có thể khẳng định

- Đã triển khai và chạy được sáu cấu hình trên hai backbone, cùng split GDUT, cùng seed và checkpoint Stage1 tương ứng.
- A6 chứa đúng các module đã chốt; manifest ghi rõ vị trí của CoordConv, RepConv và BiFormerBlockLite.
- A6 đạt khoảng 92,5% mAP50 trên val; kết quả test thấp hơn và đã được báo riêng.
- A1 là cấu hình có mAP50–95 tốt nhất trong nhóm này. A6 YOLOv8s tăng nhẹ so với A0 trên test; A6 YOLO11s cho kết quả tăng/giảm tùy chỉ số.
- Có một hạn chế rõ: hiệu quả kết hợp của A6 chưa vượt cấu hình đơn giản hơn trong điều kiện thí nghiệm hiện tại.

### Những điều chưa đủ cơ sở để khẳng định

- A6 tốt nhất, tốt hơn ở mọi lớp hoặc tốt hơn trên mọi dataset.
- Mọi module đều đóng góp dương vào kết quả cuối.
- A6 nhanh hơn nhờ RepConv trong phép đo final test hiện tại.
- Chênh lệch nhỏ đã có ý nghĩa thống kê. Các run đang dùng một seed, chưa có độ lệch chuẩn hay khoảng tin cậy qua nhiều lần train.
- A0/A1 là hai baseline từ bài báo ngoài. Trong tài liệu này chúng là mốc nội bộ; muốn kết luận với hai baseline bài báo phải dùng bảng comparison tương ứng.

### Một cách trả lời ngắn, đúng với số liệu

> Nhóm xây dựng A6 để kết hợp bốn hướng xử lý: thông tin vị trí, convolution có thể tái tham số hóa, attention có định tuyến và loss định vị box. Thí nghiệm cho thấy tổ hợp này hoạt động, nhưng không vượt A1 trên GDUT với cấu hình hiện tại. Nhóm không xem số lượng module là bảo đảm về chất lượng. Khi kiểm tra sâu hơn, chúng em thấy việc thay C2PSA làm giảm điểm khởi tạo của YOLO11s, và các cấu hình cũng chưa hoàn toàn đồng nhất về augmentation. Vì vậy, kết luận của nhóm là lợi ích của tổ hợp phụ thuộc cách tích hợp và điều kiện huấn luyện; A1 vẫn là mốc hiệu quả hơn về mAP trong các run đã đo.

Nếu được hỏi “vậy giữ A6 để làm gì?”, có thể trả lời: A6 là kiến trúc đề xuất và là đối tượng nghiên cứu đầy đủ; ablation được thực hiện để kiểm tra giả thuyết đó, kể cả khi kết quả cho thấy giả thuyết không được xác nhận hoàn toàn. Nếu mục tiêu chỉ là chọn cấu hình có mAP cao nhất trong các run đã đo, kết quả hiện tại nghiêng về A1.

Việc bài báo được chấp nhận không thay thế cho bằng chứng thực nghiệm của phiên bản này. Khi bảo vệ nên gắn từng nhận định với đúng phiên bản code, hình và bảng kết quả, tránh dùng kết quả cũ để bảo đảm một nhận định mới.

## 12. Nguồn để kiểm tra lại

- [Thư mục artifact Kaggle đã tải read-only (`nvthanh2004`)](../kaggle_verified_source/nvthanh2004) (`dataset_manifest.json`, `run_manifest.json`, `summary.csv`, `validation_summary.csv`).
- [Báo cáo đối chiếu pháp y 15 notebook Kaggle (`75/75 MATCH`)](../KAGGLE_FORENSIC_VERIFICATION_REPORT.md) và [Danh sách 15 Kaggle URL chính thức](DANH_SACH_KAGGLE_CHINH_THUC.md).
- Các notebook công khai mà nhóm cung cấp (`nvthanh2004`): [A0](https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a0), [A1](https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a1), [A2](https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a2), [A3](https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a3), [A4](https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a4), [A5](https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a5), [A6](https://www.kaggle.com/code/nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a6).

Các bảng được dựng trực tiếp từ raw outputs trên Kaggle và log đã kiểm chứng ở trên. Phần phương pháp và đối chiếu kiến trúc nằm trong [tài liệu thứ hai](02_PHUONG_PHAP_VA_KIEN_TRUC.md).
