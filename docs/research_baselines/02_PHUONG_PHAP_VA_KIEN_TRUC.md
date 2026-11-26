# Phương pháp, quy trình A0–A6 và đối chiếu kiến trúc cũ–mới

Ngày tổng hợp: 25/11/2026. Phạm vi: bộ code GDUT CLEAN đã chốt và hình overall mới giữ cách trình bày của hình cũ. Tài liệu giải thích cách triển khai đang dùng; không thay đổi kiến trúc hay kết quả đã chạy.

## 1. Trước hết, “bản cũ” và “bản mới” ở đây là gì?

Trong cuộc trao đổi trước có nhiều phiên bản. Để tránh lẫn, tài liệu dùng ba cách gọi rõ ràng:

| Cách gọi | Nguồn được đối chiếu | Dùng để kết luận điều gì? |
| --- | --- | --- |
| Hình cũ | rep_yolo11s_overall_ieee.png trong AAIML 2027/FIGURE_PACKAGE | Hình từng thể hiện vị trí module, số lớp và các đường dữ liệu như thế nào |
| Notebook gốc | Các notebook A3, A5, A6 trong FULL_TRAIN_KAGGLE | Code gốc đã có ý định thay module nào, tại đâu |
| Bản chốt | SUBMISSION_CLEAN/GDUT-HWD; manifest và source của các run đã kiểm tra; hình GDUT_A6_YOLO11s_Overall_Simplified | Kiến trúc, dữ liệu và cách huấn luyện của các kết quả đang báo cáo |

**Không được đồng nhất “hình cũ chỉ vẽ RepConv ở neck” với “code cũ chỉ có RepConv ở neck”.** Trong notebook gốc đã có vòng lặp thay các Conv 3×3 ở cả backbone lẫn neck. Hình cũ chưa diễn đạt đúng điều đó.

Vì vậy, khi nói về thay đổi, cần tách:

- **Sửa mô tả:** chỉnh hình để phản ánh đúng graph chạy thực tế.
- **Thay đổi triển khai:** chẳng hạn khởi tạo module mới từ trọng số cũ, gắn BiFormer cho YOLOv8s, hoặc dựng lại graph tùy biến trong từng worker DDP.
- **Thay đổi bài toán dữ liệu:** chuyển từ hình hai lớp sang bài toán năm lớp đang train.

Không có một cặp thí nghiệm độc lập “model của hình cũ” và “model của hình mới” trong các bảng đang tổng hợp. Vì vậy không thể lấy chênh lệch giữa các run bất kỳ rồi quy hết cho việc sửa hình.

## 2. Toàn bộ quy trình trong một lượt đọc

Quy trình bắt đầu từ ảnh và nhãn GDUT dạng VOC. Code kiểm tra nhãn, phát hiện ảnh trùng bằng nội dung pixel, giữ nguyên test chính thức, rồi tách train/val từ phần trainval còn lại. Nhãn được chuyển sang YOLO.

Từ 1.265 ảnh train, pipeline tạo thêm một bản tăng cường cho mỗi ảnh. Hai mô hình stock YOLOv8s và YOLO11s được huấn luyện Stage1 trên train gốc. Sau đó, mỗi cấu hình A1–A6 lấy lại checkpoint Stage1 tương ứng, áp dụng đúng thay đổi của mình và fine-tune trên train gốc cộng hard augmentation.

A0 chỉ đánh giá checkpoint Stage1. Sau khi từng cấu hình chọn được best.pt bằng val, notebook final test đánh giá 14 checkpoint: hai Stage1 và 12 checkpoint A1–A6.

    GDUT gốc
        → kiểm tra + xử lý ảnh trùng + chia train/val
        → prepared_gdut
            → Stage1 YOLOv8s → A0, A1, A2, A3, A4, A5, A6 của YOLOv8s
            → Stage1 YOLO11s → A0, A1, A2, A3, A4, A5, A6 của YOLO11s
        → final test: đánh giá các checkpoint đã chọn
        → bảng val đã lưu + bảng test mới, tách riêng

Các nhánh A1–A6 không nối trọng số với nhau. Việc chạy theo thứ tự số giúp quản lý thí nghiệm, không có nghĩa A6 cần output A5.

## 3. Chuẩn bị dữ liệu: đã xử lý những gì?

### 3.1. Đọc cấu trúc GDUT gốc

Pipeline đọc:

    Annotations/*.xml
    JPEGImages/*.jpg
    ImageSets/Main/trainval.txt
    ImageSets/Main/test.txt

Phiên bản này có 3.174 ảnh: 1.587 ảnh thuộc trainval chính thức và 1.587 ảnh thuộc test chính thức.

Nhãn được giữ đúng thứ tự:

| Class ID | Tên trong code | Ý nghĩa |
| --- | --- | --- |
| 0 | blue | Mũ xanh dương |
| 1 | white | Mũ trắng |
| 2 | yellow | Mũ vàng |
| 3 | red | Mũ đỏ |
| 4 | none | Đối tượng thuộc lớp không đội mũ theo nhãn GDUT |

Thứ tự này quan trọng vì class ID là thứ đi vào loss. Không được đổi thứ tự tên lớp khi đánh giá checkpoint cũ.

### 3.2. Kiểm tra trước khi chuyển định dạng

Code xác nhận danh sách trainval và test không trùng ID, không lặp ID trong từng danh sách, và toàn bộ XML phải khớp các ID đã khai báo.

Mỗi ảnh được giải mã RGB để kiểm tra có đọc được không. Kích thước ảnh thật phải trùng width/height trong XML. Với mỗi object, code kiểm tra tên lớp, số hữu hạn, box có chiều rộng và chiều cao dương, và box chuẩn hóa nằm trong ảnh.

Các object có cờ VOC difficult vẫn được giữ. Pipeline không tự bỏ nhãn khó để làm điểm đẹp hơn.

### 3.3. Chuyển box VOC sang YOLO

VOC được xử lý theo quy ước tọa độ 1-based inclusive. Với ảnh rộng W, cao H:

    x1 = max(0, xmin − 1)
    y1 = max(0, ymin − 1)
    x2 = min(W, xmax)
    y2 = min(H, ymax)

    center_x = (x1 + x2) / (2W)
    center_y = (y1 + y2) / (2H)
    width    = (x2 − x1) / W
    height   = (y2 − y1) / H

Mỗi dòng nhãn YOLO là class_id, center_x, center_y, width, height. Các tọa độ được chuẩn hóa về 0–1. Hai mép dưới được chặn ở 0 để chấp nhận cả một số annotation xuất ra có xmin/ymin bằng 0.

Ví dụ, box phủ toàn ảnh theo VOC từ (1,1) đến (W,H) sẽ trở thành tâm (0,5; 0,5), rộng 1, cao 1. Đây là chuyển quy ước tọa độ, không phải dịch box tùy ý.

### 3.4. Xử lý lỗi ảnh trùng giữa trainval và test

Pipeline không chỉ so tên file. Nó giải mã ảnh về RGB, ghép kích thước và bytes pixel, rồi tính SHA-256. Hai tên ảnh khác nhau nhưng có cùng kích thước và pixel sẽ được xem là trùng.

Trong đúng dataset đã dùng cho các run chốt, phát hiện **6 ảnh trainval trùng với ảnh test**:

| Ảnh trainval được loại | Ảnh test được giữ |
| --- | --- |
| JPEGImages/00286.jpg | JPEGImages/02142.jpg |
| JPEGImages/00641.jpg | JPEGImages/00615.jpg |
| JPEGImages/00676.jpg | JPEGImages/00239.jpg |
| JPEGImages/00801.jpg | JPEGImages/01007.jpg |
| JPEGImages/03087.jpg | JPEGImages/01726.jpg |
| JPEGImages/03365.jpg | JPEGImages/02224.jpg |

Cách xử lý là **giữ nguyên tất cả ảnh test, loại bản sao nằm ở trainval**. Không chuyển ảnh test sang train và không loại ảnh test để tăng điểm.

Sau bước này, trainval còn 1.581 ảnh. Việc kiểm tra trùng dùng nội dung ảnh để tránh rò rỉ dữ liệu; không dùng độ khó hay điểm dự đoán trên test để quyết định giữ/bỏ.

Đây là kiểm tra trùng pixel chính xác. Nó không bảo đảm phát hiện mọi ảnh gần giống, ảnh nén lại khác pixel hoặc các frame gần nhau trong cùng cảnh.

### 3.5. Chia train và val

Các ảnh trùng pixel trong phần còn lại được gom thành nhóm không được tách rời. Các nhóm được xáo trộn với seed 3407, sau đó chọn khoảng 20% số ảnh của trainval làm val.

Kết quả:

| Tập | Ảnh gốc | blue | white | yellow | red | none | Tổng box |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1265 | 978 | 1462 | 1465 | 1865 | 1767 | 7537 |
| val | 316 | 258 | 349 | 448 | 279 | 404 | 1738 |
| test | 1587 | 1361 | 1968 | 1962 | 2083 | 2200 | 9574 |

Train + val + test có tổng 3.168 ảnh, tức 3.174 ảnh ban đầu trừ 6 bản sao trainval.

Đây là **chia theo nhóm ảnh trùng**, không phải chia theo cảnh quay, người xuất hiện hay stratification cân bằng từng lớp. Code xác nhận cả năm lớp có mặt trong mỗi split, nhưng không ép tỷ lệ box từng lớp bằng nhau. Ví dụ red có 1.865 box train nhưng chỉ 279 box val, còn yellow có 1.465 box train và 448 box val.

Không nên viết trong báo cáo rằng dataset đã được “cân bằng hoàn toàn” hoặc “chia theo scene” vì code hiện tại không làm hai việc đó.

### 3.6. Lưu ảnh và nhãn

Ảnh trong base được sao chép nguyên file; chưa resize toàn bộ ảnh gốc về 640 ở bước này. Tên mới là mã hash rút gọn từ đường dẫn nguồn, giúp tránh đụng tên. Mã tên file khác với pixel hash dùng để tìm ảnh trùng.

Nhãn base được ghi với 10 chữ số thập phân. Manifest lưu nguồn ảnh, box, pixel hash, split và thống kê lớp. Nhờ vậy có thể truy ngược một file đã đổi tên về ảnh gốc.

Các run còn lưu SHA-256 của file dataset_manifest.json làm dataset_fingerprint. Đây là kiểm tra cùng phiên bản metadata/split; không phải mỗi lần train đều giải mã và hash lại toàn bộ ảnh.

## 4. Tăng cường dữ liệu: làm trước train và trong lúc train

### 4.1. Hard augmentation lưu thành ảnh

Mỗi ảnh train gốc tạo thêm đúng một file hard augmentation. Có 1.265 ảnh gốc và 1.265 bản tăng cường, tổng 2.530 file ảnh dùng cho Stage2. Đây vẫn là các biến thể từ 1.265 ảnh nguồn, không phải 2.530 cảnh độc lập.

| Phép biến đổi | Cấu hình chính | Mục đích |
| --- | --- | --- |
| Resize giữ tỷ lệ + padding | Cạnh dài tối đa 640, pad thành 640×640, giá trị nền 114 | Chuẩn hóa kích thước bản augmentation, tránh kéo méo hình |
| RandomShadow | 1–3 vùng bóng, xác suất 0,35 | Tạo biến thiên vùng tối |
| Brightness/Contrast | Giới hạn ±0,25, xác suất 0,45 | Mô phỏng thay đổi độ sáng và tương phản |
| MotionBlur | blur_limit 5, xác suất 0,12 | Mô phỏng nhòe chuyển động |
| GaussNoise | std_range 0,02–0,08, xác suất 0,18 | Tạo nhiễu ảnh |
| CoarseDropout | 1–8 vùng che, kích thước tương đối 0,03–0,18, xác suất 0,35 | Tạo che khuất một phần |

Các phép ngẫu nhiên không bắt buộc cùng xuất hiện trên một ảnh. Có ảnh chỉ đổi sáng, có ảnh thêm bóng hoặc che khuất; không phải mọi file hardaug đều trải qua tất cả biến đổi.

Nhãn đi qua cùng phép biến đổi hình học với ảnh. Cấu hình bbox dùng định dạng YOLO và min_visibility 0,20. Tuy nhiên, không nên hiểu tham số này là bảo đảm mọi object bị CoarseDropout che nhiều đều được loại; tác động lên bbox phụ thuộc phép biến đổi, còn che pixel không giống cắt hình học box.

Seed riêng của ảnh là 3407 cộng chỉ số trong danh sách tên ảnh đã sắp xếp. Nếu đầu vào và môi trường được giữ nguyên, cách này giúp tái tạo augmentation.

### 4.2. Vì sao bỏ đổi hue và saturation?

GDUT phân biệt màu mũ. Nếu đổi hue mạnh, một chiếc mũ đỏ có thể trông như màu khác trong khi class ID vẫn là red. Nhãn và ảnh sẽ mâu thuẫn.

Vì vậy pipeline bỏ biến đổi hue/saturation ở hard augmentation; lúc train đặt hsv_h = 0 và hsv_s = 0. Vẫn cho đổi độ sáng, bóng và nhiễu để học các điều kiện chụp khác nhau. Điều đó giảm nguy cơ đổi màu nhãn, nhưng không có nghĩa mọi biến đổi đều giữ nguyên cảm nhận màu tuyệt đối.

### 4.3. Online augmentation

Ngoài các file hardaug, loader còn thực hiện augmentation lúc train. Các thiết lập chính trong args.yaml gồm mosaic 1,0; fliplr 0,5; translate 0,1; scale 0,5; không xoay, không lật dọc, không shear/perspective. close_mosaic được đặt 10.

| Cấu hình | hsv_v | mixup |
| --- | --- | --- |
| Stage1, A2–A6 | 0,4 | 0 |
| A1 | 0,45 | 0,1 |

close_mosaic = 10 nghĩa là lịch tắt mosaic gắn với 10 epoch cuối của kế hoạch train. Nếu early stopping xảy ra sớm hơn mốc đó, không nên mặc định run đã trải qua một giai đoạn 10 epoch cuối không mosaic.

Val và test không có bản hard augmentation, cũng không dùng augmentation train. Chúng vẫn qua tiền xử lý cần thiết của bộ đánh giá như resize/letterbox và chuẩn hóa pixel. “Không augment test” không có nghĩa đưa ảnh thô mọi kích thước trực tiếp vào mạng.

### 4.4. Output của preparation

    prepared_gdut/
        base/images/train, val, test
        base/labels/train, val, test
        hardaug/images/train
        hardaug/labels/train
        dataset_manifest.json
        duplicate_audit.json
        augmentation_manifest.json
        preparation_complete.json

preparation_complete.json chỉ được ghi sau khi tạo hard augmentation xong. Notebook train kiểm tra dấu hoàn tất này để tránh dùng một output bị dừng giữa chừng.

Khi notebook train nhận dataset đã mount, code tạo YAML phù hợp đường dẫn Kaggle hiện tại trong working/config. Stage1/A0 dùng base; A1–A6 dùng base + hardaug cho train. Val và test luôn trỏ về base tương ứng.

## 5. Chạy các notebook theo thứ tự nào, input/output là gì?

Các tên dưới đây là tên file trong bộ nộp GDUT CLEAN. Mỗi notebook A0–A6 xử lý cả hai backbone.

| Bước | Notebook | Add Input | Công việc và output chính |
| --- | --- | --- | --- |
| 1 | GDUT_dataset_preparation.ipynb | Dataset GDUT gốc | Tạo prepared_gdut, nhãn YOLO, split, augmentation và manifest |
| 2 | GDUT_architecture_integration_check.ipynb | Không cần dataset | Kiểm tra 12 cấu hình bằng tensor giả, loss, gradient, lưu/nạp và fusion RepConv |
| 3 | GDUT_structural_re_parameterized_yolo_architecture1.ipynb | Output preparation | Train Stage1 stock YOLOv8s; lưu best.pt, results.csv, manifest |
| 4 | GDUT_structural_re_parameterized_yolo_architecture2.ipynb | Output preparation | Train Stage1 stock YOLO11s; lưu các artifact tương tự |
| 5 | GDUT_stage2_ablation_setup_full_train_run_a0.ipynb | Preparation + hai output Stage1 | Đánh giá val hai Stage1; lưu evaluation.json, không tạo checkpoint mới |
| 6 | GDUT_stage2_ablation_setup_full_train_run_a1.ipynb | Cùng ba input như A0 | Fine-tune cấu hình A1, xuất hai checkpoint và kết quả val |
| 7 | GDUT_stage2_ablation_setup_full_train_run_a2.ipynb | Cùng ba input như A0 | Fine-tune A2 |
| 8 | GDUT_stage2_ablation_setup_full_train_run_a3.ipynb | Cùng ba input như A0 | Fine-tune A3 |
| 9 | GDUT_stage2_ablation_setup_full_train_run_a4.ipynb | Cùng ba input như A0 | Fine-tune A4 |
| 10 | GDUT_stage2_ablation_setup_full_train_run_a5.ipynb | Cùng ba input như A0 | Fine-tune A5 |
| 11 | GDUT_stage2_ablation_setup_full_train_run_a6.ipynb | Cùng ba input như A0 | Fine-tune A6 đầy đủ kiến trúc |
| 12 | GDUT_final_test_evaluation.ipynb | Preparation + hai Stage1 + output A1–A6 | Test 14 checkpoint; đồng thời tổng hợp riêng bảng val từ metadata đã lưu |

Bước 2 có thể chạy độc lập trước preparation. Hai Stage1 cũng độc lập với nhau. Sau khi có cả hai Stage1, A1–A6 không phụ thuộc thứ tự chạy.

Tên architecture1/2 được giữ theo quy ước notebook cũ. Trong bộ code này, chúng là **Stage1 của hai mô hình stock**, chưa phải A6 và cũng không phải hai baseline từ bài báo ngoài.

Output của mỗi cấu hình Stage2 nằm trong runs/gdut_Ax_yolo11s và runs/gdut_Ax_yolov8s, gồm:

- weights/best.pt và weights/last.pt: checkpoint được chọn và checkpoint cuối.
- results.csv: lịch sử theo epoch.
- args.yaml: tham số train đầy đủ.
- run_manifest.json: kết quả val_check, fingerprint dataset, hash checkpoint, vị trí module và recipe.
- architecture_rank_*.json: cấu trúc được từng tiến trình train sử dụng.
- val_check: hình, đường cong và các kết quả đánh giá lại best.pt.

A2, A3, A5, A6 còn có initialization_checks để đo mô hình ngay sau khi thay module và trước fine-tune.

**Final test vẫn cần hai output Stage1.** A0 chỉ chứa kết quả đánh giá, không có best.pt mới để thay thế Stage1. Bảng val lấy số đã lưu của checkpoint, còn bảng test là phép đánh giá trên tập test; hai bảng không được ghép lẫn.

## 6. Cấu hình huấn luyện của các run đã chốt

| Thành phần | Thiết lập |
| --- | --- |
| Backbone | YOLO11s và YOLOv8s |
| Đầu vào | imgsz 640, RGB; CoordConv thêm X/Y bên trong model ở A2/A6 |
| Class | 5 |
| Seed | 3407; deterministic = True |
| Optimizer | AdamW |
| Learning rate đầu Stage1 | 0,0003 |
| Learning rate đầu Stage2 | 0,0001 |
| Epoch tối đa | 100 |
| Early stopping | patience 50 |
| Warmup | 3 epoch; warmup_bias_lr = 0 |
| momentum trong cấu hình AdamW | 0,9 |
| weight_decay trong args | 0,0005 |
| lrf | 0,01; cos_lr = False |
| Batch | 16 tổng cho cấu hình device 0,1 |
| Workers | 4 |
| AMP | True |
| Loss gains | box 7,5; cls 0,5; dfl 1,5 |
| Ultralytics | 8.4.115 |
| Albumentations của package | 2.0.8 |
| PyTorch trong manifest run | 2.11.0+cu130 |
| Thiết bị được log final test xác nhận | Tesla T4 |

device = 0,1 là cấu hình train nhiều GPU được manifest ghi lại; không dùng thông tin này để tự suy ra một benchmark tốc độ hai GPU. Final test được ghi trên GPU 0.

Stage1 bắt đầu từ checkpoint pretrained của từng họ mô hình, sau đó học bài toán năm lớp GDUT. A1–A6 bắt đầu từ best.pt Stage1, tạo optimizer/lịch fine-tune mới, không tiếp tục trạng thái optimizer của A liền trước.

“Full train” trong tên A6 chỉ phiên bản đủ thành phần và run thật, không có nghĩa đem cả val/test vào train hay bắt buộc chạy đủ 100 epoch. Các run chốt dừng sớm theo patience.

## 7. Nội dung chính xác của từng ablation

| Cấu hình | CoordConv | RepConv | BiFormerBlockLite | Box loss |
| --- | --- | --- | --- | --- |
| A0 | Không | Không | Không | CIoU của Stage1 |
| A1 | Không | Không | Không | CIoU |
| A2 | Stem, layer 0 | Không | Không | CIoU |
| A3 | Không | 7 block, gồm stem | Không | CIoU |
| A4 | Không | Không | Không | Focal-EIoU |
| A5 | Không | Không | 1 block ở P5 | CIoU |
| A6 | Stem, layer 0 | 6 block, không thay stem | 1 block ở P5 | Focal-EIoU |

Lưu ý augmentation của A1 khác A2–A6 như phần 4. Tất cả A1–A6 dùng thêm hardaug, không phải chỉ A1 có augmentation.

Vị trí module, theo chỉ số layer bắt đầu từ 0:

| Backbone và cấu hình | CoordConv | RepConv | Attention |
| --- | --- | --- | --- |
| YOLO11s A3 | — | 0, 1, 3, 5, 7, 17, 20 | — |
| YOLO11s A6 | 0 | 1, 3, 5, 7, 17, 20 | 10, thay C2PSA |
| YOLOv8s A3 | — | 0, 1, 3, 5, 7, 16, 19 | — |
| YOLOv8s A6 | 0 | 1, 3, 5, 7, 16, 19 | Trong layer 9, sau SPPF |

A5 dùng vị trí attention giống A6 của cùng backbone. A2 chỉ đổi layer 0.

Không thay toàn bộ convolution trong mạng. Vòng thay RepConv xét các layer Conv 3×3 ở cấp graph chính; nó không đi thay mọi convolution nằm sâu bên trong C3k2, C2f hay detection head.

## 8. Đọc hình overall mới từ input tới output

![Overall mới của A6 dựa trên YOLO11s](C:/Users/nvtha/Downloads/shwd/GDUT_OVERALL_OLD_STYLE/GDUT_A6_YOLO11s_Overall_Simplified.png)

Hình không ghi GDUT trong nội dung theo lựa chọn trình bày của nhóm. Nó vẫn thể hiện cấu hình head năm lớp của phiên bản đang chạy. Tên file giữ nguyên để không làm đứt các liên kết đã dùng.

### 8.1. Input và CoordConv

Hình minh họa đầu vào vuông B×3×640×640, với B là batch size. Các bảng kích thước dưới đây cũng dùng trường hợp này. imgsz = 640 là kích thước cấu hình; ở đường đánh giá có rectangular batching/letterbox, kích thước tensor thực tế có thể thay đổi theo tỷ lệ ảnh. Hai lưới tọa độ luôn được tạo theo chiều cao/rộng thực tế của tensor:

- X thay đổi từ −1 ở bên trái đến 1 ở bên phải.
- Y thay đổi từ −1 ở phía trên đến 1 ở phía dưới.

Ghép theo chiều channel cho tensor B×5×640×640. **Năm channel này là R, G, B, X, Y, không phải năm class.** Hai channel mới được tạo trong forward, không cần lưu thành ảnh năm kênh trong dataset.

Trong code, CoordConv gồm cả thao tác thêm X/Y và convolution phía sau. Hình tách hai phần để dễ đọc: thêm tọa độ ở khối input, rồi CBS 3×3 stride 2 ở stem. Đó không phải hai convolution stem liên tiếp.

Tọa độ giúp mạng có thông tin vị trí tuyệt đối một cách trực tiếp. Nó có thể hữu ích khi hình dạng hoặc vị trí tương đối trong khung ảnh mang thông tin, nhưng cũng có thể làm mạng phụ thuộc bố cục của tập train. Vì vậy đây là một giả thuyết thiết kế cần ablation, không phải lợi ích được bảo đảm trên mọi cảnh.

### 8.2. Backbone YOLO11s A6

Kích thước trong bảng là C×H×W, chưa ghi batch:

| Đoạn graph | Xử lý | Đầu ra cuối đoạn |
| --- | --- | --- |
| Layer 0, stem | CoordConv: thêm X/Y rồi Conv 3×3 stride 2 + BN + SiLU | 32×320×320 |
| Layer 1–2, P2 | RepConv stride 2 → C3k2 | 128×160×160 |
| Layer 3–4, P3 | RepConv stride 2 → C3k2 | 256×80×80 |
| Layer 5–6, P4 | RepConv stride 2 → C3k2 | 256×40×40 |
| Layer 7–10, P5 | RepConv stride 2 → C3k2 → SPPF → BiFormerBlockLite | 512×20×20 |

**“RepConv (s2) → C3k2” nghĩa là hai block nối tiếp.** RepConv làm giảm chiều cao/rộng còn một nửa; C3k2 tiếp tục xử lý và trộn đặc trưng. Ví dụ P3 nhận đặc trưng 160×160 từ P2, giảm xuống 80×80 rồi qua C3k2.

Output stride 8 của P3 nghĩa là độ phân giải đầu ra nhỏ hơn ảnh đầu vào tám lần: 640/8 = 80. Không có convolution C3k2 “stride 8”. Tương tự, output stride của P2/P4/P5 là 4/16/32.

C3k2 là block trích xuất đặc trưng có cấu trúc chia/ghép nhánh kế thừa YOLO11. Nó giữ vai trò xử lý đặc trưng trong từng stage. RepConv bổ sung cách tham số hóa cho bước convolution giảm kích thước, không xóa vai trò của C3k2.

SPPF giữ độ phân giải 20×20, dùng các phép pooling liên tiếp và ghép đặc trưng để thu nhận thông tin trên vùng rộng hơn. Nó không tạo thêm một detection head P6.

### 8.3. Tại sao P3, P4, P5 cùng được đưa sang neck?

P3 giữ lưới 80×80, còn khá nhiều chi tiết vị trí. P5 có lưới 20×20, đi qua nhiều tầng xử lý hơn. P4 ở giữa. Neck phối hợp các mức này để head không phải chỉ dựa vào một mức đặc trưng.

Theo thiết kế, mức phân giải cao giúp giữ thông tin cho vật nhỏ, mức thấp giúp cung cấp ngữ cảnh. Tuy nhiên, P3/P4/P5 không phải ba bộ lọc cứng “chỉ vật nhỏ / chỉ vật vừa / chỉ vật lớn”; việc dự đoán và gán target vẫn dựa vào dữ liệu và cơ chế assignment.

### 8.4. PAN neck: các đường nối thật trong code

Hình dùng một khung tổng hợp để tránh rối. Bên trong thực tế có đường từ trên xuống và đường từ dưới lên:

| Bước | Dữ liệu được ghép | Kết quả |
| --- | --- | --- |
| Top-down 1 | Upsample P5 từ 20 lên 40; concat với P4 backbone ở layer 6; qua C3k2 layer 13 | Đặc trưng 256×40×40 |
| Top-down 2 | Upsample layer 13 từ 40 lên 80; concat với P3 backbone ở layer 4; qua C3k2 layer 16 | N3/P3 = 128×80×80 |
| Bottom-up 1 | RepConv layer 17 giảm N3 xuống 40; concat với layer 13; qua C3k2 layer 19 | N4/P4 = 256×40×40 |
| Bottom-up 2 | RepConv layer 20 giảm N4 xuống 20; concat với P5 layer 10; qua C3k2 layer 22 | N5/P5 = 512×20×20 |

Concat ghép channel, không cộng các feature map theo từng phần tử. Upsample dùng nearest-neighbor theo YAML nền.

Trong hình, dòng “RepConv — 2 downsampling blocks” nằm trong khung PAN để chỉ hai vị trí layer 17 và 20. Không có một block RepConv tổng duy nhất mà tất cả P3/P4/P5 đều phải đi qua nối tiếp.

Đây cũng là lý do hình ghi **PAN abstraction**: nó mô tả chức năng và các cổng vào/ra ở mức module. Bảng trên mới là cách đọc chính xác các cạnh nối bên trong.

### 8.5. Detection head và Collect scales

Detect layer 23 nhận N3, N4, N5. Ở mỗi mức, head tách hai nhánh:

| Nhánh | Tên trong code | Đầu ra |
| --- | --- | --- |
| Phân loại | cv3 | 5 logits cho mỗi vị trí |
| Hồi quy box | cv2 | 4×16 = 64 logits phân bố khoảng cách cho mỗi vị trí |

Ở YOLO11s, nhánh phân loại dùng hai cụm depthwise convolution 3×3 và pointwise convolution 1×1, sau đó lớp 1×1 ra năm logits. Nhánh box dùng hai CBS 3×3 rồi lớp 1×1 ra 64 logits.

Logit phân loại chưa phải xác suất. Sigmoid chuyển nó thành score dùng trong dự đoán; BCE lúc train nhận logits gốc.

Nhánh box học phân bố cho khoảng cách từ một điểm lưới tới bốn cạnh left/top/right/bottom. Với reg_max = 16, mỗi cạnh có 16 bin. Bước decoding lấy kỳ vọng của phân bố, kết hợp điểm lưới và stride để dựng box.

Anchor-free ở đây nghĩa là không dùng các mẫu anchor box định sẵn theo chiều rộng/cao. Code vẫn dùng **anchor points**, tức các điểm tham chiếu trên lưới; hai khái niệm này không mâu thuẫn.

Với ảnh 640×640:

    P3: 80×80 = 6.400 vị trí
    P4: 40×40 = 1.600 vị trí
    P5: 20×20 =   400 vị trí
    Tổng:       8.400 vị trí

Collect scales gom dự đoán từ ba mức. Tensor trước NMS có thể biểu diễn bằng B×9×8.400: bốn tọa độ box và năm class score. Đây chưa phải 8.400 vật thể chắc chắn có trong ảnh.

Head này không thêm một nhánh objectness riêng. Output cuối cũng không có ID theo dõi.

### 8.6. NMS và output cuối

NMS nằm ngoài detection head và chỉ thuộc hậu xử lý dự đoán. Nó dùng score và mức chồng lặp box để giảm dự đoán trùng theo cấu hình đánh giá. Các run dùng NMS theo lớp, không phải gộp tất cả class thành một lớp.

Sau NMS, đầu ra là **boxes / confidence scores / classes**. Nếu ứng dụng sau này cần tracking ID, phải có thêm bộ theo dõi; đồ thị này không có phần đó.

Final detections là đầu cuối của luồng inference nên không nối về training loss. Loss dùng dự đoán trước NMS để còn đủ thông tin và gradient cho quá trình học.

## 9. RepConv hoạt động ra sao và vì sao dùng ở vị trí đó?

### 9.1. Trong lúc train

RepConv cộng đầu ra của các nhánh rồi mới qua SiLU:

    y = SiLU(BN3(Conv3×3(x)) + BN1(Conv1×1(x)) + nhánh identity nếu có)

Nhánh 3×3 xử lý lân cận không gian. Nhánh 1×1 trộn channel tại cùng vị trí lấy mẫu. Hai nhánh cùng tạo ra một đầu ra nên kích thước và số channel phải khớp.

Identity + BN chỉ có khi số channel vào/ra bằng nhau và stride = 1. **Các RepConv được gắn trong A6 hiện tại đều stride 2, nên không có nhánh identity.** Hình module RepConv tổng quát có thể vẽ nhánh identity với điều kiện này, nhưng không được nói mỗi RepConv trong A6 đều có đủ ba nhánh.

Đặt ở các bước giảm kích thước cho phép giữ graph PAN và các stage C3k2/C2f quen thuộc, đồng thời thay cách học của các convolution chuyển mức phân giải. Đây là can thiệp có phạm vi rõ, thay vì thay cả backbone bằng một họ mạng khác.

### 9.2. Khi chuyển sang deploy

Ở chế độ eval, mỗi Conv + BN có thể gộp thành kernel và bias tương đương. Kernel 1×1 được đặt vào ô giữa của một kernel 3×3; sau đó cộng các kernel và bias của các nhánh.

    Các nhánh tuyến tính trước SiLU
        → kernel 3×3 tương đương + bias
        → convolution đã gộp
        → SiLU

Mũi tên switch_to_deploy() biểu diễn **chuyển đổi cách tham số hóa**, không phải một phép forward lấy đầu ra SiLU cũ làm đầu vào convolution mới. Đó là lý do mũi tên chuyển đổi phải bắt đầu từ cụm nhánh/tổng trước activation.

Tính tương đương dựa trên trạng thái BN dùng cho inference. Không có nghĩa một convolution cố định tái hiện mọi batch-statistics động trong chế độ train.

Trong bản final test đã báo cáo, RepConv vẫn được đánh giá theo topology được lưu; chưa có bước gọi switch_to_deploy() trên toàn mô hình. Khả năng deploy là tính năng của module, chưa phải một kết quả tốc độ triển khai đã đo của nhóm.

## 10. BiFormerBlockLite: khối attention thực sự được dùng

### 10.1. Nó nằm ở đâu?

Với YOLO11s A5/A6, layer 10 vốn là C2PSA được thay bằng BiFormerBlockLite. Vị trí này ở cuối backbone, **sau SPPF và trước PAN**. Không giữ C2PSA rồi thêm một BiFormer nữa ở giữa neck.

Với YOLOv8s A5/A6, layer 9 được bọc thành SPPFWithBiFormer:

    SPPF cũ → BiFormerBlockLite → PAN

Cách đặt sau SPPF giữ mức phân giải P5 thấp, nhờ đó lượng token cần xử lý nhỏ hơn khi đặt attention trực tiếp trên P3. Khối này bổ sung khả năng tương tác giữa các vùng xa nhau mà convolution cục bộ không thực hiện trong một bước.

### 10.2. Đọc đúng thông số vùng

Trong code: channels = 512, num_heads = 4, region_size = 8, topk = 4.

region_size = 8 nghĩa là **mỗi vùng có kích thước 8×8 pixel đặc trưng**, không phải chia feature map thành 8×8 vùng.

Với feature map P5 20×20:

1. Pad về 24×24 để chia hết cho 8.
2. Chia thành 3×3 = 9 vùng.
3. Mỗi vùng chứa 64 token.
4. Với mỗi vùng query, chọn 4 vùng key/value.
5. Mỗi head dùng 512/4 = 128 chiều.

### 10.3. Hai cấp xử lý

**Cấp vùng:** convolution 1×1 tạo Q, K, V. Code lấy trung bình Q và K của các token trong mỗi vùng thành Qʳ, Kʳ. Từ độ tương đồng giữa các vùng, top-k chọn ra chỉ số Iʳ của bốn vùng cần tham khảo.

    Aʳ = Qʳ(Kʳ)ᵀ / √C
    Iʳ = chỉ số top-k theo từng hàng của Aʳ

**Cấp token:** giữ Q của vùng đang xét; gom K và V từ các vùng được chọn. Attention chi tiết được tính bằng:

    Attention(Q, K, V) = softmax[Q(K(Iʳ))ᵀ / √dₖ] V(Iʳ)

Ở đây dₖ = 128 cho mỗi head. Một vùng có 64 query token và tham khảo 4×64 = 256 key/value token. Sau đó ghép kết quả của bốn head và các vùng về feature map.

Output được cắt bỏ phần padding, đi qua projection 1×1 và BN, rồi cộng residual với đầu vào:

    output = input + BN(projection(attention_output))

Chữ r là chỉ số cấp vùng; nó không phải chữ r đứng liền cùng hàng với Q/K như một tên biến không phân biệt. Trong công thức trên, Iʳ biểu diễn chỉ số vùng, còn phép lấy K(Iʳ), V(Iʳ) là gather token theo chỉ số đó.

### 10.4. Những giới hạn cần mô tả đúng

Đây là **BiFormerBlockLite do project triển khai theo cơ chế định tuyến hai cấp**, không phải nguyên một backbone BiFormer hay toàn bộ block của một thư viện tác giả.

Code hiện dùng top-k chỉ số, vòng lặp theo vùng, projection + BN và residual; không có thêm FFN ở cuối block này. Phần padding không có attention mask riêng. Khi giải thích phiên bản hiện tại phải bám những phép toán này, không tự thêm các thành phần chỉ vì chúng xuất hiện trong một biến thể BiFormer khác.

Định tuyến giúp hạn chế tập key/value cần tham khảo ở từng vùng, nhưng việc chọn vùng cũng có thể bỏ qua thông tin hữu ích. Thêm attention không bảo đảm giảm latency hoặc tăng mAP. Kết quả A5/A6 phải được đọc từ thực nghiệm.

## 11. Training-only supervision: từng đường nối dùng để làm gì?

### 11.1. Một batch đi qua loss như thế nào?

Luồng khái quát của A6:

    Ảnh + target thật
        → backbone + neck + head
        → raw class logits và raw box distributions
        → giải mã box; sigmoid score phục vụ assignment
        → TAL gán prediction với target
        → BCE + Focal-EIoU + DFL
        → tổng loss → backward → optimizer cập nhật trọng số

TAL và các loss không tạo thêm detection lúc suy luận. Chúng hướng dẫn model học trong quá trình train.

### 11.2. Task-aligned assigner không phải NMS

Một ảnh có hàng nghìn vị trí dự đoán nhưng chỉ có một số object thật. TAL quyết định vị trí nào học object nào và target score tương ứng.

Phiên bản loss thư viện dùng top-k = 10, alpha = 0,5, beta = 6,0. Nó kết hợp score của lớp và độ phù hợp hình học để chọn các dự đoán ứng viên phù hợp, xét vị trí anchor point và xử lý xung đột gán.

TAL nhận score sau sigmoid và decoded box đã **detach**, cùng box/class thật. Detach chỉ ngắt gradient qua bước quyết định assignment. Các logits và box dùng để tính loss vẫn giữ đường gradient về model.

NMS giải quyết dự đoán trùng ở inference; TAL tạo mục tiêu học ở training. Hai khối không thay nhau.

### 11.3. Classification BCE

BCE dùng logits gốc và target score do assignment tạo ra. Target score có thể là giá trị mềm theo chất lượng gán, không đơn giản luôn là nhãn one-hot 0/1.

Nhận logits giúp BCEWithLogitsLoss xử lý sigmoid và log theo cách ổn định số học. Vì vậy hình mới ghi BCE (logits), đồng thời bổ sung đường lấy raw class logits từ ba nhánh phân loại về phần training.

### 11.4. Box regression: Focal-EIoU

A4 và A6 dùng Focal-EIoU thay phần box loss CIoU. Công thức tương ứng với code:

    EIoU = 1 − IoU
           + khoảng_cách_tâm² / đường_chéo_khung_bao²
           + (w − w_gt)² / chiều_rộng_khung_bao²
           + (h − h_gt)² / chiều_cao_khung_bao²

    Focal-EIoU = IoU^0,5 × EIoU

Code có thêm epsilon/clamp để tránh chia 0 và xử lý ổn định. Box được đưa vào hàm theo dạng xyxy trong đường train hiện tại.

Khung bao ở đây là hình chữ nhật nhỏ nhất bao cả box dự đoán lẫn box thật. EIoU tách sai số tâm, rộng và cao để cung cấp tín hiệu học hình học cụ thể. Đây là lý do nhóm thử nó thay CIoU.

Hệ số IoU^0,5 làm thay đổi trọng số theo chất lượng box. Không nên gọi nó là “focal classification loss” hay khẳng định nó tự động ưu tiên mọi mẫu khó. Trong code, hệ số này cũng không bị detach khỏi đồ thị gradient.

Loss box chỉ được tổng hợp trên các foreground được gán, nhân với trọng số target score rồi chuẩn hóa bằng tổng target score. Việc đổi công thức box không làm mất DFL.

### 11.5. Distribution focal loss

DFL học hình dạng phân bố khoảng cách cạnh box từ raw box distributions. Nếu khoảng cách target nằm giữa hai bin k và k+1, loss phân bổ trọng số cho hai bin lân cận theo vị trí target.

Do đó cần phân biệt:

- **DFL loss:** dùng raw distributions và target khoảng cách để huấn luyện.
- **DFL integral/box decoding:** lấy kỳ vọng phân bố để chuyển thành khoảng cách rồi thành box.

Ba nhánh cv2 phải giữ raw output trước decoding cho DFL. Chỉ gửi bốn tọa độ box đã giải mã về DFL sẽ thiếu dữ liệu.

### 11.6. Vì sao có đường từ Collect scales tới cả TAL và Loss inputs?

Cùng dự đoán trước NMS có hai vai trò: giúp TAL xác định target gán; đồng thời cung cấp các box dự đoán để tính sai số hồi quy sau khi đã có assignment.

Khối Loss inputs trong hình là nơi **tập hợp đầu vào cho các loss**, không phải một neural layer. Nó nhận:

| Dữ liệu | Được dùng ở đâu? |
| --- | --- |
| Raw class logits từ cv3 | BCE |
| Raw box distributions từ cv2 | DFL |
| Decoded boxes | Focal-EIoU |
| Assigned target boxes/scores và foreground mask | Các loss tương ứng |

Đường nét đứt chung biểu diễn một nhóm dữ liệu đi về training, không có nghĩa cộng số học logits và distributions với nhau. Trong code chúng là những tensor riêng.

### 11.7. Tổng training objective

    L = 0,5 × L_BCE + 7,5 × L_box + 1,5 × L_DFL

L_box là Focal-EIoU ở A4/A6, còn các cấu hình khác dùng CIoU. Đây là các hệ số trong args; code thư viện còn chuẩn hóa thành phần loss và áp dụng quy ước batch khi trả về loss cho trainer.

Ground truth chỉ xuất hiện trong train và đánh giá. Khi dùng model phát hiện trên ảnh mới, không cần nhãn thật, TAL hay loss.

## 12. YOLOv8s khác hình YOLO11s ở những chỗ nào?

Hình overall mới là hình của **A6 dựa trên YOLO11s**. Không nên thay chữ YOLO11s bằng YOLOv8s rồi giữ nguyên toàn bộ nội dung.

| Thành phần | YOLO11s A6 | YOLOv8s A6 |
| --- | --- | --- |
| Block xử lý chính | C3k2 | C2f |
| Đầu ra P2 backbone | 128×160×160 | 64×160×160 |
| Đầu ra P3 backbone | 256×80×80 | 128×80×80 |
| Đầu ra P4 backbone | 256×40×40 | 256×40×40 |
| Đầu ra P5 backbone | 512×20×20 | 512×20×20 |
| Attention | Thay C2PSA tại layer 10 | Giữ SPPF, nối BiFormer sau nó trong layer 9 |
| RepConv ở neck | Layer 17, 20 | Layer 16, 19 |
| Đầu ra neck đưa vào head | Layer 16, 19, 22 | Layer 15, 18, 21 |
| Detect layer | 23 | 22 |
| Nhánh phân loại | Depthwise + pointwise như phần 8 | Hai CBS 3×3 rồi Conv 1×1 |

C2f vẫn làm nhiệm vụ trích xuất và trộn đặc trưng qua các nhánh kế thừa từ YOLOv8. Các đầu ra neck của cả hai bản A6 có lần lượt 128/256/512 channel ở độ phân giải 80/40/20.

CoordConv, hai nhánh hồi quy/phân loại, quy tắc phân bố box, TAL, Focal-EIoU và DFL có cùng vai trò. Dù vậy, khác biệt “thay C2PSA” và “nối sau SPPF” tạo ra hai điều kiện khởi tạo khác nhau, đã thể hiện trong số liệu trước fine-tune.

## 13. Khởi tạo, DDP và checkpoint: làm sao biết module thật sự được train?

### 13.1. Không bỏ vô ích những trọng số đã học ở Stage1

| Module mới | Cách khởi tạo trong bản chốt | Điều được giữ lại |
| --- | --- | --- |
| CoordConv | Sao chép trọng số RGB, cho trọng số X/Y bằng 0; sao chép BN | Stem ban đầu vẫn cho đầu ra như Conv cũ, rồi dần học sử dụng tọa độ |
| RepConv | Sao chép Conv 3×3 và BN; scale/bias BN của nhánh mới bằng 0 | Ban đầu nhánh cũ chi phối đầu ra; nhánh mới được mở dần trong train |
| BiFormerBlockLite | Scale/bias BN cuối nhánh residual bằng 0 | Bản thân block mới bắt đầu như identity |

“Bắt đầu như identity” không có nghĩa attention bị đóng băng suốt quá trình train. Các tham số được phép học. Ở thời điểm khởi tạo, một số gradient bên trong nhánh có thể bằng 0 do scale cuối bằng 0; scale có thể cập nhật trước rồi nhánh dần đóng góp.

Với YOLOv8s, giữ SPPF rồi nối identity giúp giữ gần nguyên hàm khởi đầu. Với YOLO11s, thay C2PSA bằng identity làm mất phép biến đổi C2PSA đã học. Số liệu initialization_check đã đo được mức giảm này; không được mô tả là “bảo toàn hoàn toàn toàn mạng” cho cả hai backbone.

### 13.2. Dựng graph ở từng tiến trình train

ResearchTrainer.get_model() dựng ResearchDetectionModel, nạp trọng số và gọi patch theo đúng variant. Khi DDP khởi tạo các worker, mỗi worker chạy lại quy trình này thay vì chỉ dựa vào một object sửa thủ công ở notebook cha.

Code giữ các thuộc tính graph như chỉ số layer và nguồn đầu vào khi thay module. Điều này cần thiết để các cạnh Concat và Detect vẫn nhận đúng feature map.

Mỗi worker ghi architecture_rank_*.json. Sau train, code nạp lại best.pt, kiểm tra variant, kiểm tra inventory module và đối chiếu các file audit của worker.

Vì vậy bằng chứng không chỉ là “notebook có định nghĩa class RepConv”. Các run chốt có manifest ghi rõ module nào nằm ở layer nào, và checkpoint được kiểm tra lại sau khi lưu.

### 13.3. Gắn loss vào đường train thật

ResearchDetectionModel.init_criterion() tạo loss của model. Với A4/A6, criterion.bbox_loss được thay bằng FocalEIoUBboxLoss.

Lớp này lấy DFL từ đường loss gốc và thay riêng phần box loss bằng Focal-EIoU. Chỉ import một hàm focal_eiou_loss vào notebook sẽ không đủ; hàm phải được criterion gọi trong backward.

### 13.4. Các kiểm tra có ý nghĩa gì?

Notebook architecture integration check kiểm tra 12 tổ hợp: A1–A6 nhân hai backbone. Nó dùng tensor ảnh giả 2×3×64×64 và nhãn giả để kiểm tra:

- Forward và loss cho số hữu hạn.
- Backward tạo gradient cho các module tùy biến.
- Đúng lớp loss cho A4/A6.
- Lưu rồi nạp checkpoint không mất module, đầu ra trước/sau nạp khớp trong sai số cho phép.
- Shape sau decoding là 2×9×84 tại ảnh 64×64.
- Một RepConv mẫu cho đầu ra tương đương trước/sau fusion.

Đó là kiểm tra tích hợp phần mềm, không phải test chất lượng phát hiện. Kiểm tra fusion mẫu cũng không chứng minh toàn bộ checkpoint A6 của final test đã được chuyển sang deploy.

Trong lần viết tài liệu này không chạy train lại và không thay notebook. Kết luận về kết quả dựa vào artifact của các run đã hoàn thành.

## 14. Đối chiếu đầy đủ hình cũ và hình mới

### 14.1. Những điểm đã thay trong hình

| Chi tiết | Hình cũ | Hình mới đã chốt | Vì sao đổi? |
| --- | --- | --- | --- |
| RepConv ở backbone | Chưa thể hiện, các stage chỉ ghi C3k2 | P2/P3/P4/P5 đều thể hiện RepConv (s2) trước C3k2 | Khớp các layer 1, 3, 5, 7 của A6 |
| RepConv ở neck | Một khối “RepConv fusion” chung | Ghi rõ hai block downsampling stride 2 trong PAN | Khớp layer 17 và 20; tránh hiểu thành một phép fusion dùng chung |
| P2 output channel | 64 | 128 | Hình cần ghi đầu ra cả stage sau C3k2, không chỉ đầu ra Conv trước nó |
| P3 output channel | 128 | 256 | Cùng lý do: đầu ra stage sau C3k2 |
| P5 | SPPF + C2PSA | RepConv → C3k2 → SPPF → BiFormerBlockLite | C2PSA đã được thay; bổ sung các phép trước SPPF vốn nằm trong stage |
| BiFormer | Vẽ trong khung neck, nối đến RepConv | Đặt cuối backbone P5, trước PAN | Đúng layer 10, thay C2PSA |
| Nội dung PAN | Dễ hiểu thành chuỗi BiFormer rồi RepConv | Top-down + bottom-up; Upsample, Concat, C3k2 và hai RepConv | Mô tả đúng chức năng neck nhưng vẫn giữ bố cục gọn |
| Nhánh phân loại | 2 class scores, hat/person | 5 class logits, sigmoid thành score của 5 lớp | Đúng nc = 5 và phân biệt logits với score |
| Đầu ra thô dùng lúc train | Chỉ nhấn mạnh box distributions tới DFL | Gom raw class logits và box distributions về Loss inputs | BCE cũng cần dữ liệu thô trước sigmoid |
| TAL | Prediction-to-target assignment | Bổ sung ý detached predictions | Phản ánh code detach khi gán target |
| Nhãn BCE | BCE | BCE (logits) | Làm rõ đầu vào thực sự của loss |
| Chữ “single 3×3 deployment” ở neck | Có trong overall | Bỏ khỏi mô tả topology đang đánh giá | Không đánh đồng khả năng fusion với việc final test đã fusion |
| Tên dataset trong hình | Từng xuất hiện ở một bản trung gian | Không ghi GDUT trong nội dung hình chốt | Theo lựa chọn trình bày; dataset được mô tả ở phần thực nghiệm |

Đổi kích thước ghi ở P2/P3 không có nghĩa vừa tăng width của model để cải thiện kết quả. Đó là sửa nhãn tensor theo đúng điểm lấy đầu ra trong graph YOLO11s đang dùng.

### 14.2. Những phần được giữ

Input RGB + X/Y, kích thước 640, các mức P3/P4/P5, decoupled head, box decoding, Collect scales, NMS ngoài head, ba nhánh loss và output boxes/confidence/classes vẫn được giữ về ý tưởng.

Bố cục, bảng màu và cách nhóm khối được giữ gần hình cũ. Hình mới không mở toàn bộ PAN thành hàng loạt đường nối; các cạnh chi tiết được giải thích ở phần 8.4.

### 14.3. Trả lời đúng câu “trước RepConv chỉ ở neck, giờ đặt thế nào?”

Nếu nói về **hình**, câu trả lời là:

> Hình cũ mới thể hiện RepConv trong neck nên chưa đầy đủ. Hình mới thể hiện sáu RepConv của A6 YOLO11s: bốn ở các bước giảm kích thước trong backbone và hai ở đường bottom-up của neck. C3k2 vẫn còn phía sau các bước giảm kích thước; stem dùng CoordConv nên không đặt RepConv ở stem A6.

Nếu nói về **lịch sử code**, phải thêm:

> Notebook gốc đã có vòng thay Conv 3×3 ở cả backbone và neck. Vì vậy việc bổ sung RepConv vào phần backbone của hình chủ yếu là làm hình khớp triển khai, không phải bằng chứng nhóm vừa chuyển từ một kiến trúc chỉ có RepConv ở neck sang kiến trúc mới.

Đây là điểm phân biệt quan trọng nhất khi trình bày trước hội đồng.

## 15. Những thay đổi thực sự của triển khai so với notebook gốc

Phần này đối chiếu code nguồn, không suy đoán rằng mọi run lịch sử đều đã chạy đúng hoặc sai theo cùng một cách.

| Vấn đề | Notebook gốc được đọc | Bản chốt |
| --- | --- | --- |
| Chọn vị trí RepConv | Vòng lặp qua Conv 3×3 cấp graph, ở backbone và neck; A6 bỏ qua layer 0 | Giữ cách chọn đó, thêm kiểm tra danh sách layer mong đợi |
| Khởi tạo Conv thay thế | Khởi tạo CoordConv/RepConv mới trực tiếp trong đoạn patch | Sao chép Conv/BN cũ, cho nhánh bổ sung đóng góp ban đầu bằng 0 |
| Attention YOLO11s | Tìm C2PSA/C3k2 từ index ≥ 9; với YOLO11s gặp C2PSA layer 10 | Kiểm tra rõ layer 10 phải là C2PSA, rồi thay bằng BiFormerBlockLite |
| Attention YOLOv8s | Điều kiện tìm C2PSA/C3k2 không bao phủ graph C2f/SPPF của YOLOv8s | Dùng SPPFWithBiFormer tại layer 9; có attention thật trong A5/A6 của YOLOv8s |
| Dựng model khi DDP | Gán patched_model trên trainer ở tiến trình cha | Dựng/patch graph trong get_model của từng worker và lưu audit |
| Focal-EIoU của A6 | Cell chạy A6 có import hàm, nhưng đoạn train đó chưa gắn hàm vào criterion | Gắn FocalEIoUBboxLoss qua init_criterion; giữ DFL |
| Chọn dữ liệu Stage2 | Có nhánh dùng hardaug nếu tìm thấy thư mục, nếu không thì fallback | A1–A6 chủ động dùng prepared dataset có hardaug và dấu hoàn tất |
| Kiểm tra output | Phụ thuộc log/đường dẫn trong notebook | Có fingerprint, checkpoint hash, module inventory, val_check và kiểm tra sau nạp |

Ví dụ với DDP: sửa một model trong bộ nhớ của notebook cha chưa đủ để chứng minh các tiến trình con dùng chính object đó. Bản mới đặt việc dựng graph vào nơi trainer thực sự tạo model cho từng tiến trình, rồi kiểm tra artifact đầu ra. Đây là sửa độ tin cậy của triển khai, không phải thêm một khối mới vào overall.

Tương tự, thay cách khởi tạo không đổi vị trí hoặc topology của RepConv. Nó đổi điểm xuất phát tối ưu: dùng lại nhánh đã học thay vì bắt các convolution vừa thay phải học lại từ đầu.

Những điểm trên giải thích vì sao phải lấy **code + checkpoint metadata + log** làm căn cứ, thay vì chỉ đọc tên cell “Full Fusion” hoặc một comment “neck attention”. Với các run lịch sử chưa có checkpoint/audit tương ứng, không nên khẳng định chính xác mọi module đã được giữ qua DDP chỉ từ source.

## 16. Vì sao chọn các phương pháp này, và không nên hứa quá mức điều gì?

| Lựa chọn | Bài toán muốn xử lý | Lý do hợp lý để thử | Hạn chế hoặc đánh đổi |
| --- | --- | --- | --- |
| CoordConv | Feature thiếu biểu diễn vị trí tuyệt đối trực tiếp | Thêm X/Y với can thiệp nhỏ ở input | Có thể học thiên lệch vị trí/bố cục |
| RepConv | Muốn cách học nhiều nhánh nhưng có khả năng gộp khi deploy | Giữ đầu vào/ra tương thích các Conv bị thay | Khi chưa deploy vẫn tốn thêm nhánh; không tự bảo đảm mAP cao hơn |
| BiFormerBlockLite | Cần trao đổi thông tin giữa các vùng xa | Route vùng trước, attention token sau | Tăng chi phí; có thể chọn thiếu ngữ cảnh; thay C2PSA làm đổi khởi tạo |
| Focal-EIoU | Muốn tín hiệu riêng cho tâm, rộng, cao của box | Thử một mục tiêu định vị khác CIoU | Loss mới có thể cần điều kiện tối ưu khác; chưa chắc AP cao hơn |
| Hard augmentation | Ảnh train hạn chế về điều kiện sáng, nhiễu, che khuất | Tạo biến thể train mà không cần nhãn mới | Ảnh tổng hợp không thay thế cảnh thật; biến đổi quá mạnh có thể làm mất thông tin |
| Giữ PAN và head ba mức | Muốn so tác động của phần thay đổi trên nền YOLO quen thuộc | Giữ đường đa tỉ lệ và đầu ra tương thích | Không phải toàn bộ head/neck đều là đóng góp mới |

Không có thí nghiệm trong bộ A0–A6 này để chứng minh những lựa chọn trên tốt hơn mọi attention khác, mọi kiểu convolution khác hoặc mọi box loss khác. Cách trình bày đúng là “nhóm lựa chọn và kiểm chứng các hướng này”, sau đó chỉ rõ kết quả thực nghiệm ủng hộ phần nào, chưa ủng hộ phần nào.

Nếu đếm thay đổi về model/loss thì có bốn nhóm chính: CoordConv, RepConv, BiFormerBlockLite, Focal-EIoU. Nếu tính cả chiến lược dữ liệu thì có thêm hard augmentation, thành năm nhóm. PAN, decoupled head, TAL và DFL là các thành phần nền được giữ lại, không phải tất cả đều do nhóm phát minh.

## 17. Cách giải thích ngắn khi đứng trước hình

> Ảnh RGB được bổ sung hai kênh tọa độ X và Y. Backbone dùng RepConv ở các bước giảm kích thước, còn C3k2 vẫn xử lý đặc trưng trong từng stage. Cuối backbone, sau SPPF, nhóm thay C2PSA bằng BiFormerBlockLite để thử đưa quan hệ giữa các vùng vào feature P5. PAN kết hợp P3, P4, P5 theo hai hướng, với hai RepConv ở nhánh đi từ phân giải cao xuống thấp. Head tách phân loại và box; ba mức dự đoán được gom lại rồi qua NMS để ra box, điểm tin cậy và lớp. Trong train, TAL gán dự đoán với nhãn thật; BCE học lớp, Focal-EIoU học box và DFL học phân bố khoảng cách cạnh.

Nếu được hỏi vì sao hình khác bản trước:

> Bản mới sửa các vị trí và ký hiệu để khớp graph thực tế: làm rõ RepConv ở backbone lẫn neck, đưa BiFormer về đúng cuối backbone, bỏ C2PSA đã bị thay, cập nhật đầu ra năm lớp và đường logits về BCE. Phần bố cục được giữ gần bản cũ. Đây không phải tự ý vẽ thêm module để làm mô hình trông phức tạp hơn.

Nếu được hỏi vì sao A6 chưa cao nhất, dùng các bảng và lập luận trong [tài liệu kết quả](C:/Users/nvtha/Downloads/shwd/BAO_CAO_GDUT_CHOT/01_KET_QUA_A1_A6.md). Không dùng lý thuyết “attention tốt hơn” để phủ nhận số đo.

## 18. Nguồn và phạm vi xác minh

### Hình và code chốt

- [Overall mới PNG](C:/Users/nvtha/Downloads/shwd/GDUT_OVERALL_OLD_STYLE/GDUT_A6_YOLO11s_Overall_Simplified.png), [SVG](C:/Users/nvtha/Downloads/shwd/GDUT_OVERALL_OLD_STYLE/GDUT_A6_YOLO11s_Overall_Simplified.svg), [caption và cách đọc](C:/Users/nvtha/Downloads/shwd/GDUT_OVERALL_OLD_STYLE/FIGURE_CAPTION.txt).
- [Hình cũ](<C:/Users/nvtha/Downloads/shwd/AAIML 2027/FIGURE_PACKAGE/rep_yolo11s_overall_ieee.png>).
- [Bộ notebook GDUT CLEAN](C:/Users/nvtha/Downloads/shwd/SUBMISSION_CLEAN/GDUT-HWD).
- [Chuẩn bị dữ liệu](C:/Users/nvtha/Downloads/shwd/SUBMISSION_CLEAN/GDUT-HWD/support/dataset_pipeline.py), [augmentation](C:/Users/nvtha/Downloads/shwd/SUBMISSION_CLEAN/GDUT-HWD/support/albumentations_hardcase_policy.py).
- [Định nghĩa các module và Focal-EIoU](C:/Users/nvtha/Downloads/shwd/SUBMISSION_CLEAN/GDUT-HWD/support/custom_ablation_modules.py).
- [Tích hợp graph, DDP và criterion](C:/Users/nvtha/Downloads/shwd/SUBMISSION_CLEAN/GDUT-HWD/support/research_trainer.py), [khởi tạo từ pretrained](C:/Users/nvtha/Downloads/shwd/SUBMISSION_CLEAN/GDUT-HWD/support/pretrained_conversion.py).
- [Train và đánh giá](C:/Users/nvtha/Downloads/shwd/SUBMISSION_CLEAN/GDUT-HWD/support/run_experiment.py), [recipe optimizer](C:/Users/nvtha/Downloads/shwd/SUBMISSION_CLEAN/GDUT-HWD/support/training_recipe.py), [kiểm tra tích hợp](C:/Users/nvtha/Downloads/shwd/SUBMISSION_CLEAN/GDUT-HWD/support/preflight.py).

### Dữ liệu và run đã đối chiếu

- [Dataset manifest đúng fingerprint](C:/Users/nvtha/Downloads/shwd/tmp/report_final_20261125/evidence/dataset_manifest.json), [augmentation manifest](C:/Users/nvtha/Downloads/shwd/tmp/report_final_20261125/evidence/augmentation_manifest.json).
- [Stage1 YOLO11s manifest](C:/Users/nvtha/Downloads/shwd/tmp/report_final_20261125/evidence/S1_yolo11s_run_manifest.json), [Stage1 YOLOv8s manifest](C:/Users/nvtha/Downloads/shwd/tmp/report_final_20261125/evidence/S1_yolov8s_run_manifest.json).
- [Artifact A0–A6 đã lưu](C:/Users/nvtha/Downloads/shwd/tmp/gdut_defense_audit_20261119), trong đó A6/research_code là source tải từ run A6 để đối chiếu triển khai.
- [Log final test](<C:/Users/nvtha/.codex/attachments/b5351aaf-9ebc-4fab-b875-accb980e7b86/Pasted text.txt>).

### Code nguồn giúp xác định khác biệt với bản cũ

- [Notebook A3 gốc](C:/Users/nvtha/Downloads/shwd/FULL_TRAIN_KAGGLE/shwd-stage2-ablation-setup-full-train-run-a3.ipynb), cell index 18: vòng thay Conv 3×3 ở backbone/neck.
- [Notebook A5 gốc](C:/Users/nvtha/Downloads/shwd/FULL_TRAIN_KAGGLE/shwd-stage2-ablation-setup-full-train-run-a5.ipynb), cell index 20: tìm C2PSA/C3k2 để thay attention.
- [Notebook A6 gốc](C:/Users/nvtha/Downloads/shwd/FULL_TRAIN_KAGGLE/shwd-stage2-ablation-setup-full-train-run-a6.ipynb), cell index 21: CoordConv stem, RepConv ngoài stem và attention.
- [YAML nền YOLO11](C:/Users/nvtha/Downloads/shwd/tmp/dataset_adaptation/vendor/ultralytics/cfg/models/11/yolo11.yaml), [YAML nền YOLOv8](C:/Users/nvtha/Downloads/shwd/tmp/dataset_adaptation/vendor/ultralytics/cfg/models/v8/yolov8.yaml), [Detect head](C:/Users/nvtha/Downloads/shwd/tmp/dataset_adaptation/vendor/ultralytics/nn/modules/head.py), [loss của thư viện](C:/Users/nvtha/Downloads/shwd/tmp/dataset_adaptation/vendor/ultralytics/utils/loss.py), trong bản Ultralytics 8.4.115 được lưu local.

Các cell index ở trên bắt đầu từ 0 trong file notebook, không phải số execution count hiển thị sau khi chạy. Các đường dẫn nguồn phục vụ truy vết trên máy hiện tại; nếu chuyển riêng hai file MD sang máy khác, cần chuyển kèm nguồn/hình hoặc điều chỉnh liên kết.
