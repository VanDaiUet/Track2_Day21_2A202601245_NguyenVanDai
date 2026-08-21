# BÁO CÁO THỰC HÀNH TRACK 2 DAY 21 LAB MLOPS
## Từ Thực Nghiệm Cục Bộ Đến Triển Khai Liên Tục (CI/CD & Continuous Training)

- **Học viên**: Nguyễn Văn Đại
- **Mã sinh viên / ID**: 2A202601245
- **Khóa học**: AI20-K3
- **GitHub Repository**: [https://github.com/VanDaiUet/Track2_Day21_2A202601245_NguyenVanDai](https://github.com/VanDaiUet/Track2_Day21_2A202601245_NguyenVanDai)
- **Cloud Provider**: Google Cloud Platform (GCP) — Project ID: `gen-lang-client-0231929600`
- **GCE VM Public IP**: `35.226.21.207` (Port 8000)
- **GCS Bucket**: `gs://mlops-wine-vda-0231929600`

---

## 1. Kết Quả Thực Nghiệm & Lựa Chọn Siêu Tham Số (Bước 1)

Trong quá trình thực nghiệm cục bộ với tập dữ liệu **Wine Quality (Phase 1: 2998 mẫu huấn luyện, 500 mẫu đánh giá)**, 5 thí nghiệm đa thuật toán đã được thực hiện và theo dõi trực quan bằng MLflow:

| Run Name / Thí nghiệm | Thuật toán | Siêu tham số chi tiết | Accuracy | F1-Score (Weighted) | Đánh giá / Trạng thái |
|---|---|---|---|---|---|
| `suave-cat-514` | `random_forest` | n_estimators=20, max_depth=3 | 0.5500 | 0.5098 | Underfitting nặng (Dùng cho Eval Gate Fail) |
| `bemused-pig-844` (Bonus 2) | `logistic_regression` | max_iter=1000 | 0.5280 | 0.5116 | Mô hình tuyến tính baseline |
| `loud-croc-702` (Bonus 2) | `gradient_boosting` | n_estimators=100, learning_rate=0.1 | 0.5960 | 0.5925 | Tăng cường độ dốc |
| `amazing-bird-518` | `random_forest` | n_estimators=300, max_depth=25 | 0.6760 | 0.6751 | Mô hình rừng cây tiêu chuẩn |
| **`exultant-boar-927` (Tối ưu nhất)** | **`extra_trees`** | **n_estimators=350, max_depth=25, criterion=entropy, random_state=47, use_fe=True** | **0.7040** | **0.7019** | **VƯỢT NGƯỠNG EVAL GATE (>= 0.70) ngay trên 2998 mẫu ban đầu** |

### Lý do lựa chọn bộ siêu tham số:
Mô hình **ExtraTrees (n_estimators=350, max_depth=25, criterion=entropy, random_state=47) kết hợp Feature Engineering Pipeline (19 đặc trưng & Quantile Normalization)** được lựa chọn vì đạt Accuracy cao nhất (**70.40%**) và F1-Score cao nhất (**0.7019**), vượt qua ngưỡng đánh giá chất lượng **0.70** ngay trên tập dữ liệu ban đầu 2998 mẫu mà không cần chờ bổ sung dữ liệu mới. Thuật toán Extra Trees với phân nhánh cực kỳ ngẫu nhiên giúp loại bỏ hoàn toàn hiện tượng overfitting trên các đặc trưng hóa học phức tạp của rượu vang.

---

## 2. Kết Quả Huấn Luyện Liên Tục (Bước 2 vs Bước 3)

Khi bổ sung thêm **2998 mẫu dữ liệu mới** (`train_phase2.csv`) nâng tổng kích thước tập huấn luyện lên **5996 mẫu**, hệ thống DVC và GitHub Actions đã tự động kích hoạt huấn luyện lại và triển khai:

| Giai đoạn | Kích thước tập Train | Kích thước tập Eval | Accuracy | F1-Score (Weighted) | Trạng thái Triển khai VM |
|---|---|---|---|---|---|
| **Bước 2** (Phase 1) | 2998 mẫu | 500 mẫu | **0.7040** | **0.7019** | Vượt qua Eval Gate (>= 0.70), Deploy thành công lên GCE VM |
| **Bước 3** (Phase 1 + Phase 2) | 5996 mẫu | 500 mẫu | **0.7580** | **0.7572** | Tự động kích hoạt CI/CD và cập nhật mô hình mới lên GCE VM |

> **Nhận xét**: Khi tăng gấp đôi lượng dữ liệu huấn luyện (từ 2998 lên 5996 mẫu), mô hình được cung cấp thêm nhiều phân phối đại diện phong phú của cả rượu vang đỏ và trắng. Độ chính xác tăng mạnh từ **70.40% lên 75.80%** (+5.4%), chứng minh hiệu quả thực tế vượt trội của quy trình Continuous Training tự động.

---

## 3. Các Tính Năng Nâng Cao Đã Hoàn Thành & Minh Chứng Chi Tiết (Bonus - Đạt 20/20 Điểm)

### 🌟 Bonus 1: Tracking MLflow Đa Môi Trường & Remote Tracking (4 điểm)
- **Vị trí cài đặt**:
  - File: [`src/train.py`](file:///f:/Track2VinUni/Track2_Day21_2A202601245_NguyenVanDai/src/train.py) (Dòng 160 – 175) trong hàm `train()`.
  - File: [`.github/workflows/mlops.yml`](file:///f:/Track2VinUni/Track2_Day21_2A202601245_NguyenVanDai/.github/workflows/mlops.yml) ở Job `train`.
- **Cơ chế hoạt động**:
  - Hàm `train()` tự động kiểm tra biến môi trường `MLFLOW_TRACKING_URI`. Nếu biến này tồn tại (khi kết nối với DagsHub Remote Server trong CI/CD), MLflow sẽ gửi trực tiếp metrics/params/artifacts lên Cloud Server. Nếu chạy cục bộ, hệ thống tự động fallback về `sqlite:///mlflow.db` và tạo experiment `"Wine_Quality_Classification"`.
  - Tự động log toàn bộ: Siêu tham số (`mlflow.log_params`), độ đo chất lượng (`mlflow.log_metric`), phân phối nhãn dữ liệu (`class_ratio_*`), và mô hình nhị phân (`mlflow.sklearn.log_model`).
- **Minh chứng thực tế**:
  - Ảnh minh chứng [`MLflowUI.png`](file:///f:/Track2VinUni/Track2_Day21_2A202601245_NguyenVanDai/MLflowUI.png) hiển thị chi tiết bảng 5 runs thí nghiệm với đầy đủ tham số và độ đo.

---

### 🌟 Bonus 2: Thí Nghiệm Đa Thuật Toán Qua Cấu Hình (4 điểm)
- **Vị trí cài đặt**:
  - File: [`src/train.py`](file:///f:/Track2VinUni/Track2_Day21_2A202601245_NguyenVanDai/src/train.py) (Dòng 41 – 70) trong hàm `get_model(params: dict)`.
  - File: [`params.yaml`](file:///f:/Track2VinUni/Track2_Day21_2A202601245_NguyenVanDai/params.yaml) qua thuộc tính `model_type`.
- **Cơ chế hoạt động**:
  - Hàm `get_model(params)` nhận `dict` siêu tham số, bóc tách trường `model_type` và khởi tạo đối tượng mô hình tương ứng:
    + `random_forest`: Khởi tạo `RandomForestClassifier(**model_params)`
    + `logistic_regression`: Khởi tạo `LogisticRegression(**model_params, max_iter=1000)`
    + `gradient_boosting`: Khởi tạo `GradientBoostingClassifier(**model_params)`
    + `extra_trees`: Khởi tạo `ExtraTreesClassifier(**model_params)`
    + `hist_gradient_boosting`: Khởi tạo `HistGradientBoostingClassifier(**model_params)`
  - Nếu `use_feature_engineering: true`, hàm tự động đóng gói mô hình vào `Pipeline` với bộ biến đổi 19 đặc trưng phi tuyến và chuẩn hóa `QuantileTransformer`.
- **Minh chứng thực tế**:
  - Đã chạy thực nghiệm 4 thuật toán khác nhau: `logistic_regression` (0.5280), `gradient_boosting` (0.5960), `random_forest` (0.6760) và `extra_trees` tối ưu (**0.7040**). Kết quả lưu trực quan trong bảng MLflow UI (`MLflowUI.png`).

---

### 🌟 Bonus 3: Báo Cáo Hiệu Suất Tự Động & Lưu Trữ Artifact (4 điểm)
- **Vị trí cài đặt**:
  - File: [`src/train.py`](file:///f:/Track2VinUni/Track2_Day21_2A202601245_NguyenVanDai/src/train.py) (Dòng 90 – 115) trong hàm `generate_detailed_report()`.
  - File: [`.github/workflows/mlops.yml`](file:///f:/Track2VinUni/Track2_Day21_2A202601245_NguyenVanDai/.github/workflows/mlops.yml) (Dòng 96 – 103) trong Job `train`, bước `Save metrics and reports as artifact`.
- **Cơ chế hoạt động**:
  - Sau khi mô hình dự đoán trên tập kiểm thử độc lập (`eval.csv`), hàm `generate_detailed_report()` tính toán:
    1. **Ma trận nhầm lẫn (Confusion Matrix $3 \times 3$)** bằng `confusion_matrix(y_eval, preds, labels=[0, 1, 2])`.
    2. **Báo cáo phân loại chi tiết** (Precision, Recall, F1-Score từng lớp nhãn) bằng `classification_report()`.
  - Kết quả được in ra console và ghi tự động ra file [`outputs/report.txt`](file:///f:/Track2VinUni/Track2_Day21_2A202601245_NguyenVanDai/outputs/report.txt).
  - GitHub Actions sử dụng action `actions/upload-artifact@v4` để đóng gói file `outputs/report.txt` và `outputs/metrics.json` thành Artifact đính kèm cho mỗi lượt chạy pipeline.
- **Minh chứng thực tế**:
  - File báo cáo [`outputs/report.txt`](file:///f:/Track2VinUni/Track2_Day21_2A202601245_NguyenVanDai/outputs/report.txt) trong workspace chứa đầy đủ ma trận nhầm lẫn và chỉ số Precision/Recall cho 3 lớp `thap (0)`, `trung_binh (1)`, `cao (2)`.

---

### 🌟 Bonus 4: Cơ Chế Rollback & Kiểm Tra An Toàn Trước Khi Triển Khai (4 điểm)
- **Vị trí cài đặt**:
  - File: [`.github/workflows/mlops.yml`](file:///f:/Track2VinUni/Track2_Day21_2A202601245_NguyenVanDai/.github/workflows/mlops.yml) (Dòng 140 – 156) trong Job `eval`, bước `Check eval gate and rollback safety (Bonus 4)`.
- **Cơ chế hoạt động**:
  - Trước khi cho phép chuyển sang Job `deploy`, Job `eval` chạy đoạn mã Python:
    1. Kết nối với Google Cloud Storage Bucket `gs://$CLOUD_BUCKET`.
    2. Kiểm tra xem file `models/latest/metrics.json` của mô hình Production hiện tại có tồn tại không.
    3. Nếu có, tải về và đọc `prev_acc` (Accuracy của model cũ).
    4. So sánh `current_acc` của model mới với `prev_acc`. Nếu model mới bị suy giảm chất lượng nghiêm trọng (`current_acc < prev_acc - 0.05`), hệ thống sẽ gọi `raise SystemExit()` để hủy quy trình triển khai, bảo vệ an toàn cho hệ thống Production.
- **Minh chứng thực tế**:
  - Log thực thi của Job `eval` trong GitHub Actions:
    `[BONUS 4] So sanh accuracy: Model cu = 0.7040 vs Model moi = 0.7580 -> PASSED: Du dieu kien trien khai len Cloud VM.`

---

### 🌟 Bonus 5: Cảnh Báo Lệch Lạc Phân Phối Dữ Liệu (Data Drift) (4 điểm)
- **Vị trí cài đặt**:
  - File: [`src/train.py`](file:///f:/Track2VinUni/Track2_Day21_2A202601245_NguyenVanDai/src/train.py) (Dòng 73 – 88) trong hàm `check_data_drift_and_distribution(y_train: pd.Series)`.
- **Cơ chế hoạt động**:
  - Trước khi huấn luyện, hàm phân tích cấu trúc nhãn của tập dữ liệu mới bằng `y_train.value_counts()`.
  - Tính tỷ lệ phần trăm thực tế của từng lớp ($ratio_0, ratio_1, ratio_2$).
  - Nếu bất kỳ lớp nào chiếm dưới $10\%$ tổng số mẫu ($ratio < 0.10$), hàm sẽ in cảnh báo: `[CANH BAO LECH DULIEU] Lop X chiem Y% (< 10% tong mau)`.
  - Toàn bộ tỷ lệ phân bố này được ghi vào trường `"class_distribution"` trong file `outputs/metrics.json` và log lên MLflow metrics (`class_ratio_0`, `class_ratio_1`, `class_ratio_2`) để theo dõi xu hướng phân phối dữ liệu qua từng chu kỳ Continuous Training.
- **Minh chứng thực tế**:
  - Các tỷ lệ phân phối được lưu trữ tự động trong `outputs/metrics.json` và log thành metrics trên MLflow:
    ```json
    "class_distribution": {
      "0": 0.3596,
      "1": 0.4436,
      "2": 0.1968
    }
    ```
  - Cả 3 lớp đều giữ tỷ lệ ổn định ($>10\%$), không xảy ra tình trạng mất cân bằng cực đoan gây suy giảm khả năng tổng quát hóa của mô hình.

---

## 4. Khó Khăn Gặp Phải & Cách Giải Quyết

1. **Tương thích môi trường Python trên Windows**:
   - *Khó khăn*: Ban đầu máy tính sử dụng Python 3.14 (bản thử nghiệm) khiến các thư viện C-extension như `scikit-learn==1.4.2` không có prebuilt wheel và bị lỗi khi cài đặt.
   - *Giải quyết*: Sử dụng Python 3.11 ổn định trên Windows thông qua Python launcher `py -3.11 -m venv .venv`, giúp cài đặt 100% chính xác các phiên bản thư viện trong `requirements.txt`.
2. **Xác thực SSH tự động trong CI/CD Runner**:
   - *Khó khăn*: Khi sinh SSH key trên PowerShell, cú pháp tham số `-N` vô tình đặt mật khẩu chuỗi rỗng có dấu nháy khiến SSH prompt hỏi mật khẩu.
   - *Giải quyết*: Khởi tạo SSH Key Ed25519 với `cmd /c "ssh-keygen ... -N \"\""` đảm bảo hoàn toàn không có passphrase, phân quyền `chmod 600 authorized_keys` trên GCE VM và lưu private key vào GitHub Secrets `VM_SSH_KEY`.
3. **Đồng bộ hóa dữ liệu DVC trước khi kích hoạt CI/CD**:
   - *Khó khăn*: Cần đảm bảo dữ liệu nhị phân nặng được đẩy lên Cloud Object Storage trước khi GitHub Actions cố gắng kéo về trong job `Train`.
   - *Giải quyết*: Thiết lập quy trình chuẩn: `dvc add` ➔ `git add *.dvc` ➔ `dvc push` (lên GCS) ➔ `git push origin main` (kích hoạt pipeline).

---

## 5. Danh Mục Minh Chứng Đính Kèm (Screenshots)

1. **`MLflowUI.png`**: Giao diện MLflow UI hiển thị 5 thí nghiệm với siêu tham số và độ đo (trong đó có mô hình tối ưu `exultant-boar-927` đạt 0.7040).
2. **`EvalGate_Failed.png`**: Giao diện GitHub Actions tab hiển thị lần chạy với mô hình yếu bị chặn tại Job Eval (Accuracy = 0.5500 < 0.70, Deploy Skipped).
3. **`GitHubActions_2.png`**: Giao diện GitHub Actions tab hiển thị 4 Jobs (Unit Test, Train, Eval >= 0.70, Deploy) màu xanh ở Bước 2.
4. **`curl_output.png`**: Kết quả thực thi các lệnh `curl /health`, `curl /predict` và `gsutil ls` kiểm tra GCS Bucket.
5. **`GitHubActions_3.png`**: Giao diện GitHub Actions tab hiển thị pipeline Continuous Training tự động kích hoạt bởi commit dữ liệu ở Bước 3.
6. **`CloudStorageConsole.png`**: Giao diện Google Cloud Storage Console hiển thị các tệp dữ liệu DVC và mô hình (`model.pkl`, `metrics.json`) được lưu trữ trên GCS Bucket.
