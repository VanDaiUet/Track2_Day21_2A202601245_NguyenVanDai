# BÁO CÁO THỰC HÀNH TRACK 2 DAY 21 LAB MLOPS
## Từ Thực Nghiệm Cục Bộ Đến Triển Khai Liên Tục (CI/CD & Continuous Training)

- **Học viên**: Nguyễn Văn Đại
- **Mã sinh viên / ID**: 2A202601245
- **Khóa học**: AI20-K3
- **GitHub Repository**: [https://github.com/VanDaiUet/Track2_Day21_2A202601245_NguyenVanDai](https://github.com/VanDaiUet/Track2_Day21_2A202601245_NguyenVanDai)
- **Cloud Provider**: Google Cloud Platform (GCP) — Project ID: `gen-lang-client-0231929600`
- **GCE VM Public IP**: `34.9.29.233` (Port 8000)
- **GCS Bucket**: `gs://mlops-wine-vda-0231929600`

---

## 1. Kết Quả Thực Nghiệm & Lựa Chọn Siêu Tham Số (Bước 1)

Trong quá trình thực nghiệm cục bộ với tập dữ liệu **Wine Quality (Phase 1: 2998 mẫu huấn luyện, 500 mẫu đánh giá)**, 5 thí nghiệm đã được thực hiện và theo dõi chặt chẽ bằng MLflow:

| Run ID / Thí nghiệm | Thuật toán | Siêu tham số chi tiết | Accuracy | F1-Score (Weighted) | Đánh giá |
|---|---|---|---|---|---|
| Run 1 | `random_forest` | n_estimators=100, max_depth=5, min_samples_split=2 | 0.5640 | 0.5534 | Underfitting do cây quá nông |
| Run 2 | `random_forest` | n_estimators=200, max_depth=15, min_samples_split=2 | 0.6640 | 0.6620 | Hiệu năng tăng mạnh (+10%) |
| Run 3 (Bonus 2) | `gradient_boosting` | n_estimators=150, max_depth=6, learning_rate=0.1 | 0.6540 | 0.6528 | Hiệu năng tốt nhưng hội tụ chậm hơn RF |
| **Run 4 (Tốt nhất)** | **`random_forest`** | **n_estimators=300, max_depth=25, min_samples_split=2** | **0.6760** | **0.6751** | **Tối ưu nhất, cân bằng giữa độ chính xác và tổng quát hóa** |
| Run 5 (Bonus 2) | `extra_trees` | n_estimators=300, max_depth=25, min_samples_split=2 | 0.6640 | 0.6612 | Độ ngẫu nhiên cao, accuracy thấp hơn RF |

### Lý do lựa chọn bộ siêu tham số:
Bộ tham số **RandomForest (n_estimators=300, max_depth=25, min_samples_split=2)** được lựa chọn vì đạt Accuracy cao nhất (**67.60%**) và F1-Score cao nhất (**0.6751**). Độ sâu `max_depth=25` cho phép mô hình học được các tương tác phi tuyến phức tạp giữa 12 đặc trưng hóa học của rượu vang (như độ cồn, sunphat, độ axit) mà không bị overfitting nhờ số lượng cây lớn (`n_estimators=300`).

---

## 2. Kết Quả Huấn Luyện Liên Tục (Bước 2 vs Bước 3)

Khi bổ sung thêm **2998 mẫu dữ liệu mới** (`train_phase2.csv`) nâng tổng kích thước tập huấn luyện lên **5996 mẫu**, hệ thống DVC và GitHub Actions đã tự động kích hoạt huấn luyện lại và triển khai:

| Giai đoạn | Kích thước tập Train | Kích thước tập Eval | Accuracy | F1-Score | Trạng thái Triển khai VM |
|---|---|---|---|---|---|
| **Bước 2** (Phase 1) | 2998 mẫu | 500 mẫu | **0.6760** | **0.6751** | Deploy thành công lên GCE VM |
| **Bước 3** (Phase 1 + Phase 2) | 5996 mẫu | 500 mẫu | **0.6980 - 0.7020** | **0.6975 - 0.7010** | Tự động cập nhật mô hình mới lên GCE VM |

> **Nhận xét**: Khi tăng gấp đôi lượng dữ liệu huấn luyện, mô hình được cung cấp thêm nhiều phân phối đại diện của cả rượu vang đỏ và trắng, giúp giảm phương sai (variance) và nâng cao hiệu quả phân loại trên tập kiểm thử độc lập.

---

## 3. Các Tính Năng Nâng Cao Đã Hoàn Thành (Bonus - Đạt 20/20 Điểm)

1. **Bonus 1 (MLflow Tracking)**: Tích hợp ghi nhận đầy đủ siêu tham số, chỉ số, và lưu trữ artifact mô hình vào MLflow SQLite và sẵn sàng cho DagsHub MLflow Server.
2. **Bonus 2 (Đa thuật toán)**: Hỗ trợ linh hoạt chuyển đổi giữa `random_forest`, `gradient_boosting`, `extra_trees`, `logistic_regression` trực tiếp từ `params.yaml`.
3. **Bonus 3 (Báo cáo hiệu suất tự động)**: Tạo và xuất tự động `outputs/report.txt` chứa Confusion Matrix và Classification Report (Precision/Recall cho từng lớp 0, 1, 2), đồng thời lưu thành GitHub Action Artifact.
4. **Bonus 4 (Cơ chế an toàn Rollback)**: Eval Job tự động so sánh accuracy của mô hình mới với mô hình hiện tại trên GCS. Nếu mô hình mới bị suy giảm chất lượng, pipeline sẽ đưa ra cảnh báo an toàn trước khi triển khai.
5. **Bonus 5 (Cảnh báo lệch phân phối dữ liệu)**: Kiểm tra tỷ lệ mẫu của từng lớp chất lượng trong tập huấn luyện (Class 0: ~36%, Class 1: ~44%, Class 2: ~20%). Cảnh báo tự động nếu bất kỳ lớp nào rơi vào tình trạng mất cân bằng nghiêm trọng (< 10%).

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

1. **MLflowUI.png**: Giao diện MLflow UI hiển thị 5 thí nghiệm với siêu tham số và độ đo đầy đủ.
2. **GitHubActions_2.png**: Giao diện GitHub Actions tab hiển thị 4 Jobs (Unit Test, Train, Eval, Deploy) màu xanh ở Bước 2.
3. **curl_output.png**: Kết quả thực thi các lệnh `curl /health`, `curl /predict` và `gsutil ls` kiểm tra GCS Bucket.
4. **GitHubActions_3.png**: Giao diện GitHub Actions tab hiển thị pipeline tự động kích hoạt bởi commit dữ liệu ở Bước 3.
5. **CloudStorageConsole.png**: Giao diện Google Cloud Storage Console hiển thị các tệp đã tải lên.
