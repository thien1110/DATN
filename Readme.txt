# Nghiên cứu và phát triển khung học sâu đa biến có khả năng giải thích trong dự báo lưu lượng bệnh nhân và công suất sử dụng giường bệnh tại bệnh viện công

Kho lưu trữ này chứa toàn bộ mã nguồn thực nghiệm, kịch bản truy vấn dữ liệu và biểu đồ phân tích của chuyên đề tốt nghiệp Thạc sĩ ngành Khoa học máy tính:
* **Học viên thực hiện:** Nguyễn Hữu Thiên (Lớp K30MCS)
* **Giảng viên hướng dẫn:** TS. Nguyễn Đức Mận
* **Cơ sở thực nghiệm:** Bệnh viện C Đà Nẵng (Dữ liệu HIS từ 01/10/2023 đến 31/07/2026)

---

## 📂 Cấu trúc thư mục dự án


DATN/
├── figures/                    # Thư mục lưu trữ tự động toàn bộ Hình 1 – Hình 9 (300 DPI)
├── T-SQL.txt                   # Kịch bản T-SQL trích xuất và ẩn danh hóa dữ liệu HIS
├── raw_his_data.csv            # Dữ liệu bệnh án nội trú thô sau khi trích xuất và băm SHA-256
├── features.csv                # Bảng chuỗi thời gian theo ngày (tự động sinh từ pipeline)
├── removed_cases_log.csv       # Nhật ký các hồ sơ ảo (< 15 phút) bị loại bỏ
├── run_pipeline.py             # Kịch bản Python thực thi toàn bộ quy trình từ đầu đến cuối
├── Requirements.txt            # Danh sách cố định phiên bản các thư viện phụ thuộc
└── Readme.TXT                   # Hướng dẫn cài đặt và chạy thực nghiệm

⚙️ Hướng dẫn cài đặt và tái lập kết quả
Để đảm bảo tính tái lập (Reproducibility) với hạt giống ngẫu nhiên cố định (seed = 42), hãy thực hiện theo các bước sau trên giao diện dòng lệnh (Terminal / Command Prompt):

Bước 1: Cài đặt môi trường Python và các thư viện phụ thuộc
Khuyến nghị sử dụng Python 3.10.x. Chạy lệnh sau để cài đặt các gói phần mềm có phiên bản cố định:

	pip install -r Requirements.txt

Bước 2: Thực thi kịch bản Pipeline khép kín
Chạy tệp điều phối chính để hệ thống tự động làm sạch dữ liệu, trích xuất đặc trưng, huấn luyện LSTM, ARIMA, XGBoost Surrogate, chạy TreeSHAP và xuất trọn bộ Hình 1 đến Hình 9 vào thư mục figures/:

	python run_pipeline.py



📊 Tóm tắt kết quả thực nghiệm chính
Hiệu năng dự báo LSTM (Tập kiểm thử độc lập 200 ngày / 193 mẫu chuỗi):
	Sai số tuyệt đối trung bình: MAE = 38.45 giường   
	Sai số toàn phương trung bình: RMSE = 47.13 giường   
	Vượt trội hơn đáng kể so với mô hình cơ sở ARIMA(7,1,1) (MAE = 103.14 giường).   
Độ trung thành của Mô hình đại diện (Global Surrogate Model):
	Hệ số xác định trên tập kiểm thử độc lập: R^2 = 0.9750.   
Các yếu tố chi phối mạnh nhất theo TreeSHAP (tính theo đơn vị giường bệnh):
	Quán tính công suất lưu trú (Lag_1: 40.58 giường, Rolling_Mean_7: 18.03 giường).   
	Áp lực bệnh nhân cấp cứu (Emergency_Count: 17.71 giường).   
	Lưu lượng xuất viện (Discharges: 8.08 giường).
	Số bệnh nhân cao tuổi nhập viện (Total_Elderly_Admitted: 7.43 giường).
	Chu kỳ hành chính tuần (DayOfWeek: 6.91 giường).