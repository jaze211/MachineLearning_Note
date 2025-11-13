Tiền xử lý dữ liệu
    xử lý khuyết thiếu
    mã hóa dữ liệu (encoding)
    chuẩn hóa dữ liệu (normalization/scaling)
    thống kê mô tả
    trực quan hóa dữ liệu
        biểu đồ histogram, scatter, boxplot
    phân tích tương quan 

Khám phá & trực quan hóa dữ liệu (EDA - Exploratory Data Analysis)
    Mục tiêu
        Hiểu cấu trúc, phân phối, mối quan hệ và bất thường trong dữ liệu.
    Checklist kỹ thuật
        Kiểm tra kích thước, kiểu dữ liệu.
        Thống kê mô tả (mean, median, std).
        Xác định outlier.
        Phân tích tương quan giữa các biến.
        Trực quan hóa (hist, boxplot, heatmap).
    Công cụ/Thư viện
        pandas, matplotlib, seaborn, ydata-profiling

Huấn luyện mô hình (Model Training)
    Mục tiêu
        Xây dựng mô hình ML phù hợp và huấn luyện trên dữ liệu đã xử lý.
    Checklist kỹ thuật
        Chọn thuật toán phù hợp (SVM, Random Forest, XGBoost, v.v.).
        Thiết lập pipeline huấn luyện.
        Chạy huấn luyện và lưu model.
        Theo dõi loss/metric.
        Kiểm tra overfitting/underfitting.
    Công cụ/Thư viện
        scikit-learn, xgboost, lightgbm, tensorflow, pytorch

Fine-tuning mô hình (Model Optimization)
    Mục tiêu
        Tối ưu siêu tham số để đạt hiệu năng cao nhất.
    Checklist kỹ thuật
        Chọn tham số cần tối ưu.
        Dùng GridSearchCV / RandomizedSearchCV / Optuna.
        Đánh giá bằng cross-validation.
        Theo dõi metric trung bình và độ lệch chuẩn.
        Lưu kết quả tối ưu.
    Công cụ/Thư viện    
        scikit-learn, optuna, ray[tune]        

Vận hành & bảo dưỡng (Deployment & Maintenance)
    Mục tiêu
        Đưa mô hình vào môi trường sản xuất, giám sát và cập nhật định kỳ.
        đóng hói mô tình 
        triển khai 
        giám sát 
        drift (trôi dạt dữ liệu)
        hiệu suất
        tái huấn luyện 
    Checklist kỹ thuật
        Lưu model (joblib, onnx, pickle).
        Triển khai API bằng FastAPI hoặc Flask.
        Theo dõi performance thực tế (drift detection).
        Cập nhật model định kỳ.
        Quản lý version và logs.
        Bảo mật và kiểm soát truy cập.
    Công cụ/Thư viện
        FastAPI, MLflow, Docker, Prometheus, Grafana


from sklearn import datasets
digits = datasets.load_digits()
feature = digits.data
target = digits.target
feature[0]
