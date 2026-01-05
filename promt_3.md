Dưới đây là tập **prompts sẵn dùng** (bằng tiếng Việt, kỹ thuật, dễ dán vào ChatGPT/LLM khác) để bạn **lấy toàn bộ kiến thức, tài liệu, bài tập, slide, code, quiz** cho buổi học máy theo từng bước bạn liệt kê. Mỗi prompt có mục đích rõ ràng — chọn/ghép tùy nhu cầu.

---

## 1) Prompt tổng quát — tóm tắt buổi học (toàn cảnh)

```
Bạn là giảng viên học máy. Tóm tắt toàn cảnh một buổi học gồm các phần: 
1) Xem xét yêu cầu bài toán, 2) Lấy dữ liệu, 3) Khám phá & trực quan hóa, 4) Tiền xử lý, 5) Huấn luyện model, 6) Fine-tune, 7) Vận hành & bảo dưỡng. 
Cho mỗi phần: mục tiêu, checklist kỹ thuật (5–8 bước), công cụ/thư viện gợi ý (Python), ví dụ lệnh/đoạn code ngắn, và tài liệu tham khảo (sách/bài báo/nguồn trực tuyến).
Trả lời dạng mục lục có tiêu đề và phần mô tả ngắn cho từng bước.
```

---

## 2) Prompt đào sâu từng phần — ví dụ: Khám phá & trực quan hóa

```
Bạn là chuyên gia Data Science. Viết hướng dẫn chi tiết cho phần "Khám phá và trực quan hóa dữ liệu" trong một buổi lab 90 phút. Bao gồm:
- Mục tiêu học phần,
- Các bước thực hành (đọc dữ liệu, kiểm tra missing, thống kê mô tả, phân phối, correlation, outliers, time series decomposition nếu có),
- Code Python cụ thể (pandas + matplotlib/seaborn + plotly), mỗi bước 6–10 dòng code minh họa,
- Các câu hỏi thảo luận để đánh giá hiểu biết (5 câu),
- Kết luận & những "red flags" cần chú ý khi EDA.
```

---

## 3) Prompt cho bài lab lấy dữ liệu (hands-on)

```
Tạo một bài lab thực hành "Lấy dữ liệu" dài ~60 phút cho sinh viên: 
- Mục tiêu: thu thập dữ liệu từ (1) file CSV, (2) API REST (JSON paginated), (3) cơ sở dữ liệu PostgreSQL; 
- Yêu cầu bài tập: viết script Python để tải/stream dữ liệu, xử lý pagination, và lưu vào Parquet; 
- Cung cấp template code cho mỗi nguồn (requests + pandas, sqlalchemy), và test cases kiểm tra dữ liệu hợp lệ (schema checks). 
Kèm checklist đánh giá và gợi ý lỗi thường gặp.
```

---

## 4) Prompt tiền xử lý dữ liệu (data preprocessing recipes)

```
Liệt kê các bước tiền xử lý phổ biến cho supervised learning (numerical + categorical + datetime + text). Với mỗi loại dữ liệu, cho:
- Các transform cụ thể (imputation, scaling, encoding, feature engineering),
- Ví dụ code sklearn/CategoryEncoders/Pandas (pipeline),
- Khi nào dùng RobustScaler vs StandardScaler vs MinMax,
- Các chỉ số kiểm thử (leakage checks, label leakage detection).
Trả lời kỹ thuật, có ví dụ code.
```

---

## 5) Prompt huấn luyện model (model training pipeline)

```
Mô tả pipeline huấn luyện model cho classification/regression: từ train/val/test split, cross-validation strategy, metric selection, baseline models, hyperparameter tuning (GridSearchCV vs Randomized vs Bayesian), cách log experiments (MLflow), và lưu model (joblib/torch.save). Cho ví dụ code scikit-learn cho RandomForest + example for PyTorch training loop (skeleton).
```

---

## 6) Prompt fine-tune (transfer learning / hyperparam)

```
Viết hướng dẫn fine-tuning model theo hai kịch bản: (A) fine-tune pretrained CNN (PyTorch) cho image classification, (B) fine-tune transformer (Hugging Face) cho NLP classification. Cho checklist: freeze layers, lr scheduling, discriminative lr, gradual unfreezing, early stopping, mixed precision. Kèm đoạn code mẫu ngắn cho mỗi kịch bản.
```

---

## 7) Prompt vận hành & bảo dưỡng model (MLOps)

```
Chuẩn bị checklist triển khai model production: containerization (Dockerfile), serving (FastAPI + Uvicorn / TorchServe), CI/CD (GitHub Actions), monitoring (prometheus + grafana), model versioning (DVC/MLflow), data drift detection, periodic retraining schedule. Cho architecture diagram mô tả (dạng text), và các lệnh mẫu để build & deploy container.
```

---

## 8) Prompt tạo slide/giáo án cho 60–90 phút

```
Tạo outline slide 20–25 slide cho buổi học kết hợp lý thuyết & lab theo các phần: problem framing, data acquisition, EDA, preprocessing, model training, fine-tune, deployment. Mỗi slide: tiêu đề + 1–2 bullet points nội dung + 1 hình minh hoạ gợi ý (ví dụ: confusion matrix, pipeline diagram). Trả về định dạng markdown phù hợp để dán vào slide tool.
```

---

## 9) Prompt sinh câu hỏi kiểm tra / quiz

```
Tạo 15 câu hỏi trắc nghiệm + 5 câu tự luận cho chủ đề buổi học (từ problem framing tới deployment). Trắc nghiệm có 4 lựa chọn, đánh dấu đáp án đúng và giải thích ngắn (1–2 câu). Tự luận yêu cầu viết code hoặc phân tích kết quả EDA.
```

---

## 10) Prompt flashcards / ôn nhanh

```
Sinh bộ 30 flashcards (Q/A) tóm tắt các khái niệm chính: bias-variance, cross-validation, overfitting, regularization, imputation, one-hot vs target encoding, AUROC vs AUPRC, calibration, CI/CD, canary deploy, data drift, etc. Trả về JSON array: [{"q":"...","a":"..."}].
```

---

## 11) Prompt code template end-to-end (skeleton)

```
Sinh một skeleton repo README + file Python chính (notebook or script) cho pipeline end-to-end: data_ingest.py, eda.ipynb, preprocess.py, train.py, evaluate.py, serve.py. Mỗi file có header comment mô tả inputs/outputs và sample CLI args. Dùng sklearn + pandas. Trả về nội dung file README và các snippet code mẫu.
```

---

## 12) Prompt cho mentor review (nhận xét writeup / code)

```
Bạn là mentor. Đọc đoạn writeup hoặc code (paste nội dung). Cho feedback kỹ thuật: điểm mạnh, lỗi/anti-pattern, cách tối ưu hoá (3–5 gợi ý), và checklist cải thiện trước khi nộp. Yêu cầu trả lời ngắn gọn, tập trung vào reproducibility, metric selection, và data leakage.
```

---

### Hướng dẫn sử dụng

* Dán trực tiếp một prompt vào ChatGPT để nhận output chi tiết.
* Muốn có kết quả bằng **Tiếng Anh**? Thêm `Please answer in English.` vào cuối prompt.
* Muốn code sẵn chạy được? Thêm `Provide runnable Python code with requirements.txt and small synthetic dataset generation.`

---

Nếu bạn muốn, mình có thể:

* Tạo ngay **slide outline** (prompt #8) cho buổi học bạn mô tả; hoặc
* Sinh **bộ quiz 20 câu** và file `eda.ipynb` mẫu.

Bạn muốn mình chạy prompt nào trước (ví dụ: tạo slide hay lab)?
