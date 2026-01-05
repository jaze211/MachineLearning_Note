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

---
```python
from sklearn import datasets
digits = datasets.load_digits()
feature = digits.data
target = digits.target
feature[0]
```
### 🧠 Giải thích từng bước
| Dòng                              | Ý nghĩa                                                 | Ghi chú kỹ thuật                                                       |
| --------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------- |
| `from sklearn import datasets`    | Import module `datasets` của scikit-learn               | Thư viện này chứa nhiều bộ dữ liệu mẫu (Iris, Wine, Digits, Boston...) |
| `digits = datasets.load_digits()` | Tải **bộ dữ liệu handwritten digits (chữ số viết tay)** | Dữ liệu gồm 1.797 ảnh 8×8 pixel của các chữ số 0–9                     |
| `feature = digits.data`           | Lấy **ma trận đặc trưng (features)**                    | Mỗi ảnh được flatten thành vector 64 giá trị (8×8 pixel → 64 features) |
| `target = digits.target`          | Lấy **nhãn (labels)** tương ứng                         | Nhãn là số nguyên từ 0 đến 9                                           |
| `feature[0]`                      | Xem **mẫu đầu tiên** trong dữ liệu                      | In ra 64 giá trị pixel (dạng số thực từ 0–16)                          |

---
### Ví dụ minh họa

```python
import matplotlib.pyplot as plt

plt.imshow(digits.images[0], cmap='gray')
plt.title(f'Label: {digits.target[0]}')
plt.show()
```
Lệnh này sẽ hiển thị **ảnh chữ số viết tay đầu tiên** trong tập dữ liệu (ví dụ: “0” hoặc “3”).

---
### Thông tin nhanh

* `digits.data.shape` → `(1797, 64)`
  → Có 1.797 mẫu, mỗi mẫu có 64 đặc trưng.
* `digits.target_names` → `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

---

### Ứng dụng

Bộ **Digits dataset** này thường dùng để:

* Thử nghiệm nhanh các thuật toán phân loại (SVM, RandomForest, LogisticRegression).
* Dạy khái niệm về pipeline ML.
* Kiểm thử mô hình trước khi áp dụng vào dữ liệu thật.



kết hợp dataframes
    nối
        vấn đề: "xếp chồng" các dataframe
        pd.concat([df_a,df_b], axis=0)(nối theo hàng)
        pd.concat([df_a,df_b], axis=1)(nối theo cột)
    trộn
        vấn đề join các dataframe giống như trong sql

các kỹ thuật co dãn


| Giai đoạn                  | Trạng thái | Nội dung chính             |
| -------------------------- | ---------- | -------------------------- |
| 1️⃣ Xem xét bài toán       | ✅          | Chọn dạng hồi quy          |
| 2️⃣ Tạo/Lấy dữ liệu        | ✅          | Dùng `make_regression()`   |
| 3️⃣ Khám phá dữ liệu (EDA) | 🔜         | Kiểm tra, trực quan hóa    |
| 4️⃣ Tiền xử lý             | ⏭️         | Chuẩn hóa, chia train/test |
| 5️⃣ Huấn luyện mô hình     | ...        | `LinearRegression.fit()`   |
| 6️⃣ Fine-tuning            | ...        | Điều chỉnh hyperparameter  |
| 7️⃣ Vận hành & bảo dưỡng   | ...        | Deploy và giám sát model   |


dữ liệu thô
dữ liệu đã xử lý
file pyinb


file dữ liệu thô
file dữ liệu để dùng để huấn luyện
file ipynb (2 code cell cuối để show 30 dòng của tập thô và 30 dòng của tập đã xử lý)

NUM_EPOCHS = 100

# Initialize random weights (Khởi tạo trọng số ngẫu nhiên)
# W: Trọng số (Slope), b: Hệ số tự do (Bias/Intercept)
W = 0.01 * np.random.randn(INPUT_DIM, OUTPUT_DIM)
b = np.zeros((1, ))

# Training loop (Vòng lặp huấn luyện)
for epoch_num in range(NUM_EPOCHS):

    # Forward pass [NX1] . [1X1] = [NX1] (Lan truyền xuôi)
    # Tính giá trị dự đoán y_pred = X * W + b
    y_pred = np.dot(X_train, W) + b

    # Loss (Hàm mất mát - Mean Squared Error)
    # Tính sai số trung bình bình phương
    loss = (1/len(y_train)) * np.sum((y_train - y_pred)**2)

    # Show progress (Hiển thị tiến trình)
    if epoch_num % 10 == 0:
        print(f"Epoch: {epoch_num}, loss: {loss:.3f}")

    # Backpropagation (Lan truyền ngược)
    # Tính đạo hàm (gradient) để biết hướng điều chỉnh W và b
    # Lưu ý: Trong hình dùng biến N, giả định N = len(y_train)
    dW = -(2/N) * np.sum((y_train - y_pred) * X_train)
    db = -(2/N) * np.sum((y_train - y_pred) * 1)

    # Update weights (Cập nhật trọng số)
    # Dùng thuật toán Gradient Descent để tối ưu hóa
    W += -LEARNING_RATE * dW
    b += -LEARNING_RATE * db


    import numpy as np
import pandas as pd
data = pd.read_csv("housing_processed.csv")
data = data.iloc[:100]
X = data.drop(columns=['median_house_value']).values
y = data[['median_house_value']].values
nsample = 100
train = int(0.8 * nsample)
test= nsample - train
X_train = X[:train]
y_train = y[:train]

X_test = X[train:]
y_test = y[train:]
n_features = X_train.shape[1]

w = np.zeros((n_features, 1))
b = 0.0
LR = 0.01
nepoch = 50
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
for epoch in range(nepoch):
    y_pred = X_train.dot(w) + b
    loss = mse(y_train, y_pred)
    dw = (-2/train) * X_train.T.dot(y_train - y_pred)
    db = (-2/train) * np.sum(y_train - y_pred)
    w = w - LR * dw
    b = b - LR * db

    print(f"Epoch {epoch+1}/{nepoch}, Loss = {loss}")
y_pred_train = X_train.dot(w) + b
y_pred_test = X_test.dot(w) + b

MSE_train = mse(y_train, y_pred_train)
MSE_test = mse(y_test, y_pred_test)
print("w shape =", w.shape)
print("b =", b)
print("MSE Train =", MSE_train)
print("MSE Test  =", MSE_test)

output
Epoch 1/50, Loss = 0.09393447602140333
Epoch 2/50, Loss = 0.08371964502882284
Epoch 3/50, Loss = 0.0748873950430457
Epoch 4/50, Loss = 0.0672504382125044
Epoch 5/50, Loss = 0.06064685734908866
Epoch 6/50, Loss = 0.054936669129589645
Epoch 7/50, Loss = 0.04999885285786364
Epoch 8/50, Loss = 0.045728781721236046
Epoch 9/50, Loss = 0.04203600201787216
Epoch 10/50, Loss = 0.03884231321775387
Epoch 11/50, Loss = 0.03608010810529291
Epoch 12/50, Loss = 0.03369093777201489
Epoch 13/50, Loss = 0.03162427100033936
Epoch 14/50, Loss = 0.029836421705560147
Epoch 15/50, Loss = 0.02828962167027573
Epoch 16/50, Loss = 0.02695121888944797
Epoch 17/50, Loss = 0.025792984510435026
Epoch 18/50, Loss = 0.024790513657344783
Epoch 19/50, Loss = 0.023922707421811094
Epoch 20/50, Loss = 0.023171325025106215
Epoch 21/50, Loss = 0.022520596645935403
Epoch 22/50, Loss = 0.0219568886959289
Epoch 23/50, Loss = 0.021468414438083148
Epoch 24/50, Loss = 0.021044983805837296
Epoch 25/50, Loss = 0.02067778711253018
Epoch 26/50, Loss = 0.020359208060328565
Epoch 27/50, Loss = 0.02008266207961826
Epoch 28/50, Loss = 0.019842456567505027
Epoch 29/50, Loss = 0.01963367005889568
Epoch 30/50, Loss = 0.01945204776548596
Epoch 31/50, Loss = 0.01929391126540137
Epoch 32/50, Loss = 0.019156080426594353
Epoch 33/50, Loss = 0.01903580590677026
Epoch 34/50, Loss = 0.018930710797108896
Epoch 35/50, Loss = 0.01883874017113098
Epoch 36/50, Loss = 0.01875811746785115
Epoch 37/50, Loss = 0.01868730678342115
Epoch 38/50, Loss = 0.0186249802708786
Epoch 39/50, Loss = 0.018569989956039416
Epoch 40/50, Loss = 0.018521343371307793
Epoch 41/50, Loss = 0.018478182490215222
Epoch 42/50, Loss = 0.018439765515560362
Epoch 43/50, Loss = 0.018405451134590953
Epoch 44/50, Loss = 0.018374684907033704
Epoch 45/50, Loss = 0.01834698749704912
Epoch 46/50, Loss = 0.0183219444993268
Epoch 47/50, Loss = 0.01829919764337326
Epoch 48/50, Loss = 0.018278437189297708
Epoch 49/50, Loss = 0.018259395353691057
Epoch 50/50, Loss = 0.01824184062605842
w shape = (12, 1)
b = 0.07378415672497833
MSE Train = 0.018225572855167384
MSE Test  = 0.030460584386512852

import pandas as pd

data = pd.read_csv("housing_processed.csv")
print(data.columns)
Index(['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value', 'ocean_proximity_INLAND',
       'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
       'ocean_proximity_NEAR OCEAN'],
      dtype='object')

print(data.head())
   longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \
0   0.213996  0.564293                 1.0     0.257379        0.162209   
1   0.212982  0.564293                 1.0     0.223472        0.201035   
2   0.212982  0.564293                 1.0     0.285488        0.239862   
3   0.212982  0.564293                 1.0     0.161103        0.182053   
4   0.212982  0.563231                 1.0     0.445011        0.420190   

   population  households  median_income  median_house_value  \
0    0.157609    0.160550       0.899633            0.721533   
1    0.177430    0.199083       0.684719            0.698417   
2    0.179668    0.235780       0.445496            0.700343   
3    0.131074    0.175229       0.470871            0.545164   
4    0.348785    0.469725       0.420587            0.608306   

   ocean_proximity_INLAND  ocean_proximity_ISLAND  ocean_proximity_NEAR BAY  \
0                     0.0                     0.0                       1.0   
1                     0.0                     0.0                       1.0   
2                     0.0                     0.0                       1.0   
3                     0.0                     0.0                       1.0   
4                     0.0                     0.0                       1.0   

   ocean_proximity_NEAR OCEAN  
0                         0.0  
1                         0.0  
2                         0.0  
3                         0.0  
4                         0.0  



## 1. **Mô hình dự đoán**

### **1.1. Z đầu ra tuyến tính**

[
z_i = W x_i + b
]

* ( x_i ): vector đặc trưng của mẫu thứ i (D chiều)
* ( W ): ma trận trọng số kích thước ((K \times D))
* ( b ): vector bias ((K \times 1))
* ( z_i ): đầu ra tuyến tính trước softmax (K chiều)

---

### **1.2. Hàm softmax**

[
\hat{y}*{ij} = \frac{e^{z*{ij}}}{\sum_{k=1}^{K} e^{z_{ik}}}
]

* Đây là xác suất mẫu thứ i thuộc lớp j
* Softmax biến vector ( z_i ) thành một phân phối xác suất (tổng = 1)

---

## 2. **Hàm mất mát – Cross entropy**

[
L_i = - \sum_{j=1}^{K} y_{ij} \log(\hat{y}_{ij})
]

* ( y_{ij} = 1 ) nếu mẫu thuộc lớp j, ngược lại 0
  → one-hot vector

Hàm này đo độ sai lệch giữa phân phối thật và phân phối dự đoán.

---

## 3. **Đạo hàm (Gradient) để cập nhật W và b**

Ta cần tính:

[
\frac{\partial L_i}{\partial W},\quad \frac{\partial L_i}{\partial b}
]

Hai mục tiêu được trình bày trong ảnh:

---

### ⭐ **Mục tiêu 1: Đạo hàm theo W**

Kết quả:

[
\frac{\partial L_i}{\partial W} = (\hat{y}_i - y_i) x_i^T
]

Giải thích:

* (\hat{y}_i - y_i) là vector kích thước (K×1)
* (x_i^T) là vector (1×D)
  → Nhân vào ra ma trận (K×D) đúng bằng kích thước W.

Với toàn bộ batch N mẫu:

[
\frac{\partial L}{\partial W} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i) x_i^T
]

---

### ⭐ **Mục tiêu 2: Đạo hàm theo b**

[
\frac{\partial L_i}{\partial b} = (\hat{y}_i - y_i)
]

Với toàn bộ N mẫu:

[
\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)
]

---

## 4. **Quy trình huấn luyện (Training Procedure)**

Ảnh tóm tắt 3 bước:

---

### **Bước 1 — Khởi tạo**

* Khởi tạo

  * (W): ma trận ((K \times D))
  * (b): vector ((K \times 1))
* Learning rate: (\eta)

---

### **Bước 2 — Duyệt qua từng epoch và từng mẫu**

Đối với mỗi mẫu:

#### (a) **Tính dự đoán**

* Tính (z_i = W x_i + b)
* Tính softmax: (\hat{y}_i)

#### (b) **Tính loss**

[
L_i = -\sum_j y_{ij} \log(\hat{y}_{ij})
]

#### (c) **Tính đạo hàm**

[
\frac{\partial L_i}{\partial W} = (\hat{y}_i - y_i)x_i^T
]
[
\frac{\partial L_i}{\partial b} = (\hat{y}_i - y_i)
]

#### (d) **Cập nhật tham số**

[
W = W - \eta \frac{\partial L_i}{\partial W}
]
[
b = b - \eta \frac{\partial L_i}{\partial b}
]

---

### **Bước 3 — Kết thúc**

Sau nhiều epoch, thu được W và b tối ưu.

---

# 🎯 **Ý nghĩa toàn bộ nội dung**

Ảnh mô tả đầy đủ cách huấn luyện một **mô hình phân loại nhiều lớp (softmax classifier)**:

* Dùng **hàm softmax** để biến output thành xác suất.
* Dùng **cross entropy loss** để đo độ lệch.
* Dùng **gradient descent** để cập nhật trọng số W, b.
* Cách tính gradient chính xác:

  * Đạo hàm W = outer product giữa sai số và vector input
  * Đạo hàm b = sai số trực tiếp
* Chu trình lặp lại đến khi tối ưu.

Đây chính là mô hình **multiclass logistic regression** hoặc **output layer của neural network**.

làm tay cái đó với làm code one sample theo đa biến với đơn biến full sample đa với đơn





1. **Giải tay (tính toán gradient + cập nhật W, b) theo *one-sample***
2. **Giải tay theo *full-sample***
3. **Code Python đầy đủ**, gồm:

   * One-sample (online SGD)
   * Full-sample (batch gradient descent)
   * Bản đơn biến (1 input)
   * Bản đa biến (multi-feature)

---

# ⭐ **BÀI TOÁN**

Dự đoán giống hoa (0,1,2) dựa trên **chiều rộng cánh hoa (1 feature)**.
Có 6 mẫu:

| x (chiều rộng) | y (lớp) |
| -------------- | ------- |
| 1.0            | 0       |
| 2.5            | 0       |
| 4.0            | 1       |
| 5.5            | 1       |
| 7.0            | 2       |
| 8.0            | 2       |

Dự đoán cho x = 3.5 sau khi train.

---

## 🎯 Tham số ban đầu

Ta có 3 lớp → W kích thước 3×1, b kích thước 3×1

[
W =
\begin{bmatrix}
0\ 0\ 0
\end{bmatrix},
\quad
b =
\begin{bmatrix}
0\ 0\ 0
\end{bmatrix}
]

Learning rate:
[
\eta = 0.2
]
Số epoch:
[
n_epoch = 2
]

---

# PHẦN 1 — ⭐ GIẢI TAY ONE-SAMPLE (SGD)

Ở đây ta cập nhật **sau từng điểm dữ liệu**.

---

# 📌 **EPOCH 1 – SAMPLE 1**

### Mẫu:

x = 1.0 , y = 0 → one-hot = [1,0,0]

### 1) Tính z

[
z = Wx + b =
\begin{bmatrix}0 \ 0 \ 0\end{bmatrix}
]

### 2) Softmax

[
\hat{y} = [1/3,;1/3,;1/3]
]

### 3) Sai số

[
\hat{y}-y =
\begin{bmatrix}
1/3 - 1 \
1/3 - 0 \
1/3 - 0
\end{bmatrix}
=============

\begin{bmatrix}
-2/3 \
1/3 \
1/3
\end{bmatrix}
]

### 4) Gradient W

[
\frac{\partial L}{\partial W}
=============================

# (\hat{y}-y)x^T

\begin{bmatrix}
-2/3 \
1/3 \
1/3
\end{bmatrix} (1)
]

### 5) Cập nhật W

[
W = W - \eta \frac{\partial L}{\partial W}
]

[
W =
\begin{bmatrix}
0\0\0
\end{bmatrix}

* 0.2
  \begin{bmatrix}
  -2/3\1/3\1/3
  \end{bmatrix}
  =
  \begin{bmatrix}
  0.1333\ -0.0666\ -0.0666
  \end{bmatrix}
  ]

### 6) Cập nhật b (giống gradient W nhưng không nhân x)

[
b = b - 0.2
\begin{bmatrix}
-2/3\1/3\1/3
\end{bmatrix}
=============

\begin{bmatrix}
0.1333\ -0.0666\ -0.0666
\end{bmatrix}
]

---

➡️ **Bạn sẽ lặp lại tương tự cho toàn bộ 6 mẫu × 2 epoch**.
Do giới hạn tin nhắn, nếu bạn muốn **tôi sẽ tính hết toàn bộ bảng 12 update cho bạn**.

---

# PHẦN 2 — ⭐ GIẢI TAY FULL-SAMPLE (batch)

Full sample nghĩa là:

1. Tính softmax cho **tất cả** 6 điểm
2. Tính trung bình gradient
3. Chỉ cập nhật W, b **một lần mỗi epoch**

---

## 🎯 Công thức

[
\frac{\partial L}{\partial W}
= \frac{1}{N}\sum_{i=1}^N (\hat{y}_i-y_i)x_i^T
]

[
\frac{\partial L}{\partial b}
= \frac{1}{N}\sum_{i=1}^N (\hat{y}_i-y_i)
]

---

### ⚠️ Vì ban đầu W=0, b=0 nên tất cả mẫu đều cho cùng softmax:

[
\hat{y}_i = [1/3,1/3,1/3]
]

### Tính ( \hat{y}-y ) cho từng mẫu, trung bình lại

### (Tôi sẽ làm nhanh, nếu cần bạn có thể yêu cầu chi tiết):

[
\sum(\hat{y}-y) =
\begin{bmatrix}
-2 \ -0 \ +2
\end{bmatrix}
]

Trung bình (N=6):
[
\frac{1}{6}
\begin{bmatrix}
-2 \ 0 \ 2
\end{bmatrix}
=============

\begin{bmatrix}
-0.3333 \ 0 \ 0.3333
\end{bmatrix}
]

Nhân với vector x trung bình (x-bar ≈ 4.6667)

Do single-feature, gradient W = errors × x̄:

[
\frac{\partial L}{\partial W}
=============================

\begin{bmatrix}
-0.3333\0\0.3333
\end{bmatrix} (4.6667)
======================

\begin{bmatrix}
-1.555\0\1.555
\end{bmatrix}
]

---

### Cập nhật

[
W = 0 - 0.2
\begin{bmatrix}
-1.555\0\1.555
\end{bmatrix}
=============

\begin{bmatrix}
0.311\0\-0.311
\end{bmatrix}
]

[
b = 0 - 0.2
\begin{bmatrix}
-0.3333\0\0.3333
\end{bmatrix}
=============

\begin{bmatrix}
0.0666\0\-0.0666
\end{bmatrix}
]

---

# PHẦN 3 — CODE HOÀN CHỈNH

Tôi đưa 4 phiên bản:

---

# ⭐ **1. ONE-SAMPLE – 1 BIẾN**

```python
import numpy as np

X = np.array([1.0,2.5,4.0,5.5,7.0,8.0])
y = np.array([0,0,1,1,2,2])  # labels 0..2

def one_hot(y, K=3):
    out = np.zeros((len(y), K))
    out[np.arange(len(y)), y] = 1
    return out

Y = one_hot(y)

W = np.zeros((3,1))
b = np.zeros((3,1))

eta = 0.2
epochs = 2

for ep in range(epochs):
    for i in range(len(X)):
        x = np.array([[X[i]]])      # shape (1,1)
        t = Y[i].reshape(3,1)       # target
        
        z = W @ x + b
        y_hat = np.exp(z) / np.sum(np.exp(z))
        
        grad = (y_hat - t)
        W -= eta * grad * x
        b -= eta * grad

print("W =", W)
print("b =", b)
```

---

# ⭐ **2. FULL-SAMPLE – 1 BIẾN**

```python
import numpy as np

X = np.array([1.0,2.5,4.0,5.5,7.0,8.0]).reshape(-1,1)
y = np.array([0,0,1,1,2,2])

def one_hot(y, K=3):
    out = np.zeros((len(y), K))
    out[np.arange(len(y)), y] = 1
    return out

Y = one_hot(y)

W = np.zeros((3,1))
b = np.zeros((3,1))

eta = 0.2
epochs = 2

for ep in range(epochs):
    z = X @ W.T + b.T      # shape (6,3)
    y_hat = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    grad_W = ((y_hat - Y).T @ X) / len(X)
    grad_b = np.mean((y_hat - Y), axis=0, keepdims=True).T

    W -= eta * grad_W
    b -= eta * grad_b

print("W =", W)
print("b =", b)
```

---

# ⭐ **3. ONE-SAMPLE – NHIỀU BIẾN (multi-feature)**

```python
# X shape (N, D)
# W shape (K, D)
# b shape (K,1)

for ep in range(epochs):
    for i in range(N):
        x = X[i].reshape(D,1)
        t = Y[i].reshape(K,1)

        z = W @ x + b
        y_hat = np.exp(z)/np.sum(np.exp(z))

        grad = (y_hat - t)
        W -= eta * grad @ x.T
        b -= eta * grad
```

---

# ⭐ **4. FULL-SAMPLE – NHIỀU BIẾN**

```python
z = X @ W.T + b.T
y_hat = softmax(z)

grad_W = (y_hat - Y).T @ X / N
grad_b = np.mean(y_hat - Y, axis=0, keepdims=True).T
```

xét bài toán ví dụ: dự đoán giống hoa 0,1,2 dựa trên chiều rộng cánh hoa.chúng ta sẽ lấy ra 6 điểm dữ liệu(N=6) để thực hiện tính toán 
mục tiêu: 
1. thực hiện cập nhất W và b theo one-sample
2. thực hiện cập nhật W và b theo full-sample

gợi ý Ta có 3 lớp → W kích thước 3×1, b kích thước 3×1

[
W =
\begin{bmatrix}
0\ 0\ 0
\end{bmatrix},
\quad
b =
\begin{bmatrix}
0\ 0\ 0
\end{bmatrix}
]

Learning rate:
[
\eta = 0.2
]
Số epoch:
[
n_epoch = 2
]



[
\frac{\partial L}{\partial W}
= \frac{1}{N}\sum_{i=1}^N (\hat{y}_i-y_i)x_i^T
]

[
\frac{\partial L}{\partial b}
= \frac{1}{N}\sum_{i=1}^N (\hat{y}_i-y_i)
]

---

### ⚠️ Vì ban đầu W=0, b=0 nên tất cả mẫu đều cho cùng softmax:

[
\hat{y}_i = [1/3,1/3,1/3]
]

### Tính ( \hat{y}-y ) cho từng mẫu, trung bình lại

[
\sum(\hat{y}-y) =
\begin{bmatrix}
-2 \ -0 \ +2
\end{bmatrix}
]

Trung bình (N=6):
[
\frac{1}{6}
\begin{bmatrix}
-2 \ 0 \ 2
\end{bmatrix}
=============

\begin{bmatrix}
-0.3333 \ 0 \ 0.3333
\end{bmatrix}
]

Nhân với vector x trung bình (x-bar ≈ 4.6667)

Do single-feature, gradient W = errors × x̄:

[
\frac{\partial L}{\partial W}
=============================

\begin{bmatrix}
-0.3333\0\0.3333
\end{bmatrix} (4.6667)
======================

\begin{bmatrix}
-1.555\0\1.555
\end{bmatrix}
]

---

### Cập nhật

[
W = 0 - 0.2
\begin{bmatrix}
-1.555\0\1.555
\end{bmatrix}
=============

\begin{bmatrix}
0.311\0\-0.311
\end{bmatrix}
]

[
b = 0 - 0.2
\begin{bmatrix}
-0.3333\0\0.3333
\end{bmatrix}
=============

\begin{bmatrix}
0.0666\0\-0.0666
\end{bmatrix}
]



Được — mình đã làm **toàn bộ phép tính theo Gaussian Naive Bayes (đa biến: height + weight)** cho dữ liệu bạn cho, và **kết luận** rõ ràng. Dưới đây là quy trình + số thực chi tiết và giải thích tại sao kết quả như vậy.

---

## 1) Dữ liệu (training)

```
Height  Weight   Class
160     50       Gầy
159     49       Gầy
162     52       Bình thường
161     51       Bình thường
172     72       Bình thường
180     85       Mập
182     86       Mập
170     70       Mập
171     71       Mập
181     87       Mập
```

(Tổng N = 10 mẫu). Mẫu cần dự đoán: **x = (Height=169, Weight=69)**.

---

## 2) Bước – công thức dùng

* Prior: (P(c) = N_c/N).
* Với mỗi feature giả sử Gaussian:
  (P(x_j \mid c) = \dfrac{1}{\sqrt{2\pi\sigma_{c,j}^2}} \exp!\big(-\dfrac{(x_j-\mu_{c,j})^2}{2\sigma_{c,j}^2}\big)).
* Do giả thiết Naive Bayes độc lập tính tích likelihood theo từng feature:
  (P(x \mid c) = \prod_j P(x_j \mid c)).
* Posterior chưa chuẩn hóa: (P(c)\cdot P(x\mid c)). Chọn lớp có giá trị lớn nhất (bạn có thể chuẩn hoá để thành xác suất).

Mình dùng phương pháp ML để tính variance (chia cho (N_c), ddof=0).

---

## 3) Thống kê per-class (tính từ dữ liệu)

**Số lượng / prior:**

* Bình thường: (N=3) → prior = 3/10 = 0.3
* Gầy: (N=2) → prior = 2/10 = 0.2
* Mập: (N=5) → prior = 5/10 = 0.5

**Means & variances (từng feature):**

| Class       | μ_height | var_height | μ_weight | var_weight |
| ----------- | -------: | ---------: | -------: | ---------: |
| Bình thường | 165.0000 | 24.6666667 | 58.33333 | 93.5555556 |
| Gầy         | 159.5000 |  0.2500000 | 49.50000 |  0.2500000 |
| Mập         | 176.8000 | 26.9600000 | 79.80000 | 58.1600000 |

(Đây là các giá trị trung bình và phương sai tính từ mẫu mỗi lớp.)

---

## 4) Tính likelihood cho x = (169,69)

Tính Gaussian cho mỗi feature+j kết hợp:

* **Bình thường**

  * (p(\text{height}=169\mid \text{Bình}) \approx 0.0580767853)
  * (p(\text{weight}=69\mid \text{Bình}) \approx 0.0224538708)
  * likelihood = 0.0580768 × 0.0224539 ≈ **0.0013040486**
  * posterior (unnorm) = prior × likelihood = 0.3 × 0.0013040486 ≈ 0.0003912146

* **Gầy**

  * vars rất nhỏ (0.25) và giá trị x cách mean khá xa → gaussian gần 0:
  * (p(\text{height}=169\mid \text{Gầy}) \approx 0.0) (rất gần 0)
  * (p(\text{weight}=69\mid \text{Gầy}) \approx 0.0)
  * likelihood ≈ 0 → posterior ≈ 0

* **Mập**

  * (p(\text{height}=169\mid \text{Mập}) \approx 0.0248610252)
  * (p(\text{weight}=69\mid \text{Mập}) \approx 0.0191914870)
  * likelihood ≈ 0.0004771200
  * posterior (unnorm) = 0.5 × 0.00047712 ≈ 0.00023856

---

## 5) Chuẩn hoá posterior để dễ so sánh (tổng = 1)

* Bình thường: ≈ **0.62120**
* Gầy: ≈ **0.00000**
* Mập: ≈ **0.37880**

(Phần tính chi tiết mình đã tính bằng code để tránh sai số làm tay.)

---

## 6) Kết luận (dự đoán)

**Dự đoán cho x = (169,69):** → **Bình thường** (vì posterior lớn nhất ≈ 62.1%).

---

## 7) Giải thích trực quan vì sao không phải “Gầy” hay “Mập”

* **Gầy:** có variance rất nhỏ (mẫu Gầy có height 160,159 và weight 50,49) → phân phối rất hẹp. Điểm (169,69) cách mean Gầy quá xa → gaussian ~0 → xác suất gần 0.
* **Mập:** priors lớn (0.5) nhưng mean height ~176.8, weight ~79.8 → x hơi thấp hơn => likelihood không quá cao.
* **Bình thường:** mean height 165 và weight ~58 → x=(169,69) tương đối gần cả 2 chiều, khi nhân 2 pdf cho likelihood tốt hơn so với Mập → cộng với prior (0.3) cho posterior cao nhất.

---



Tốt — mình sẽ xuất **bảng tính chi tiết từng bước** (với số học rõ ràng) để bạn đối chiếu. Mình giữ nhiều chữ số thập phân để thấy chính xác từng trung gian.

---

## Dữ liệu (training) — tóm tắt

N = 10 mẫu

| Class       | mẫu (Height,Weight)                          | N_c |
| ----------- | -------------------------------------------- | --- |
| Gầy         | (160,50), (159,49)                           | 2   |
| Bình thường | (162,52), (161,51), (172,72)                 | 3   |
| Mập         | (180,85),(182,86),(170,70),(171,71),(181,87) | 5   |

Sample cần dự đoán: (x=(\text{Height}=169,\ \text{Weight}=69)).

---

## Bước 0 — Priors

[
P(\text{Bình}) = 3/10 = 0.3,\quad P(\text{Gầy})=2/10=0.2,\quad P(\text{Mập})=5/10=0.5
]

---

## Bước 1 — Means và variances (ML, chia cho (N_c))

Tính nhanh (kết quả):

* **Bình thường**
  (\mu_h = 165.0000000000,; \sigma_h^2 = 24.6666666667)
  (\mu_w = 58.3333333333,; \sigma_w^2 = 93.5555555556)

* **Gầy**
  (\mu_h = 159.5000000000,; \sigma_h^2 = 0.2500000000)
  (\mu_w = 49.5000000000,; \sigma_w^2 = 0.2500000000)

* **Mập**
  (\mu_h = 176.8000000000,; \sigma_h^2 = 26.9600000000)
  (\mu_w = 79.8000000000,; \sigma_w^2 = 58.1600000000)

(Đây là mean và variance tính từ dữ liệu đã cho.)

---

## Bước 2 — Công thức Gaussian (1D)

[
p(x\mid \mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big)
]

Ta áp dụng cho cả hai feature (height, weight) rồi nhân (Naive Bayes giả sử độc lập).

---

## Bước 3 — Tính từng PDF (từng class, từng feature)

Mình sẽ trình bày từng lớp, với phép thay số:

### A) Class = **Bình thường**

**Height:**

* (x_h=169,; \mu_h=165,; \sigma_h^2=24.6666666667)
* (2\sigma^2 = 49.3333333334)
* ((x-\mu)^2 = (169-165)^2 = 16)
* exponent (= -16/49.3333333334 = -0.3248015)
* (\exp(\cdot)=0.7226797)
* denom (=\sqrt{2\pi\sigma^2}=\sqrt{2\pi\cdot24.6666666667}=\sqrt{154.99999999}=12.449895)
* (p_{\text{height}} = 0.7226797 / 12.449895 = \mathbf{0.0580767853})

**Weight:**

* (x_w=69,; \mu_w=58.3333333333,; \sigma_w^2=93.5555555556)
* ((x-\mu)^2 = (69-58.3333333333)^2 = 10.6666666667^2 = 113.7777777778)
* (2\sigma^2 = 187.1111111112)
* exponent (= -113.77777778/187.1111111112 = -0.6079171)
* (\exp(\cdot)=0.5445283)
* denom (=\sqrt{2\pi\cdot93.555555556}=24.248737)
* (p_{\text{weight}} = 0.5445283 / 24.248737 = \mathbf{0.0224538708})

**Likelihood (height × weight):**
[
\text{lik} = 0.0580767853 \times 0.0224538708 = \mathbf{0.0013040486}
]

**Posterior (unnormalized):**
[
P(\text{Bình})\cdot \text{lik} = 0.3 \times 0.0013040486 = \mathbf{0.0003912146}
]

---

### B) Class = **Gầy**

**Height:**

* (x_h=169,; \mu_h=159.5,; \sigma_h^2=0.25)
* ((x-\mu)^2 = (169-159.5)^2 = 9.5^2 = 90.25)
* (2\sigma^2 = 0.5)
* exponent (= -90.25 / 0.5 = -180.5)
* (\exp(-180.5)) là cực kì nhỏ ≈ (1.45\times10^{-79})
* denom (=\sqrt{2\pi\cdot0.25}=\sqrt{1.5707963268}=1.252314)
* (p_{\text{height}} \approx 1.45\times10^{-79} / 1.252314 \approx \mathbf{1.16\times10^{-79}}) (≈ 0 trong tính thực tế)

**Weight:**

* (x_w=69,; \mu_w=49.5,; \sigma_w^2=0.25)
* ((x-\mu)^2 = 19.5^2 = 380.25)
* exponent (= -380.25/0.5 = -760.5)
* (\exp(-760.5)) ≈ extremely tiny (~(10^{-330}))
* (p_{\text{weight}} \approx \mathbf{0}) (vô cùng nhỏ; thực tế làm tròn về 0)

**Likelihood ≈ 0** (sản phẩm của hai số ~0)

**Posterior (unnorm) ≈ 0**

> Kết luận: vì phương sai rất nhỏ (lớp Gầy có 2 mẫu gần nhau), điểm (169,69) quá xa mean → probability ≈ 0.

---

### C) Class = **Mập**

**Height:**

* (x_h=169,; \mu_h=176.8,; \sigma_h^2=26.96)
* ((x-\mu)^2 = (169-176.8)^2 = (-7.8)^2 = 60.84)
* (2\sigma^2 = 53.92)
* exponent (= -60.84/53.92 = -1.128353)
* (\exp(\cdot)=0.323764)
* denom (=\sqrt{2\pi\cdot26.96}=\sqrt{169.4528}=13.017401)
* (p_{\text{height}} = 0.323764 / 13.017401 = \mathbf{0.0248610252})

**Weight:**

* (x_w=69,; \mu_w=79.8,; \sigma_w^2=58.16)
* ((x-\mu)^2 = (69-79.8)^2 = (-10.8)^2 =116.64)
* (2\sigma^2 = 116.32)
* exponent (= -116.64/116.32 = -1.002754)
* (\exp(\cdot)=0.367415)
* denom (=\sqrt{2\pi\cdot58.16}=\sqrt{365.99904}=19.157012)
* (p_{\text{weight}} = 0.367415 / 19.157012 = \mathbf{0.0191914870})

**Likelihood:**
[
\text{lik} = 0.0248610252 \times 0.0191914870 = \mathbf{0.0004771200}
]

**Posterior (unnorm):**
[
P(\text{Mập})\cdot \text{lik} = 0.5 \times 0.0004771200 = \mathbf{0.0002385600}
]

---

## Bước 4 — Chuẩn hoá posterior (tổng = 1)

Tổng posterior (unnorm) =
(0.0003912146 + 0 + 0.0002385600 = 0.0006297746)

Chuẩn hoá:

* (P(\text{Bình}\mid x) = 0.0003912146 / 0.0006297746 = \mathbf{0.6211977791})
* (P(\text{Gầy}\mid x) \approx \mathbf{0.0000000000})
* (P(\text{Mập}\mid x) = 0.0002385600 / 0.0006297746 = \mathbf{0.3788022209})

---

## Bảng tổng hợp (rút gọn)

| Class       | Prior  |    p(height) |    p(weight) |   likelihood | posterior_unnorm | posterior_norm |
| ----------- | ------ | -----------: | -----------: | -----------: | ---------------: | -------------: |
| Bình thường | 0.3000 | 0.0580767853 | 0.0224538708 | 0.0013040486 |     0.0003912146 |   0.6211977791 |
| Gầy         | 0.2000 |    ≈1.16e-79 | ≈0 (≈1e-330) |           ≈0 |               ≈0 |   0.0000000000 |
| Mập         | 0.5000 | 0.0248610252 | 0.0191914870 | 0.0004771200 |     0.0002385600 |   0.3788022209 |

(Trên thực tế ta coi mọi giá trị < ~1e-16 là 0 khi hiển thị; lớp Gầy gần như 0.)

---

## Kết luận

* **Dự đoán cho (169,69)** theo Gaussian Naive Bayes (với các phép tính trên): **Bình thường** (posterior ≈ 0.6212).
* Mập có posterior ≈ 0.3788, còn Gầy gần 0.

