# K-means trong học máy — giải thích chi tiết và các phần liên quan

K-means là một thuật toán **phân cụm (clustering)** không giám sát rất phổ biến. Mục tiêu của K-means là **chia N mẫu thành K cụm** sao cho các mẫu cùng cụm càng giống nhau (về khoảng cách tới tâm cụm) càng tốt.

---

## 1. Mục tiêu toán học (objective)

Gọi (X={x_1,\dots,x_N}), (x_i\in\mathbb{R}^d). K-means tìm các centroid ({\mu_1,\dots,\mu_K}) và phân cụm (C_1,\dots,C_K) bằng cách tối thiểu hàm mất mát:

[
\min_{C,\mu} ; J = \sum_{k=1}^K \sum_{x_i \in C_k} |x_i - \mu_k|_2^2
]

Đây là tổng phương sai trong các cụm (within-cluster sum of squares).

---

## 2. Thuật toán (Lloyd’s algorithm — phổ biến)

Lặp cho đến khi hội tụ:

1. **Gán nhãn (Assignment step)**: với mỗi điểm (x_i), gán vào cụm gần nhất:
   [
   c_i = \arg\min_{k} |x_i - \mu_k|^2
   ]
2. **Cập nhật centroid (Update step)**: với mỗi cụm k, tính trung bình mẫu trong cụm:
   [
   \mu_k = \frac{1}{|C_k|}\sum_{x_i\in C_k} x_i
   ]
3. Dừng khi nhãn không thay đổi hoặc khi giảm J rất nhỏ.

Thuật toán luôn giảm hoặc giữ nguyên giá trị J, nên hội tụ về một điểm cực trị (cục bộ), nhưng không đảm bảo cực trị toàn cục.

---

## 3. Khoảng cách & không gian đặc trưng

* K-means mặc định dùng **khoảng cách Euclid (L2)**. Điều này tương đương với giả định rằng cụm có hình cầu trong không gian đặc trưng.
* Do vậy **scale của các feature rất quan trọng** → luôn cân nhắc chuẩn hóa (StandardScaler, MinMaxScaler) trước khi chạy K-means.

---

## 4. Khởi tạo centroid và k-means++

* Khởi tạo ngẫu nhiên centroid có thể dẫn tới nghiệm kém (local minima).
* **k-means++** là phương pháp khởi tạo phổ biến: chọn centroid đầu tiên ngẫu nhiên, các centroid tiếp theo với xác suất tỉ lệ khoảng cách² tới centroid gần nhất. k-means++ cải thiện tính ổn định và thường hội tụ nhanh hơn.

---

## 5. Vấn đề chọn K (số cụm)

Một số kỹ thuật:

* **Elbow method**: vẽ tổng within-cluster SSE (J) theo K; chọn K tại “khuỷu” (điểm giảm lợi ích bắt đầu giảm).
* **Silhouette score**: cho mỗi điểm (s_i=(b_i-a_i)/\max(a_i,b_i)) với (a_i)=avg dist tới cụm mình, (b_i)=min avg dist tới cụm khác. Trung bình silhouette gần 1 tốt, gần 0 ranh giới, âm là sai.
* **Gap statistic**, **BIC/AIC (với mô hình hỗn hợp Gaussian)**, hoặc domain knowledge.

---

## 6. Độ phức tạp và hiệu năng

* Mỗi vòng lặp assignment: (O(N K d)) (tính khoảng cách N×K), update: (O(N d)). Tổng: (O(N K d \cdot I)) với I số vòng lặp.
* **Mini-batch K-means** giảm chi phí cho dữ liệu lớn bằng cách cập nhật bằng mẫu ngẫu nhiên nhỏ (scikit-learn cung cấp).

---

## 7. Hạn chế & lưu ý thực tế

* **Chỉ hoạt động tốt khi cụm dạng hình cầu** — không tốt với cụm có hình dạng phức tạp, mật độ khác nhau, hoặc dữ liệu có nhiễu/outliers.
* **Nhạy với outliers** (outlier kéo centroid).
* **Yêu cầu K cố định**; nếu không biết K, dùng kỹ thuật chọn K.
* **Không xử lý tốt dữ liệu categorical** (phải transform: one-hot, target encoding, hoặc dùng k-modes / k-prototypes).
* **Kết quả phụ thuộc khởi tạo** → dùng nhiều lần với random_state khác nhau và chọn kết quả tốt nhất.

---

## 8. Các biến thể & thuật toán liên quan

* **Mini-batch K-means**: cho dữ liệu lớn, cập nhật theo lô nhỏ.
* **K-medoids (PAM)**: dùng medoid (mẫu thực) thay vì centroid, robust với outliers.
* **K-modes / k-prototypes**: cho dữ liệu categorical hoặc hỗn hợp.
* **Gaussian Mixture Models (GMM)**: mô phỏng mỗi cụm bằng Gaussian → soft assignment (EM algorithm).
* **Spectral clustering, DBSCAN, Hierarchical clustering**: khi dữ liệu không suit K-means (non-convex shapes, varying density).
* **Bisecting K-means**: hierarchical variant.

---

## 9. Đánh giá kết quả clustering

Không có nhãn → dùng metrics không giám sát:

* **Within-cluster sum of squares (SSE)** — nhỏ tốt.
* **Silhouette score** — giá trị ∈ [−1,1].
* **Davies-Bouldin index**, **Calinski-Harabasz index**.
  Nếu có nhãn ground truth: **Adjusted Rand Index (ARI)**, **Normalized Mutual Information (NMI)**.

---

## 10. Checklist thực hành trước khi chạy K-means

* Chuẩn hóa dữ liệu (StandardScaler / MinMax) nếu feature khác scale.
* Loại/ xử lý outliers nếu cần.
* Chọn số cụm K (elbow / silhouette / domain).
* Dùng k-means++ khởi tạo.
* Thực hiện nhiều lần (n_init) và chọn best inertia.
* Kiểm tra silhouette và visualize (PCA/TSNE projection).
* Nếu dữ liệu lớn, cân nhắc mini-batch K-means.

---

## 11. Ví dụ code (scikit-learn, Python)

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

X = ...  # numpy array shape (N,d)

# scale
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# KMeans with k-means++ and multiple inits
k = 4
km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
km.fit(Xs)

labels = km.labels_
centroids = km.cluster_centers_
inertia = km.inertia_  # SSE

# silhouette
sil = silhouette_score(Xs, labels)
print("Inertia:", inertia, "Silhouette:", sil)
```

---

## 12. Khi nào dùng K-means (use cases)

* Khảo sát dữ liệu (exploratory analysis)
* Tách nhóm khách hàng (customer segmentation)
* Phân cụm ảnh/điểm dữ liệu (preprocessing)
* Vector quantization, compression
* Làm tiền xử lý cho algorithms khác (coarsening, prototype selection)

---

## 13. Mẹo & thủ thuật nâng cao

* Nếu dữ liệu sparse (TF-IDF text), có thể dùng **KMeans trên cosine distance** (bình thường scikit-learn điều chỉnh bằng chuẩn hoá l2 → cosine similarity).
* Với TF-IDF, **bình thường không nên tiêu chuẩn hóa bằng StandardScaler**; thay vào đó, dùng normalize (l2) để tương thích cosine.
* Visualize clusters qua PCA/TSNE/UMAP để kiểm tra cấu trúc.
* Nếu muốn soft cluster assignments / probabilistic → dùng GMM.

---

### Kết luận ngắn gọn

K-means là công cụ phân cụm đơn giản, nhanh, phù hợp khi cụm gần hình cầu và dữ liệu đã chuẩn hóa. Quan trọng là **lựa chọn K, khởi tạo tốt (k-means++), chuẩn hóa** và cân nhắc các biến thể (mini-batch, k-medoids, GMM) nếu dữ liệu có đặc tính phức tạp hơn.

Nếu bạn muốn, mình sẽ:

* Viết **notebook minh hoạ** với ví dụ (synthetic + real data), elbow plot, silhouette, PCA visualization, và mini-batch;
* Hoặc chuyển ví dụ sang dữ liệu của bạn (upload một sample) và chạy clustering cụ thể.

Bạn muốn mình làm tiếp phần nào?

Cảm ơn bạn đã cung cấp hình ảnh! Hình ảnh này đang trình bày một ví dụ **Thực hành** cụ thể về thuật toán **K-Means Clustering** (Phân cụm K-Means) trên một bộ dữ liệu nhỏ.

Đây là bản tóm tắt các bước đang được thực hiện, dựa trên nội dung slide:

## 📊 Ví dụ K-Means Clustering

Bộ dữ liệu gồm 6 sinh viên (SV) với 2 đặc trưng (feature): **Điểm học tập** và **Điểm rèn luyện**.

| SV | Điểm học tập ($x_1$) | Điểm rèn luyện ($x_2$) |
| :---: | :---: | :---: |
| **S01** | 85 | 83 |
| **S02** | 70 | 59 |
| **S03** | 90 | 50 |
| **S04** | 50 | 85 |
| **S05** | 50 | 50 |
| **S06** | 90 | 85 |

### 🎯 Bước 1: Lựa chọn $k$

* **Chọn $k=3$.** (Tức là thuật toán sẽ chia dữ liệu thành 3 cụm).

### 🚀 Bước 2: Khởi tạo trọng tâm ban đầu

Các trọng tâm (centroids) ban đầu ($\mu_i^0$) được chọn **theo phương pháp Forgy** (thường là chọn ngẫu nhiên $k$ điểm dữ liệu làm trọng tâm).

* $\mu_1^0 = (85, 83)$ (Trùng với dữ liệu của SV S01)
* $\mu_2^0 = (70, 59)$ (Trùng với dữ liệu của SV S02)
* $\mu_3^0 = (90, 50)$ (Trùng với dữ liệu của SV S03)

### 🔗 Bước 3: Gán điểm dữ liệu vào cụm gần nhất (Vòng lặp 1)

Đây là bước tính toán **khoảng cách (K/C)** từ mỗi điểm dữ liệu đến 3 trọng tâm $\mu_1^0, \mu_2^0, \mu_3^0$ và gán điểm đó vào trọng tâm gần nhất. Khoảng cách thường dùng là **khoảng cách Euclidean** ($d = \sqrt{(x_1 - \mu_1)^2 + (x_2 - \mu_2)^2}$).

Kết quả gán cụm (theo slide) là:

* **Cụm 1 (gần $\mu_1^0$):** S1, S6
    * *Lưu ý: S6(90, 85) gần S1(85, 83) hơn là S3(90, 50) hoặc S2(70, 59).*
* **Cụm 2 (gần $\mu_2^0$):** S2, S4, S5
    * *Lưu ý: S4(50, 85) và S5(50, 50) được gán vào Cụm 2 (gần S2(70, 59)).*
* **Cụm 3 (gần $\mu_3^0$):** S3
    * *Lưu ý: Chỉ có S3(90, 50) ở cụm này.*

---

## ⏭️ Bước Tiếp theo (Chưa có trong hình)

Sau Bước 3, thuật toán sẽ chuyển sang **Bước 4: Cập nhật trọng tâm mới ($\mu_i^1$)**.

**Yêu cầu của Bước 4:** Tính lại vị trí trọng tâm mới bằng cách lấy **giá trị trung bình** của tất cả các điểm trong cụm đó.

* **$\mu_1^1$ (Cụm S1, S6):** $\mu_1^1 = \left( \frac{85+90}{2}, \frac{83+85}{2} \right) = \left( 87.5, 84 \right)$
* **$\mu_2^1$ (Cụm S2, S4, S5):** $\mu_2^1 = \left( \frac{70+50+50}{3}, \frac{59+85+50}{3} \right) = \left( \frac{170}{3}, \frac{194}{3} \right) \approx (56.67, 64.67)$
* **$\mu_3^1$ (Cụm S3):** $\mu_3^1 = (90, 50)$

Sau đó, thuật toán sẽ **lặp lại** Bước 3 (Gán dữ liệu) với các trọng tâm mới ($\mu_1^1, \mu_2^1, \mu_3^1$) cho đến khi không còn sự thay đổi nào trong việc gán cụm.


bước 5 kiểm tra hội tụ 
     thuật toán lặp lại 2 bước 3 và 4 cho tới khi trọng tâm không thay đổi nhỏ hơn một ngưỡng cho trước