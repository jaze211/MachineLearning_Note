# Multinomial Naive Bayes — Quy trình & ví dụ (bài trong ảnh)
## Ý tưởng ngắn gọn

* Mỗi lớp có **prior** (P(Y=c)).
* Với Multinomial NB ta xem mỗi tài liệu là một túi từ (bag of words); likelihood của tài liệu là tích các xác suất từ (theo tần suất) (P(w\mid Y=c)).
* Ta dùng **Laplace smoothing** (thêm 1) để tránh xác suất 0.
* Chọn lớp có (P(Y=c),P(\text{document}\mid Y=c)) lớn nhất.

---

## Dữ liệu (training) trong ảnh

Các câu huấn luyện (4 câu):

1. `"buy cheap now"` — **Spam**
2. `"limited offer buy"` — **Spam**
3. `"meet me now"` — **Not Spam**
4. `"let's catch up"` — **Not Spam**

Cần dự đoán câu 5: `"buy now"`.

---

## Bước 1 — Xây vocab và đếm từ

Tập từ (vocabulary) = union của tất cả từ xuất hiện:

```
V = {buy, cheap, now, limited, offer, meet, me, let's, catch, up}  → |V| = 10
```

Đếm số token từng lớp (Multinomial):

* **Spam** (2 câu): tokens = `buy, cheap, now, limited, offer, buy` → tổng tokens = 6
  counts_spam: buy:2, cheap:1, now:1, limited:1, offer:1, các từ khác:0
* **Not Spam** (2 câu): tokens = `meet, me, now, let's, catch, up` → tổng tokens = 6
  counts_not: meet:1, me:1, now:1, let's:1, catch:1, up:1, các từ khác:0

---

## Bước 2 — Priors

Số tài liệu mỗi lớp: Spam = 2, Not = 2, tổng = 4
[
P(Y=\text{spam}) = 2/4 = 0.5,\quad P(Y=\text{not}) = 0.5
]

---

## Bước 3 — Xác suất từ có smoothing (Laplace, α=1)

Công thức:
[
P(w \mid Y=c) = \frac{\text{count}(w,c) + 1}{\text{total_tokens_in_class } c + |V|}
]

Ở đây: denominator cho cả hai lớp = total_tokens + |V| = (6 + 10 = 16).

Tính những từ cần thiết:

* **Spam**

  * (P(\text{buy}\mid spam) = (2+1)/16 = 3/16)
  * (P(\text{now}\mid spam) = (1+1)/16 = 2/16)

* **Not Spam**

  * (P(\text{buy}\mid not) = (0+1)/16 = 1/16)
  * (P(\text{now}\mid not) = (1+1)/16 = 2/16)

(Các từ không xuất hiện sẽ có ((0+1)/16=1/16).)

---

## Bước 4 — Likelihood cho tài liệu mới `"buy now"`

Multinomial (giả sử mỗi từ xuất hiện 1 lần):
[
P(\text{document}\mid c) = \prod_{w\in doc} P(w\mid c)
]

* **Spam**:
  [
  \text{lik}_{spam} = P(buy\mid spam)\cdot P(now\mid spam) = \frac{3}{16}\cdot\frac{2}{16} = \frac{6}{256}
  ]

* **Not Spam**:
  [
  \text{lik}_{not} = \frac{1}{16}\cdot\frac{2}{16} = \frac{2}{256}
  ]

---

## Bước 5 — Posterior chưa chuẩn hóa và quyết định

Multiply by prior:

* Spam: (P(Y=spam)\cdot \text{lik}_{spam} = 0.5 \cdot \frac{6}{256} = \frac{3}{256})
* Not:  (0.5 \cdot \frac{2}{256} = \frac{1}{256})

So sánh:
[
\frac{3}{256} > \frac{1}{256} \Rightarrow \textbf{Dự đoán: Spam}
]

(Chuẩn hoá nếu cần: spam ≈ 0.75, not ≈ 0.25 — nhưng so sánh tỷ lệ unnormalized là đủ.)

