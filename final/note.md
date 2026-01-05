Chào bạn, việc chuẩn bị kỹ lưỡng trước khi báo cáo là cực kỳ quan trọng. Dựa trên nội dung bạn cung cấp và danh sách các câu hỏi "tủ", tôi sẽ giúp bạn **hệ thống hóa lại kiến thức**, làm sắc bén các câu trả lời và **dự đoán thêm các câu hỏi hóc búa** mà cô giáo có thể xoay dựa trên mục lục (TOC) của bạn.

Dưới đây là kịch bản chuẩn bị toàn diện để bạn tự tin 100%.

---

### PHẦN 1: LÀM SẮC BÉN CÁC CÂU HỎI "TỦ" BẠN ĐÃ CÓ

*Bạn đã có câu trả lời, nhưng tôi sẽ giúp bạn diễn đạt "học thuật" và thuyết phục hơn.*

#### 1. Boosting khác gì Bagging (Random Forest)?

* **Điểm mấu chốt cần nói:**
* **Bagging (Random Forest):** Làm việc **Song song**. Các cây độc lập nhau. Mục tiêu chính là giảm **Phương sai (Variance)**  Giúp mô hình không bị học vẹt (Overfitting).
* **Boosting:** Làm việc **Tuần tự**. Cây sau sửa sai cho cây trước. Mục tiêu chính là giảm **Độ lệch (Bias)**  Giúp mô hình học được những ca khó mà mô hình trước bỏ sót.



#### 2. Tại sao chọn Gini thay vì Entropy?

* **Câu trả lời của bạn:** Đúng nhưng cần bổ sung.
* **Nói thêm:** "Thưa cô, ngoài việc Entropy phải tính logarit (tốn chi phí tính toán), thì Gini Index có giá trị nằm trong khoảng [0, 0.5] trong khi Entropy là [0, 1]. Với bài toán đơn giản như Iris, sự khác biệt về hiệu năng phân loại giữa hai độ đo này là không đáng kể, nên em chọn Gini để **tối ưu hóa tốc độ thực thi** khi code thủ công ạ."

#### 3. Tại sao chọn 3 thuật toán (KNN, Logistic, Decision Tree) cho Voting?

* **Câu trả lời của bạn:** Rất hay (ví dụ kiềng 3 chân).
* **Bổ sung thuật ngữ:** "Đây gọi là **Diversity (Sự đa dạng mô hình)**.
* Logistic Regression: Là mô hình **Tham số (Parametric)**, giỏi vẽ đường biên giới tuyến tính.
* KNN: Là mô hình **Phi tham số (Non-parametric)**, giỏi phát hiện các cụm cục bộ.
* Decision Tree: Là mô hình **Dựa trên luật (Rule-based)**, giỏi cắt không gian dữ liệu thành các hình khối chữ nhật.
*  Khi kết hợp lại, chúng bù trừ khuyết điểm cho nhau."



#### 4. Tại sao không chọn SVM?

* **Câu trả lời thành thật:** Code khó.
* **Câu trả lời khi báo cáo:** "Thưa cô, vì yêu cầu đồ án là **cài đặt thủ công (From Scratch)**. Thuật toán SVM yêu cầu giải bài toán tối ưu lồi (Quadratic Programming) với các điều kiện KKT, việc cài đặt thủ công phần này rất phức tạp và dễ phát sinh lỗi số học. Trong khi đó, KNN dựa trên khoảng cách, phù hợp với năng lực cài đặt thủ công mà vẫn đảm bảo hiệu quả trên tập Iris nhỏ ạ."

#### 5. Tại sao Boosting không hiệu quả lắm với Iris nhưng vẫn làm?

* **Câu trả lời "ghi điểm":** "Dạ, việc áp dụng Boosting vào Iris giống như **'dùng dao mổ trâu để giết gà'**. Iris quá đơn giản và các lớp phân tách khá rõ, nên một mô hình đơn giản cũng đã đạt 96%. Boosting cố gắng học sâu vào các sai số (residuals), với dữ liệu nhỏ và sạch như Iris, điều này dễ dẫn đến việc model cố học nhiễu (noise) gây ra Overfitting nhẹ hoặc không tăng thêm được độ chính xác nào. Tuy nhiên, em vẫn thực hiện để **chứng minh tính đúng đắn của thuật toán** em tự viết, và để so sánh xem liệu trên dữ liệu đơn giản, mô hình phức tạp có thực sự cần thiết không."

---

### PHẦN 2: CÁC CÂU HỎI MỚI DỰA TRÊN MỤC LỤC & CODE CỦA BẠN

*Cô giáo sẽ nhìn vào Mục lục (TOC) và code để hỏi xoáy những chỗ bạn chưa chuẩn bị.*

#### 🔸 Liên quan đến CHƯƠNG 3 (Methodology)

**Câu 1: "Em tiền xử lý dữ liệu (Scaling) như thế nào? Tại sao Tree cần Scaling?"**

* **Cú lừa:** Thực ra Decision Tree và Random Forest **KHÔNG** cần chuẩn hóa dữ liệu (Scaling) vì nó cắt dựa trên ngưỡng giá trị.
* **Cách trả lời:** "Dạ, với Random Forest hay Boosting thì không bắt buộc phải Scaling. TUY NHIÊN, trong mô hình **Voting Classifier** của em có chứa **KNN và Logistic Regression**. Hai thuật toán này cực kỳ nhạy cảm với khoảng cách và độ lớn dữ liệu, nên bắt buộc em phải chuẩn hóa (StandardScaler/MinMaxScaler) toàn bộ dữ liệu đầu vào để đảm bảo công bằng cho Voting ạ."

**Câu 2: "Trong code Gradient Boosting thủ công, em xử lý bài toán Phân loại (Classification) như thế nào khi dùng cây Hồi quy?"**

* *Đây là điểm yếu trong code của bạn (dùng Regression Tree cho bài toán phân loại), cô rất dễ hỏi.*
* **Trả lời:** "Dạ, để đơn giản hóa việc cài đặt thủ công, em đã tiếp cận theo hướng **Hồi quy trên nhãn số**. Em coi các lớp (0, 1, 2) là các giá trị liên tục. Mô hình sẽ dự đoán ra một số thực (ví dụ 1.8), sau đó em dùng hàm **làm tròn (round)** để đưa về nhãn gần nhất (thành 2). Em biết cách chuẩn nhất là dùng hàm loss *Multinomial Deviance* (Softmax), nhưng cách tiếp cận hồi quy này vẫn hoạt động tốt trên Iris do đặc thù thứ tự kích thước của 3 loài hoa ạ."

**Câu 3: "One-vs-Rest trong AdaBoost/Logistic của em hoạt động sao?"**

* **Trả lời:** "Dạ Iris có 3 lớp. Với One-vs-Rest, em huấn luyện 3 mô hình con:
1. Setosa (1) vs Không phải Setosa (0).
2. Versicolor (1) vs Không phải Versicolor (0).
3. Virginica (1) vs Không phải Virginica (0).
Khi dự đoán, mẫu dữ liệu sẽ được đưa qua cả 3 mô hình, mô hình nào cho xác suất/điểm số cao nhất thì em chọn lớp đó."



#### 🔸 Liên quan đến CHƯƠNG 4 (Kết quả thực nghiệm)

**Câu 4: "Ablation Study (Nghiên cứu cắt bỏ) trong mục 4.2.3 là em làm gì?"**

* **Trả lời:** "Dạ, phần này em thử nghiệm xem nếu bỏ bớt một thành phần trong hệ thống thì kết quả thay đổi sao. Ví dụ: Trong Voting, nếu em bỏ Decision Tree ra, chỉ còn KNN và Logistic thì độ chính xác có giảm không? Mục đích để chứng minh vai trò của từng 'chuyên gia' trong hội đồng Voting ạ."

**Câu 5: "Hyperparameters (Tham số siêu hình) em chọn n_estimators=50 hay Learning rate=0.1 là dựa vào đâu?"**

* **Trả lời:** "Dạ ban đầu em chọn theo kinh nghiệm (mặc định). Sau đó ở phần Tối ưu tham số, em có thử thay đổi các giá trị này (Grid Search thủ công) để xem bộ tham số nào cho Accuracy cao nhất trên tập Test ạ."

---

### PHẦN 3: TỔNG HỢP KIẾN THỨC CẦN NẮM ĐỂ "QUA MÔN"

Để hiểu rõ tất cả mọi thứ trong Project này, bạn chỉ cần nhớ một bức tranh tổng thể sau:

1. **Đầu vào:** Dữ liệu Iris (4 đặc trưng, 3 lớp).
2. **Thách thức:** Cần độ chính xác cao nhưng phải code tay (không dùng thư viện core).
3. **Giải pháp 1 (Voting):** Gọi 3 ông chuyên gia khác nhau (KNN - nhìn gần, Logistic - nhìn thẳng, Tree - nhìn luật) vào họp bàn (Voting).  **Mạnh về sự ổn định.**
4. **Giải pháp 2 (Bagging - Random Forest):** Tạo ra 20 ông chuyên gia giống nhau (Cây quyết định), nhưng mỗi ông chỉ được xem một phần dữ liệu khác nhau (Bootstrap) và một phần đặc trưng khác nhau. Sau đó lấy ý kiến số đông.  **Giảm bệnh học vẹt (Overfitting).**
5. **Giải pháp 3 (Boosting):** Tạo ra 50 ông học sinh. Ông thứ nhất làm bài, sai đâu thì ông thứ 2 tập trung sửa đó (AdaBoost: tăng trọng số / Gradient Boosting: học sai số).  **Giảm sai lệch (Bias).**
6. **Kết luận:** Với Iris, cả 3 cách đều tốt (vì dữ liệu dễ). Nhưng Project thành công ở chỗ nhóm đã **hiểu và tự code được** toán học bên dưới của cả 3 phương pháp phức tạp này.

### LỜI KHUYÊN CUỐI CÙNG KHI BÁO CÁO:

* Nếu cô hỏi câu nào khó quá về toán (đạo hàm, tích phân): Hãy lái câu trả lời về **Code**. Ví dụ: *"Dạ về mặt toán học sâu sắc thì em cần tìm hiểu thêm, nhưng trong code em đã hiện thực hóa công thức đó bằng dòng lệnh update weights này ạ..."* (Chỉ vào code).
* Luôn nhấn mạnh chữ **"FROM SCRATCH"**. Đây là tấm khiên bảo vệ bạn. Nếu mô hình chạy chậm hơn thư viện, hay thiếu tính năng nâng cao  *"Do em code from scratch để học thuật toán là chính"*.

Chúc bạn báo cáo thành công rực rỡ! Bạn đã chuẩn bị rất kỹ rồi, tự tin lên nhé! 💪