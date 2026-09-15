<h2 align="center">
    <a href="https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin">
        🎓 Faculty of Information Technology - Dai Nam University
    </a>
</h2>

<h2 align="center">
    🩺 PHÂN LOẠI VIÊM PHỔI TỪ ẢNH X-QUANG NGỰC <br/>
    (Chest X-Ray Pneumonia Classification using a Hybrid DenseNet121–Vision Transformer)
</h2>

<div align="center">
    <p align="center">
        <img src="docs/aiotlab_logo.png" alt="AIoTLab Logo" width="170"/>
        <img src="docs/fitdnu_logo.png" alt="FIT Logo" width="180"/>
        <img src="docs/dnu_logo.png" alt="DaiNam University Logo" width="200"/>
    </p>

[![AIoTLab](https://img.shields.io/badge/AIoTLab-green?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Faculty of Information Technology](https://img.shields.io/badge/Faculty%20of%20Information%20Technology-blue?style=for-the-badge)](https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-orange?style=for-the-badge)](https://dainam.edu.vn)

</div>

---

## 📘 GIỚI THIỆU ĐỀ TÀI

**PneumoAI** là hệ thống hỗ trợ sàng lọc viêm phổi từ ảnh X-quang ngực, sử dụng mô hình học sâu lai ghép giữa **DenseNet121** (trích xuất đặc trưng cục bộ) và **Vision Transformer** (mô hình hóa quan hệ toàn cục). Bài toán được đặt ở dạng **phân loại nhị phân**: `NORMAL` và `PNEUMONIA`.

Hệ thống cung cấp đồng thời hai giao diện người dùng — **web app** (FastAPI + Jinja2 + Bootstrap 5) và **ứng dụng di động** (Flutter) — cùng chia sẻ một backend suy luận duy nhất. Mỗi dự đoán đều kèm bản đồ nhiệt **Grad-CAM** để bác sĩ kiểm chứng vùng ảnh ảnh hưởng đến quyết định của mô hình.

> ⚠️ Kết quả của hệ thống chỉ mang tính **hỗ trợ tham khảo**, không thay thế chẩn đoán của nhân viên y tế có thẩm quyền.

---

## 🧠 KIẾN TRÚC MÔ HÌNH (HybridViT)

| Thành phần | Cấu hình |
| :--- | :--- |
| Backbone | DenseNet121 (ImageNet), cắt đến `denseblock3` — 1024 kênh |
| Patch embedding | Conv 1×1 → BatchNorm → chuỗi token, cộng CLS token + positional embedding |
| Transformer encoder | 6 layer, 4 head, `embed_dim = 512`, MLP ratio 4, Pre-Norm |
| Regularization | Dropout 0.1, Attention dropout 0.1, DropPath (stochastic depth) 0.1 |
| Classifier head | Linear(512 → 256) → GELU → Dropout → Linear(256 → 2) |
| Input | Ảnh RGB 224×224, chuẩn hóa theo mean/std của ImageNet |
| Framework | PyTorch + torchvision + einops |
| Giải thích kết quả | Grad-CAM tại lớp `backbone.denseblock3` |

---

## 🧩 KIẾN TRÚC HỆ THỐNG

```text
┌──────────────────────┐        ┌────────────────────────┐
│  Web Browser         │        │  Flutter Mobile App    │
│  (Jinja2 + Bootstrap)│        │  (Android / iOS)       │
└──────────┬───────────┘        └───────────┬────────────┘
           │  HTML form                     │  HTTP multipart
           └───────────────┬────────────────┘
                           v
              ┌────────────────────────────┐
              │   FastAPI Backend          │
              │   SQLAlchemy ORM  ──────►  │ ──► MySQL (diagnosis_history)
              └────────────┬───────────────┘
                           v
        ┌──────────────────────────────────────┐
        │  Tiền xử lý → HybridViT → Softmax    │
        │            → Grad-CAM overlay        │
        └──────────────────────────────────────┘
```

---

## ⚙️ TÍNH NĂNG CHÍNH

- Tải ảnh X-quang và nhận dự đoán nhãn kèm xác suất từng lớp.
- Sinh **Grad-CAM heatmap** và ảnh **overlay** (`cv2.COLORMAP_JET`, tỉ lệ trộn 0.6/0.4).
- Ghi nhận thời gian **tiền xử lý / suy luận / sinh Grad-CAM** cho từng ca.
- Lưu và tra cứu **lịch sử chẩn đoán** (phân trang, thống kê, xóa bản ghi).
- Nhập thông tin bệnh nhân tùy chọn (tên, tuổi, giới tính, ghi chú) với **kiểm tra đầu vào** (tuổi 0–120, định dạng ảnh hợp lệ, file rỗng).
- **Bảo vệ dữ liệu bệnh nhân**: tên và ghi chú được mã hóa đối xứng trước khi lưu DB, họ tên được ẩn danh khi hiển thị.

---

## 🔌 API ENDPOINTS

| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `POST` | `/predict` | Suy luận cho app Flutter, trả về JSON đầy đủ |
| `POST` | `/api_predict` | Suy luận cho web, trả về `diagnosis_id` để chuyển trang |
| `GET` | `/history` | Danh sách lịch sử dạng JSON (mobile) |
| `DELETE` | `/history/{id}` | Xóa một bản ghi (mobile) |
| `GET` | `/api_stats` | Thống kê: số ca Normal/Pneumonia, độ tin cậy trung bình |
| `GET` | `/`, `/about`, `/web-history`, `/result/{id}` | Các trang giao diện web |
| `GET` | `/delete/{id}` | Xóa bản ghi và các file ảnh liên quan (web) |

---

## 📊 KẾT QUẢ ĐÁNH GIÁ

| Chỉ số | Giá trị |
| :--- | :---: |
| Accuracy | 97.95% |
| Thời gian suy luận | < 200 ms / ảnh |

> Bảng Precision / Recall / F1-score theo từng lớp và kết quả ablation study được trình bày chi tiết trong báo cáo đồ án.

---

## 🔧 CÀI ĐẶT & CHẠY DỰ ÁN

### 1️⃣ Clone project

```bash
git clone https://github.com/username/pneumo-ai.git
cd pneumo-ai
```

### 2️⃣ Cài thư viện backend

```bash
pip install -r requirements.txt
# fastapi uvicorn torch torchvision einops opencv-python pillow
# sqlalchemy pymysql numpy python-multipart jinja2
```

### 3️⃣ Cấu hình môi trường

Tạo file `.env` cùng cấp với `server.py`:

```env
DATABASE_URL=mysql+pymysql://root:password@localhost:3306/pneumonia_db
DB_ENCRYPTION_KEY=your_secret_key
```

Database sẽ được tự động tạo nếu chưa tồn tại. Đặt file trọng số `hybrid_vit_pneumonia_best_v10.pth` cùng thư mục với `server.py`.

### 4️⃣ Chạy server

```bash
python server.py     # hoặc: uvicorn server:app --host 0.0.0.0 --port 8000
```

Truy cập web tại `http://localhost:8000`, tài liệu API tại `http://localhost:8000/docs`.

### 5️⃣ Chạy ứng dụng Flutter

```bash
flutter pub get
flutter run
```

`baseUrl` trong `homescreen.dart` và `historyscreen.dart` đang trỏ tới `http://10.0.2.2:8000` (địa chỉ localhost của máy chủ khi chạy trên Android Emulator). Khi chạy trên thiết bị thật, đổi thành IP LAN của máy chủ.

---

## 📁 CẤU TRÚC THƯ MỤC

```text
├── server.py                    # FastAPI backend + HybridViT + Grad-CAM
├── hybrid_vit_pneumonia_best_v10.pth
├── webapp/
│   ├── templates/               # base.html, index.html, result.html, history.html, about.html
│   └── static/
│       ├── css/style.css
│       ├── uploads/             # ảnh gốc
│       └── gradcam/             # heatmap & overlay
└── lib/                         # Flutter app
    ├── main.dart                # splash + điều hướng tab
    ├── shared.dart              # hằng số giao diện & model dữ liệu
    ├── homescreen.dart          # chọn ảnh & gửi chẩn đoán
    ├── resultscreen.dart        # hiển thị kết quả & Grad-CAM
    ├── historyscreen.dart       # lịch sử chẩn đoán
    └── aboutscreen.dart         # giới thiệu hệ thống
```

---

## 💡 ĐỊNH HƯỚNG PHÁT TRIỂN

- Mở rộng sang phân loại đa lớp (vi khuẩn / virus / COVID-19).
- Bổ sung đánh giá mức độ tổn thương theo thang điểm Brixia (lược đồ dữ liệu đã có sẵn trường `severity_json`).
- Thay mã hóa XOR bằng thuật toán chuẩn (AES) và bổ sung xác thực người dùng.
- Xuất báo cáo PDF cho từng ca chẩn đoán.

---

## ✉️ LIÊN HỆ

**Tác giả**: Nguyễn Đào Nguyên Giáp

📧 **Email**: nguyennguyenvh09@gmail.com
🏫 **Trường**: Đại học Đại Nam - Khoa Công nghệ Thông tin

---

<p align="center">
  <b>© 2026 Faculty of Information Technology - Dai Nam University</b><br>
  Developed with ❤️ by <b>AIoT Lab</b>
</p>
