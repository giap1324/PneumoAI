import os
import io
import sys
from PIL import Image
from sqlalchemy.orm import Session
from datetime import datetime

# Thêm đường dẫn backend vào sys.path
backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app flutter", "backend")
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from server import DiagnosisHistory, CLASSES

def test_web_routes_availability(client):
    """Kiểm tra tính khả dụng của các trang giao diện Web (phản hồi mã HTTP 200)."""
    # Test trang chủ
    response = client.get("/")
    assert response.status_code == 200
    assert "PneumoAI" in response.text
    
    # Test trang Giới thiệu (About)
    response = client.get("/about")
    assert response.status_code == 200
    
    # Test trang Lịch sử (Web History)
    response = client.get("/web-history")
    assert response.status_code == 200

def test_api_stats_endpoint(client, db_session: Session):
    """Kiểm tra endpoint tính toán thống kê lịch sử chẩn đoán."""
    # Khi DB trống
    response = client.get("/api_stats")
    assert response.status_code == 200
    data = response.json()
    assert data["normal_count"] == 0
    assert data["pneumonia_count"] == 0
    assert data["avg_confidence"] == 0.0
    
    # Thêm dữ liệu mẫu vào SQLite test DB
    entry1 = DiagnosisHistory(
        image_url="/static/uploads/test1.png",
        label="NORMAL",
        confidence=0.95,
        normal_prob=0.95,
        pneumonia_prob=0.05,
        timestamp=datetime.utcnow()
    )
    entry2 = DiagnosisHistory(
        image_url="/static/uploads/test2.png",
        label="PNEUMONIA",
        confidence=0.85,
        normal_prob=0.15,
        pneumonia_prob=0.85,
        timestamp=datetime.utcnow()
    )
    db_session.add(entry1)
    db_session.add(entry2)
    db_session.commit()
    
    # Kiểm tra lại stats
    response = client.get("/api_stats")
    assert response.status_code == 200
    data = response.json()
    assert data["normal_count"] == 1
    assert data["pneumonia_count"] == 1
    # Average of 0.95 and 0.85 is 0.90 -> 90.0%
    assert abs(data["avg_confidence"] - 90.0) < 0.01

def test_mobile_history_endpoints(client, db_session: Session):
    """Kiểm tra các endpoint phục vụ ứng dụng Flutter lấy lịch sử."""
    # Thêm dữ liệu mẫu
    entry = DiagnosisHistory(
        image_url="/static/uploads/mobile.png",
        label="PNEUMONIA",
        confidence=0.99,
        normal_prob=0.01,
        pneumonia_prob=0.99,
        timestamp=datetime.utcnow()
    )
    db_session.add(entry)
    db_session.commit()
    db_session.refresh(entry)
    
    # Test GET /history
    response = client.get("/history")
    assert response.status_code == 200
    history_list = response.json()
    assert len(history_list) == 1
    assert history_list[0]["label"] == "PNEUMONIA"
    assert history_list[0]["confidence"] == 0.99
    
    # Test DELETE /history/{record_id}
    delete_response = client.delete(f"/history/{entry.id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["message"] == "Deleted successfully"
    
    # Kiểm tra DB đã bị xóa
    assert db_session.query(DiagnosisHistory).count() == 0

def test_prediction_api_flow(client):
    """Kiểm thử tích hợp luồng dự đoán qua ảnh tải lên cho Mobile API."""
    # Tạo một ảnh nhị phân giả lập trong bộ nhớ
    file_bytes = io.BytesIO()
    Image.new('RGB', (224, 224), color='white').save(file_bytes, format='PNG')
    file_bytes.seek(0)
    
    # Thực hiện gửi POST request kèm file
    response = client.post(
        "/predict",
        files={"file": ("test_image.png", file_bytes, "image/png")}
    )
    
    # Kiểm tra kết quả phản hồi của API
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert data["label"] in CLASSES
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "image_url" in data
    assert "heatmap_url" in data
    assert "gradcam_url" in data
    
    # Dọn dẹp tệp tin vật lý được sinh ra trong quá trình test
    for path_key in ["image_url", "heatmap_url", "gradcam_url"]:
        rel_path = data[path_key].lstrip("/")
        abs_path = os.path.join(os.getcwd(), "webapp", rel_path)
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
            except Exception:
                pass


def test_prediction_missing_file(client):
    """Kiểm tra lỗi đầu vào: Thiếu tệp tin hình ảnh."""
    # POST /predict không có file
    response = client.post("/predict", data={"patient_name": "Test"})
    assert response.status_code == 422  # Unprocessable Entity từ FastAPI
    
    # POST /api_predict không có file
    response = client.post("/api_predict", data={"patient_name": "Test"})
    assert response.status_code == 422


def test_prediction_invalid_age(client):
    """Kiểm tra lỗi đầu vào: Độ tuổi bệnh nhân không hợp lệ."""
    file_bytes = io.BytesIO()
    Image.new('RGB', (224, 224), color='white').save(file_bytes, format='PNG')
    file_bytes.seek(0)

    # Trường hợp tuổi âm (-5) cho /predict
    response = client.post(
        "/predict",
        files={"file": ("test.png", file_bytes, "image/png")},
        data={"patient_age": "-5"}
    )
    assert response.status_code == 400
    assert "Tuổi bệnh nhân phải nằm trong khoảng" in response.json()["detail"]

    # Trường hợp tuổi chữ ("abc") cho /api_predict
    file_bytes.seek(0)
    response = client.post(
        "/api_predict",
        files={"file": ("test.png", file_bytes, "image/png")},
        data={"patient_age": "abc"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "Tuổi bệnh nhân phải là một số nguyên hợp lệ" in data["error"]


def test_prediction_invalid_images(client):
    """Kiểm tra lỗi ảnh không hợp lệ (tệp rỗng, tệp văn bản, tệp hỏng)."""
    # 1. Tệp rỗng (0 bytes)
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/predict",
        files={"file": ("empty.png", empty_file, "image/png")}
    )
    assert response.status_code == 400
    assert "bị trống hoặc không tồn tại" in response.json()["detail"]

    # 2. Tệp văn bản không phải ảnh
    txt_file = io.BytesIO(b"This is not a chest X-ray image")
    response = client.post(
        "/predict",
        files={"file": ("info.txt", txt_file, "text/plain")}
    )
    assert response.status_code == 400
    assert "không phải là ảnh hợp lệ" in response.json()["detail"]

    # 3. Tệp ảnh bị hỏng (corrupted bytes)
    corrupted_file = io.BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRrandomgarbage")
    response = client.post(
        "/api_predict",
        files={"file": ("corrupt.png", corrupted_file, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "không phải là ảnh hợp lệ" in data["error"]


def test_api_routing_errors(client):
    """Kiểm tra các lỗi API liên quan đến phương thức và tài nguyên không tồn tại."""
    # 1. Sai phương thức HTTP
    response = client.get("/predict")
    assert response.status_code == 405  # Method Not Allowed

    response = client.post("/history")
    assert response.status_code == 405

    # 2. Xóa bản ghi không tồn tại
    response = client.delete("/history/999999")
    assert response.status_code == 404
    assert "Record not found" in response.json()["detail"]

    # 3. Xem bản ghi web không tồn tại
    response = client.get("/result/999999")
    assert response.status_code == 404

    # 4. Yêu cầu xóa web bản ghi không tồn tại
    response = client.get("/delete/999999", follow_redirects=False)
    assert response.status_code == 303  # Redirect về /web-history


def test_prediction_execution_times(client):
    """Kiểm chứng thời gian suy luận AI và thời gian tạo Grad-CAM được đo đạc."""
    file_bytes = io.BytesIO()
    Image.new('RGB', (224, 224), color='white').save(file_bytes, format='PNG')
    file_bytes.seek(0)
    
    response = client.post(
        "/predict",
        files={"file": ("test_image.png", file_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Xác minh sự tồn tại của trường thời gian hiệu năng
    assert "inference_time" in data
    assert "gradcam_time" in data
    assert "preprocess_time" in data
    assert isinstance(data["inference_time"], float)
    assert isinstance(data["gradcam_time"], float)
    assert isinstance(data["preprocess_time"], float)
    assert data["inference_time"] > 0.0
    assert data["gradcam_time"] > 0.0
    assert data["preprocess_time"] > 0.0

    # Dọn dẹp
    for path_key in ["image_url", "heatmap_url", "gradcam_url"]:
        rel_path = data[path_key].lstrip("/")
        abs_path = os.path.join(os.getcwd(), "webapp", rel_path)
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
            except Exception:
                pass


def test_patient_data_security(client, db_session: Session):
    """Kiểm chứng tính năng bảo mật dữ liệu bệnh nhân (Mã hóa tĩnh at-rest & Ẩn danh hiển thị)."""
    file_bytes = io.BytesIO()
    Image.new('RGB', (224, 224), color='white').save(file_bytes, format='PNG')
    file_bytes.seek(0)
    
    # Gửi dự đoán kèm thông tin bệnh nhân
    response = client.post(
        "/predict",
        files={"file": ("test_patient.png", file_bytes, "image/png")},
        data={"patient_name": "Nguyễn Văn Anh"}
    )
    assert response.status_code == 200
    res_data = response.json()
    diag_id = res_data["id"]

    # 1. Kiểm tra trong cơ sở dữ liệu xem thông tin đã được mã hóa chưa (Security at rest)
    record = db_session.query(DiagnosisHistory).filter(DiagnosisHistory.id == diag_id).first()
    assert record is not None
    # Tên trong DB phải khác hoàn toàn văn bản gốc "Nguyễn Văn Anh" và có cấu trúc mã hóa Base64
    assert record.patient_name != "Nguyễn Văn Anh"
    
    # 2. Kiểm tra giải mã khi lấy lịch sử cho Mobile client
    history_response = client.get("/history")
    assert history_response.status_code == 200
    history_data = history_response.json()
    target_record = next(r for r in history_data if r["id"] == diag_id)
    # API lịch sử cho Mobile phải trả về tên giải mã đầy đủ
    assert target_record["patient_name"] == "Nguyễn Văn Anh"

    # 3. Kiểm tra ẩn danh hóa trên giao diện Web (Wrapper)
    from server import WebDiagnosisWrapper
    wrapper = WebDiagnosisWrapper(record)
    # Tên hiển thị trên web phải được mặt nạ hóa ẩn danh
    assert wrapper.patient_name == "N. V. Anh"

    # Dọn dẹp
    for path_key in ["image_url", "heatmap_url", "gradcam_url"]:
        rel_path = res_data[path_key].lstrip("/")
        abs_path = os.path.join(os.getcwd(), "webapp", rel_path)
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
            except Exception:
                pass
