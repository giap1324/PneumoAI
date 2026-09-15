import os
import sys
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Thêm đường dẫn backend vào sys.path để import dễ dàng
backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app flutter", "backend")
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Đảm bảo thiết lập môi trường test sử dụng file sqlite cố định để tránh mất bảng khi đóng connection
TEST_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_pneumonia.db")
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

from server import app, Base, get_db

# Thiết lập Database SQLite file cho quá trình test
engine = create_engine(f"sqlite:///{TEST_DB_PATH}", connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    # Xóa file cũ nếu còn tồn tại để đảm bảo test cô lập
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass
            
    # Tạo lại bảng trước mỗi ca kiểm thử
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        # Xóa file test sau khi test xong
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except Exception:
                pass

@pytest.fixture(scope="function")
def client(db_session):
    # Override get_db của FastAPI để trả về session từ test db
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
            
    app.dependency_overrides[get_db] = override_get_db
    from fastapi.testclient import TestClient
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
