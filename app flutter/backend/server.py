import json
import os
import io
import uuid
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from einops.layers.torch import Rearrange
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, inspect, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Custom lightweight .env loader
def load_env():
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.getcwd(), ".env"),
        ".env"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            parts = line.split("=", 1)
                            if len(parts) == 2:
                                key = parts[0].strip()
                                val = parts[1].strip()
                                os.environ[key] = val
                print(f"Loaded environment variables from: {path}")
                break
            except Exception as e:
                print(f"Warning: Could not read .env file at {path}: {e}")

load_env()

# MySQL/SQLite Connection
MYSQL_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost:3306/pneumonia_db")

# Auto-create MySQL database if it doesn't exist
if "mysql" in MYSQL_URL:
    import pymysql
    try:
        parts = MYSQL_URL.split("://")[1].split("/")
        creds_host = parts[0]
        dbname = parts[1].split("?")[0]
        
        user_pass, host_port = creds_host.split("@")
        user = user_pass.split(":")[0]
        password = user_pass.split(":")[1] if ":" in user_pass else ""
        
        host = host_port.split(":")[0]
        port = int(host_port.split(":")[1]) if ":" in host_port else 3306
        
        conn = pymysql.connect(host=host, port=port, user=user, password=password)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {dbname} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Database '{dbname}' verified/created successfully.")
    except Exception as e:
        print(f"Warning: Could not automatically verify/create MySQL database: {e}")

STATIC_DIR = "webapp/static"
UPLOAD_DIR = "webapp/static/uploads"
GRADCAM_DIR = "webapp/static/gradcam"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

CLASSES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hybrid_vit_pneumonia_best_v10.pth")

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT DATA SECURITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
import base64

DB_ENCRYPTION_KEY = os.getenv("DB_ENCRYPTION_KEY", "PneumoScanSecretKey")

def encrypt_value(val: str) -> str:
    """Mã hóa đối xứng dữ liệu bệnh nhân sử dụng XOR và Base64 để bảo vệ dữ liệu tĩnh."""
    if not val:
        return val
    key = DB_ENCRYPTION_KEY
    cipher = "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(val))
    return base64.b64encode(cipher.encode('utf-8', errors='ignore')).decode('utf-8')

def decrypt_value(val: str) -> str:
    """Giải mã đối xứng dữ liệu bệnh nhân."""
    if not val:
        return val
    try:
        decoded = base64.b64decode(val.encode('utf-8')).decode('utf-8', errors='ignore')
        key = DB_ENCRYPTION_KEY
        return "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(decoded))
    except Exception:
        return val

def mask_patient_name(name: str) -> str:
    """Ẩn danh họ tên bệnh nhân (Ví dụ: 'Nguyễn Văn Anh' -> 'N. V. Anh') để tăng tính bảo mật hiển thị."""
    if not name:
        return ""
    parts = name.strip().split()
    if len(parts) == 0:
        return ""
    if len(parts) == 1:
        return parts[0][0] + "*" * max(1, len(parts[0]) - 1)
    
    masked_parts = [p[0] + "." for p in parts[:-1]]
    masked_parts.append(parts[-1])
    return " ".join(masked_parts)

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE SETUP (SQLAlchemy)
# ─────────────────────────────────────────────────────────────────────────────

Base = declarative_base()

class DiagnosisHistory(Base):
    __tablename__ = "diagnosis_history"
    id = Column(Integer, primary_key=True, index=True)
    image_url = Column(String(255))
    heatmap_url = Column(String(255), nullable=True)
    gradcam_url = Column(String(255), nullable=True)
    label = Column(String(50))
    confidence = Column(Float)
    normal_prob = Column(Float)
    pneumonia_prob = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    severity_json = Column(Text, nullable=True)
    patient_name = Column(String(100), nullable=True)
    patient_age = Column(Integer, nullable=True)
    patient_gender = Column(String(20), nullable=True)
    notes = Column(Text, nullable=True)
    inference_time = Column(Float, nullable=True)
    gradcam_time = Column(Float, nullable=True)
    preprocess_time = Column(Float, nullable=True)



engine = create_engine(MYSQL_URL, connect_args={"check_same_thread": False} if "sqlite" in MYSQL_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Tự động cập nhật cấu trúc database nếu thiếu cột
def run_migrations():
    inspector = inspect(engine)
    if 'diagnosis_history' in inspector.get_table_names():
        columns = [c['name'] for c in inspector.get_columns('diagnosis_history')]
        with engine.connect() as conn:
            modified = False
            if 'heatmap_url' not in columns:
                conn.execute(text("ALTER TABLE diagnosis_history ADD COLUMN heatmap_url VARCHAR(255)"))
                print("Successfully added missing column: heatmap_url")
                modified = True
            if 'patient_name' not in columns:
                conn.execute(text("ALTER TABLE diagnosis_history ADD COLUMN patient_name VARCHAR(100)"))
                print("Successfully added missing column: patient_name")
                modified = True
            if 'patient_age' not in columns:
                conn.execute(text("ALTER TABLE diagnosis_history ADD COLUMN patient_age INT"))
                print("Successfully added missing column: patient_age")
                modified = True
            if 'patient_gender' not in columns:
                conn.execute(text("ALTER TABLE diagnosis_history ADD COLUMN patient_gender VARCHAR(20)"))
                print("Successfully added missing column: patient_gender")
                modified = True
            if 'notes' not in columns:
                conn.execute(text("ALTER TABLE diagnosis_history ADD COLUMN notes TEXT"))
                print("Successfully added missing column: notes")
                modified = True
            if 'inference_time' not in columns:
                conn.execute(text("ALTER TABLE diagnosis_history ADD COLUMN inference_time FLOAT"))
                print("Successfully added missing column: inference_time")
                modified = True
            if 'gradcam_time' not in columns:
                conn.execute(text("ALTER TABLE diagnosis_history ADD COLUMN gradcam_time FLOAT"))
                print("Successfully added missing column: gradcam_time")
                modified = True
            if 'preprocess_time' not in columns:
                conn.execute(text("ALTER TABLE diagnosis_history ADD COLUMN preprocess_time FLOAT"))
                print("Successfully added missing column: preprocess_time")
                modified = True
            if modified:
                conn.commit()

run_migrations()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─────────────────────────────────────────────────────────────────────────────
# AI MODEL ARCHITECTURE (HybridViT)
# ─────────────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training: return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep * (rand < keep).float()

class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_dropout(attn.softmax(dim=-1))
        x = self.proj_dropout(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout2(self.fc2(self.dropout1(self.act(self.fc1(x)))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4, dropout=0.1, attn_dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = EfficientMultiHeadAttention(embed_dim, num_heads, attn_dropout)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CNNBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        feat = densenet121(weights=w).features
        self.conv0 = feat.conv0
        self.norm0 = feat.norm0
        self.relu0 = feat.relu0
        self.pool0 = feat.pool0
        self.denseblock1 = feat.denseblock1
        self.transition1 = feat.transition1
        self.denseblock2 = feat.denseblock2
        self.transition2 = feat.transition2
        self.denseblock3 = feat.denseblock3
        self.out_channels = 1024
    def forward(self, x):
        x = self.pool0(self.relu0(self.norm0(self.conv0(x))))
        x = self.transition1(self.denseblock1(x))
        x = self.transition2(self.denseblock2(x))
        x = self.denseblock3(x)
        return x

class HybridViT(nn.Module):
    def __init__(self, num_classes=2, embed_dim=512, depth=6, num_heads=4, mlp_ratio=4, dropout=0.1, attn_dropout=0.1, drop_path_rate=0.1, pretrained_backbone=False):
        super().__init__()
        self.backbone = CNNBackbone(pretrained=pretrained_backbone)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(self.backbone.out_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            Rearrange('b c h w -> b (h w) c')
        )
        self.num_patches = 14 * 14
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.transformer = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim // 2, num_classes))

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone(x)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.transformer: x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM & SEVERITY LOGIC
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output): self.activations = output.detach()
    def _save_gradient(self, module, grad_in, grad_out): self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        if class_idx is None: class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probs.detach().cpu().numpy()[0]

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP & ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="PneumoScan AI Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount webapp static folder at /static
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

# Setup Jinja2 templates from webapp/templates
templates = Jinja2Templates(directory="webapp/templates")

# Custom url_for to be compatible with Flask templates
def web_url_for(name: str, **path_params):
    if name == 'index':
        return "/"
    elif name == 'history':
        return "/web-history"
    elif name == 'about':
        return "/about"
    elif name == 'api_predict':
        return "/api_predict"
    elif name == 'api_stats':
        return "/api_stats"
    elif name == 'clear_history_on_exit':
        return "/clear_history_on_exit"
    elif name == 'result':
        diag_id = path_params.get("diagnosis_id", 0)
        return f"/result/{diag_id}"
    elif name == 'static':
        filename = path_params.get("filename", "")
        return f"/static/{filename}"
    return "#"

templates.env.globals['url_for'] = web_url_for
templates.env.globals['get_flashed_messages'] = lambda *args, **kwargs: []

# Load Model
model = HybridViT(depth=6, num_heads=4, embed_dim=512).to(DEVICE)
if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()
gradcam_tool = GradCAM(model, model.backbone.denseblock3)

infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─── Flutter Mobile endpoints ─────────────────────────────────────────────────

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_name: str = Form(None),
    patient_age: str = Form(None),
    patient_gender: str = Form(None),
    db: Session = Depends(get_db)
):
    import time
    # Measure preprocess time (T1) starting from request input read
    start_preprocess = time.perf_counter()
    content = await file.read()
    if not content or len(content) == 0:
        raise HTTPException(status_code=400, detail="Tệp tin tải lên bị trống hoặc không tồn tại")

    # Validation 2: check age format and range
    p_age = None
    if patient_age and patient_age.strip():
        try:
            p_age = int(patient_age)
            if p_age < 0 or p_age > 120:
                raise HTTPException(status_code=400, detail="Tuổi bệnh nhân phải nằm trong khoảng từ 0 đến 120")
        except ValueError:
            raise HTTPException(status_code=400, detail="Tuổi bệnh nhân phải là một số nguyên hợp lệ")

    # Validation 3: check image format
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Tệp tin không phải là ảnh hợp lệ hoặc định dạng không được hỗ trợ")

    try:
        tensor = infer_transform(image).unsqueeze(0).to(DEVICE)
        preprocess_time = time.perf_counter() - start_preprocess
        
        # Measure inference time (forward pass)
        start_infer = time.perf_counter()
        with torch.no_grad():
            _ = model(tensor)
        inference_time = time.perf_counter() - start_infer
        
        # Measure gradcam generation time (forward + backward + interpolation)
        start_gradcam = time.perf_counter()
        with torch.enable_grad():
            heatmap, pred_idx, probs = gradcam_tool.generate(tensor)
        gradcam_time = time.perf_counter() - start_gradcam
        
        label = CLASSES[pred_idx]
        
        # Save original image to static/uploads
        filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(UPLOAD_DIR, filename)
        image.save(image_path)
        
        # Create Grad-CAM Overlay
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        heatmap_colored = cv2.resize(heatmap_colored, (img_bgr.shape[1], img_bgr.shape[0]))
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
        
        # Save Heatmap alone to static/gradcam
        heatmap_filename = f"heatmap_{filename}"
        heatmap_path = os.path.join(GRADCAM_DIR, heatmap_filename)
        cv2.imwrite(heatmap_path, heatmap_colored)

        # Save Overlay to static/gradcam
        gradcam_filename = f"gradcam_{filename}"
        gradcam_path = os.path.join(GRADCAM_DIR, gradcam_filename)
        cv2.imwrite(gradcam_path, overlay)
        
        # Clean inputs
        p_name = patient_name.strip() if (patient_name and patient_name.strip()) else None
        p_gender = patient_gender.strip() if (patient_gender and patient_gender.strip()) else None
        
        # Save to DB (Name encrypted at rest)
        new_entry = DiagnosisHistory(
            image_url=f"/static/uploads/{filename}",
            heatmap_url=f"/static/gradcam/{heatmap_filename}",
            gradcam_url=f"/static/gradcam/{gradcam_filename}",
            label=label,
            confidence=float(probs[pred_idx]),
            normal_prob=float(probs[0]),
            pneumonia_prob=float(probs[1]),
            severity_json=None,
            patient_name=encrypt_value(p_name),
            patient_age=p_age,
            patient_gender=p_gender,
            notes=encrypt_value(None),
            inference_time=float(inference_time),
            gradcam_time=float(gradcam_time),
            preprocess_time=float(preprocess_time)
        )
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)
        
        return {
            "id": new_entry.id,
            "label": label,
            "confidence": float(probs[pred_idx]),
            "probabilities": {"NORMAL": float(probs[0]), "PNEUMONIA": float(probs[1])},
            "image_url": f"/static/uploads/{filename}",
            "heatmap_url": f"/static/gradcam/{heatmap_filename}",
            "gradcam_url": f"/static/gradcam/{gradcam_filename}",
            "severity": None,
            "inference_time": float(inference_time),
            "gradcam_time": float(gradcam_time),
            "preprocess_time": float(preprocess_time)
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    records = db.query(DiagnosisHistory).order_by(DiagnosisHistory.timestamp.desc()).all()
    return [
        {
            "id": r.id,
            "image_url": r.image_url,
            "heatmap_url": r.heatmap_url,
            "gradcam_url": r.gradcam_url,
            "label": r.label,
            "confidence": r.confidence,
            "normal_prob": r.normal_prob,
            "pneumonia_prob": r.pneumonia_prob,
            "timestamp": r.timestamp.isoformat(),
            "patient_name": decrypt_value(r.patient_name),
            "patient_age": r.patient_age,
            "patient_gender": r.patient_gender,
            "severity": json.loads(r.severity_json) if r.severity_json else None,
            "inference_time": r.inference_time,
            "gradcam_time": r.gradcam_time,
            "preprocess_time": r.preprocess_time
        } for r in records
    ]

@app.delete("/history/{record_id}")
def delete_history(record_id: int, db: Session = Depends(get_db)):
    record = db.query(DiagnosisHistory).filter(DiagnosisHistory.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    # Delete files
    for path in [record.image_url, record.heatmap_url, record.gradcam_url]:
        if path:
            abs_path = os.path.join(os.getcwd(), "webapp", path.lstrip("/"))
            if os.path.exists(abs_path):
                os.remove(abs_path)
                
    db.delete(record)
    db.commit()
    return {"message": "Deleted successfully"}

# ─────────────────────────────────────────────────────────────────────────────
# WEB ROUTE HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

class WebDiagnosisWrapper:
    def __init__(self, record):
        self._record = record

    @property
    def id(self): return self._record.id
    @property
    def created_at(self): return self._record.timestamp
    @property
    def patient_name(self): 
        decrypted = decrypt_value(self._record.patient_name)
        return mask_patient_name(decrypted) if decrypted else None
    @property
    def patient_age(self): return self._record.patient_age
    @property
    def patient_gender(self): return self._record.patient_gender
    @property
    def notes(self): 
        return decrypt_value(self._record.notes)
    @property
    def prediction(self): return "Viêm phổi" if self._record.label == "PNEUMONIA" else "Bình thường"
    @property
    def prediction_vi(self): return "Viêm phổi" if self._record.label == "PNEUMONIA" else "Bình thường"
    
    @property
    def filename(self):
        return os.path.basename(self._record.image_url) if self._record.image_url else ""
    
    @property
    def gradcam_path(self):
        return os.path.basename(self._record.gradcam_url) if self._record.gradcam_url else None

    @property
    def confidence(self): return self._record.confidence * 100
    @property
    def prob_normal(self): return self._record.normal_prob * 100
    @property
    def prob_pneumonia(self): return self._record.pneumonia_prob * 100

    @property
    def inference_time(self): return self._record.inference_time
    @property
    def gradcam_time(self): return self._record.gradcam_time
    @property
    def preprocess_time(self): return self._record.preprocess_time

    # Severity fields (return None to disable Brixia display safely)
    @property
    def severity_grade_vi(self): return None
    @property
    def severity_grade_en(self): return None
    @property
    def severity_score(self): return None
    @property
    def affected_area_pct(self): return None
    @property
    def left_area_pct(self): return None
    @property
    def right_area_pct(self): return None
    @property
    def mean_intensity(self): return None
    @property
    def severity_description(self): return None

class MockPagination:
    def __init__(self, items, total, page, per_page):
        self.items = items
        self.total = total
        self.page = page
        self.per_page = per_page
        self.pages = (total + per_page - 1) // per_page if total > 0 else 0
        self.has_prev = page > 1
        self.has_next = page < self.pages
        self.prev_num = page - 1
        self.next_num = page + 1

    def iter_pages(self):
        return list(range(1, self.pages + 1))

@app.get("/", response_class=HTMLResponse)
async def web_index(request: Request, db: Session = Depends(get_db)):
    request.scope["endpoint"] = "index"
    total_analyses = db.query(DiagnosisHistory).count()
    return templates.TemplateResponse(request, "index.html", {
        "total_analyses": total_analyses
    })

@app.get("/about", response_class=HTMLResponse)
async def web_about(request: Request):
    request.scope["endpoint"] = "about"
    return templates.TemplateResponse(request, "about.html")

@app.get("/web-history", response_class=HTMLResponse)
async def web_history(request: Request, page: int = 1, db: Session = Depends(get_db)):
    request.scope["endpoint"] = "history"
    per_page = 10
    total = db.query(DiagnosisHistory).count()
    offset = (page - 1) * per_page
    records = db.query(DiagnosisHistory).order_by(DiagnosisHistory.timestamp.desc()).offset(offset).limit(per_page).all()
    
    web_records = [WebDiagnosisWrapper(r) for r in records]
    diagnoses = MockPagination(web_records, total, page, per_page)
    return templates.TemplateResponse(request, "history.html", {
        "diagnoses": diagnoses
    })

@app.get("/result/{diagnosis_id}", response_class=HTMLResponse)
async def web_result(request: Request, diagnosis_id: int, db: Session = Depends(get_db)):
    request.scope["endpoint"] = "result"
    diagnosis = db.query(DiagnosisHistory).filter(DiagnosisHistory.id == diagnosis_id).first()
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Diagnosis not found")
    web_diag = WebDiagnosisWrapper(diagnosis)
    return templates.TemplateResponse(request, "result.html", {
        "diagnosis": web_diag
    })

@app.post("/api_predict")
async def api_predict(
    file: UploadFile = File(...),
    patient_name: str = Form(None),
    patient_age: str = Form(None),
    patient_gender: str = Form(None),
    notes: str = Form(None),
    db: Session = Depends(get_db)
):
    try:
        import time
        start_preprocess = time.perf_counter()
        content = await file.read()
        # Validation 1: Check empty file
        if not content or len(content) == 0:
            return {"success": False, "error": "Tệp tin tải lên bị trống hoặc không tồn tại"}

        # Validation 2: Check patient age range
        p_age = None
        if patient_age and patient_age.strip():
            try:
                p_age = int(patient_age)
                if p_age < 0 or p_age > 120:
                    return {"success": False, "error": "Tuổi bệnh nhân phải nằm trong khoảng từ 0 đến 120"}
            except ValueError:
                return {"success": False, "error": "Tuổi bệnh nhân phải là một số nguyên hợp lệ"}

        # Validation 3: Check image validation
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            return {"success": False, "error": "Tệp tin không phải là ảnh hợp lệ hoặc định dạng không được hỗ trợ"}

        tensor = infer_transform(image).unsqueeze(0).to(DEVICE)
        preprocess_time = time.perf_counter() - start_preprocess
        
        # Measure inference time
        start_infer = time.perf_counter()
        with torch.no_grad():
            _ = model(tensor)
        inference_time = time.perf_counter() - start_infer
        
        # Measure gradcam generation time
        start_gradcam = time.perf_counter()
        with torch.enable_grad():
            heatmap, pred_idx, probs = gradcam_tool.generate(tensor)
        gradcam_time = time.perf_counter() - start_gradcam
        
        label = CLASSES[pred_idx]
        
        # Save images
        filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(UPLOAD_DIR, filename)
        image.save(image_path)
        
        # Create Grad-CAM Overlay
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        heatmap_colored = cv2.resize(heatmap_colored, (img_bgr.shape[1], img_bgr.shape[0]))
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
        
        # Save Heatmap alone
        heatmap_filename = f"heatmap_{filename}"
        heatmap_path = os.path.join(GRADCAM_DIR, heatmap_filename)
        cv2.imwrite(heatmap_path, heatmap_colored)

        # Save Overlay
        gradcam_filename = f"gradcam_{filename}"
        gradcam_path = os.path.join(GRADCAM_DIR, gradcam_filename)
        cv2.imwrite(gradcam_path, overlay)
        
        # Process optional text inputs to store clean NULL values
        p_name = patient_name.strip() if (patient_name and patient_name.strip()) else None
        p_gender = patient_gender.strip() if (patient_gender and patient_gender.strip()) else None
        p_notes = notes.strip() if (notes and notes.strip()) else None
        
        # Save to DB (Name and notes encrypted)
        new_entry = DiagnosisHistory(
            image_url=f"/static/uploads/{filename}",
            heatmap_url=f"/static/gradcam/{heatmap_filename}",
            gradcam_url=f"/static/gradcam/{gradcam_filename}",
            label=label,
            confidence=float(probs[pred_idx]),
            normal_prob=float(probs[0]),
            pneumonia_prob=float(probs[1]),
            severity_json=None,
            patient_name=encrypt_value(p_name),
            patient_age=p_age,
            patient_gender=p_gender,
            notes=encrypt_value(p_notes),
            inference_time=float(inference_time),
            gradcam_time=float(gradcam_time),
            preprocess_time=float(preprocess_time)
        )
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)
        
        return {
            "success": True,
            "diagnosis_id": new_entry.id
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api_stats")
def api_stats(db: Session = Depends(get_db)):
    normal_count = db.query(DiagnosisHistory).filter(DiagnosisHistory.label == "NORMAL").count()
    pneumonia_count = db.query(DiagnosisHistory).filter(DiagnosisHistory.label == "PNEUMONIA").count()
    
    records = db.query(DiagnosisHistory.confidence).all()
    avg_confidence = 0.0
    if records:
        avg_confidence = sum([r[0] for r in records]) / len(records) * 100.0
        
    return {
        "normal_count": normal_count,
        "pneumonia_count": pneumonia_count,
        "avg_confidence": avg_confidence
    }

@app.get("/delete/{record_id}")
def web_delete_history(record_id: int, db: Session = Depends(get_db)):
    record = db.query(DiagnosisHistory).filter(DiagnosisHistory.id == record_id).first()
    if record:
        for path in [record.image_url, record.heatmap_url, record.gradcam_url]:
            if path:
                abs_path = os.path.join(os.getcwd(), "webapp", path.lstrip("/"))
                if os.path.exists(abs_path):
                    try:
                        os.remove(abs_path)
                    except Exception:
                        pass
        db.delete(record)
        db.commit()
    return RedirectResponse(url="/web-history", status_code=303)

@app.post("/clear_history_on_exit")
def clear_history_on_exit():
    return {"message": "session cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
