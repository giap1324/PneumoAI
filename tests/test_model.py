import os
import sys
import torch
import numpy as np

# Thêm đường dẫn backend vào sys.path
backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app flutter", "backend")
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from server import HybridViT, GradCAM, CLASSES

def test_model_initialization():
    """Kiểm tra mô hình HybridViT khởi tạo thành công với cấu hình chuẩn."""
    model = HybridViT(depth=2, num_heads=2, embed_dim=128)  # Dùng cấu hình nhỏ để chạy test nhanh
    assert model is not None
    assert isinstance(model, torch.nn.Module)

def test_model_forward_pass():
    """Kiểm tra lan truyền xuôi của mô hình HybridViT."""
    model = HybridViT(depth=2, num_heads=2, embed_dim=128)
    model.eval()
    
    # Tạo tensor giả lập ảnh đầu vào (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    # Đầu ra phải có dạng (batch_size, num_classes) -> (1, 2)
    assert output.shape == (1, 2)
    assert not torch.isnan(output).any(), "Kết quả chứa giá trị NaN"

def test_gradcam_generation():
    """Kiểm tra giải thuật sinh bản đồ giải thích GradCAM."""
    model = HybridViT(depth=2, num_heads=2, embed_dim=128)
    model.eval()
    
    # Khởi tạo GradCAM nhắm vào lớp denseblock3
    gradcam = GradCAM(model, model.backbone.denseblock3)
    
    # Tạo tensor đầu vào giả lập
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Chạy GradCAM
    with torch.enable_grad():
        heatmap, pred_idx, probs = gradcam.generate(dummy_input)
        
    # Kiểm tra kiểu và hình dạng dữ liệu đầu ra
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert pred_idx in [0, 1]
    assert len(probs) == 2
    assert 0.0 <= probs[0] <= 1.0
    assert 0.0 <= probs[1] <= 1.0
    assert np.allclose(sum(probs), 1.0)
    
    # Kiểm tra bản đồ nhiệt đã được chuẩn hóa trong khoảng [0, 1]
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
