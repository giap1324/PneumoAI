import os
import io
import sys
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add backend directory to sys.path
backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app flutter", "backend")
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from server import HybridViT, GradCAM, infer_transform

def run_benchmark(num_runs=50):
    print("Configuring device...")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    print("Initializing HybridViT model...")
    model = HybridViT(depth=6, num_heads=4, embed_dim=512).to(DEVICE)
    model.eval()

    # Load weights if available
    model_path = os.path.join(backend_dir, "hybrid_vit_pneumonia_best_v10.pth")
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            print("Successfully loaded model weights from best_v10.pth")
        except Exception as e:
            print(f"Failed to load model weights: {e}. Using random weights.")
    else:
        print("Model weights not found. Using random weights.")

    gradcam_tool = GradCAM(model, model.backbone.denseblock3)

    t1_list = []
    t2_list = []
    t3_list = []
    total_list = []

    # Mock image representation
    img_data = io.BytesIO()
    Image.new('RGB', (1024, 1024), color='white').save(img_data, format='PNG')
    img_bytes = img_data.getvalue()

    print("Starting warm-up (5 runs)...")
    for _ in range(5):
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = infer_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _ = model(tensor)
        with torch.enable_grad():
            _, _, _ = gradcam_tool.generate(tensor)

    print(f"Starting benchmark ({num_runs} runs)...")
    for i in range(num_runs):
        start_total = time.perf_counter()

        # T1: Preprocess
        start_t1 = time.perf_counter()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = infer_transform(image).unsqueeze(0).to(DEVICE)
        t1_time = time.perf_counter() - start_t1
        t1_list.append(t1_time * 1000)

        # T2: Inference
        start_t2 = time.perf_counter()
        with torch.no_grad():
            _ = model(tensor)
        t2_time = time.perf_counter() - start_t2
        t2_list.append(t2_time * 1000)

        # T3: Grad-CAM
        start_t3 = time.perf_counter()
        with torch.enable_grad():
            _, _, _ = gradcam_tool.generate(tensor)
        t3_time = time.perf_counter() - start_t3
        t3_list.append(t3_time * 1000)

        total_time = time.perf_counter() - start_total
        total_list.append(total_time * 1000)

    print("\n================ BENCHMARK LATENCY STATS ================")
    print(f"Total runs:            {num_runs}")
    print(f"T1. Preprocess:        {np.mean(t1_list):.2f} ms +/- {np.std(t1_list):.2f} ms")
    print(f"T2. AI Inference:      {np.mean(t2_list):.2f} ms +/- {np.std(t2_list):.2f} ms")
    print(f"T3. Grad-CAM Gen:      {np.mean(t3_list):.2f} ms +/- {np.std(t3_list):.2f} ms")
    print(f"Total Latency:         {np.mean(total_list):.2f} ms +/- {np.std(total_list):.2f} ms")
    print("==========================================================\n")

if __name__ == "__main__":
    run_benchmark(50)
