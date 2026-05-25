"""One-time script: export a trained .pt checkpoint to a TensorRT .engine file
for faster inference. The .engine is GPU-architecture-specific — rebuild on
whatever machine you intend to run inference on.

Run:  python export_engine.py
Output: a .engine file next to the .pt weights, ready to use in predict.py.
"""

import os
from ultralytics import YOLO

WEIGHTS = os.path.join(os.path.dirname(__file__), r"runs\pose\runs\gate-pose-5\weights\best.pt")
IMG_SIZE = 640        # must match what the model was trained at
USE_HALF = True       # FP16 — about 2x faster than FP32 on RTX 30/40 series

if __name__ == "__main__":
    print(f"Loading {WEIGHTS}")
    model = YOLO(WEIGHTS)

    print(f"Exporting to TensorRT engine (imgsz={IMG_SIZE}, half={USE_HALF})...")
    print("This takes ~2-10 minutes. TensorRT is recompiling kernels for your GPU.")
    engine_path = model.export(format="engine", half=USE_HALF, imgsz=IMG_SIZE)
    print(f"\nEngine written to: {engine_path}")
    print("Update predict.py's _WEIGHTS to point at this .engine file.")
