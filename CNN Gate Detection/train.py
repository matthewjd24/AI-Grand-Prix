"""Train a YOLO-pose CNN to detect drone-racing gate corners.

Auto-runs convert.py at startup, which generates .txt labels for any PNG
that doesn't have one yet. Up-to-date splits are skipped, so re-running
this script is cheap.

Run:  c:\\python313\\python.exe train.py
"""

import torch
from ultralytics import YOLO

import convert  # generates YOLO .txt labels from the Unity JSON sidecars

if __name__ == "__main__":
    # Bring labels in sync with the current PNG set before training starts.
    convert.main()

    # CPU if no CUDA GPU is available, otherwise GPU 0.
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | device: {device}")

    # Pretrained nano pose model -> fine-tuned on our gate dataset.
    # 'n' (nano) is the fastest; bump to yolo11s-pose.pt for more accuracy.
    model = YOLO("yolo11n-pose.pt")

    results = model.train(
        data="data.yaml",   # dataset config in this folder
        epochs=100,
        imgsz=640,          # matches sim's 640x360 inference; bigger would blow latency budget
        batch=16,
        workers=0,
        device=device,
        pose=30.0,
        multi_scale=True,
        patience=20,        # early-stop if val metrics plateau for 20 epochs
        project="runs",     # output folder
        name="gate-pose",   # runs/gate-pose/
    )
    print("Training complete. Best weights: runs/gate-pose/weights/best.pt")
