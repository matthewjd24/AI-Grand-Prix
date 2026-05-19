"""Train a YOLO-pose CNN to detect drone-racing gate corners.

Prerequisites (must be done BEFORE running this):
  1. data.yaml is correct and points at a dataset root with this layout:
       dataset/images/train, dataset/images/val
       dataset/labels/train, dataset/labels/val
  2. The label .txt files exist (converted from the Unity JSON sidecars).
     Empty/negative frames still need an empty .txt file.

Run:  c:\\python313\\python.exe test.py
"""

import torch
from ultralytics import YOLO

if __name__ == "__main__":
    # CPU if no CUDA GPU is available, otherwise GPU 0.
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | device: {device}")

    # Pretrained nano pose model -> fine-tuned on our gate dataset.
    # 'n' (nano) is the fastest; bump to yolo11s-pose.pt for more accuracy.
    model = YOLO("yolo11n-pose.pt")

    results = model.train(
        data="data.yaml",   # dataset config in this folder
        epochs=100,
        imgsz=640,          # competition camera feed is 640x360; YOLO pads to square
        batch=16,           # lower this (e.g. 4) if you run out of memory on CPU
        device=device,
        patience=20,        # early-stop if val metrics plateau for 20 epochs
        project="runs",     # output folder
        name="gate-pose",   # runs/gate-pose/
    )
    print("Training complete. Best weights: runs/gate-pose/weights/best.pt")
