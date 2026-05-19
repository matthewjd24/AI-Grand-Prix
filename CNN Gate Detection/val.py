"""Evaluate a trained checkpoint on the validation set."""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"runs\pose\runs\gate-pose\weights\last.pt")   # or best.pt
    metrics = model.val(data="data.yaml", split="val")
    print("Box mAP50-95:", metrics.box.map)
    print("Pose mAP50-95:", metrics.pose.map)
