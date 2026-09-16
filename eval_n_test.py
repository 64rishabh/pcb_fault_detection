#!/usr/bin/env python3
"""Evaluate YOLOv8n best weights on test split explicitly."""
from ultralytics import YOLO

model = YOLO("runs/detect/train-5/weights/best.pt")
results = model.val(
    data="PCB Defect.v3i.yolov8/data.yaml",
    split="test",
    imgsz=640,
    workers=0,
)
print("\n=== TEST EVALUATION RESULTS ===")
print(f"mAP@50: {results.box.map50:.4f}")
print(f"mAP@50-95: {results.box.map:.4f}")
