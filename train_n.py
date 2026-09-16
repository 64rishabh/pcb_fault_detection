#!/usr/bin/env python3
"""Run YOLOv8n baseline training."""
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    cfg="configs/yolov8n_baseline.yaml",
    data="PCB Defect.v3i.yolov8/data.yaml",
)
