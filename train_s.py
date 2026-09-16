#!/usr/bin/env python3
"""Run YOLOv8s baseline training using config source of truth."""
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    cfg="configs/yolov8s_baseline.yaml",
    data="PCB Defect.v3i.yolov8/data.yaml",
    workers=0,
)
