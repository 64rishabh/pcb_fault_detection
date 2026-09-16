#!/usr/bin/env python3
"""Benchmark YOLOv8s best weights on test split."""
import json
from ultralytics import YOLO

CLASS_NAMES = [
    "missing_hole", "mouse_bite", "open_circuit",
    "short", "spur", "spurious_copper",
]

def main():
    model = YOLO("models/yolov8s_best.pt")
    res = model.val(
        data="PCB Defect.v3i.yolov8/data.yaml",
        split="test",
        imgsz=640,
        workers=0,
        verbose=False,
    )

    mean_res = res.mean_results()
    P_all, R_all, mAP50_all, mAP50_95_all = mean_res
    F1_all = 2 * P_all * R_all / (P_all + R_all + 1e-6)

    per_class = []
    for i in range(len(CLASS_NAMES)):
        cr = res.class_result(i)
        P, R, mAP50, mAP50_95 = cr
        F1 = 2 * P * R / (P + R + 1e-6)
        per_class.append({
            "class_id": i,
            "name": CLASS_NAMES[i],
            "P": float(P),
            "R": float(R),
            "mAP50": float(mAP50),
            "mAP50-95": float(mAP50_95),
            "F1": float(F1),
        })

    result = {
        "model": "yolov8s",
        "best_weights": "models/yolov8s_best.pt",
        "split": "test",
        "overall": {
            "P": float(P_all),
            "R": float(R_all),
            "mAP50": float(mAP50_all),
            "mAP50-95": float(mAP50_95_all),
            "F1": float(F1_all),
        },
        "per_class": per_class,
    }

    import os
    os.makedirs("results", exist_ok=True)
    with open("benchmarks/benchmark_s.json", "w") as f:
        json.dump(result, f, indent=2)

    print("=== YOLOv8s Benchmark (test) ===")
    print(f"Overall  | mAP50={mAP50_all:.4f} mAP50-95={mAP50_95_all:.4f} P={P_all:.4f} R={R_all:.4f} F1={F1_all:.4f}")
    for c in per_class:
        print(f"Class {c['class_id']:>1d} | {c['name']:>18s} | mAP50={c['mAP50']:.4f} mAP50-95={c['mAP50-95']:.4f} P={c['P']:.4f} R={c['R']:.4f} F1={c['F1']:.4f}")

if __name__ == "__main__":
    main()
