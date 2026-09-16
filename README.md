# PCB Defect Detection — YOLOv8 Fine-Tuning

Automated deep-learning system for detecting manufacturing defects on printed circuit boards (PCBs). Uses YOLOv8 (`yolov8n` and `yolov8s`) fine-tuned on the PKU-Market-PCB dataset (6 defect classes).

## Quick Start

```bash
# 1. Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # ultralytics, torch, gradio, numpy, pandas, matplotlib

# 2. Verify dataset
ls "PCB Defect.v3i.yolov8/"

# 3. Run training (uses config YAMLs as source of truth)
python train_n.py   # YOLOv8n baseline (batch=8, amp=True, seed=42)
python train_s.py   # YOLOv8s baseline (batch=6, amp=True, seed=42, workers=0)

# 4. Evaluate on test set
python eval_n_test.py

# 5. Benchmark (overall + per-class metrics)
python benchmark_n.py
python benchmark_s.py

# 6. Launch Gradio demo
python app_gradio.py  # Opens at http://localhost:7860
```

## Project Structure

| Path | Purpose |
|---|---|
| `bible.md` | Source of truth: problem, data details, model specs, hyperparameter policy, benchmark results, limitations |
| `configs/` | `yolov8n_baseline.yaml`, `yolov8s_baseline.yaml` — reproducible run configs |
| `PCB Defect.v3i.yolov8/` | Dataset (train/valid/test, 640×640, YOLOv8 format) |
| `models/` | Best fine-tuned weights (`yolov8n_best.pt`, `yolov8s_best.pt`) |
| `benchmarks/` | `benchmark_n.json`, `benchmark_s.json` (test-set metrics) |
| `frontend/app_gradio.py` | Gradio dashboard: upload image → side-by-side predictions (n vs s) with bounding boxes, class labels, and confidence scores |
| `runs/` | Training outputs (excluded from git) |

## Key Results (Test Set — `split="test"`)

| Metric | YOLOv8n (nano) | YOLOv8s (small) |
|---|---|---|
| mAP@0.50 | 0.9908 | 0.9924 |
| mAP@0.50:0.95 | 0.5738 | 0.6212 |
| Precision | 0.9808 | 0.9820 |
| Recall | 0.9856 | 0.9912 |
| F1 | 0.9832 | 0.9866 |

Full per-class breakdown: see `bible.md` section 3 and `benchmarks/benchmark_*.json`.

## References

- Dataset provenance: `PCB Defect.v3i.yolov8/README.roboflow.txt`
- Dataset config: `PCB Defect.v3i.yolov8/data.yaml`
- Model weights: `models/yolov8n_best.pt`, `models/yolov8s_best.pt`
- Config source of truth: `configs/yolov8n_baseline.yaml`, `configs/yolov8s_baseline.yaml`
- Benchmark scripts: `benchmark_n.py`, `benchmark_s.py`
