# Bible — Source of Truth for PCB Defect Detection Agent

> This file is the single source of truth for the project agent. It must be updated before any action changes. All clarifying questions must be resolved here.

---

## 1. Problem Statement (Fixed from PRD)

Manual visual inspection of printed circuit boards (PCBs) for manufacturing defects — open circuits, short circuits, missing holes, mouse bites, spurs, and spurious copper — is slow, inconsistent, and does not scale in high-volume electronics manufacturing.

This project builds an automated deep-learning-based system that detects and localizes these 6 defect types directly from PCB images, enabling faster and more reliable quality inspection.

**Defect classes (6):** open circuit, short circuit, missing hole, mouse bite, spur, spurious copper.

---

## 2. Data Acquisition (Confirmed — 2026-09-15)

**Dataset folder:** `PCB Defect.v3i.yolov8/` (located in project root; path contains spaces — reference with quotes in scripts).

**Source / provenance:** PKU-Market-PCB dataset, exported via Roboflow (workspace `ta-4ixjm`, project `pcb-defect-gad22`, version 3, CC BY 4.0). URL referenced in `data.yaml`.

**Format on disk:** `train/`, `valid/`, `test/` folders, each with `images/` and `labels/` subfolders. `data.yaml` defines class names, split paths, and metadata.

**Image specs:** Uniform 640x640 (Resize — Stretch), confirmed by `README.roboflow.txt`. Pre-processing: auto-orientation (EXIF stripped), resize to 640x640, auto-contrast via contrast stretching.

**Augmentation applied (from README.roboflow.txt):**
- 50% horizontal flip
- 50% vertical flip
- Random rotation: -15° to +15°
- Random brightness: -15% to +15%
- 3 versions of each source image created

Note: Augmentation details live in README files and will be captured in per-run config YAMLs during experimentation; this file references them but does not duplicate inline values.

**Split ratios (confirmed by file counts):**
- Train: 20,370 images (82%)
- Valid: 2,265 images (9%)
- Test: 2,266 images (9%)
- Total: 24,901 images

**Class counts — train labels (recorded for imbalance tracking):**

| Class ID | Defect Name | Count |
|----------|-------------|-------|
| 0 | missing_hole | 7,406 |
| 1 | mouse_bite | 7,249 |
| 2 | open_circuit | 7,135 |
| 3 | short | 7,220 |
| 4 | spur | 7,275 |
| 5 | spurious_copper | 7,610 |

**Class imbalance:** Confirmed — counts vary from 7,135 (open_circuit) to 7,610 (spurious_copper). Difference ~475 instances (~6.6%). This will be reflected in the final evaluation comparison table and may influence augmentation or class-weight choices during hyperparameter experimentation.

---

## 3. Model Fine-Tuning (Confirmed — 2026-09-15)

**Architecture:**
- Baseline: YOLOv8n (nano) — ~3.2M params, COCO-pretrained (`yolov8n.pt`).
- Comparison: YOLOv8s (small) — ~11.2M params (~3.5× larger), COCO-pretrained (`yolov8s.pt`).

**Head replacement:** Detection head replaced for 6-class task (`nc: 6` in `data.yaml`).

**Fine-tuning strategy:** Full fine-tune (unfreeze all layers). No backbone frozen.

**Batch / workers (Fedora, 4GB allocated from 6GB GPU):**
- YOLOv8n: `batch=8`, `amp=True`, `seed=42`, `workers=2`
- YOLOv8s: `batch=6`, `amp=True`, `seed=42`, `workers=2`

**Hyperparameter policy (confirmed):**
- Each experiment uses one small YAML config file (e.g., `configs/yolov8n_baseline.yaml`, `configs/yolov8s_baseline.yaml`).
- Configs contain: epochs, lr, img_size, batch, workers, augmentation flags, `amp`, and any other run-specific settings.
- Hyperparameters are NOT hardcoded into this planning document; inline values would need constant edits and go stale.
- Config files are the source of truth for re-running any specific experiment.
- Once experimentation concludes and a final configuration is chosen, this file (`bible.md`) will record ONLY:
  - The winning model (n or s) and its best hyperparameter variant
  - A comparison table: model | epochs | LR | augmentation | batch | mAP@0.50 | mAP@0.50:0.95
  - The winning hyperparameter values
- Benchmark results (test split, `split="test"`) recorded after experimentation:

| Metric | YOLOv8n (nano) | YOLOv8s (small) |
|---|---|---|
| **mAP@0.50** | 0.9908 | 0.9924 |
| **mAP@0.50:0.95** | 0.5738 | 0.6212 |
| **Precision (P)** | 0.9808 | 0.9820 |
| **Recall (R)** | 0.9856 | 0.9912 |
| **F1** | 0.9832 | 0.9866 |
| Per-class F1 (missing_hole / mouse_bite / open_circuit / short / spur / spurious_copper) | 0.984 / 0.977 / 0.985 / 0.984 / 0.983 / 0.985 | 0.984 / 0.982 / 0.989 / 0.987 / 0.991 / 0.987 |

- Source: `results/benchmark_n.json`, `results/benchmark_s.json` (extracted via `YOLO.val(split="test")`).
- Winner: `YOLOv8s` outperforms `YOLOv8n` on all overall metrics (`mAP@0.50`: +0.0016, `mAP@0.50:0.95`: +0.0474, `F1`: +0.0034), with tighter per-class F1 spread (0.984–0.991 vs 0.977–0.985). Given the user's instruction to ignore class imbalance and compare only `n` vs `s`, `s` is the stronger baseline.
- Reproducibility: `seed=42` set in both YAML configs for consistent weight initialization and data shuffling.
- Evaluation must use `split="test"` explicitly (`model.val(data=..., split="test")`) since default `val()` uses the validation split.

**Standard YAML filenames (confirmed):** `configs/yolov8n_baseline.yaml`, `configs/yolov8s_baseline.yaml`, etc.

**Configuration directory:** `configs/` (to be created).

**VRAM assessment — YOLOv8s at 4GB (user-requested check):**
- YOLOv8s (~11.2M params) with full unfreeze, batch=6, 640×640 images, `amp=True`: estimated feasible within 4GB, but tight.
- If OOM occurs during the s run: reduce batch to 4, enable `workers=0` temporarily, or switch to gradient accumulation (`batch=6` simulated with `accum=2` at `batch=3`).
- YOLOv8n (`batch=8`) is comfortably within 4GB.
- Recommendation for experimentation: start `yolov8s_baseline.yaml` with `batch: 6`, `amp: true`, `workers: 2`. Monitor GPU memory; adjust downward if needed.

**Evaluation metrics (confirmed):**
- Default / primary: `mAP@0.50`
- Secondary: `mAP@0.50:0.95` (COCO standard)
- Both will be reported; final model selection uses the stronger validation metric, with `mAP@0.50` as the default decision criterion.

**Model weights storage:** Final best weights saved to `models/` (directory excluded from `.gitignore` for large files; weights may be tracked separately or downloaded on-demand).

**Runs / training outputs:** `runs/` (excluded from git via `.gitignore`).

---

## 4. Environment (Fixed — 2026-09-15)

- **OS:** Fedora Workstation (Linux)
- **GPU:** 6GB total VRAM → 4GB allocated to project
- **Virtual env:** Python `venv` (`.venv/`)
- **Batch size:** 8 (YOLOv8n), 6 with `amp=True` (YOLOv8s)
- **Workers:** 2 (Linux-safe, avoids training hangs)
- **Packages:** `ultralytics`, `torch` (CUDA), `gradio`, `jupyter`, `numpy`, `pandas`, `matplotlib`
- **Git repo:** Initialized (`64cdaa2` initial commit: `.gitignore`, `bible.md`)
- **.gitignore exclusions:** `.venv/`, `data/`, `models/`, `runs/`, `results/`, `.env`, Jupyter checkpoints, OS artifacts

---

## 5. Hyperparameter Experimentation — Config Policy (Confirmed)

- Config files live in `configs/`.
- One YAML per run (standard filenames: `yolov8n_baseline.yaml`, `yolov8s_baseline.yaml`, etc.).
- Each YAML specifies: `model`, `data` path, `epochs`, `imgsz`, `batch`, `workers`, `amp`, `lr0`, `lrf`, augmentation flags, and any other run variables.
- Config YAMLs are the reproducible source of truth; `bible.md` only reflects the final winning values and comparison table.
- After experimentation concludes (step 5 of PRD), this section will be expanded with:
  - Final selected model
  - Winning hyperparameters (from the winning YAML)
  - Comparison table (all variants: model, epochs, LR, augmentation, batch, mAP@50, mAP@50:0.95)

---

## 6. Dataset Reference Details

- `PCB Defect.v3i.yolov8/data.yaml`: defines `nc: 6`, class names, split paths relative to dataset root (`../train/images`, etc.).
- `README.roboflow.txt`: augmentation details, export metadata, dataset statistics (24,901 images).
- `README.dataset.txt`: additional dataset-level notes (if present).
- Storage: dataset remains in project root (`PCB Defect.v3i.yolov8/`) and is excluded from git via `.gitignore` (`data/` pattern does not cover this folder name; note: actual folder is in root, not `data/`). If needed, add folder name to `.gitignore` explicitly.

---

## 7. Demo Interface (Confirmed — brief spec; elaboration later)

**Interface type:** Simple React web application.

**Core functionality (confirmed):**
- User uploads an image via browser/file picker.
- The model (final selected YOLOv8n or YOLOv8s) detects defects.
- Results displayed visually with bounding boxes overlaid on the uploaded image.

**Pending elaboration (user will specify later):**
- Deployment context (local server, Gradio wrapper, or standalone React build?)
- Confidence threshold for displaying detections
- Additional UI details (class labels, confidence scores, download result, multi-image batch upload)
- Integration approach: will React call a Python backend (e.g., Flask/FastAPI serving the YOLO model), or will the model run within the browser (e.g., ONNX/WebAssembly)?

**Location:** `frontend/` directory at project root (to be created when elaboration is provided).

---

## 8. Documentation & Reporting (Pending — user will specify)
> Section to be expanded when user provides details for final report structure, literature references on PCB defect detection, and any additional conclusions or limitations to document. References to `README.roboflow.txt` and `data.yaml` will be included in reporting.

---

*Created: 2026-09-15. Updated: 2026-09-15. Updated by agent after all clarifying questions answered (batch=8 for n, batch=6+amp for s; configs in `configs/`; class table added; hyperparameter policy defined; evaluation metrics confirmed).*
