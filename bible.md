# Bible — Source of Truth for PCB Defect Detection Agent

> This file is the single source of truth for the project agent. It must be updated before any action changes. All clarifying questions must be resolved here.

---

## 1. Problem Statement (Fixed from PRD)

Manual visual inspection of printed circuit boards (PCBs) for manufacturing defects — open circuits, short circuits, missing holes, mouse bites, spurs, and spurious copper — is slow, inconsistent, and does not scale in high-volume electronics manufacturing.

This project builds an automated deep-learning-based system that detects and localizes these 6 defect types directly from PCB images, enabling faster and more reliable quality inspection.

**Defect classes (6):** open circuit, short circuit, missing hole, mouse bite, spur, spurious copper.

---

## 2. Data Acquisition — CLARIFYING QUESTIONS (Pending)

Before writing the dataset plan, please confirm:

- **Dataset source:** PKU-Market-PCB via Roboflow API? Or a manual zip upload? What is the exact dataset URL / API endpoint?
- **Format on disk:** Will the dataset arrive as `train/valid/test` folders with `images/` and `labels/` in YOLO format, plus a `data.yaml`?
- **Image specs:** What are the image dimensions / resolution? Are they all uniform?
- **Class counts / imbalance:** Should we assume class imbalance exists? Do you want a class-distribution table recorded here?
- **Split ratios:** Confirm train/valid/test split percentages (default from dataset?)
- **Storage location:** `data/PKU-Market-PCB/` — confirm path.

---

## 3. Model Fine-Tuning — CLARIFYING QUESTIONS (Pending)

Before defining the training protocol, please confirm:

- **Architecture:** YOLOv8n (nano) as baseline, YOLOv8s (small) as comparison — is that final?
- **Pretrained weights:** COCO-pretrained from Ultralytics (`yolov8n.pt`, `yolov8s.pt`) — confirm.
- **Head replacement:** Replace detection head for 6-class task — confirm.
- **Fine-tuning strategy:** Full fine-tune (unfreeze all) or freeze backbone? What layers frozen (if any)?
- **Batch / workers (Fedora, 4GB allocated, 6GB GPU):** Confirm `batch=8`, `workers=2` (Linux-safe).
- **Epochs / LR / augmentation:** Should hyperparameters be listed in this file once decided, or kept in a separate config?
- **Evaluation metric:** mAP@0.50? mAP@0.50:0.95? Confirm target metric for final model selection.

---

## 4. Environment (Fixed from User Input)

- **OS:** Fedora Workstation (Linux)
- **GPU:** 6GB total VRAM → allocate 4GB to project
- **Virtual env:** Python `venv` (`.venv/`)
- **Batch size:** 8 (4GB allocation)
- **Workers:** 2 (Linux-safe, avoids hangs)
- **Packages:** `ultralytics`, `torch` (CUDA), `gradio`, `jupyter`, `numpy`, `pandas`, `matplotlib`
- **Git repo:** Initialized; `.gitignore` excludes `.venv/`, `data/`, `models/`, `runs/`, `results/`, `.env`

---

## 5. Pending User Inputs (Rest Details)

> The user will specify remaining details for documentation, reporting, demo interface, and literature references. Once specified, this section will expand into sections 6, 7, 8 (Demo, Documentation, References) matching the PRD steps 6–8.

---

*Created: 2026-09-15. Updated by agent only after clarifying questions are answered.*
