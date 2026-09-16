#!/usr/bin/env python3
"""Gradio dashboard: upload PCB image, get annotated predictions from both YOLOv8n and YOLOv8s."""
import gradio as gr
from ultralytics import YOLO
import numpy as np
import cv2

# Load fine-tuned best weights
model_n = YOLO("models/yolov8n_best.pt")
model_s = YOLO("models/yolov8s_best.pt")

CLASS_NAMES = [
    "missing_hole", "mouse_bite", "open_circuit",
    "short", "spur", "spurious_copper",
]

def predict(image_path):
    # Run inference (conf=0.25 default)
    res_n = model_n.predict(source=image_path, conf=0.25, save=False, verbose=False)
    res_s = model_s.predict(source=image_path, conf=0.25, save=False, verbose=False)

    # Get annotated images
    annotated_n = res_n[0].plot()
    annotated_s = res_s[0].plot()

    # Also collect text results
    results_n = res_n[0].boxes
    results_s = res_s[0].boxes

    def fmt(boxes):
        lines = []
        if boxes is not None and len(boxes.cls) > 0:
            for cls, conf in zip(boxes.cls, boxes.conf):
                name = CLASS_NAMES[int(cls)]
                lines.append(f"{name}: {conf:.2f}")
        return "\n".join(lines) if lines else "No detections"

    return annotated_n, annotated_s, fmt(results_n), fmt(results_s)

with gr.Blocks(title="PCB Defect Detection Dashboard") as demo:
    gr.Markdown("# PCB Defect Detection — YOLOv8n vs YOLOv8s")
    gr.Markdown("Upload a PCB image. The dashboard runs both fine-tuned models and shows annotated results with class labels + confidence scores.")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="filepath", label="Upload PCB Image")
            btn = gr.Button("Detect Faults")
        with gr.Column():
            gr.Markdown("### YOLOv8n (Nano — ~3.2M params)")
            out_n_img = gr.Image(label="Annotated (n)")
            out_n_text = gr.Textbox(label="Detected classes + scores (n)", lines=6)

    with gr.Row():
        gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### YOLOv8s (Small — ~11.2M params)")
            out_s_img = gr.Image(label="Annotated (s)")
            out_s_text = gr.Textbox(label="Detected classes + scores (s)", lines=6)

    btn.click(
        fn=predict,
        inputs=img_input,
        outputs=[out_n_img, out_s_img, out_n_text, out_s_text],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
