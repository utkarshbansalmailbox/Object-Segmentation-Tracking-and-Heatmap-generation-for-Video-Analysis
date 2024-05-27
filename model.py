import os
from ultralytics import YOLO

def load_model(model_name='yolov8n-seg.pt'):
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_path):
        YOLO(model_name)

    return YOLO(model_path)
