import cv2
import numpy as np
import base64
import os
from ultralytics import YOLO

# Load YOLO model
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/license_plate_detector.pt")
_YOLO_MODEL = YOLO(_MODEL_PATH)

def load_image(image_input):
    """Load an image from a file path or base64 string."""
    if isinstance(image_input, str) and os.path.exists(image_input):
        return cv2.imread(image_input)
    else:
        img_bytes = base64.b64decode(image_input.split(",")[-1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def detect_plate_region(img):
    """Detect license plate in image using YOLO model."""
    results = _YOLO_MODEL(img, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None
    box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]
