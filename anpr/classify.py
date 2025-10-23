import cv2
import numpy as np
import base64
import os
from ultralytics import YOLO

_CLASSIFY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolo11n.pt")
_model = YOLO(_CLASSIFY_MODEL_PATH)  

def _load_image(image_input):
    """Load an image from a file path or base64 string."""
    if isinstance(image_input, str) and os.path.exists(image_input):
        return cv2.imread(image_input)
    else:
        img_bytes = base64.b64decode(image_input.split(",")[-1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def detect_vehicle(image_input):
    """
    Detects vehicles in an image.
    Returns:
        - 'car' if class_id == 2
        - 'bike' if class_id == 3
        - None if no relevant object detected
    """
    img = _load_image(image_input)
    if img is None:
        raise ValueError("Could not read the image.")

    results = _model(img, verbose=False)
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            if class_id == 2:
                return "car"
            elif class_id == 3:
                return "bike"
    return None

# Example usage
if __name__ == "__main__":
    result = detect_vehicle("images/c.jpeg")  # or base64 string
    print(result)
