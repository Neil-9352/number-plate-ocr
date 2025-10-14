"""
anpr.py — Automatic Number Plate Recognition Library

Public API:
    detect_and_ocr(image_input)

All other functions are private (internal use only).
"""

import cv2
import pytesseract
from PIL import Image
import re
from ultralytics import YOLO
import base64
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# Regex for Indian plate formats
_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$")

# Load YOLO model (your trained weights)
_YOLO_MODEL = YOLO("models/license_plate_detector.pt")


# -----------------------------
# INTERNAL OCR HELPERS
# -----------------------------
def _ocr_image(img, psm=7):
    pil_img = Image.fromarray(img)
    config = (
        f'--oem 3 --psm {psm} '
        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
        '--tessdata-dir ./models '
        '-l plates'
    )
    return pytesseract.image_to_string(pil_img, config=config).strip()


def _normalize_plate(text: str) -> str:
    clean = re.sub(r'[^A-Z0-9\n]', '', text)
    return clean.replace("\n", "")


def _preprocess_and_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    variants = {
        "raw_gray": gray,
        "adaptive": cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 2
        )
    }

    results = []
    for variant_img in variants.values():
        for psm in [6, 7, 8]:
            raw = _ocr_image(variant_img, psm=psm)
            merged = _normalize_plate(raw)
            if merged:
                results.append(merged)

    if not results:
        return None

    # Filter by regex
    valid = [r for r in results if _PLATE_REGEX.match(r)]
    if valid:
        return max(valid, key=len)  # longest valid plate
    else:
        return None


# -----------------------------
# PUBLIC API
# -----------------------------
def detect_and_ocr(image_input):
    """
    Detects and recognizes license plate from an image.
    Accepts:
        - File path or base64 image string.
    Returns:
        - License plate text if valid
        - "Invalid plate" otherwise
    """
    try:
        # Load image
        if isinstance(image_input, str) and os.path.exists(image_input):
            img = cv2.imread(image_input)
        else:
            img_bytes = base64.b64decode(image_input.split(",")[-1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return "Invalid plate"

        # YOLO detection
        results = _YOLO_MODEL(img, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return "Invalid plate"

        box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        plate_crop = img[y1:y2, x1:x2]

        # OCR
        best_plate = _preprocess_and_ocr(plate_crop)

        # Final regex check
        if best_plate and _PLATE_REGEX.match(best_plate):
            return best_plate
        else:
            return "Invalid plate"

    except Exception:
        return "Invalid plate"


# -----------------------------
# OPTIONAL TEST ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    print(detect_and_ocr("images/abhi.jpg"))
