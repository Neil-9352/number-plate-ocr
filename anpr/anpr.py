"""
ANPR — Automatic Number Plate Recognition Library

Public API:
    detect_and_ocr(image_input)
"""

from .detect import load_image, detect_plate_region
from .ocr import preprocess_and_ocr
from .utils import PLATE_REGEX

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
        img = load_image(image_input)
        if img is None:
            return "Invalid plate"

        plate_crop = detect_plate_region(img)
        if plate_crop is None:
            return "Invalid plate"

        best_plate = preprocess_and_ocr(plate_crop)
        return best_plate if best_plate and PLATE_REGEX.match(best_plate) else "Invalid plate"

    except Exception:
        return "Invalid plate"


if __name__ == "__main__":
    print(detect_and_ocr("images/sample.jpg"))
