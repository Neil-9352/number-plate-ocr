import cv2
import pytesseract
from PIL import Image
import os
from .utils import PLATE_REGEX, normalize_plate

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# Path to local trained data
_TESSDATA_DIR = os.path.join(os.path.dirname(__file__), "models/tessdata")

def _ocr_image(img, psm=7):
    pil_img = Image.fromarray(img)
    config = (
        f'--oem 3 --psm {psm} '
        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
        f'--tessdata-dir "{_TESSDATA_DIR}" '
        '-l plates'
    )
    return pytesseract.image_to_string(pil_img, config=config).strip()


def preprocess_and_ocr(img):
    """Preprocess image and perform OCR, returning best valid plate."""
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
            raw = _ocr_image(variant_img, psm)
            merged = normalize_plate(raw)
            if merged:
                results.append(merged)

    if not results:
        return None

    valid = [r for r in results if PLATE_REGEX.match(r)]
    return max(valid, key=len) if valid else None
