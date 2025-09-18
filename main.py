# import cv2
# import pytesseract
# from PIL import Image
# import re

# # Path to tesseract binary
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# # Regex for Indian plate formats
# # Example: MH12AB1234, KA01AB1234
# PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$")

# def ocr_image(img, psm=7):
#     """Run OCR with custom plates traineddata and whitelist"""
#     pil_img = Image.fromarray(img)
#     custom_config = (
#         f'--oem 3 --psm {psm} '
#         '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
#         '--tessdata-dir ./models '
#         '-l plates'
#     )
#     return pytesseract.image_to_string(pil_img, config=custom_config).strip()

# def normalize_plate(text: str) -> str:
#     """Clean OCR text and merge 2-line plates"""
#     # Remove non-alphanumeric
#     clean = re.sub(r'[^A-Z0-9\n]', '', text)
#     # Merge if multi-line
#     merged = clean.replace("\n", "")
#     return merged

# def preprocess_and_ocr(path):
#     img = cv2.imread(path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

#     # Variants: raw grayscale, adaptive threshold
#     variants = {
#         "raw_gray": gray,
#         "adaptive": cv2.adaptiveThreshold(
#             gray, 255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
#             31, 2
#         )
#     }

#     results = []
#     for name, variant in variants.items():
#         for psm in [6, 7, 8]:  # allow multi-line (6), single-line (7), sparse (8)
#             raw_text = ocr_image(variant, psm=psm)
#             merged = normalize_plate(raw_text)
#             results.append((name, psm, raw_text, merged))

#     # First, filter results by regex validity
#     valid = [r for r in results if PLATE_REGEX.match(r[3])]

#     if valid:
#         best = valid[0]  # first valid
#     else:
#         # Fall back to longest merged candidate
#         best = max(results, key=lambda r: len(r[3])) if results else None

#     return results, best

# if __name__ == "__main__":
#     path = "images/3.jpeg"
#     results, best = preprocess_and_ocr(path)

#     print("=== OCR Variants Tried ===")
#     for name, psm, raw, merged in results:
#         print(f"[{name}, psm={psm}] -> Raw: '{raw}' | Merged: '{merged}'")

#     print("\n=== Best Guess ===")
#     if best:
#         print(best[3])  # merged clean plate string
#     else:
#         print("No text detected")





import cv2
import pytesseract
from PIL import Image
import re
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# Regex for Indian plate formats
PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$")

# Load YOLO model (your trained weights)
yolo_model = YOLO("models/license_plate_detector.pt")  # <-- put your YOLO weights here


# -----------------------------
# OCR FUNCTIONS
# -----------------------------
def ocr_image(img, psm=7):
    """Run OCR with custom plates traineddata and whitelist"""
    pil_img = Image.fromarray(img)
    custom_config = (
        f'--oem 3 --psm {psm} '
        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
        '--tessdata-dir ./models '
        '-l plates'
    )
    return pytesseract.image_to_string(pil_img, config=custom_config).strip()


def normalize_plate(text: str) -> str:
    """Clean OCR text and merge 2-line plates"""
    clean = re.sub(r'[^A-Z0-9\n]', '', text)
    merged = clean.replace("\n", "")
    return merged


def preprocess_and_ocr(img):
    """Run OCR with multiple preprocessing variants"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    variants = {
        "raw_gray": gray,
        "adaptive": cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            31, 2
        )
    }

    results = []
    for name, variant in variants.items():
        for psm in [6, 7, 8]:  # multi-line, single-line, sparse
            raw_text = ocr_image(variant, psm=psm)
            merged = normalize_plate(raw_text)
            results.append((name, psm, raw_text, merged))

    valid = [r for r in results if PLATE_REGEX.match(r[3])]
    if valid:
        best = valid[0]
    else:
        best = max(results, key=lambda r: len(r[3])) if results else None

    return results, best


# -----------------------------
# YOLO + OCR PIPELINE
# -----------------------------
def detect_and_ocr(path):
    img = cv2.imread(path)

    # Run YOLO detection
    results = yolo_model(img)

    if not results or len(results[0].boxes) == 0:
        print("No plate detected with YOLO")
        return None

    # Take the first detection (highest confidence)
    box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    plate_crop = img[y1:y2, x1:x2]

    # OCR on cropped plate
    results_ocr, best = preprocess_and_ocr(plate_crop)

    print("=== OCR Variants Tried ===")
    for name, psm, raw, merged in results_ocr:
        print(f"[{name}, psm={psm}] -> Raw: '{raw}' | Merged: '{merged}'")

    print("\n=== Best Guess ===")
    if best:
        print(best[3])
        return best[3]
    else:
        print("No text detected")
        return None


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    path = "images/bordoloi.jpeg"  # full uncropped image
    plate_text = detect_and_ocr(path)
