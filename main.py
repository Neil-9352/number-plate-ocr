# import cv2
# import pytesseract
# from PIL import Image

# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# img = cv2.imread("dataset/cropped/00000000_0.jpg")

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Preprocess (simple for now)
# gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# # Convert to PIL before passing to pytesseract
# pil_img = Image.fromarray(thresh)

# custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# text = pytesseract.image_to_string(pil_img, config=custom_config)

# print("Detected:", text)





# import cv2
# import pytesseract
# from PIL import Image
# import re

# # Path to tesseract binary
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# # Regex for Indian plate format (2 letters, 2 digits, 1–2 letters, 1–4 digits)
# PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$")

# def ocr_image(img, psm=7):
#     """Run OCR with custom plates traineddata and whitelist"""
#     pil_img = Image.fromarray(img)
#     custom_config = (
#         f'--oem 3 --psm {psm} '
#         '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
#         '--tessdata-dir ./output '
#         '-l plates'
#     )
#     return pytesseract.image_to_string(pil_img, config=custom_config).strip()

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
#         for psm in [7, 8]:  # single line and sparse text
#             text = ocr_image(variant, psm=psm)
#             # Clean to only alphanumeric uppercase
#             clean = re.sub(r'[^A-Z0-9]', '', text)
#             results.append((name, psm, text, clean))

#     # First, filter results by regex validity
#     valid = [r for r in results if PLATE_REGEX.match(r[3])]

#     if valid:
#         # Prefer the first valid one
#         best = valid[0]
#     else:
#         # Fall back to longest cleaned string
#         best = max(results, key=lambda r: len(r[3])) if results else None

#     return results, best

# if __name__ == "__main__":
#     path = "images/rishab.jpeg"
#     results, best = preprocess_and_ocr(path)

#     print("=== OCR Variants Tried ===")
#     for name, psm, text, clean in results:
#         print(f"[{name}, psm={psm}] -> Raw: '{text}' | Clean: '{clean}'")

#     print("\n=== Best Guess ===")
#     if best:
#         print(best[2])  # Raw OCR output of best guess
#     else:
#         print("No text detected")


import cv2
import pytesseract
from PIL import Image
import re

# Path to tesseract binary
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# Regex for Indian plate formats
# Example: MH12AB1234, KA01AB1234
PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$")

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
    # Remove non-alphanumeric
    clean = re.sub(r'[^A-Z0-9\n]', '', text)
    # Merge if multi-line
    merged = clean.replace("\n", "")
    return merged

def preprocess_and_ocr(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Variants: raw grayscale, adaptive threshold
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
        for psm in [6, 7, 8]:  # allow multi-line (6), single-line (7), sparse (8)
            raw_text = ocr_image(variant, psm=psm)
            merged = normalize_plate(raw_text)
            results.append((name, psm, raw_text, merged))

    # First, filter results by regex validity
    valid = [r for r in results if PLATE_REGEX.match(r[3])]

    if valid:
        best = valid[0]  # first valid
    else:
        # Fall back to longest merged candidate
        best = max(results, key=lambda r: len(r[3])) if results else None

    return results, best

if __name__ == "__main__":
    path = "images/liza.jpg"
    results, best = preprocess_and_ocr(path)

    print("=== OCR Variants Tried ===")
    for name, psm, raw, merged in results:
        print(f"[{name}, psm={psm}] -> Raw: '{raw}' | Merged: '{merged}'")

    print("\n=== Best Guess ===")
    if best:
        print(best[3])  # merged clean plate string
    else:
        print("No text detected")
