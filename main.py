import cv2
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

img = cv2.imread("images/geet.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Preprocess (simple for now)
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Convert to PIL before passing to pytesseract
pil_img = Image.fromarray(thresh)

custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
text = pytesseract.image_to_string(pil_img, config=custom_config)

print("Detected:", text)

