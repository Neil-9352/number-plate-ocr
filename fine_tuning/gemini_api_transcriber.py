import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
import shutil

# ----------------------------
# CONFIGURATION
# ----------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY_4")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

client = genai.Client(api_key=API_KEY)

INPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset", "cropped")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "training_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL = "gemini-2.5-flash-lite"     # use Lite for higher limits
RPM_LIMIT = 15                      # requests per minute
RPD_LIMIT = 1000                    # requests per day
SLEEP_TIME = 60 / RPM_LIMIT         # ~4 sec between requests

# ----------------------------
# HELPER FUNCTION
# ----------------------------
def transcribe_plate(image_path):
    """Send a single cropped plate image to Gemini and return the plate text."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    else:
        raise ValueError(f"Unsupported image format: {image_path}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            "Extract the license plate number. Return only A-Z and 0-9, no spaces or punctuation."
        ]
    )
    # Normalize: remove spaces, uppercase
    return response.text.strip().replace(" ", "").upper()

# ----------------------------
# MAIN LOOP
# ----------------------------
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
processed_today = 0

for idx, fname in enumerate(sorted(image_files), start=1):
    base = os.path.splitext(fname)[0]
    gt_path = os.path.join(OUTPUT_DIR, f"{base}.gt.txt")
    out_img_path = os.path.join(OUTPUT_DIR, fname)

    # Skip if both GT and image already exist (resume capability)
    if os.path.exists(gt_path) and os.path.exists(out_img_path):
        print(f"[{idx}/{len(image_files)}] Skipping {fname}, already processed.")
        continue

    if processed_today >= RPD_LIMIT:
        print("Reached daily request limit (1000). Stop here and resume tomorrow.")
        break

    image_path = os.path.join(INPUT_DIR, fname)

    try:
        plate_text = transcribe_plate(image_path)
    except Exception as e:
        print(f"[{idx}/{len(image_files)}] Error processing {fname}: {e}")
        continue

    # Save GT file if missing
    if not os.path.exists(gt_path):
        with open(gt_path, "w") as f:
            f.write(plate_text)

    # Copy image if missing
    if not os.path.exists(out_img_path):
        shutil.copy(image_path, out_img_path)

    processed_today += 1
    print(f"[{idx}/{len(image_files)}] {fname} → {plate_text}")

    # Respect RPM limit
    time.sleep(SLEEP_TIME)

print(f"Done. Processed {processed_today} new images today.")
