import os
import time
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ----------------------------
# CONFIGURATION
# ----------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

client = genai.Client(api_key=API_KEY)

# Paths
BASE_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "car", "clean_images")
LABELS_DIR = os.path.join(BASE_DIR, "dataset", "car", "labels")

os.makedirs(LABELS_DIR, exist_ok=True)

MODEL = "gemini-2.5-flash-lite"
RPM_LIMIT = 15
SLEEP_TIME = 60 / RPM_LIMIT  # ~4 sec between requests

# ----------------------------
# HELPER FUNCTION
# ----------------------------
def get_car_bounding_boxes(image_path):
    """
    Send an image to Gemini and return a list of YOLO-format bounding boxes:
    [ (class_id, x_center, y_center, width, height), ... ]
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    else:
        raise ValueError(f"Unsupported image format: {image_path}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    prompt = (
        "Detect all cars in this image. "
        "Return a JSON array of objects, each with normalized YOLO coordinates "
        "(x_center, y_center, width, height) between 0 and 1, "
        "and use class_id = 0 for all cars. "
        "Output ONLY the JSON, no markdown, no text before or after it."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            prompt,
        ]
    )

    text = response.text.strip()

    # --- CLEANING STEP ---
    if "```" in text:
        # Remove markdown fences and labels like ```json
        text = text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    # Extract JSON substring if extra text is included
    if not (text.startswith("[") and text.endswith("]")):
        start = text.find("[")
        end = text.rfind("]") + 1
        text = text[start:end] if start != -1 and end != -1 else "[]"

    try:
        boxes = json.loads(text)
        valid_boxes = []
        for box in boxes:
            if all(k in box for k in ["x_center", "y_center", "width", "height"]):
                valid_boxes.append((
                    box.get("class_id", 0),
                    float(box["x_center"]),
                    float(box["y_center"]),
                    float(box["width"]),
                    float(box["height"])
                ))
            elif "box_2d" in box:  # Fallback for x1,y1,x2,y2
                x1, y1, x2, y2 = box["box_2d"]
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                valid_boxes.append((0, x_center, y_center, width, height))
        return valid_boxes
    except Exception as e:
        print(f"⚠️ Failed to parse JSON for {os.path.basename(image_path)}:\n{text}\nError: {e}\n")
        return []

# ----------------------------
# MAIN LOOP (with resume)
# ----------------------------
def generate_yolo_labels():
    image_files = [f for f in os.listdir(IMAGES_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("No images found in:", IMAGES_DIR)
        return

    processed_count = 0
    total_files = len(image_files)

    for i, filename in enumerate(sorted(image_files), 1):
        image_path = os.path.join(IMAGES_DIR, filename)
        label_path = os.path.join(LABELS_DIR, os.path.splitext(filename)[0] + ".txt")

        # Skip already processed files
        if os.path.exists(label_path):
            print(f"[{i}/{total_files}] Skipping {filename} (already labeled).")
            continue

        print(f"[{i}/{total_files}] Processing {filename} ...")

        try:
            boxes = get_car_bounding_boxes(image_path)
            if boxes:
                with open(label_path, "w", encoding="utf-8") as f:
                    for box in boxes:
                        f.write(" ".join(map(str, box)) + "\n")
                print(f" → {len(boxes)} car(s) detected and labeled.")
            else:
                # Write an empty label file to mark as processed
                open(label_path, "w").close()
                print(" → No cars detected (empty label).")
            processed_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")

        time.sleep(SLEEP_TIME)

    print(f"\n✅ All done. Processed {processed_count} new image(s).")

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    generate_yolo_labels()
