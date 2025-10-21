import os
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
OUTPUT_DIR = os.path.join(DATASET_DIR, "cropped")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_EXTS = [".jpg", ".jpeg", ".png"]

# ----------------------------
# HELPER FUNCTION
# ----------------------------
def yolo_to_bbox(yolo_line, img_width, img_height):
    """
    Convert YOLO normalized format to absolute bounding box (x_min, y_min, x_max, y_max)
    YOLO format: class x_center y_center width height (normalized 0-1)
    """
    parts = yolo_line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO label line: {yolo_line}")
    
    _, x_center, y_center, w, h = map(float, parts)
    
    x_min = int((x_center - w/2) * img_width)
    y_min = int((y_center - h/2) * img_height)
    x_max = int((x_center + w/2) * img_width)
    y_max = int((y_center + h/2) * img_height)
    
    # Clamp to image size
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)
    
    return x_min, y_min, x_max, y_max

# ----------------------------
# MAIN LOOP
# ----------------------------
for fname in os.listdir(IMAGES_DIR):
    if not any(fname.lower().endswith(ext) for ext in VALID_EXTS):
        continue

    image_path = os.path.join(IMAGES_DIR, fname)
    label_path = os.path.join(LABELS_DIR, os.path.splitext(fname)[0] + ".txt")
    
    if not os.path.exists(label_path):
        print(f"Label file not found for {fname}, skipping.")
        continue

    with Image.open(image_path) as img:
        w, h = img.size

        with open(label_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines, start=1):
            try:
                x_min, y_min, x_max, y_max = yolo_to_bbox(line, w, h)
                cropped = img.crop((x_min, y_min, x_max, y_max))
                
                out_fname = f"{os.path.splitext(fname)[0]}_plate{i}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_fname)
                cropped.save(out_path)
                print(f"Cropped {out_fname}")
            except Exception as e:
                print(f"Error processing line in {label_path}: {e}")

print("Cropping complete.")
