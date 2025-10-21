import os
import shutil
import re

# ----------------------------
# CONFIGURATION
# ----------------------------
TRAINING_DIR = os.path.join(os.path.dirname(__file__), "training_data")
FILTERED_DIR = os.path.join(os.path.dirname(__file__), "filtered_data")
os.makedirs(FILTERED_DIR, exist_ok=True)

VALID_EXTS = [".jpg", ".jpeg", ".png"]

# Regex for Indian license plates (normal + Delhi)
PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$")

def is_valid_plate(text: str) -> bool:
    """Check if text matches Indian license plate pattern (incl. Delhi)."""
    text = text.strip().upper()
    return bool(PLATE_REGEX.match(text))

# ----------------------------
# MAIN LOOP
# ----------------------------
counter = 1
for fname in sorted(os.listdir(TRAINING_DIR)):
    if not fname.endswith(".gt.txt"):
        continue

    # Base name without .gt.txt
    base = fname[:-7]
    gt_path = os.path.join(TRAINING_DIR, fname)

    # Read the plate text
    with open(gt_path, "r") as f:
        plate_text = f.read().strip().upper()

    if not is_valid_plate(plate_text):
        print(f"Skipping {fname} (invalid plate: '{plate_text}')")
        continue

    # Zero-padded new filename (7 digits)
    new_id = f"{counter:07d}"

    # Copy .gt.txt with new name
    out_gt = os.path.join(FILTERED_DIR, new_id + ".gt.txt")
    shutil.copy(gt_path, out_gt)

    # Copy matching image
    copied = False
    for ext in VALID_EXTS:
        img_path = os.path.join(TRAINING_DIR, base + ext)
        if os.path.exists(img_path):
            out_img = os.path.join(FILTERED_DIR, new_id + ext.lower())
            shutil.copy(img_path, out_img)
            copied = True
            break

    if copied:
        print(f"{new_id}{ext} + {new_id}.gt.txt → {plate_text}")
        counter += 1
    else:
        print(f"Warning: No matching image found for {fname}")

print(f"\nFiltering complete. {counter-1} samples written to {FILTERED_DIR}/")
