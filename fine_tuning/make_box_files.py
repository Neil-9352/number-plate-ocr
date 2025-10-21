import os
import shutil
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
SOURCE_DIR = os.path.join(os.path.dirname(__file__), "filtered_data")   # contains images + .gt.txt
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "boxed_data")      # output with images + .box
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_EXTS = {".png", ".jpg", ".jpeg"}

# ----------------------------
# MAIN
# ----------------------------
for fname in sorted(os.listdir(SOURCE_DIR)):
    if not fname.endswith(".gt.txt"):
        continue

    base = fname[:-7]  # Remove .gt.txt
    gt_path = os.path.join(SOURCE_DIR, fname)

    # Read label
    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip().upper()
    if not text:
        print(f"Empty label in {fname}; skipping")
        continue

    # Find matching image
    img_path = None
    for ext in VALID_EXTS:
        candidate = os.path.join(SOURCE_DIR, base + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break

    if not img_path:
        print(f"No image found for {fname}; skipping")
        continue

    # Get image size
    with Image.open(img_path) as im:
        w, h = im.size

    # Write .box file
    out_box = os.path.join(OUTPUT_DIR, f"{base}.box")
    with open(out_box, "w", encoding="utf-8") as f:
        f.write(f"{text} 0 0 {w} {h} 0\n")

    # Copy image
    out_img = os.path.join(OUTPUT_DIR, base + os.path.splitext(img_path)[1].lower())
    shutil.copy2(img_path, out_img)

    print(f"{os.path.basename(img_path)} + {fname} → {os.path.basename(out_box)}: '{text}' [{w}x{h}]")

print(f"\nDone. Output written to: {OUTPUT_DIR}")
