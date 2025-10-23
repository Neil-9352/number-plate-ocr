import os
import shutil

# ----------------------------
# CONFIGURATION
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "car", "images")
CLEAN_DIR = os.path.join(BASE_DIR, "dataset", "car", "clean_images")

# Create destination folder if not exists
os.makedirs(CLEAN_DIR, exist_ok=True)

# Supported image extensions
VALID_EXTS = (".jpg", ".jpeg", ".png")

# ----------------------------
# MAIN SCRIPT
# ----------------------------
def copy_and_rename_images():
    # Collect image files
    image_files = sorted(
        [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(VALID_EXTS)]
    )

    if not image_files:
        print(f"No images found in: {IMAGES_DIR}")
        return

    for idx, filename in enumerate(image_files, start=1):
        src_path = os.path.join(IMAGES_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()
        new_name = f"{idx:06d}{ext}"
        dest_path = os.path.join(CLEAN_DIR, new_name)

        shutil.copy2(src_path, dest_path)
        print(f"Copied {filename} → {new_name}")

    print(f"\n✅ Done! {len(image_files)} images copied to {CLEAN_DIR}")

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    copy_and_rename_images()
