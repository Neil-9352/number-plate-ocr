import os
import cv2

# Paths
images_dir = "dataset/images"
labels_dir = "dataset/labels"
output_dir = "dataset/cropped"

os.makedirs(output_dir, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

for img_file in image_files:
    # Load image
    img_path = os.path.join(images_dir, img_file)
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # Load corresponding label
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_file)

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        for idx, line in enumerate(f.readlines()):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            # YOLO format: class, x_center, y_center, width, height (all normalized)
            cls, x_center, y_center, bw, bh = map(float, parts)

            # Convert to pixel coords
            x_center, y_center, bw, bh = x_center * w, y_center * h, bw * w, bh * h
            x1 = int(x_center - bw / 2)
            y1 = int(y_center - bh / 2)
            x2 = int(x_center + bw / 2)
            y2 = int(y_center + bh / 2)

            # Crop safely
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img[y1:y2, x1:x2]

            # Save crop
            out_name = f"{os.path.splitext(img_file)[0]}_{idx}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), crop)

print("✅ Cropping completed. Cropped plates saved in:", output_dir)
