import os
import yaml
from ultralytics import YOLO

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
BASE_DIR = os.path.dirname(__file__)

# Dataset directories
CAR_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "car", "clean_images")
CAR_LABELS_DIR = os.path.join(BASE_DIR, "dataset", "car", "labels")

BIKE_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "bike", "train", "images")
BIKE_LABELS_DIR = os.path.join(BASE_DIR, "dataset", "bike", "train", "labels")

# Class names
CLASSES = ["car", "bike"]

# YOLO training parameters
MODEL_NAME = "yolov8n.pt"        
EPOCHS = 100
IMG_SIZE = 640

# Training output folder (runs will go inside this)
TRAINING_DIR = os.path.join(BASE_DIR)
os.makedirs(TRAINING_DIR, exist_ok=True)  # ensure it exists

PROJECT_NAME = os.path.join(TRAINING_DIR, "runs")  # <- runs/ will be created inside training/

# -------------------------------------------------
# VERIFY DIRECTORIES
# -------------------------------------------------
for d in [CAR_IMAGES_DIR, CAR_LABELS_DIR, BIKE_IMAGES_DIR, BIKE_LABELS_DIR]:
    if not os.path.exists(d):
        raise FileNotFoundError(f"Missing directory: {d}")

print("Dataset directories found.")

# -------------------------------------------------
# CREATE DATA YAML FILE
# -------------------------------------------------
train_yaml = {
    "train": [CAR_IMAGES_DIR, BIKE_IMAGES_DIR],
    "val": [CAR_IMAGES_DIR, BIKE_IMAGES_DIR],
    "nc": len(CLASSES),
    "names": CLASSES
}

yaml_path = os.path.join(BASE_DIR, "dataset", "car_bike_dataset.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(train_yaml, f, default_flow_style=False)

print(f"Dataset YAML file created at: {yaml_path}")

# -------------------------------------------------
# TRAIN YOLO MODEL
# -------------------------------------------------
def main():
    print("Starting YOLO model training...")

    # Load YOLO model
    model = YOLO(MODEL_NAME)

    # Train and save results in BASE_DIR/training/runs/
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=PROJECT_NAME,
        name="car_bike_yolo",
        exist_ok=True
    )

    print("\nTraining complete!")
    print("The trained model weights (.pt) are saved in:")
    print(f"   {os.path.join(PROJECT_NAME, 'train', 'car_bike_yolo', 'weights', 'best.pt')}")

if __name__ == "__main__":
    main()
