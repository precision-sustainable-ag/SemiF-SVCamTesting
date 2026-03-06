from ultralytics import YOLO
from pathlib import Path

# Load a pretrained YOLO26n model
model = YOLO("runs/obb/train4/weights/best.pt")

# Define a glob search for all JPG files in a directory
source = Path("data/NC_2026-03-06").glob("*.png")  # generator of Path objects for all JPG files in the directory

for img_path in source:
    print(f"Running inference on {img_path}...")
    # Run inference on the image with arguments
    model.predict(str(img_path), save=True, imgsz=960, conf=0.40, iou=0.05)