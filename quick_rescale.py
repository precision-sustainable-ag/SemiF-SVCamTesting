import cv2
from pathlib import Path

lts_developed_dir = Path("/mnt/research-projects/s/screberg/longterm_images2/semifield-developed-images/NC_2025-08-19/images")
imgs = lts_developed_dir.glob("*.jpg")

outputdir = Path("data/NC_2025-08-19") / "simple_resized"
for img in imgs:
    image = cv2.imread(str(img))
    # Resize the image to make it easier to view while keep ratios
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_width = 800
    new_height = int(new_width / aspect_ratio)
    image = cv2.resize(image, (new_width, new_height))
    