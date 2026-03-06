from ultralytics import YOLO

model = YOLO("yolo26n-obb.pt")

results = model.train(
    data="job_3702562_annotations_2026_03_06_14_15_05/data.yaml",
    epochs=100,
    imgsz=960,

    # Stronger lighting augmentation
    hsv_h=0.02,
    hsv_s=0.8,
    hsv_v=0.65,   # push brightness jitter harder for bright/dark robustness

    scale=0.5,
    degrees=30,

    # Optional but often helpful generalization
    fliplr=0.5,
    mosaic=1.0, mixup=0.1,  # if your version supports/uses these well
)