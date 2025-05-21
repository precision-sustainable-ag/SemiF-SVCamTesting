import numpy as np
import cv2
import random
import argparse
from pathlib import Path
import json
import csv

WINDOW_HEIGHT = 800
WINDOW_WIDTH = 1200

def parse_arguments():
    parser = argparse.ArgumentParser(description="RAW image processor and color checker crop tool")
    parser.add_argument("batch", type=str, help="Subdirectory under base directory")
    parser.add_argument("selection_mode", choices=["random", "last", "first", "index"], help="Image selection mode")
    parser.add_argument("sample_number", type=int, help="Number of images or index (if mode is 'index')")
    parser.add_argument("bit_depth", type=int, choices=[8, 16], help="Bit depth for output image")
    # parser.add_argument("demosaicing_method", choices=["cv2"], help="Demosaicing method")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save outputs")
    parser.add_argument("--show_downscaled", action="store_true", help="Show downscaled image for selecting crop coords")
    parser.add_argument("--draw_grid", action="store_true", help="Draw grid on downscaled image")
    parser.add_argument("--save_fullres", action="store_true", help="Save full resolution image")
    parser.add_argument("--save_downscaled", action="store_true", help="Save downscaled image for selecting crop coords")
    parser.add_argument("--scale_factor", type=float, default=0.1, help="Downscale factor for preview (default: 0.25)")
    parser.add_argument("--get_bboxes", action="store_true", help="Get bounding boxes for crops")
    parser.add_argument("--save_crops", action="store_true", help="Save the 400x400 crops")
    parser.add_argument("--show_crops", action="store_true", help="Show the 400x400 crops")
    parser.add_argument("--bbox_file", type=str, default=None, help="Path to JSON file to save/load bounding boxes")
    parser.add_argument("--use_saved_bboxes", action="store_true", help="Use previously saved bounding boxes from file")
    
    parser.add_argument("--log_csv", type=str, default="data/MD_svcam_config_log.csv", help="Path to the CSV file to append image metadata logs")
    parser.add_argument("--log_metadata", action="store_true", help="Log metadata for each image")
    
    
    return parser.parse_args()

def draw_scaled_grid(image, scale_factor, step=500, color=(255, 0, 0), thickness=1):
    """
    Draws a grid on the downscaled image that represents full-resolution spacing.
    
    Args:
        image (np.ndarray): Downscaled image
        scale_factor (float): Downscale factor used
        step (int): Grid spacing in full-resolution pixels
        color (tuple): BGR color for grid lines
        thickness (int): Line thickness
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    scaled_step = int(step * scale_factor)

    for x in range(0, w, scaled_step):
        cv2.line(annotated, (x, 0), (x, h), color, thickness)
        cv2.putText(annotated, str(int(x / scale_factor)), (x + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    for y in range(0, h, scaled_step):
        cv2.line(annotated, (0, y), (w, y), color, thickness)
        cv2.putText(annotated, str(int(y / scale_factor)), (5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return annotated

def log_image_metadata(csv_path, image_name, f_number, focus, flash_power, z_height, note=""):
    file_exists = Path(csv_path).exists()
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["image_name", "f_number", "focus", "flash_power", "z_height", "note"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "image_name": image_name,
            "f_number": f_number,
            "focus": focus,
            "flash_power": flash_power,
            "z_height": z_height,
            "note": note
        })

def select_raw_files(raw_files, mode, number):
    if mode == "first":
        return raw_files[:number]
    elif mode == "last":
        return raw_files[-number:]
    elif mode == "random":
        return random.sample(raw_files, min(number, len(raw_files)))
    elif mode == "index":
        return [raw_files[number]]
    return []


def draw_xy_axes(image, step=500, color=(0, 255, 0), thickness=1):
    annotated = image.copy()
    h, w = annotated.shape[:2]
    for x in range(0, w, step):
        cv2.line(annotated, (x, 0), (x, h), color, thickness)
        cv2.putText(annotated, str(x), (x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for y in range(0, h, step):
        cv2.line(annotated, (0, y), (w, y), color, thickness)
        cv2.putText(annotated, str(y), (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated


def show_image(image, window_name="Image"):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def get_multiple_boxes_from_clicks(image, scale_factor, save_file=None, load_file=None, window_name="Select boxes"):
    """
    Allows user to select multiple bounding boxes using 4-point clicks.
    Returns a list of (x1, y1, x2, y2) in full-resolution coordinates.
    """

    boxes = []
    current_box = []
    clone = image.copy()

    if load_file and Path(load_file).exists():
        print(f"📂 Loading box coordinates from {load_file}")
        with open(load_file, "r") as f:
            boxes = json.load(f)
        return boxes  # Already full-res coords, skip GUI

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_box
        if event == cv2.EVENT_LBUTTONDOWN and len(current_box) < 4:
            current_box.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(clone, f"{x},{y}", (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow(window_name, clone)

            if len(current_box) == 4:
                xs = [p[0] for p in current_box]
                ys = [p[1] for p in current_box]
                cv2.rectangle(clone, (min(xs), min(ys)), (max(xs), max(ys)), (0, 0, 255), 2)
                boxes.append([
                    int(min(xs) / scale_factor), int(min(ys) / scale_factor),
                    int(max(xs) / scale_factor), int(max(ys) / scale_factor)
                ])
                current_box = []
                print(f"📦 Added box #{len(boxes)}. Press 'n' for another or Enter to finish.")

    print("\n🖱️  Click 4 points to define a bounding box.")
    print("➡️ Press [n] to start another box, [Enter] to finish, or [Esc] to cancel all.\n")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.imshow(window_name, clone)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            break
        elif key == ord("n"):
            continue
        elif key == 27:  # Esc
            print("❌ Cancelled selection.")
            boxes = []
            break

    cv2.destroyWindow(window_name)

    if save_file and boxes:
        with open(save_file, "w") as f:
            json.dump(boxes, f)
        print(f"💾 Saved box coordinates to {save_file}")

    return boxes


def process_raw_image(raw_path, bit_depth, method="cv2", im_height=9528, im_width=13376):
    raw_data = np.fromfile(raw_path, dtype=np.uint16).reshape((im_height, im_width))
    if method == "cv2":
        rgb = cv2.cvtColor(raw_data, cv2.COLOR_BayerBG2RGB_EA).astype(np.float64) / 65535.0
    else:
        raise NotImplementedError("Only 'cv2' demosaicing supported.")
    clipped = np.clip(rgb, 0, 1)
    if bit_depth == 8:
        print("Converting to 8-bit")
        output = (clipped * 255).astype(np.uint8)
    else:
        output = (clipped * 65535).astype(np.uint16)
    bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    print(f"Image shape: {bgr.shape}, dtype: {bgr.dtype}")
    return bgr


def main():
    args = parse_arguments()

    base_dir = Path("/mnt/research-projects/s/screberg/longterm_images2/semifield-upload")
    # base_dir = Path("/home/benchbot/benchbot_app/mini_computer_api/")
    input_dir = base_dir / args.batch
    assert input_dir.exists(), f"❌ Input directory does not exist: {input_dir}"

    raw_files = sorted(input_dir.glob("*.RAW"))
    selected = select_raw_files(raw_files, args.selection_mode, args.sample_number)
    output_dir = Path(args.output_dir) / args.batch
    output_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in selected:
        
        output_path = output_dir / f"{raw_path.stem}_{args.bit_depth}bit.png"
        log_flag = False
        if output_path.exists():
            print(f"⏩ Skipping processing, already exists: {output_path.name}")
            full_img = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)

        else:
            full_img = process_raw_image(raw_path, args.bit_depth)
            # Save full image if needed
            if args.save_fullres:
                cv2.imwrite(str(output_path), full_img)
                print(f"✅ Saved full image: {output_path.name}")
            log_flag = True

        if args.show_downscaled or args.save_downscaled:
            print(f"DType: {full_img.dtype}, Shape: {full_img.shape}")
            preview = cv2.resize(full_img, (0, 0), fx=args.scale_factor, fy=args.scale_factor, interpolation=cv2.INTER_AREA)
            if args.draw_grid:
                preview = draw_scaled_grid(preview, args.scale_factor, step=int(3000 * args.scale_factor))

            if args.save_downscaled:
                downscaled_path = output_dir / f"{raw_path.stem}_{args.bit_depth}bit_downscaled.png"
                cv2.imwrite(str(downscaled_path), preview)
                print(f"✅ Saved downscaled image: {downscaled_path.name}")
            # preview = draw_xy_axes(preview, step=int(500 * args.scale_factor))
        else:
            preview = cv2.resize(full_img, (0, 0), fx=args.scale_factor, fy=args.scale_factor, interpolation=cv2.INTER_AREA)
        
        if args.get_bboxes:
            boxes = get_multiple_boxes_from_clicks(
                preview,
                scale_factor=args.scale_factor,
                save_file=args.bbox_file if args.save_crops else None,
                load_file=args.bbox_file if args.use_saved_bboxes else None
            )

            if boxes and (args.save_crops or args.show_crops):
                crop_images = []
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    crop = full_img[y1:y2, x1:x2]
                    crop_images.append(crop)

                    if args.save_crops:
                        out_path = output_dir / f"{raw_path.stem}_crop_box_{i+1}_{x1}_{y1}_{x2}_{y2}_{args.bit_depth}bit.png"
                        cv2.imwrite(str(out_path), crop)
                        print(f"✅ Saved crop: {out_path}")

                    if args.show_crops:
                        show_image(crop, f"{raw_path.stem}: Crop #{i+1}: ({x1},{y1}) to ({x2},{y2})")
                # Combine all crops into one image (grid-style layout)
                if crop_images:
                    # Auto-determine grid shape (e.g., 2 columns)
                    num_cols = 2
                    num_rows = (len(crop_images) + num_cols - 1) // num_cols

                    max_width = max(c.shape[1] for c in crop_images)
                    max_height = max(c.shape[0] for c in crop_images)

                    grid_height = num_rows * max_height
                    grid_width = num_cols * max_width

                    combined_image = np.zeros((grid_height, grid_width, 3), dtype=full_img.dtype)

                    for idx, crop in enumerate(crop_images):
                        row = idx // num_cols
                        col = idx % num_cols

                        y_offset = row * max_height
                        x_offset = col * max_width

                        h, w = crop.shape[:2]
                        combined_image[y_offset:y_offset+h, x_offset:x_offset+w] = crop

                    combined_output_path = output_dir / f"{raw_path.stem}_combined_crops_{args.bit_depth}bit.png"
                    cv2.imwrite(str(combined_output_path), combined_image)
                    print(f"🧩 Combined image of all crops saved: {combined_output_path}")
        
        if log_flag and args.log_metadata:
            # Prompt user for image metadata
            print("\n📷 Please enter metadata for this image:")
            f_number = input("f/number: ")
            focus = 20.2 #input("focus: ")
            flash_power = input("flash power: ")
            z_height = "170" #input("z-axis height: ")
            note = input("notes: ")

            # Log it
            log_image_metadata(
                csv_path=args.log_csv,
                image_name=raw_path.stem,
                f_number=f_number,
                focus=focus,
                flash_power=flash_power,
                z_height=z_height,
                note=note,
            )
                
            

if __name__ == "__main__":
    main()

# To run this script, use the command line:
# python quick_correct.py MD_2025-04-09 last 1 8 cv2 benchbot_app/mini_computer_api/images/MD_2025-04-09 --show_downscaled --save_crops --show_crops --bbox_file bboxes.json
