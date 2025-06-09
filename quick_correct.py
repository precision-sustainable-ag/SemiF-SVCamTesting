import numpy as np
import cv2
import random
import argparse
from pathlib import Path

WINDOW_HEIGHT = 800
WINDOW_WIDTH = 1200

def parse_arguments():
    parser = argparse.ArgumentParser(description="RAW image processor and color checker crop tool")
    parser.add_argument("batch", type=str, help="Subdirectory under base directory")
    parser.add_argument("selection_mode", choices=["random", "last", "first", "index"], help="Image selection mode")
    parser.add_argument("sample_number", type=int, help="Number of images or index (if mode is 'index')")
    parser.add_argument("bit_depth", type=int, choices=[8, 16], help="Bit depth for output image")
    parser.add_argument("--brightness", type=float, default=1.3, help="Brightness adjustment factor (default: 1.3)")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save outputs")
    parser.add_argument("--save_fullres", action="store_true", help="Save full resolution image")
    parser.add_argument("--save_downscaled", action="store_true", help="Save downscaled image for selecting crop coords")
    parser.add_argument("--scale_factor", type=float, default=0.15, help="Downscaling factor for preview images (default: 0.15)")
    return parser.parse_args()

def adjust_brightness(img, factor):
    """Brighten or darken an image by a given factor (float)."""
    img = img.astype(np.float32) * factor
    if img.dtype == np.uint16:
        np.clip(img, 0, 65535, out=img)
        return img.astype(np.uint16)
    else:
        np.clip(img, 0, 255, out=img)
        return img.astype(np.uint8)

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

def process_raw_image(raw_path, bit_depth, im_height=9528, im_width=13376):
    raw_data = np.fromfile(raw_path, dtype=np.uint16).reshape((im_height, im_width))
    # Only cv2 demosaicing is kept (hard-coded)
    rgb = cv2.cvtColor(raw_data, cv2.COLOR_BayerBG2RGB_EA).astype(np.float64) / 65535.0
    clipped = np.clip(rgb, 0, 1)
    if bit_depth == 8:
        output = (clipped * 255).astype(np.uint8)
    else:
        output = (clipped * 65535).astype(np.uint16)
    bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    print(f"Image shape: {bgr.shape}, dtype: {bgr.dtype}")
    return bgr

def main():
    args = parse_arguments()

    base_dir = Path("/mnt/research-projects/s/screberg/longterm_images2/semifield-upload")
    input_dir = base_dir / args.batch
    assert input_dir.exists(), f"❌ Input directory does not exist: {input_dir}"

    raw_files = sorted(input_dir.glob("*.RAW"))
    selected = select_raw_files(raw_files, args.selection_mode, args.sample_number)
    output_dir = Path(args.output_dir) / args.batch
    output_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in selected:
        output_path = output_dir / f"{raw_path.stem}_{args.bit_depth}bit.png"
        if output_path.exists():
            print(f"⏩ Skipping processing, already exists: {output_path.name}")
            full_img = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        else:
            full_img = process_raw_image(raw_path, args.bit_depth)
            full_img = adjust_brightness(full_img, args.brightness)
            if args.save_fullres:
                cv2.imwrite(str(output_path), full_img)
                print(f"✅ Saved full image: {output_path.name}")

        if args.save_downscaled:
            preview = cv2.resize(full_img, (0, 0), fx=args.scale_factor, fy=args.scale_factor, interpolation=cv2.INTER_AREA)
            downscaled_path = output_dir / f"{raw_path.stem}_{args.bit_depth}bit_downscaled.png"
            cv2.imwrite(str(downscaled_path), preview)
            print(f"✅ Saved downscaled image: {downscaled_path.name}")

if __name__ == "__main__":
    main()
