import numpy as np
import cv2
import random
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear

def print_image_properties(image, print_statement=""):
    print(f"\n{print_statement}")
    print("Image properties:")
    print(f"  - Shape: {image.shape}")
    print(f"  - Data type: {image.dtype}")
    print(f"  - Range: {image.min()} - {image.max()}")

def filter_by_epoch(raw_files, start_epoch, end_epoch):
    filtered_files = []
    for file in raw_files:
        try:
            epoch_timestamp = int(file.stem.split('_')[1])
            if start_epoch <= epoch_timestamp <= end_epoch:
                filtered_files.append(file)
        except (IndexError, ValueError):
            print(f"Skipping file {file} due to invalid format.")
    return filtered_files

def select_raw_files(raw_files, selection_mode, sample_number):
    if selection_mode == "first":
        return raw_files[:sample_number]
    elif selection_mode == "last":
        return raw_files[-sample_number:]
    elif selection_mode == "random":
        return random.sample(raw_files, min(sample_number, len(raw_files)))
    return []

def process_image(raw_file, im_height, im_width, bit_depth, output_dir):
    nparray = np.fromfile(raw_file, dtype=np.uint16).astype(np.uint16)
    org_reshaped = nparray.reshape((im_height, im_width))
    image_data = org_reshaped.astype(np.float32) / 65535.

    # Apply color demosaicing
    colour_image_gamma_adjusted = demosaicing_CFA_Bayer_bilinear(image_data, pattern="RGGB")
    
    # Clip data
    colour_image_gamma_adjusted = np.clip(colour_image_gamma_adjusted, 0, 1)

    # Convert image data to the appropriate bit depth and integer format for color conversion
    if bit_depth == 8:
        bgr_colour_image = (colour_image_gamma_adjusted * 255).astype(np.uint8)
    else:
        bgr_colour_image = (colour_image_gamma_adjusted * 65535).astype(np.uint16)

    # Perform color conversion (needs integer image)
    bgr_colour_image = cv2.cvtColor(bgr_colour_image, cv2.COLOR_RGB2BGR)
    
    # Save the image
    output_file = output_dir / f"{raw_file.stem}_{bit_depth}bit.png"
    cv2.imwrite(str(output_file), bgr_colour_image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    print(f"Saved image to {output_file} with {bit_depth}-bit depth")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process RAW images in a given directory")
    parser.add_argument("batch", type=str, help="Input directory path")
    parser.add_argument("selection_mode", choices=["random", "last", "first"],
                        help="Selection mode for choosing images: 'random', 'last', or 'first'")
    parser.add_argument("sample_number", type=int, help="Number of images to process")
    parser.add_argument("bit_depth", type=int, choices=[8, 16], help="Bit depth for saving images: 8 or 16")
    parser.add_argument("--start", type=int, default=0, help="Starting epoch timestamp")
    parser.add_argument("--end", type=int, default=2147483647, help="Ending epoch timestamp")
    parser.add_argument("--concurrent", action="store_true", help="Enable concurrent processing")
    return parser.parse_args()

def main():
    args = parse_arguments()
    # main_dir = Path("/mnt/research-projects/s/screberg/longterm_images2/semifield-upload")
    main_dir = Path("data/temp")
    input_dir = Path(main_dir, args.batch)
    assert input_dir.exists(), "Input directory does not exist"
    
    output_dir = Path("data/results", args.batch)
    output_dir.mkdir(exist_ok=True, parents=True)


    # Get the raw image files and filter by epoch
    raw_files = sorted(list(input_dir.rglob("*.RAW")))
    raw_files = filter_by_epoch(raw_files, args.start, args.end)
    print(f"Found {len(raw_files)} files in the directory")
    raw_files = select_raw_files(raw_files, args.selection_mode, args.sample_number)

    # Remove already demosaiced files from the list
    demosaiced_files = {file.stem.replace(f"_{args.bit_depth}bit", "") for file in output_dir.glob("*.png")}

    raw_files = [file for file in raw_files if file.stem not in demosaiced_files]
    print(f"{len(raw_files)} files remaining after filtering out already demosaiced files")
    
    im_height = 9528
    im_width = 13376

    if args.concurrent:
        # Use ProcessPoolExecutor for parallel processing
        max_workers = min(8, len(raw_files))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_image, raw_file, im_height, im_width, args.bit_depth, output_dir)
                for raw_file in raw_files
            ]
            for future in futures:
                future.result()  # This will raise any exceptions caught during processing
    else:
        for raw_file in raw_files:
            process_image(raw_file, im_height, im_width, args.bit_depth, output_dir)
    

if __name__ == "__main__":
    main()
