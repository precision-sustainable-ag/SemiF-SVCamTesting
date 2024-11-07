import numpy as np
import cv2
import random
import argparse
from pathlib import Path
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process RAW images in a given directory")
    parser.add_argument("batch", type=str, help="Input directory path")
    parser.add_argument("selection_mode", choices=["random", "last", "first"],
                        help="Selection mode for choosing images: 'random', 'last', or 'first'")
    parser.add_argument("sample_number", type=int, help="Number of images to process")
    parser.add_argument("bit_depth", type=int, choices=[8, 16], help="Bit depth for saving images: 8 or 16")
    return parser.parse_args()

def print_image_properties(image, print_statement=""):
    print(f"\n{print_statement}")
    print("Image properties:")
    print(f"  - Shape: {image.shape}")
    print(f"  - Data type: {image.dtype}")
    print(f"  - Range: {image.min()} - {image.max()}")

def select_raw_files(raw_files, selection_mode, sample_number):
    if selection_mode == "first":
        return raw_files[:sample_number]
    elif selection_mode == "last":
        return raw_files[-sample_number:]
    elif selection_mode == "random":
        return random.sample(raw_files, min(sample_number, len(raw_files)))
    return []

def main():
    args = parse_arguments()
    main_dir = Path("/home/benchbot/benchbot-brain-app/mini_computer_api/images")
    input_dir = Path(main_dir, args.batch)
    assert input_dir.exists(), "Input directory does not exist"
    
    # Get the last folder in the input directory
    last_folder = sorted(input_dir.iterdir())[-1]
    raw_files = list(last_folder.rglob("*.RAW"))
    raw_files = select_raw_files(raw_files, args.selection_mode, args.sample_number)
    
    im_height = 9528
    im_width = 13376
    output_dir = Path("data/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    for raw_file in raw_files:
        # Read the raw file
        nparray = np.fromfile(raw_file, dtype=np.uint16).astype(np.uint16)
        org_reshaped = nparray.reshape((im_height, im_width))
        image_data = org_reshaped.astype(np.float32) / 65535.

        # Apply color demosaicing
        colour_image_gamma_adjusted = demosaicing_CFA_Bayer_bilinear(image_data, pattern="RGGB")
        
        # Clip data
        colour_image_gamma_adjusted = np.clip(colour_image_gamma_adjusted, 0, 1)

        # Convert to BGR and save with specified bit depth
        bgr_colour_image = cv2.cvtColor(colour_image_gamma_adjusted, cv2.COLOR_RGB2BGR)
        output_file = output_dir / f"{raw_file.stem}.png"
        
        if args.bit_depth == 8:
            # Save as 8-bit
            cv2.imwrite(str(output_file), (bgr_colour_image * 255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            # Save as 16-bit
            cv2.imwrite(str(output_file), (bgr_colour_image * 65535).astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        print(f"Saved image to {output_file} with {args.bit_depth}-bit depth")

if __name__ == "__main__":
    main()
