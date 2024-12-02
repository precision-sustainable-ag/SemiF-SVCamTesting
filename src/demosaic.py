import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
import logging
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


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
    log.info(f"Saved image to {output_file} with {bit_depth}-bit depth")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    
    # main_dir = Path("/mnt/research-projects/s/screberg/longterm_images2/semifield-upload")
    main_dir = Path("temp_data/semifield-upload")
    input_dir = Path(main_dir, cfg.batch_id)
    
    assert input_dir.exists(), "Input directory does not exist"
    
    output_dir = Path("data/results", cfg.batch_id)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get the raw image files and filter by epoch
    raw_files = sorted(list(input_dir.rglob("*.RAW")))
    
    # Remove already demosaiced files from the list
    demosaiced_files = {file.stem.replace(f"_{cfg.demosaic.bit_depth}bit", "") for file in output_dir.glob("*.png")}

    raw_files = [file for file in raw_files if file.stem not in demosaiced_files]
    log.info(f"{len(raw_files)} files remaining after filtering out already demosaiced files")
    
    im_height = 9528
    im_width = 13376

    if cfg.demosaic.concurrent:
        # Use ProcessPoolExecutor for parallel processing
        max_workers = min(cfg.demosaic.concurrent_workers, len(raw_files))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_image, raw_file, im_height, im_width, cfg.demosaic.bit_depth, output_dir)
                for raw_file in raw_files
            ]
            for future in futures:
                future.result()  # This will raise any exceptions caught during processing
    else:
        for raw_file in raw_files:
            process_image(raw_file, im_height, im_width, cfg.demosaic.bit_depth, output_dir)
    

if __name__ == "__main__":
    main()
