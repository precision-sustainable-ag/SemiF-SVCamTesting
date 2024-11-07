import numpy as np
import imageio
import time
from pathlib import Path
from imageio.v2 import imread
import colour
from colour_demosaicing import (

    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)

import cv2
import random

def print_image_properties(image, print_statement=""):
    print(f"\n{print_statement}")
    print(f"Image properties:")
    print(f"  - Shape: {image.shape}")
    print(f"  - Data type: {image.dtype}")
    print(f"  - Range: {image.min()} - {image.max()}")

data_ath = Path("/home/benchbot/benchbot-brain-app/mini_computer_api/images/NC_2024-11-06")
assert data_ath.exists(), "data_ath"
raw_files = list(data_ath.rglob("*.RAW"))
# raw_files = random.sample(raw_files, 10)
raw_files = sorted(raw_files)[:1]

im_height = 9528
im_width = 13376

output_dir = Path("data/results")
output_dir.mkdir(exist_ok=True, parents=True)
    
for raw_file in raw_files: 
    # Read the raw file
    nparray = np.fromfile(raw_file, dtype=np.uint16).astype(np.uint16)
    
    # Reshape the raw data to the original image shape
    org_reshaped = nparray.reshape((im_height, im_width))
    
    # Normalize the image data
    image_data = (org_reshaped).astype(np.float32)/65535.
    
    # Apply the colour encoding
    # image_data = colour.cctf_encoding(image_data)

    # colour_image_gamma_adjusted = demosaicing_CFA_Bayer_Malvar2004(image_data, "RGGB")
    colour_image_gamma_adjusted = demosaicing_CFA_Bayer_bilinear(image_data, pattern="RGGB")
    
    # Clip data
    colour_image_gamma_adjusted[colour_image_gamma_adjusted<0] = 0
    colour_image_gamma_adjusted[colour_image_gamma_adjusted>1] = 1

    # Save in 8 bit for quicker viewing
    # bgr_colour_image = cv2.cvtColor(colour_image_gamma_adjusted.astype(np.float32), cv2.COLOR_RGB2BGR)
    # bgr_colour_image = (bgr_colour_image*255).astype(np.uint8)
    # cv2.imwrite(f'{output_dir}/{raw_file.stem}_CFA_Bayer_bilinear.png', bgr_colour_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Save as 16 bit
    bgr_colour_image = cv2.cvtColor(colour_image_gamma_adjusted.astype(np.float32), cv2.COLOR_RGB2BGR)
    print(f"Writing {Path(output_dir, raw_file.stem)}.png")
    cv2.imwrite(f'{output_dir}/{raw_file.stem}.png', (bgr_colour_image*65535).astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
