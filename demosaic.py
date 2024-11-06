import numpy as np
import imageio
import time
from pathlib import Path
from imageio.v2 import imread
import colour
from colour_demosaicing import (
    ROOT_RESOURCES_EXAMPLES,
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)

import cv2

def print_image_properties(image, print_statement=""):
    print(f"\n{print_statement}")
    print(f"Image properties:")
    print(f"  - Shape: {image.shape}")
    print(f"  - Data type: {image.dtype}")
    print(f"  - Range: {image.min()} - {image.max()}")

data_ath = Path("data")

raw_files = list(data_ath.rglob("*.RAW"))[4:]

im_height = 9528
im_width = 13376


for raw_file in raw_files: 
    # Read the raw file
    nparray = np.fromfile(raw_file, dtype=np.uint16).astype(np.uint16)
    print_image_properties(nparray, f"Raw file: {raw_file}")
    
    # Reshape the raw data to the original image shape
    org_reshaped = nparray.reshape((im_height, im_width))
    print_image_properties(org_reshaped, "Original image")
    
    # Normalize the image data
    image_data = (org_reshaped).astype(np.float32)/65535.
    print_image_properties(image_data, "Normalized image data")
    
    # Apply the colour encoding
    image_data = colour.cctf_encoding(image_data)
    print_image_properties(image_data, "Colour encoded image data")
    

    # colour_image_gamma_adjusted = demosaicing_CFA_Bayer_Malvar2004(image_data, "RGGB")
    colour_image_gamma_adjusted = demosaicing_CFA_Bayer_bilinear(image_data, pattern="RGGB")
    print_image_properties(colour_image_gamma_adjusted, "Demosaiced image")

    colour_image_gamma_adjusted[colour_image_gamma_adjusted<0] = 0
    colour_image_gamma_adjusted[colour_image_gamma_adjusted>1] = 1

    print_image_properties(colour_image_gamma_adjusted, "Demosaiced and clipped image")
    
        
    # test = cv2.cvtColor(colour_image_gamma_adjusted.astype(np.float32), cv2.COLOR_RGB2BGR)
    bgr_colour_image = cv2.cvtColor(colour_image_gamma_adjusted.astype(np.float32), cv2.COLOR_RGB2BGR)
    # bgr_colour_image = colour_image_gamma_adjusted.astype(np.float32)

    bgr_colour_image = (bgr_colour_image*255).astype(np.uint8)
    
    print_image_properties(bgr_colour_image, "RGB 2 BGR image.")

    cv2.imwrite(f'data/results/{raw_file.stem}_CFA_Bayer_bilinear.png', bgr_colour_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])