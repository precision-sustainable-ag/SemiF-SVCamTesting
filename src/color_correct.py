import numpy as np
import cv2
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def apply_transformation_matrix(source_img: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    """Apply a transformation matrix to the source image to correct its color space."""
    if transformation_matrix.shape != (9, 9):
        log.error("Transformation matrix must be a 9x9 matrix.")
        return None

    if source_img.ndim != 3:
        log.error("Source image must be an RGB image.")
        return None

    red, green, blue, *_ = np.split(transformation_matrix, 9, axis=1)

    source_dtype = source_img.dtype
    max_val = np.iinfo(source_dtype).max if source_dtype.kind == 'u' else 1.0

    source_flt = source_img.astype(np.float64) / max_val
    source_b, source_g, source_r = cv2.split(source_flt)

    source_b2, source_b3 = source_b**2, source_b**3
    source_g2, source_g3 = source_g**2, source_g**3
    source_r2, source_r3 = source_r**2, source_r**3

    b = (source_r * blue[0] + source_g * blue[1] + source_b * blue[2] +
         source_r2 * blue[3] + source_g2 * blue[4] + source_b2 * blue[5] +
         source_r3 * blue[6] + source_g3 * blue[7] + source_b3 * blue[8])
    
    g = (source_r * green[0] + source_g * green[1] + source_b * green[2] +
         source_r2 * green[3] + source_g2 * green[4] + source_b2 * green[5] +
         source_r3 * green[6] + source_g3 * green[7] + source_b3 * green[8])
    
    r = (source_r * red[0] + source_g * red[1] + source_b * red[2] +
         source_r2 * red[3] + source_g2 * red[4] + source_b2 * red[5] +
         source_r3 * red[6] + source_g3 * red[7] + source_b3 * red[8])

    corrected_img = cv2.merge([b, g, r])
    corrected_img = np.clip(corrected_img * max_val, 0, max_val).astype(source_dtype)
    return corrected_img

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    batch_id = cfg.batch_id
    demosiac_dir = Path(cfg.paths.demosaiced, batch_id)
    demosiac_imgs = list(demosiac_dir.glob("*.png"))

    corrected_dir = Path(cfg.paths.colorcorrected_upload, batch_id)
    corrected_dir.mkdir(parents=True, exist_ok=True)

    color_matrix_path = cfg.paths.color_matrix

    downscale_factor = cfg.color_correct.scale_factor
    
    if Path(color_matrix_path).exists():
        with np.load(color_matrix_path) as data:
            transformation_matrix = data['matrix']
    else:
        log.error(f"Transformation matrix file {color_matrix_path} not found.")
        return
    
    for demosiac_img in demosiac_imgs:
        image = cv2.imread(str(demosiac_img), cv2.IMREAD_UNCHANGED)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        corrected_img = apply_transformation_matrix(image, transformation_matrix)

        bit8_corrected_img = (corrected_img * 255).astype(np.uint8)
        
        image_stem = demosiac_img.stem.replace("_16bit", "")
        
        if downscale_factor == 1:
            cv2.imwrite(f"{corrected_dir}/{image_stem}.jpg", cv2.cvtColor(bit8_corrected_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
            continue
        
        downscaled_img = cv2.resize(bit8_corrected_img, (0, 0), fx=downscale_factor, fy=downscale_factor)
        cv2.imwrite(f"{corrected_dir}/{image_stem}.jpg", cv2.cvtColor(downscaled_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])

if __name__ == "__main__":
    main()
