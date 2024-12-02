import numpy as np
import cv2
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from pprint import pprint

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


def get_matrix_m(target_matrix: np.ndarray, source_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate components required for generating a transformation matrix."""
    if target_matrix.shape == source_matrix.shape:
        _, t_r, t_g, t_b = np.split(target_matrix, 4, axis=1)
        _, s_r, s_g, s_b = np.split(source_matrix, 4, axis=1)
    else:
        combined_matrix = []
        for target, source in zip(target_matrix, source_matrix):
            if target[0] == source[0]:
                combined_matrix.append(np.hstack([target, source[1:]]))
        combined_matrix = np.array(combined_matrix)
        _, t_r, t_g, t_b, s_r, s_g, s_b = np.split(combined_matrix, 7, axis=1)

    matrix_a = np.hstack([s_r, s_g, s_b, s_r**2, s_g**2, s_b**2, s_r**3, s_g**3, s_b**3])
    matrix_m = np.linalg.solve(matrix_a.T @ matrix_a, matrix_a.T)
    matrix_b = np.hstack([t_r, t_r**2, t_r**3, t_g, t_g**2, t_g**3, t_b, t_b**2, t_b**3])
    return matrix_a, matrix_m, matrix_b


def calc_transformation_matrix(matrix_m: np.ndarray, matrix_b: np.ndarray) -> tuple[float, np.ndarray]:
    """Calculate the transformation matrix and its deviance."""
    t_r, t_r2, t_r3, t_g, t_g2, t_g3, t_b, t_b2, t_b3 = np.split(matrix_b, 9, 1)

    # multiply each 22x1 matrix from target color space by matrix_m
    red = np.matmul(matrix_m, t_r)
    green = np.matmul(matrix_m, t_g)
    blue = np.matmul(matrix_m, t_b)

    red2 = np.matmul(matrix_m, t_r2)
    green2 = np.matmul(matrix_m, t_g2)
    blue2 = np.matmul(matrix_m, t_b2)

    red3 = np.matmul(matrix_m, t_r3)
    green3 = np.matmul(matrix_m, t_g3)
    blue3 = np.matmul(matrix_m, t_b3)

    # concatenate each product column into 9X9 transformation matrix
    transformation_matrix = np.concatenate((red, green, blue, red2, green2, blue2, red3, green3, blue3), 1)

    # find determinant of transformation matrix
    t_det = np.linalg.det(transformation_matrix)

    return 1-t_det, transformation_matrix

def save_matrix(matrix: np.ndarray, filename: str) -> None:
    """Save a matrix to a file."""
    np.savez(filename, matrix=matrix)


def array_info(array: np.ndarray) -> dict:
    """Provide metadata about a numpy array."""
    return {
        "shape": array.shape,
        "dtype": array.dtype,
        "min": np.min(array),
        "max": np.max(array),
    }


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    img_path = Path("/home/mkutuga/SemiF-SVCamTesting/data/results/NC_2024-12-02/NC_1733153354_16bit.png")
    image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0

    print("Image info:")
    pprint(array_info(image))

    reference_colors = np.array([
        [idx] + [x / 255.0 for x in ref['rgb']]
        for idx, ref in enumerate(cfg.colorchecker.reference_colors, start=1)
    ])

    measured_colors = np.array([
        [idx] + [x / 255.0 for x in meas['rgb']]
        for idx, meas in enumerate(cfg.colorchecker.image_colors, start=1)
    ])

    # print("\nReference colors:")
    # pprint(array_info(reference_colors))
    # print("\nMeasured colors:")
    # pprint(array_info(measured_colors))

    _, matrix_m, matrix_b = get_matrix_m(reference_colors, measured_colors)
    _, transformation_matrix = calc_transformation_matrix(matrix_m, matrix_b)

    save_matrix(transformation_matrix, "transformation_matrix.npz")

    corrected_img = apply_transformation_matrix(image, transformation_matrix)

    bit8_corrected_img = (corrected_img * 255).astype(np.uint8)
    cv2.imwrite(f"{img_path}_corrected_image_8bit.png", cv2.cvtColor(bit8_corrected_img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
