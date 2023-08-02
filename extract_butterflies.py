"""
Lazily extracts butterflies from iPhone XR photos using image masks
"""
import os
import logging
import argparse
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image
from pillow_heif import register_heif_opener

logger = logging.getLogger(__name__)
register_heif_opener()

IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "", "images"))
LEFT_MASK_PATH = os.path.join(IMAGE_DIR, "butterfly-left-mask.png")
RIGHT_MASK_PATH = os.path.join(IMAGE_DIR, "butterfly-right-mask.png")
BODY_MASK_PATH = os.path.join(IMAGE_DIR, "butterfly-body-mask.png")
EXPECTED_SHAPE = (3024, 4032)


def load_mask(mask_path: str) -> np.array:
    logger.debug(f"Loading {mask_path}")
    img = Image.open(mask_path)
    mask = np.array(img, dtype="bool")
    assert mask.shape == EXPECTED_SHAPE, f"Mask is not expected shape {EXPECTED_SHAPE}, instead {mask.shape}"
    return mask


def get_heic_file_names_and_paths(dir_path: str) -> List[Tuple[str, str]]:
    names_and_paths = list()
    for file_name in os.listdir(dir_path):
        name, ext = os.splitext(file_name)
        if ext != ".HEIC":
            continue
        file_path = os.path.join(dir_path, file_name)
        if not os.path.isfile(file_path):
            continue
        names_and_paths.append((name, file_path))
    return names_and_paths


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    return parser


def main(flags: List[str] = None) -> None:
    parser = get_arg_parser()
    args = parser.parse_args(flags)

    logger.debug("Loading masks")
    masks = {
        "left": load_mask(LEFT_MASK_PATH),
        "right": load_mask(RIGHT_MASK_PATH),
        "body": load_mask(BODY_MASK_PATH),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    for heic_name, heic_path in tqdm(get_heic_file_names_and_paths(args.photo_dir), desc="Extracting Images"):
        logger.debug(f"Converting {heic_name}")
        heic_image = Image.open(heic_path)
        for mask_name, mask in masks.items():
            pass  # apply mask, then save to output


if __name__ == '__main__':
    main()
