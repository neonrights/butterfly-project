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

MASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "masks"))
EXPECTED_SHAPE = (4032, 3024)


def load_mask(mask_path: str) -> np.array:
    logger.debug(f"Loading {mask_path}")
    img = Image.open(mask_path)
    mask = np.array(img, dtype="bool")
    assert mask.shape == EXPECTED_SHAPE, f"Mask is not expected shape {EXPECTED_SHAPE}, instead {mask.shape}"
    return mask


def load_mask_dir(dir_path: str):
    masks = dict()
    for file_name in os.listdir(dir_path):
        mask_name, ext = os.path.splitext(file_name)
        if not ext == ".png":
            continue
        file_path = os.path.join(dir_path, file_name)
        masks[mask_name] = load_mask(file_path)

    return masks


def get_heic_file_names_and_paths(dir_path: str) -> List[Tuple[str, str]]:
    names_and_paths = list()
    for file_name in os.listdir(dir_path):
        name, ext = os.path.splitext(file_name)
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
    masks = load_mask_dir(MASK_DIR)
    os.makedirs(args.output_dir, exist_ok=True)
    for heic_name, heic_path in tqdm(get_heic_file_names_and_paths(args.photo_dir), desc="Extracting Images"):
        logger.debug(f"Converting {heic_name}")
        heic_image = Image.open(heic_path)
        heic_array = np.array(heic_image)
        for mask_name, mask in masks.items():
            masked_image = Image.fromarray(heic_array * np.expand_dims(mask, -1))
            output_path = os.path.join(args.output_dir, f"{heic_name}-{mask_name}.png")
            masked_image.save(output_path)


if __name__ == '__main__':
    main()
