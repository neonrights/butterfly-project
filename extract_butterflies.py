"""
Lazily extracts butterflies from iPhone XR photos using image masks
"""
import os
import logging
import argparse
from typing import List, Tuple

import PIL
import numpy as np
from tqdm import tqdm
from PIL import Image
from pillow_heif import register_heif_opener

logger = logging.getLogger(__name__)
register_heif_opener()

MASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "masks"))
XR_DIM = (4032, 3024)


def load_mask(mask_path: str) -> np.array:
    logger.debug(f"Loading {mask_path}")
    img = Image.open(mask_path)
    mask = np.array(img, dtype="bool")
    assert mask.shape == XR_DIM, f"Mask is not expected shape {XR_DIM}, instead {mask.shape}"
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


def get_file_names_and_paths(dir_path: str) -> List[Tuple[str, str]]:
    names_and_paths = list()
    for file_name in os.listdir(dir_path):
        name, ext = os.path.splitext(file_name)
        file_path = os.path.join(dir_path, file_name)
        if not os.path.isfile(file_path):
            continue
        names_and_paths.append((name, file_path))
    return names_and_paths


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--flip_output", action="store_true", default=False)

    return parser


def main(flags: List[str] = None) -> None:
    parser = get_arg_parser()
    args = parser.parse_args(flags)

    logger.debug("Loading masks")
    masks = load_mask_dir(MASK_DIR)
    os.makedirs(args.output_dir, exist_ok=True)
    for image_name, image_path in tqdm(get_file_names_and_paths(args.photo_dir), desc="Extracting images"):
        logger.debug(f"Converting {image_name}")
        try:
            image = Image.open(image_path)
        except PIL.UnidentifiedImageError:
            logger.warning(f"{image_path} is not a supported PIL file. Skipping")
            continue
        image_array = np.array(image)
        for mask_name, mask in masks.items():
            if image_array.ndim > 2:
                mask = np.expand_dims(mask, -1)
            output_array = image_array * mask
            if args.flip_output:
                # flip on y-axis
                output_array = np.flip(output_array, 1)
                output_name = f"{image_name}-{mask_name}-flipped.png"
            else:
                output_name = f"{image_name}-{mask_name}.png"
            masked_image = Image.fromarray(output_array)
            output_path = os.path.join(args.output_dir, output_name)
            masked_image.save(output_path)


if __name__ == '__main__':
    main()
