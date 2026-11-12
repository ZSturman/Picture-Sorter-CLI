#!/usr/bin/env python3
"""
sort_images_frcnn.py

Batch-sort images (including HEIC) into slugified category folders
using TorchVision's Faster R-CNN, with a tqdm progress bar.
"""

import argparse
import os
import sys
from PIL import Image, UnidentifiedImageError
import pillow_heif         # HEIC/HEIF plugin for Pillow
import torch
from tqdm import tqdm
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from slugify import slugify
from media.base import BaseHandler, validate_directories
from media import IMAGE_EXTS
import math

# Register HEIC/HEIF opener so Pillow can read .HEIC/.HEIF files
pillow_heif.register_heif_opener()

# Load COCO class names from the pretrained weights metadata
WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
COCO_CLASS_NAMES = WEIGHTS.meta["categories"]

# STOPWORDS to strip out of final slug
STOPWORDS = {'a', 'an', 'the', 'is', 'are'}


def is_image(path: str) -> bool:
    """Return True if Pillow can open and verify the file as an image."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def get_detections(model, preprocess, img_path: str, threshold: float) -> set[str]:
    """
    Run Faster R-CNN on a single image and return
    a set of COCO class names above the confidence threshold.
    """
    image = Image.open(img_path).convert("RGB")
    tensor = preprocess(image)
    with torch.no_grad():
        outputs = model([tensor])[0]
    labels = outputs['labels'].tolist()
    scores = outputs['scores'].tolist()
    return {
        COCO_CLASS_NAMES[label]
        for label, score in zip(labels, scores)
        if score >= threshold
    }


def categorize(detected: set[str]) -> str:
    """
    Build a slug from the raw label set (or "others"):
      1. Join into space-separated text
      2. slugify() & truncate to 50 chars
      3. replace '-' with ' ', split, strip STOPWORDS
    """
    text = " ".join(sorted(detected)) if detected else "others"
    raw = slugify(text)[:50] or "uncategorized"
    words = raw.replace('-', ' ').split()
    filtered = [w for w in words if w not in STOPWORDS]
    return " ".join(filtered) or "uncategorized"


def main():
    p = argparse.ArgumentParser(
        description="Batch-sort images (including HEIC) into slugified folders"
    )
    p.add_argument('-s', '--src',  required=True, help='Source directory of files')
    p.add_argument('-d', '--dest', required=True, help='Destination base directory')
    p.add_argument('-c', '--conf', type=float, default=0.5,
                   help='Confidence threshold (0–1)')
    p.add_argument('--dry-run', action='store_true', help='Classify without moving any files')
    args = p.parse_args()
    try:
        validate_directories(args.src, args.dest)
        if not math.isfinite(args.conf) or not 0 <= args.conf <= 1:
            raise ValueError('Confidence must be between 0 and 1')
    except ValueError as error:
        p.error(str(error))

    # Load the model + default preprocessing
    model = fasterrcnn_resnet50_fpn(weights=WEIGHTS, progress=False)
    model.eval()
    preprocess = WEIGHTS.transforms()

    # Gather all files in src
    all_files = [
        f for f in os.listdir(args.src)
        if os.path.isfile(os.path.join(args.src, f))
    ]

    # Process with tqdm, showing current file & its slug folder
    pbar = tqdm(all_files, desc="Sorting images", unit="file")
    failures = 0
    for fname in pbar:
        src_path = os.path.join(args.src, fname)
        if not is_image(src_path):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                failures += 1
                pbar.write(f"Cannot open image: {fname}", file=sys.stderr)
            continue

        try:
            detected = get_detections(model, preprocess, src_path, args.conf)
            slug     = categorize(detected)
        except Exception as e:
            failures += 1
            pbar.write(f"Skipping {fname}: error {e}", file=sys.stderr)
            continue

        try:
            if not args.dry_run:
                BaseHandler().save_to(args.dest, src_path, slug)
        except Exception as error:
            failures += 1
            pbar.write(f"Failed to move {fname}: {error}", file=sys.stderr)

        pbar.set_postfix(file=fname, folder=slug)

    pbar.close()
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
