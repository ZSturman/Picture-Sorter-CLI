#!/usr/bin/env python3

import os
# Silence HuggingFace tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import argparse
from collections import Counter
from tqdm import tqdm

from media import collect_media_files, get_handler_for_extension, VIDEO_EXTS
from media.base import validate_directories

def main(argv=None, media_kind=None):
    parser = argparse.ArgumentParser(
        description="Video sorting CLI" if media_kind == 'video' else "Generic media-sorting CLI (images + video, audio to come)"
    )
    parser.add_argument('--src',     required=True, help="Source directory to scan")
    parser.add_argument('--dest',    required=True, help="Destination base directory")
    parser.add_argument('--dry-run', action='store_true', help="Don't actually move files")
    parser.add_argument('--image-mode', choices=['auto', 'objects'], default='auto',
                        help='auto: CLIP/OCR/object detection; objects: object detection only')
    parser.add_argument('--frames', type=int, default=5, help='Video frames to caption (1–100)')
    args = parser.parse_args(argv)
    try:
        validate_directories(args.src, args.dest)
        if not 1 <= args.frames <= 100:
            raise ValueError('Frames must be between 1 and 100')
    except ValueError as error:
        parser.error(str(error))

    # 1. Gather and classify by extension
    all_files = list(collect_media_files(args.src))
    if media_kind == 'video':
        all_files = [(path, ext) for path, ext in all_files if ext in VIDEO_EXTS]
    if not all_files:
        print(f"No supported media files found in {args.src}")
        sys.exit(0)

    # 2. Print counts per media type
    counts = Counter(ext for _, ext in all_files)
    print("Found:", dict(counts))

    # 3. Process with a single progress bar
    pbar = tqdm(all_files, desc="Processing media", unit="file", file=sys.stdout)
    failures = 0
    handlers = {}
    for src_path, ext in pbar:
        fn = os.path.basename(src_path)
        category = None
        try:
            if ext not in handlers:
                handlers[ext] = get_handler_for_extension(ext, args.image_mode, args.frames)
            handler = handlers[ext]
            media_obj = handler.load(src_path)
            category  = handler.categorize(media_obj)
            if not args.dry_run:
                handler.save_to(args.dest, src_path, category)
        except Exception as e:
            failures += 1
            tqdm.write(f"Error on {fn}: {e}", file=sys.stderr)
        pbar.set_postfix(file=fn, category=category)
    pbar.close()
    print(f"Processed {len(all_files)} supported files; {failures} failed; dry_run={args.dry_run}")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
