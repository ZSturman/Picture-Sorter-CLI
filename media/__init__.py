# media/__init__.py
import os
from pathlib import Path

# future: from .audio import AudioHandler

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.m4v'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.heic', '.heif'}
AUDIO_EXTS = set()  # to be defined later

def collect_media_files(src_dir: str):
    for root, _, files in os.walk(src_dir):
        for fn in files:
            ext = Path(fn).suffix.lower()
            if ext in VIDEO_EXTS | IMAGE_EXTS | AUDIO_EXTS:
                yield os.path.join(root, fn), ext

def get_handler_for_extension(ext: str, image_mode="auto", frames=5):
    if ext in VIDEO_EXTS:
        from .video import VideoHandler
        return VideoHandler(frames=frames)
    if ext in IMAGE_EXTS:
        from .image import ImageHandler
        return ImageHandler(mode=image_mode)
    # if ext in AUDIO_EXTS:
    #    return AudioHandler()
    raise ValueError(f"Unsupported media type: {ext}")