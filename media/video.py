# media/video.py
from .base import BaseHandler
import subprocess
import tempfile
import os
import re
import math
from contextlib import contextmanager
from functools import lru_cache
from collections import Counter
@lru_cache(maxsize=1)
def captioner():
    from transformers import pipeline, logging as hf_logging
    hf_logging.set_verbosity_error()
    return pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.m4v'}
STOPWORDS = {'a','an','the','is','are','and','of','for','to','in','on','with','at','by','from'}

def get_video_duration(path: str) -> float:
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        path
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.PIPE, timeout=30)
    duration = float(out.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration must be positive and finite")
    return duration

@contextmanager
def extract_frames(path: str, num_frames: int = 5):
    if not 1 <= num_frames <= 100:
        raise ValueError("Frames must be between 1 and 100")
    duration = get_video_duration(path)
    interval = max(duration / num_frames, 0.01)
    with tempfile.TemporaryDirectory(prefix='frames_') as tmpdir:
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', path,
               '-vf', f"fps=1/{interval:.6f}", '-frames:v', str(num_frames),
               '-qscale:v', '2', os.path.join(tmpdir, 'frame_%03d.jpg')]
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        frames = sorted(os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith('.jpg'))
        if not frames:
            raise ValueError("No video frames could be decoded")
        yield frames

class VideoHandler(BaseHandler):
    def __init__(self, frames=5):
        if not 1 <= frames <= 100:
            raise ValueError("Frames must be between 1 and 100")
        self.frames = frames

    def load(self, path: str):
        # existence + basic probe
        if not os.path.isfile(path):
            raise RuntimeError("File not found")
        return path

    def categorize(self, media_obj: str) -> str:
        words = []
        with extract_frames(media_obj, num_frames=self.frames) as frames:
            model = captioner()
            for img in frames:
                out = model(img)[0]
                txt = out.get('generated_text') or out.get('caption') or ''
                for w in re.findall(r'\w+', txt.lower()):
                    if w not in STOPWORDS:
                        words.append(w)
        if not words:
            return 'uncategorized'
        top2 = [w for w,_ in Counter(words).most_common(2)]
        return " ".join(top2)[:50] or 'uncategorized'