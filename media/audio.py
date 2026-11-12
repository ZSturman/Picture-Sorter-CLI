# media/audio.py
from .base import BaseHandler

class AudioHandler(BaseHandler):
    def load(self, path: str):
        # stub for future audio support
        raise NotImplementedError("Audio support coming soon")

    def categorize(self, media_obj):
        raise NotImplementedError