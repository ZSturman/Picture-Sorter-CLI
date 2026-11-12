from .base import BaseHandler
from PIL import Image, UnidentifiedImageError
import pillow_heif      # HEIC/HEIF support
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from slugify import slugify
from typing import Optional
from pytesseract import image_to_string, TesseractNotFoundError
from functools import lru_cache
import re

pillow_heif.register_heif_opener()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
COCO_CLASSES = WEIGHTS.meta["categories"]
STOPWORDS = {'a','an','the','is','are','and','of','for','to','in','on','with','at','by','from'}

@lru_cache(maxsize=1)
def detection_resources():
    model = fasterrcnn_resnet50_fpn(weights=WEIGHTS, progress=False).to(DEVICE)
    model.eval()
    return model, WEIGHTS.transforms()

@lru_cache(maxsize=1)
def clip_resources():
    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    model.eval()
    return processor, model

@lru_cache(maxsize=1)
def summarizer():
    from transformers import pipeline
    return pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)


def summarize_ocr(text: str) -> str:
    # Short text already fits a folder label; abstractive generation can invent content.
    if len(text.split()) <= 50:
        return text
    return summarizer()(text, max_length=30, min_length=5,
                        do_sample=False, truncation=True)[0]['summary_text']


class ImageHandler(BaseHandler):
    def __init__(self, mode="auto"):
        if mode not in {"auto", "objects"}:
            raise ValueError("Image mode must be auto or objects")
        self.mode = mode

    def load(self, path: str) -> str:
        try:
            with Image.open(path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Cannot open image: {e}")
        return path

    def crop_screenshot(self, image: Image.Image) -> Image.Image:
        """
        Remove top menu bar and bottom dock from screenshots.
        """
        w, h = image.size
        top = int(h * 0.08)      # drop top 8%
        bottom = int(h * 0.95)   # drop bottom 5%
        return image.crop((0, top, w, bottom))

    def clean_summary(self, summary: str) -> str:
        """
        Strip out UI noise (dates, times, URLs, usernames), punctuation, and very short tokens.
        """
        patterns = [
            r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b",
            r"\d{1,2}:\d{2}(?:AM|PM|am|pm)?",
            r"\b\d+%|\bpercent\b",
            r"https?://\S+",
            r"[A-Z][a-z]+ [A-Z][a-z]+",   # crude remove of “First Last” names
        ]
        for pat in patterns:
            summary = re.sub(pat, " ", summary)
        # remove any non-alphanumeric/spaces
        summary = re.sub(r"[^A-Za-z0-9 ]+", " ", summary)
        # drop 1–2 character tokens
        tokens = [t for t in summary.split() if len(t) > 2]
        return " ".join(tokens) or summary

    def categorize_text_image(self, image: Image.Image) -> Optional[str]:
        """
        OCR + summarization slug for screenshots/docs
        """
        try:
            text = image_to_string(image).strip()
        except TesseractNotFoundError as error:
            raise RuntimeError('Auto image mode requires the tesseract executable (macOS: brew install tesseract); use --image-mode objects for object detection only') from error
        if not text:
            return None

        summary = summarize_ocr(text)

        cleaned = self.clean_summary(summary)
        slug = slugify(
            cleaned,
            max_length=50,
            word_boundary=True
        ) or "uncategorized"
        return slug

    def categorize_visual_image(self, image: Image.Image) -> str:
        """
        Object-detection-based slug for camera photos
        """
        model, preprocess = detection_resources()
        tensor = preprocess(image).to(DEVICE)
        with torch.no_grad():
            out = model([tensor])[0]

        labels = out["labels"].tolist()
        scores = out["scores"].tolist()

        # keep only high-confidence labels
        raw_labels = [
            COCO_CLASSES[lbl]
            for lbl, sc in zip(labels, scores)
            if sc >= 0.5
        ]

        # preserve order, drop duplicates & stopwords
        seen = set()
        unique = []
        for lbl in raw_labels:
            lower = lbl.lower()
            if lower not in seen and lower not in STOPWORDS:
                seen.add(lower)
                unique.append(lbl)

        text = " ".join(unique) or "others"
        slug = slugify(
            text,
            max_length=50,
            word_boundary=True
        ) or "uncategorized"
        return slug

    def categorize(self, media_obj: str) -> str:
        """
        1) Distinguish camera vs screenshot vs document via CLIP
        2) For screenshots/docs, crop UI and try OCR summarization
        3) Else fall back to visual detection slug
        """
        with Image.open(media_obj) as source:
            img = source.convert("RGB")
        if self.mode == "objects":
            return self.categorize_visual_image(img)
        processor, model = clip_resources()

        # 1. Classify type
        inputs = processor(
            text=["camera photo", "screenshot", "document scan"],
            images=img,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        with torch.no_grad():
            clip_out = model(**inputs)
        probs = clip_out.logits_per_image.softmax(dim=1)[0]
        labels = ["camera photo", "screenshot", "document scan"]
        clip_label = labels[probs.argmax().item()]

        # 2. If screenshot or document, crop and OCR
        if clip_label in ("screenshot", "document scan"):
            ocr_image = self.crop_screenshot(img) if clip_label == "screenshot" else img
            text_cat = self.categorize_text_image(ocr_image)
            if text_cat:
                return text_cat

        # 3. Otherwise fallback to visual detection
        return self.categorize_visual_image(img)
