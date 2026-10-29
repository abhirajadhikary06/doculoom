"""Targeted OCR: full-page OCR only on zero-text pages; text extraction otherwise."""
from __future__ import annotations
import hashlib
import io
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from config import Config

logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_text_pypdf(path: Path) -> Tuple[List[str], bool]:
    """
    Returns (list of page texts, had_any_text).
    Empty string for a page means likely scanned / needs OCR.
    """
    pages: List[str] = []
    had_text = False
    try:
        reader = PdfReader(str(path))
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            t = t.strip()
            if t:
                had_text = True
            pages.append(t)
    except PdfReadError as e:
        logger.warning("pypdf failed for %s: %s", path, e)
        return [], False
    return pages, had_text


def ocr_space_image(image_bytes: bytes, filename: str = "page.png", overlay: bool = False) -> Tuple[str, list]:
    """
    Call free OCR.space API (requires API key for higher limits).
    If overlay=True, returns (text, word_boxes) where word_boxes is a list of
    {text, left, top, width, height} dicts for bounding-box awareness.
    """
    if not Config.OCR_SPACE_API_KEY:
        return "", []
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                Config.OCR_SPACE_URL,
                data={
                    "apikey": Config.OCR_SPACE_API_KEY,
                    "language": "eng",
                    "isOverlayRequired": overlay,
                    "OCREngine": 2,
                    "scale": True,
                },
                files={"file": (filename, image_bytes, "image/png")},
            )
            r.raise_for_status()
            data = r.json()
            if data.get("IsErroredOnProcessing"):
                logger.warning("OCR.space error: %s", data.get("ErrorMessage"))
                return "", []
            parsed = data.get("ParsedResults") or []
            if not parsed:
                return "", []
            text = (parsed[0].get("ParsedText") or "").strip()
            word_boxes = []
            if overlay:
                lines = parsed[0].get("TextOverlay", {}).get("Lines", [])
                for line in lines:
                    for word in line.get("WordObjects", line.get("Words", [])):
                        word_boxes.append({
                            "text": word.get("WordText", ""),
                            "left": int(word.get("Left", 0)),
                            "top": int(word.get("Top", 0)),
                            "width": int(word.get("Width", 0)),
                            "height": int(word.get("Height", 0)),
                        })
            return text, word_boxes
    except Exception as e:
        logger.warning("OCR.space request failed: %s", e)
        return "", []


def ocr_tesseract_page(image) -> str:
    """Local tesseract fallback."""
    try:
        import pytesseract
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.open(io.BytesIO(image)) if isinstance(image, bytes) else image
        return pytesseract.image_to_string(image, lang="eng").strip()
    except Exception as e:
        logger.debug("tesseract unavailable or failed: %s", e)
        return ""


def render_pdf_pages(path: Path, dpi: int = 200) -> List:
    """Render PDF pages to PIL images. Requires poppler + pdf2image."""
    try:
        from pdf2image import convert_from_path

        return convert_from_path(str(path), dpi=dpi)
    except Exception as e:
        logger.warning("pdf2image failed (is poppler installed?): %s", e)
        return []


def _merge_ocr_into_page(pypdf_text: str, ocr_text: str, word_boxes: list) -> str:
    """
    Merge OCR text into a page that already has pypdf-extracted text.
    Only adds OCR words that appear to come from image regions (i.e. not
    already captured by pypdf) by checking word-level overlap.
    """
    if not ocr_text:
        return pypdf_text
    # Build a set of pypdf words for quick membership test
    pypdf_words = set(w.lower().strip(".,;:!?()[]{}\"'") for w in pypdf_text.split())
    # Find OCR words not in pypdf text (image-region text pypdf missed)
    extra_words = []
    for wb in word_boxes:
        w = wb.get("text", "").lower().strip(".,;:!?()[]{}\"'")
        if w and w not in pypdf_words and len(w) > 1:
            extra_words.append(wb.get("text", ""))
    if not extra_words:
        return pypdf_text
    # Append image-region text at the end of the page
    extra = " ".join(extra_words)
    return pypdf_text + "\n[OCR image text] " + extra


def extract_document(path: Path) -> Tuple[str, List[str], dict]:
    """
    Full pipeline:
    1. Try pypdf text extraction.
    2. For pages with zero text → full-page OCR (OCR.space or tesseract).
    3. For pages WITH text → bounding-box overlay OCR to capture image-region
       text that pypdf missed (tables, figures, embedded images).
    4. Join with page separators; return full text, per-page texts, meta.
    """
    path = Path(path)
    pages, had_text = extract_text_pypdf(path)
    meta = {
        "page_count": len(pages),
        "ocr_pages": [],
        "image_ocr_pages": [],
        "extraction": "pypdf",
        "checksum": sha256_file(path),
    }

    zero_text_pages = [i for i, t in enumerate(pages) if not t]
    digital_pages = [i for i, t in enumerate(pages) if t]

    # Render images once if any OCR is needed
    need_images = bool(zero_text_pages or (digital_pages and Config.OCR_SPACE_API_KEY))
    images = render_pdf_pages(path) if need_images else []

    # Full-page OCR for zero-text (scanned) pages
    for i in zero_text_pages:
        text = ""
        if i < len(images):
            img = images[i]
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            # Prefer OCR.space if key present
            text, _ = ocr_space_image(img_bytes, f"page_{i+1}.png", overlay=False)
            if not text:
                text = ocr_tesseract_page(img)
        pages[i] = text
        if text:
            meta["ocr_pages"].append(i + 1)
    if meta["ocr_pages"]:
        meta["extraction"] = "hybrid_ocr"

    # Bounding-box OCR for digital pages — capture image-region text pypdf missed
    if digital_pages and Config.OCR_SPACE_API_KEY and images:
        for i in digital_pages:
            if i >= len(images):
                continue
            img = images[i]
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            ocr_text, word_boxes = ocr_space_image(
                img_bytes, f"page_{i+1}.png", overlay=True
            )
            if word_boxes:
                merged = _merge_ocr_into_page(pages[i], ocr_text, word_boxes)
                if merged != pages[i]:
                    pages[i] = merged
                    meta["image_ocr_pages"].append(i + 1)
        if meta["image_ocr_pages"]:
            meta["extraction"] = "hybrid_ocr"

    # Join pages
    full = "\n\n".join(pages)
    return full, pages, meta
