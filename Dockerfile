FROM python:3.11-slim

# ─── System deps for PDF / OCR processing ───────────────────
#   poppler-utils  → pdf2image (PDF→image rendering for OCR)
#   tesseract-ocr  → local OCR fallback
#   gcc python3-dev → build fallback for any non-wheel packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Install Python deps first (cached layer) ──────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Copy application code + corpus ────────────────────────
COPY . .

# Runtime directories (app also calls Config.ensure_dirs at startup)
RUN mkdir -p uploads corpus

EXPOSE 5000

# Production WSGI — mirrors the Procfile command
CMD ["gunicorn", \
     "-b", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "app:app"]
