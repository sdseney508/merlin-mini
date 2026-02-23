FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps:
# - ca-certificates: HTTPS trust
# - python3/pip/venv: runtime
# - tesseract-ocr(+eng): OCR
# - libgl1 + libglib2.0-0: common deps for image libs / headless rendering
# - build essentials + image libs: helps if Pillow/matplotlib wheels need compilation on arm64
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    python3 python3-pip python3-venv \
    tesseract-ocr tesseract-ocr-eng \
    libgl1 libglib2.0-0 \
    build-essential gcc \
    libjpeg-turbo8 zlib1g libfreetype6 libpng16-16 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create venv for cleaner installs
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Optional: run as non-root
RUN useradd -m -u 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=8000"]