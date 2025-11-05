# Dockerfile
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Minimal libs needed by numpy/opencv/paddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Perf/env defaults (tweak as needed)
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    UVICORN_WORKERS=1 \
    PADDLEOCR_LANG=en \
    PADDLEOCR_USE_ANGLE_CLS=false \
    PORT=8080

# Pre-download models so cold starts are faster
RUN python - <<'PY'
from paddleocr import PaddleOCR
PaddleOCR(use_angle_cls=True, lang="en")
print("PaddleOCR models cached.")
PY

COPY app.py .

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]