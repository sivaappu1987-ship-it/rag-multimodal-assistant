# Stage 1: Builder
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Prevent pip from failing on slow/unstable network connections
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=20

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list 2>/dev/null || true \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY wheels/ ./wheels/

RUN pip install --no-cache-dir --timeout 1000 --retries 20 --upgrade pip \
    && pip install --no-cache-dir --timeout 1000 --retries 20 --find-links=wheels/ --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list 2>/dev/null || true \
    && apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY app/ ./app/

# Pre-cache whisper-small and embedding model weights
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8'); from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]