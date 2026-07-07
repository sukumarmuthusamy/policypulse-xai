FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POLICIES_DIR=data/policies \
    FAISS_INDEX_PATH=storage/faiss.index \
    BM25_INDEX_PATH=storage/bm25.pkl \
    CHUNKS_PATH=storage/chunks.json \
    LOG_PATH=storage/structured_logs.jsonl

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt python-multipart

COPY app ./app
COPY scripts ./scripts

RUN mkdir -p data/policies storage

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
