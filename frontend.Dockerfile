FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BACKEND_URL=http://backend:8000

WORKDIR /app

RUN pip install --no-cache-dir streamlit httpx

COPY frontend ./frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
