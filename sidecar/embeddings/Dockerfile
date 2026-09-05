FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first so the large CUDA wheels are never pulled.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ENV HF_HOME=/app/.cache

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
