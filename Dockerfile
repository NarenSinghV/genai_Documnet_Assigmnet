FROM python:3.12-slim

WORKDIR /app

# Install native system build toolchain layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Optimize pip footprint matrix execution
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir google-genai python-dotenv langchain-community sentence-transformers PyPDF2 pandas openpyxl

# Bring across complete project folder structure paths
COPY . /app

# Build explicit sandbox storage directory structures
RUN mkdir -p /app/chromadb /app/uploads/status /app/hf_cache && \
    chmod -R 777 /app/chromadb /app/uploads /app/hf_cache

# Platform Runtime Environment Configurations
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HF_HOME=/app/hf_cache

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
