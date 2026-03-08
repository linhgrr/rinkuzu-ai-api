# Use an official Python runtime as a parent image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install requirements globally as root
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache models during build (no network needed at runtime)
ENV HF_HOME=/app/model_cache
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')" && \
    python3 -c "import underthesea; underthesea.pos_tag('Chào thế giới')"

# Create non-root user AFTER pre-caching with root
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

USER user
ENV PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Copy all project files
COPY --chown=user . .

# CRITICAL: Hardcoded absolute paths (no shell variable expansion)
ENV PYTHONPATH=/home/user/app:/home/user/app/api:/home/user/app/content-processor/src
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

