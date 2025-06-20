FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies, fixing GPG key issues
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates gnupg && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    curl \
    espeak-ng \
    git && \
    rm -rf /var/lib/apt/lists/*

# Install Rust (required for some Python packages)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/models/whisper \
    /app/models/coqui \
    /app/models/huggingface \
    /app/parler_models \
    /app/output \
    /app/static

# Copy requirements and install dependencies
COPY requirements.txt .

# Install Python packages with timeout and retries
RUN pip install --no-cache-dir --timeout 100 --retries 3 -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    # Set proper ownership and permissions
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /root/.cache && \
    chmod -R 755 /root/.cache && \
    chmod -R 755 /app/models && \
    chmod -R 755 /app/parler_models && \
    chmod -R 755 /app/models/whisper && \
    chmod -R 755 /app/models/coqui && \
    chmod -R 755 /app/models/huggingface && \
    chmod -R 755 /app/output && \
    chmod -R 755 /app/static && \
    mkdir -p /home/appuser/.cache && \
    chown -R appuser:appuser /home/appuser/.cache

# Copy application code
COPY . /app

# Set proper ownership of the copied files
RUN chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/models/huggingface
ENV TORCH_HOME=/app/models/coqui
ENV HF_HOME=/app/models/huggingface
ENV WHISPER_CACHE_DIR=/app/models/whisper
ENV PARLER_CACHE_DIR=/app/parler_models

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8003

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"] 