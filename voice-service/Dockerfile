FROM python:3.10-slim

# Install system dependencies including FFmpeg, Rust, and espeak-ng
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    libsndfile1 \
    git \
    curl \
    espeak-ng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Install Rust for sudachipy and TTS dependencies
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . $HOME/.cargo/env

# Make Rust available for all build steps
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper models
RUN python -c "import whisper; whisper.load_model('small')"
RUN python -c "import whisper; whisper.load_model('base')"

# Pre-download Coqui TTS models
RUN python -c "from TTS.api import TTS; TTS(model_name='tts_models/en/ljspeech/tacotron2-DDC')"
RUN python -c "from TTS.api import TTS; TTS(model_name='tts_models/en/vctk/vits')"
# German
RUN python -c "from TTS.api import TTS; TTS(model_name='tts_models/de/thorsten/tacotron2-DCA')"


# Create directories for data, models, and outputs
RUN mkdir -p /app/models /app/output/audio

# Copy application code
COPY main.py .
COPY .env .

# Set up environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Whisper models will be downloaded on first use to /root/.cache/whisper
# TTS models will be downloaded on first use to /root/.cache/torch/hub

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"] 