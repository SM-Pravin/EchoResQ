# Emergency AI - Production Deployment Dockerfile
# Multi-stage build for optimized production image with preinstalled models

# Stage 1: Model Download and Preparation
FROM python:3.10-slim as model-builder

# Install system dependencies for model downloading
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /models

# Create model download script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Downloading Emergency AI models..."\n\
\n\
# Create model directories\n\
mkdir -p vosk-models wav2vec2 yamnet distilroberta\n\
\n\
# Download Vosk models (English)\n\
echo "Downloading Vosk models..."\n\
wget -q https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip\n\
unzip -q vosk-model-small-en-us-0.15.zip -d vosk-models/\n\
rm vosk-model-small-en-us-0.15.zip\n\
\n\
wget -q https://alphacephei.com/vosk/models/vosk-model-medium-en-us-0.22.zip\n\
unzip -q vosk-model-medium-en-us-0.22.zip -d vosk-models/\n\
rm vosk-model-medium-en-us-0.22.zip\n\
\n\
# Download YAMNet for sound classification\n\
echo "Downloading YAMNet model..."\n\
wget -q https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed -O yamnet.tar.gz\n\
tar -xzf yamnet.tar.gz -C yamnet/\n\
rm yamnet.tar.gz\n\
\n\
# Download class labels\n\
wget -q https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv -O yamnet/yamnet_class_map.csv\n\
\n\
echo "Model downloads completed!"\n\
' > download_models.sh && chmod +x download_models.sh

# Download models
RUN ./download_models.sh

# Stage 2: Python Dependencies
FROM python:3.10-slim as python-builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libasound2-dev \
    portaudio19-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY pyproject.toml /tmp/
WORKDIR /tmp

# Install production dependencies only (excluding dev dependencies)
RUN pip install --no-cache-dir -e .[production] && \
    pip install --no-cache-dir --no-deps gunicorn uvicorn

# Stage 3: Production Image
FROM python:3.10-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TF_CPP_MIN_LOG_LEVEL=3 \
    VOSK_LOG_LEVEL=-1 \
    CUDA_VISIBLE_DEVICES="" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libasound2 \
    portaudio19-dev \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r emergency && useradd -r -g emergency -d /app -s /bin/bash emergency

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy models from model-builder stage
COPY --from=model-builder /models /app/models
RUN chown -R emergency:emergency /app/models

# Copy application code
COPY WORKING_FILES/ /app/WORKING_FILES/
COPY pyproject.toml README.md /app/
COPY config.yaml /app/

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/tmp_chunks /app/uploads && \
    chown -R emergency:emergency /app && \
    chmod +x /app/WORKING_FILES/main.py /app/WORKING_FILES/app_streamlit.py

# Switch to non-root user
USER emergency

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose ports
EXPOSE 8501 8000

# Default command (can be overridden)
CMD ["streamlit", "run", "/app/WORKING_FILES/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Labels for metadata
LABEL maintainer="Emergency AI Team <contact@emergency-ai.com>" \
      version="1.0.0" \
      description="Emergency AI - Real-time audio analysis system" \
      org.opencontainers.image.title="Emergency AI" \
      org.opencontainers.image.description="Advanced real-time emergency audio analysis system with AI-powered distress detection" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.source="https://github.com/SM-Pravin/EchoResQ" \
      org.opencontainers.image.documentation="https://github.com/SM-Pravin/EchoResQ/wiki" \
      org.opencontainers.image.licenses="MIT"