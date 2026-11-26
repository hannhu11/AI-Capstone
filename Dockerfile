# ==============================================================================
# INDUSTRIAL CCTV SAFETY SURVEILLANCE & ALERTING DOCKERFILE
# Base: PyTorch 2.0.0 with CUDA 11.7 & CUDNN8 Runtime (NVIDIA GPU Accelerated)
# Author: Capstone AI Research Team
# ==============================================================================

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

LABEL maintainer="Capstone AI Research Team"
LABEL description="Rep-YOLO11s Multi-Object Tracking & PPE Safety Compliance Container"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for OpenCV and video decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency requirements
COPY requirements_docker.txt /app/requirements_docker.txt

# Install python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_docker.txt

# Pre-cache MobileNetV3-Small weights for offline Deep Re-ID in isolated containers
RUN python -c "import torchvision.models as m; m.mobilenet_v3_small(weights=m.MobileNet_V3_Small_Weights.DEFAULT)"

# Copy application source code and configs
COPY src/ /app/src/
COPY custom_ablation_modules.py /app/custom_ablation_modules.py
COPY scripts/ /app/scripts/
COPY rep_yolo11s*.yaml /app/
# Copy non-secret environment template (real secrets injected via env_file / runtime)
COPY .env.example /app/.env.example
RUN cp /app/.env.example /app/.env

# Create directories for mounted data
RUN mkdir -p /app/video_test /app/Output /app/snapshots

# Expose web dashboard port
EXPOSE 8000

# Healthcheck to ensure FastAPI service is responsive on dynamic WEB_PORT
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD sh -c 'curl -f http://localhost:${WEB_PORT:-8000}/api/status || exit 1'

# Default command: launch FastAPI CCTV Surveillance Server
CMD ["python", "-m", "src.web.app"]
