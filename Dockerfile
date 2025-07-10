# Autonomous Driving Perception System - Docker Image
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash vehi && \
    echo "vehi ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install -r requirements.txt

# Install additional packages for specific functionality
RUN pip3 install \
    ultralytics \
    detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch1.13/index.html \
    open3d \
    wandb \
    mlflow

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data logs outputs models static && \
    chown -R vehi:vehi /app

# Set up environment variables
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# Switch to non-root user
USER vehi

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python3", "src/api/main.py"] 