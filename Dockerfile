# Multi-stage Dockerfile for DP-Flash-Attention
# Supports development and production environments with CUDA

# Base CUDA image with PyTorch support
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    ninja-build \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Development stage
FROM base as development

# Install development dependencies
RUN python -m pip install \
    pytest \
    pytest-cov \
    pytest-benchmark \
    black \
    ruff \
    mypy \
    pre-commit \
    sphinx \
    jupyter

# Set working directory
WORKDIR /workspaces/dp-flash-attention

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
RUN python -m pip install -r requirements.txt -r requirements-dev.txt

# Copy source code
COPY . .

# Install in development mode
RUN python -m pip install -e ".[dev,test,docs,cuda]"

# Set up pre-commit
RUN pre-commit install || true

# Production stage
FROM base as production

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash dpuser

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

# Copy only necessary source files
COPY src/ ./src/
COPY pyproject.toml README.md LICENSE ./

# Install in production mode
RUN python -m pip install .

# Switch to non-root user
USER dpuser

# Default command
CMD ["python", "-c", "import dp_flash_attention; print('DP-Flash-Attention ready')"]

# Testing stage for CI/CD
FROM development as testing

# Copy test files
COPY tests/ ./tests/

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "-v"]