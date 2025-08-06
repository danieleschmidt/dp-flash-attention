# DP-Flash-Attention Production Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONPATH=/app/src
ENV DP_FLASH_ATTENTION_LOG_LEVEL=INFO
ENV DP_FLASH_ATTENTION_SECURITY_MODE=strict

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY *.py ./
COPY README.md ./

# Create non-root user for security
RUN useradd -m -u 1000 dpuser && \
    chown -R dpuser:dpuser /app
USER dpuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.path.insert(0, '/app/src'); from dp_flash_attention.utils import health_check; health_check()" || exit 1

# Expose ports
EXPOSE 8080 8081

# Default command
CMD ["python3", "-m", "dp_flash_attention.server"]
