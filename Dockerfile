# Production Dockerfile for DP-Flash-Attention
FROM python:3.11-slim as builder

# Security: Create non-root user
RUN groupadd -r dpflash && useradd -r -g dpflash dpflash

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Security: Create non-root user
RUN groupadd -r dpflash && useradd -r -g dpflash dpflash

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=dpflash:dpflash . .

# Security: Remove unnecessary files
RUN find . -name "*.pyc" -delete && \
    find . -name "__pycache__" -delete && \
    rm -rf tests/ docs/ .git/

# Create necessary directories with correct permissions
RUN mkdir -p /app/logs /app/tmp && \
    chown -R dpflash:dpflash /app

# Switch to non-root user
USER dpflash

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Entry point
CMD ["python", "-m", "src.dp_flash_attention.cli", "--host", "0.0.0.0", "--port", "8000"]