# Development container for DP-Flash-Attention
FROM nvidia/cuda:12.0-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    wget \
    vim \
    sudo \
    software-properties-common \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Create vscode user for development
RUN groupadd --gid 1000 vscode && \
    useradd --uid 1000 --gid vscode --shell /bin/bash --create-home vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up directories with proper permissions
RUN mkdir -p /workspaces/dp-flash-attention && \
    chown -R vscode:vscode /workspaces

# Switch to vscode user
USER vscode
WORKDIR /workspaces/dp-flash-attention

# Create virtual environment
RUN python -m venv .venv && \
    .venv/bin/pip install --upgrade pip setuptools wheel

# Set up environment for GPU development
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV PYTHONPATH=/workspaces/dp-flash-attention/src:$PYTHONPATH

# Install basic Python packages that are commonly needed
RUN .venv/bin/pip install \
    torch==2.3.0 \
    triton==2.3.0 \
    pytest \
    black \
    ruff \
    mypy \
    jupyter \
    ipykernel

# Set default shell to bash with venv activation
RUN echo "source /workspaces/dp-flash-attention/.venv/bin/activate" >> ~/.bashrc

# Default command
CMD ["/bin/bash"]