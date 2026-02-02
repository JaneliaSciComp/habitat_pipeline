# Habitat Pipeline - Multi-Animal Electrophysiology Analysis Platform
FROM python:3.10-slim

LABEL maintainer="Habitat Pipeline Contributors"
LABEL description="Multi-Animal Electrophysiology and Behavior Analysis Pipeline"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package files
COPY setup.py .
COPY pyproject.toml .
COPY src/ src/
COPY config/ config/
COPY examples/ examples/

# Install package
RUN pip install -e .

# Create directories for data and output
RUN mkdir -p /data /output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HABITAT_PIPELINE_CONFIG=/app/config/default_config.yaml

# Default command
CMD ["python", "-c", "import habitat_pipeline; print(f'Habitat Pipeline v{habitat_pipeline.__version__} ready!')"]
