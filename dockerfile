# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
COPY requirements-optional.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    pandas \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-optional.txt

# Copy the rest of the application
COPY . .

# Install transitkit
RUN pip install --no-cache-dir .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Create data directory
RUN mkdir -p /home/appuser/data
VOLUME /home/appuser/data

# Expose port for web interface
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import transitkit; print('OK')" || exit 1

# Default command (run Streamlit web interface)
CMD ["transitkit-web", "--server.port=8501", "--server.address=0.0.0.0"]