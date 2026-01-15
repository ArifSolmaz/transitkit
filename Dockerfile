# TransitKit Docker Image
FROM python:3.11-slim

LABEL maintainer="ArifSolmaz"
LABEL description="TransitKit - Professional Exoplanet Transit Analysis"
LABEL version="2.0.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e "."

RUN useradd -m transitkit
USER transitkit

CMD ["python", "-c", "import transitkit; print(f'TransitKit v{transitkit.__version__} ready')"]