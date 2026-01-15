FROM python:3.11-slim
LABEL maintainer="Arif Solmaz <arif.solmaz@gmail.com>"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .
CMD ["python", "-c", "import transitkit; print(f'TransitKit v{transitkit.__version__}')"]
