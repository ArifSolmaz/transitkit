FROM python:3.11
WORKDIR /app
RUN pip install jupyterlab
COPY . .
RUN pip install -e ".[dev,full]"
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
