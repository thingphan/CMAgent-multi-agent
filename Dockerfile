FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY webapp/ ./webapp/
COPY data/mapping/ ./data/mapping/
RUN mkdir -p /app/data/processed /inputs /outputs /resources \
    && chown -R 10001:10001 /app/data/processed /outputs

USER 10001:10001
ENTRYPOINT ["python", "src/pipelines/cellmarker_v2_6_strict/run_pipeline.py"]
CMD ["--help"]
