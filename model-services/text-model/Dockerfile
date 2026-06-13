FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi[standard]>=0.115 \
    sentence-transformers>=3.0 \
    torch

COPY app.py .

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
