FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gcc \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY pyproject.toml .
RUN uv pip install .

COPY . .

RUN chmod +x run.sh

CMD ["./run.sh"]
