# tradebot production image — dashboard + CLI entrypoints.
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# DuckDB wheels are prebuilt; gcc only needed if a dep falls back to sdist.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY config config/
COPY core core/
COPY data data/
COPY indicators indicators/
COPY strategies strategies/
COPY analyst analyst/
COPY risk risk/
COPY execution execution/
COPY backtest backtest/
COPY orchestrator orchestrator/
COPY journal journal/
COPY screener screener/
COPY tuner tuner/
COPY dashboard dashboard/

RUN pip install .

RUN useradd --create-home --uid 1000 tradebot \
    && mkdir -p /app/runtime /app/logs /app/config \
    && chown -R tradebot:tradebot /app

USER tradebot

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -sf http://127.0.0.1:8765/ >/dev/null || exit 1

# TRADEOFF: uvicorn directly avoids the CLI localhost bind guard; Caddy
# handles auth on the public edge.
CMD ["uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "8765"]
