FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . \
    && useradd --system --uid 10001 --create-home appuser \
    && mkdir -p /data && chown 10001:10001 /data

# Anonim denetim günlüğü (yalnız sayılar); compose.yaml /data'yı kalıcı volume olarak bağlar.
ENV TURKCE_DLP_AUDIT_DB=/data/audit.sqlite3
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "turkce_llm_dlp.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
