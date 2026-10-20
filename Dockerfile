# syntax=docker/dockerfile:1

# --------------------------------------------------------------------------- #
# r/indonesia analytics dashboard
#
# The image carries code and dependencies only. Data is mounted at runtime
# (see docker-compose.yml) because the parquet cache is ~150 MB and the raw
# CSVs are ~400 MB: baking them in would turn every data refresh into an image
# rebuild, and every pull into a half-gigabyte download.
# --------------------------------------------------------------------------- #
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# Dependencies go in before the code so that editing app.py reuses the slow
# wheel layer instead of reinstalling everything.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Runtime payload: the dashboard, its theme, and the cache builders that the
# in-app "Build cache now" button shells out to.
COPY app.py ./
COPY .streamlit/ ./.streamlit/
COPY src/ ./src/

# Run unprivileged. /app/data is created up front so the bind mount has a
# target, and so a run *without* one fails with the app's own "cache missing"
# message rather than a permission error.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
