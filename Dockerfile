# Dashboard + API in one container. Data, models and reports are mounted read-only at run time
# (see compose.yaml); nothing here downloads or generates weather data.

FROM node:24-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 RAINPP_STATIC_DIR=/app/web
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install -e ".[api,geo]"
COPY config/ config/
COPY --from=web /web/dist /app/web
RUN useradd --system --no-create-home rainpp
USER rainpp
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["uvicorn", "rainpp.api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
