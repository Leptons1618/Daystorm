# Two stages: wheels are built once, the runtime image carries no toolchain.
FROM python:3.12-slim AS build
WORKDIR /build
RUN pip install --no-cache-dir uv
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu \
 && uv pip install --system --no-cache ".[serve]"

FROM python:3.12-slim AS runtime
# Never run the service as root.
RUN useradd --create-home --uid 10001 daystorm
WORKDIR /app
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY src ./src
COPY ckpt ./ckpt
ENV PYTHONPATH=/app/src \
    DAYSTORM_CKPT=/app/ckpt/stage_a \
    PYTHONUNBUFFERED=1
USER daystorm
EXPOSE 8000
# Liveness only. Readiness is /ready, which the orchestrator should poll before
# routing traffic - a container loading weights is alive but not ready.
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "daystorm.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
