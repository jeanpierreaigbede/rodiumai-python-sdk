FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY rodiumai/ rodiumai/

RUN pip install --no-cache-dir build && \
    python -m build --wheel


FROM python:3.12-slim-bookworm AS runner

RUN groupadd -r rodiumai && \
    useradd -r -g rodiumai -d /app -s /sbin/nologin rodiumai

WORKDIR /app

COPY --from=builder /build/dist/*.whl /tmp/

RUN pip install --no-cache-dir /tmp/*.whl && \
    rm /tmp/*.whl && \
    rm -rf /root/.cache

USER rodiumai

ENTRYPOINT ["python", "-c", "import rodiumai; print(f'RodiumAI SDK v{rodiumai.VERSION}')"]
