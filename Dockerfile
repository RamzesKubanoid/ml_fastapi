# ── Stage: runtime image ──────────────────────────────────────────────────────
# python:3.12-slim is the smallest official image that includes pip.
# Using a pinned minor version makes builds reproducible.
FROM python:3.12-slim

# Keeps Python from buffering stdout/stderr so log lines appear in
# `docker logs` immediately without any manual flush.
ENV PYTHONUNBUFFERED=1 \
    # Prevents Python from writing .pyc files into the image layer
    PYTHONDONTWRITEBYTECODE=1 \
    # Tell the app where to find the project root inside the container
    PYTHONPATH=/app

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
# Copy only the requirements file first so Docker can cache this layer.
# The layer is invalidated only when requirements.txt changes, not on every
# code edit — this makes rebuilds significantly faster during development.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY src/      ./src/
COPY data/     ./data/

# models/ is intentionally NOT copied — it is mounted as a volume at runtime
# so trained models survive container restarts and rebuilds.
# If no volume is mounted, the service starts untrained (model_store handles
# this gracefully via the FileNotFoundError catch on startup).

# ── Runtime ───────────────────────────────────────────────────────────────────
# Expose the port uvicorn listens on (informational — must match --port below).
EXPOSE 8000

# --host 0.0.0.0 makes uvicorn accept connections from outside the container.
# Without this it only listens on localhost and the port is unreachable.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]