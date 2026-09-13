# Multi-stage: the builder carries compilers and pip's build machinery, the runtime does not.
# TensorFlow dominates the image either way, so the win here is modest but real - roughly the
# size of build-essential plus pip's caches.

# ---------------------------------------------------------------------------- builder
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# tensorflow-cpu, not tensorflow. Same 2.21.0 code, minus ~1.5GB of nvidia CUDA wheels that
# would never load: the package hardcodes CUDA_VISIBLE_DEVICES=-1, so the GPU build is dead
# weight in a container.
#
# Numerics were checked rather than assumed. The container is deterministic against itself
# (12 identical responses to the same request), but its predictions differ from the Windows
# host's in the last few float32 digits - ~1.5e-9 on a VaR of -0.0227, a relative difference
# near float32 epsilon. That is cross-platform kernel variation (different SIMD paths and
# reduction orders), not a calibration change: it is seven orders of magnitude below the
# return scale a breach test compares against. Reproducing the *research* numbers exactly
# still means the pinned host environment, which is why the pins in pyproject.toml stay as
# they are and this substitution lives only in the serving image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install \
        "numpy==2.5.2" \
        "pandas==3.0.5" \
        "scikit-learn==1.9.0" \
        "scipy==1.18.1" \
        "tensorflow-cpu==2.21.0" \
        "mlflow==3.16.0" \
        "fastapi==0.121.2" \
        "uvicorn[standard]==0.39.0"

COPY pyproject.toml README.md* ./
COPY src/ ./src/
# --no-deps: every dependency is already pinned and installed above, and resolving them again
# here would pull the GPU tensorflow back in via the pyproject pin.
RUN pip install --no-deps .

# ---------------------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=sqlite:////app/mlruns/mlflow.db \
    MTL_VAR_CSV=/app/SPX500-18Y_scaled_y.csv

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# The model itself. mlruns/ is the local registry (SQLite + artifacts, ~4MB), so the image is
# self-contained and `docker run` serves immediately. Mount over /app/mlruns to serve a
# different registry without rebuilding.
COPY mlruns/ /app/mlruns/

# MLflow resolves artifact locations to absolute paths when an experiment is created, so the
# copied store still points at the host's directories and every lookup fails with
# "No such artifact: ''". Rewrite them to the container's root.
RUN python -m mtl_var.cli rebase-registry --db /app/mlruns/mlflow.db --to /app/mlruns

# Serving refits the fold's scalers from this csv at startup and checks its SHA-256 against
# the hash recorded on the registered model version, so it is required, not optional.
COPY SPX500-18Y_scaled_y.csv /app/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "mtl_var.serve:app", "--host", "0.0.0.0", "--port", "8000"]
