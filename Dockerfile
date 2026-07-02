# syntax=docker/dockerfile:1
#
# Tau-sorting — container image for the Q_rad explorer webapp (serves on port 8771).
#
# The large runtime inputs (~2.8 GB: the ODF, continuum, and data/ reference tables)
# are gitignored and are NOT baked into the image (see .dockerignore) — bind-mount them
# at run time. The tracked default atmosphere models/G2_1D.dat IS shipped in the image.
#
# Build:
#   docker build -t tau-sorting .
#
# Run (mount the ODF, continuum, and data/ reference tables read-only):
#   docker run --rm -p 8771:8771 \
#       -v "$PWD/ODF_format.npy:/app/ODF_format.npy:ro" \
#       -v "$PWD/continuumabs.dat:/app/continuumabs.dat:ro" \
#       -v "$PWD/data:/app/data:ro" \
#       ghcr.io/lively-stars/tau-sorting:latest
#   # then open http://localhost:8771
#   # (use ODF_nc_format.nc in place of ODF_format.npy if you only have the .nc input)

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# uv: copy (not hardlink) into the venv and precompile bytecode for faster startup.
# MPLCONFIGDIR keeps matplotlib's cache in a writable, throwaway location.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /app

# Install dependencies first (cached layer) from the lockfile only; the app source is
# copied afterwards so code edits don't bust the dependency cache. --no-install-project
# skips building the root project (it's run directly, not installed as a package).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Application source + the tracked default atmosphere models/G2_1D.dat.
# (The big gitignored data files are excluded by .dockerignore.)
COPY . .

# Run the venv's interpreter directly — no uv resolution at container start.
ENV PATH="/app/.venv/bin:$PATH"

# Bind on all interfaces inside the container so the published port is reachable.
ENV HOST=0.0.0.0 \
    PORT=8771

EXPOSE 8771

CMD ["python", "webapp/server.py"]
