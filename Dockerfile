# ==============================================================================
# Multi-Stage Dockerfile for Zero-Decimation 3D Model Optimization Pipeline
# Base: Debian Bookworm (Python 3.11, Node.js 20 LTS, basisu CLI)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build basisu CLI from source
# ------------------------------------------------------------------------------
FROM debian:bookworm-slim AS basisu-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/BinomialLLC/basis_universal.git /tmp/basis_universal \
    && cd /tmp/basis_universal \
    && cmake -DCMAKE_BUILD_TYPE=Release -B build \
    && cmake --build build --parallel $(nproc) \
    && if [ -f bin/basisu ]; then cp bin/basisu /usr/local/bin/basisu; else cp build/bin/basisu /usr/local/bin/basisu; fi \
    && chmod +x /usr/local/bin/basisu

# ------------------------------------------------------------------------------
# Stage 2: Build Python dependencies in virtualenv
# ------------------------------------------------------------------------------
FROM debian:bookworm-slim AS python-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 3: Minimal production runtime image
# ------------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

LABEL maintainer="Braitoli <dev@braitoli.com>"
LABEL description="Zero-Decimation 3D Model (.glb) Optimization Pipeline (Meshopt + KTX2 UASTC + UV Atlas)"

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"
ENV NODE_ENV=production

# Install Python 3 runtime & base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    ca-certificates \
    curl \
    gnupg \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS via official NodeSource repository
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy basisu CLI binary from builder stage
COPY --from=basisu-builder /usr/local/bin/basisu /usr/local/bin/basisu

# Copy Python virtual environment from builder stage
COPY --from=python-builder /opt/venv /opt/venv

WORKDIR /app

# Install Node.js dependencies with layer caching
COPY package.json package-lock.json* ./
RUN npm install --omit=dev --no-audit --no-fund

# Copy application source code
COPY . .

# Symlink .venv for CLI path resolution and ensure executables
RUN ln -sf /opt/venv /app/.venv \
    && chmod +x /app/bin/optimize-3d /app/optimizer/node/optimize_meshopt.mjs

# Dedicated workspace directory for user volume mounts
RUN mkdir -p /data

# Default entrypoint for direct usage:
# docker run --rm -v $(pwd):/data braitoli/poc-optimize-3d /data/input.glb /data/output.glb
ENTRYPOINT ["/app/bin/optimize-3d"]
CMD ["--help"]
