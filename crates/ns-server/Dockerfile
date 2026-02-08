# ============================================================
# NextStat Server — multi-stage Docker build
# ============================================================
# Build:
#   docker build -t nextstat-server .                   # CPU only
#   docker build --build-arg FEATURES=cuda -t nextstat-server:cuda .
#
# Run:
#   docker run -p 3742:3742 nextstat-server
#   docker run -p 3742:3742 --gpus all nextstat-server:cuda --gpu cuda
# ============================================================

# ---------- stage 1: builder ----------
FROM rust:1.93-bookworm AS builder

WORKDIR /src

# Copy workspace manifests first (better layer caching)
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ns-core/Cargo.toml        crates/ns-core/Cargo.toml
COPY crates/ns-ad/Cargo.toml          crates/ns-ad/Cargo.toml
COPY crates/ns-compute/Cargo.toml     crates/ns-compute/Cargo.toml
COPY crates/ns-translate/Cargo.toml   crates/ns-translate/Cargo.toml
COPY crates/ns-inference/Cargo.toml   crates/ns-inference/Cargo.toml
COPY crates/ns-server/Cargo.toml      crates/ns-server/Cargo.toml
COPY crates/ns-cli/Cargo.toml         crates/ns-cli/Cargo.toml
COPY crates/ns-report/Cargo.toml      crates/ns-report/Cargo.toml
COPY crates/ns-ntuple/Cargo.toml      crates/ns-ntuple/Cargo.toml
COPY crates/ns-rootio/Cargo.toml      crates/ns-rootio/Cargo.toml
COPY crates/ns-plot/Cargo.toml        crates/ns-plot/Cargo.toml
COPY bindings/ns-py/Cargo.toml        bindings/ns-py/Cargo.toml
COPY bindings/ns-wasm/Cargo.toml      bindings/ns-wasm/Cargo.toml

# Create stub lib.rs / main.rs for dependency caching
RUN for d in crates/ns-core crates/ns-ad crates/ns-compute crates/ns-translate \
            crates/ns-inference crates/ns-report crates/ns-ntuple crates/ns-rootio \
            crates/ns-plot bindings/ns-py bindings/ns-wasm; do \
        mkdir -p "$d/src" && echo "" > "$d/src/lib.rs"; \
    done && \
    mkdir -p crates/ns-cli/src && echo "fn main() {}" > crates/ns-cli/src/main.rs && \
    mkdir -p crates/ns-server/src && echo "fn main() {}" > crates/ns-server/src/main.rs

# Build-arg for optional features (cuda, metal)
ARG FEATURES=""

# Pre-build dependencies (cached unless Cargo.toml changes)
RUN if [ -n "$FEATURES" ]; then \
        cargo build --release -p ns-server --features "$FEATURES" 2>/dev/null || true; \
    else \
        cargo build --release -p ns-server 2>/dev/null || true; \
    fi

# Copy real source
COPY crates/ crates/
COPY bindings/ bindings/

# Touch sources to invalidate stubs
RUN find crates/ bindings/ -name "*.rs" -exec touch {} +

# Final build
RUN if [ -n "$FEATURES" ]; then \
        cargo build --release -p ns-server --features "$FEATURES"; \
    else \
        cargo build --release -p ns-server; \
    fi

# ---------- stage 2: runtime ----------
FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libssl3 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/release/nextstat-server /usr/local/bin/nextstat-server

# Non-root user
RUN useradd --create-home --shell /bin/bash nextstat
USER nextstat

EXPOSE 3742

ENTRYPOINT ["nextstat-server"]
CMD ["--host", "0.0.0.0", "--port", "3742"]
