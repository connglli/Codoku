# rypuz:latest — Sandbox for AI-agent puzzle solving
#
# Build:
#   docker build -t rypuz:latest \
#     --build-arg UID=$(id -u) --build-arg GID=$(id -g) \
#     -f puzzle/bench/Dockerfile .
#
# The build context is the repo root so we can COPY sources for the
# RefractIR toolchain build.

FROM ubuntu:24.04 AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000

# — System packages ——————————————————————————————————————————
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential g++ cmake ninja-build pkg-config \
      m4 python3-mesonpy libmpfr6 libmpfr-dev \
      git curl wget ca-certificates \
      python3 python3-pip python3-venv \
      nodejs npm \
      jq file ripgrep \
    && apt-get autoremove -y && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# — SMT Solvers ——————————————————————————————————————————————

# Z3
RUN git clone --depth 1 --branch z3-4.16.0 \
      https://github.com/Z3Prover/z3.git /tmp/z3-src \
    && cd /tmp/z3-src \
    && python3 scripts/mk_make.py --prefix=/usr/local \
    && cd build && make -j$(nproc) && make install \
    && rm -rf /tmp/z3-src

# CVC5
RUN git clone --depth 1 --branch cvc5-1.3.4 \
      https://github.com/cvc5/cvc5.git /tmp/cvc5-src \
    && cd /tmp/cvc5-src \
    && ./configure.sh --prefix=/usr/local --auto-download \
    && cd build && make -j$(nproc) && make install \
    && rm -rf /tmp/cvc5-src

# Bitwuzla
RUN git clone --depth 1 --branch 0.9.1 \
      https://github.com/bitwuzla/bitwuzla.git /tmp/bitwuzla-src \
    && cd /tmp/bitwuzla-src \
    && python3 ./configure.py --prefix /usr/local \
    && cd build && ninja install \
    && rm -rf /tmp/bitwuzla-src

# — ReifyPuzzle ——————————————————————————————————————————————

# ReifyPuzzle
RUN git clone --depth 1 --branch main \
      https://github.com/connglli/ReifyPuzzle /tmp/reifypuzzle-src \
    && cd /tmp/reifypuzzle-src \
    # Build and install the toolchain
    && mkdir -p /opt/rypuz \
    && INSTALL_PREFIX=/opt/rypuz make -j$(nproc) install \
    && python3 -m venv /opt/rypuz/venv \
    && . /opt/rypuz/venv/bin/activate \
    && pip install --no-cache-dir -r requirements.txt \
    # Save references
    && mkdir -p /opt/rypuz/references/ \
    && cp docs/SPEC_v0.2.2.md   /opt/rypuz/references/SPEC.md \
    && cp docs/float.md         /opt/rypuz/references/float.md \
    && cp docs/intrinsics.md    /opt/rypuz/references/intrinsics.md \
    && cp docs/undefined.md     /opt/rypuz/references/undefined.md \
    && cp -r examples/          /opt/rypuz/references/examples/ \
    && cp -r test/interp/       /opt/rypuz/references/interp/ \
    # Save benchmark files
    && mkdir -p /opt/rypuz/puzzle \
    && cp puzzle/bench/*        /opt/rypuz/puzzle/ \
    && chmod +x /opt/rypuz/puzzle/*.sh \
    && rm -rf /tmp/reifypuzzle-src

# Claude Code & OpenCode
RUN npm install -g @anthropic-ai/claude-code opencode-ai

# — Solver user (matches host UID/GID) ————————————————————
RUN groupadd -g ${GID} solver 2>/dev/null || true \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash solver

# — Workspace —————————————————————————————————————————————
RUN mkdir -p /workspace && chown ${UID}:${GID} /workspace
WORKDIR /workspace
USER solver

ENTRYPOINT ["/opt/rypuz/puzzle/entrypoint.sh"]
