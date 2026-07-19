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
    && make -j$(nproc) symiri symirc symirsolve rypuzchk \
    && mkdir -p /opt/rypuz/tools \
    && cp symiri symirc symirsolve rypuzchk /opt/rypuz/tools/ \
    && rm -rf /tmp/reifypuzzle-src

# Reference material
# These are the docs/examples the puzzle banner points solvers to.
COPY docs/SPEC_v0.2.2.md   /opt/rypuz/references/SPEC.md
COPY docs/float.md         /opt/rypuz/references/float.md
COPY docs/intrinsics.md    /opt/rypuz/references/intrinsics.md
COPY docs/undefined.md     /opt/rypuz/references/undefined.md
COPY examples/             /opt/rypuz/references/examples/
COPY test/interp/          /opt/rypuz/references/interp/

# Others
COPY puzzle/bench/system.md /opt/rypuz/system.md
COPY puzzle/bench/common.sh /opt/rypuz/common.sh

# Claude Code & OpenCode
RUN npm install -g @anthropic-ai/claude-code opencode-ai

# — Solver user (matches host UID/GID) ————————————————————
RUN groupadd -g ${GID} solver 2>/dev/null || true \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash solver

# — Entrypoint ————————————————————————————————————————————
COPY puzzle/bench/entrypoint.sh /opt/rypuz/entrypoint.sh
RUN chmod +x /opt/rypuz/entrypoint.sh

# — Workspace —————————————————————————————————————————————
RUN mkdir -p /workspace && chown solver:solver /workspace
WORKDIR /workspace
USER solver

ENTRYPOINT ["/opt/rypuz/entrypoint.sh"]
