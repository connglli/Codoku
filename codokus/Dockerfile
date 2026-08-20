# codoku:latest - Self-contained puzzle generation & evaluation environment
#
# Build:
#   docker build -t codoku:latest \
#     --build-arg UID=$(id -u) --build-arg GID=$(id -g) \
#     -f codokus/Dockerfile .
#
# This image ships what is needed to generate a codoku puzzle and to evaluate
# a solution to one. The initial version supports the Python target only. It
# is meant to be started and then driven from the outside via `docker exec`.
#
# Usage:
#   docker run --rm -it -v $PWD:/workspace codoku:latest
#   docker run --rm -v $PWD:/workspace codoku:latest \
#       codoku create --profile easy --seed 42
#   docker run --rm -v $PWD:/workspace codoku:latest \
#       codoku check puzzle.py solution.py
#   docker exec <container> codoku
#   docker exec <container> codoku check puzzle.py solution.py
#
# Files written under /workspace are owned by the UID/GID passed at build
# time.
#
# Layout inside the image:
#   /opt/rypuz/bin          toolchain binaries (on PATH) + the codoku modules
#   /opt/rypuz/venv         Python venv (python3 on PATH resolves here)
#   /opt/rypuz/references   puzzle.md, float.md, intrinsics.md, undefined.md,
#                           examples/

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000

# - System packages ------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential g++ cmake ninja-build pkg-config \
      m4 python3-mesonpy libmpfr6 libmpfr-dev \
      git curl wget ca-certificates \
      python3 python3-pip python3-venv \
      jq file ripgrep \
    && apt-get autoremove -y && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# - SMT solvers ---------------------------------------------

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

# - ReifyPuzzle toolchain + codoku -------------------------
RUN git clone --depth 1 --branch main \
      https://github.com/connglli/ReifyPuzzle /tmp/reifypuzzle-src \
    && cd /tmp/reifypuzzle-src \
    # Build and install the toolchain
    && mkdir -p /opt/rypuz \
    && INSTALL_PREFIX=/opt/rypuz make -j$(nproc) install \
    # Python venv
    && python3 -m venv /opt/rypuz/venv \
    && /opt/rypuz/venv/bin/pip install --no-cache-dir -r requirements.txt \
    # Codoku modules (from the same checkout)
    && cp codokus/codoku.py            /opt/rypuz/bin/codoku \
    && cp codokus/codoku_creator.py    /opt/rypuz/bin/codoku_creator.py \
    && cp codokus/codoku_checker.py    /opt/rypuz/bin/codoku_checker.py \
    && cp codokus/codoku_common.py     /opt/rypuz/bin/codoku_common.py \
    && cp codokus/codoku_complexity.py /opt/rypuz/bin/codoku_complexity.py \
    && chmod +x /opt/rypuz/bin/codoku \
    && rm -rf /tmp/reifypuzzle-src

# - Environment ----------------------------------------------
# venv on PATH; PYTHONPATH for the codoku modules.
ENV VIRTUAL_ENV="/opt/rypuz/venv" \
    PATH="/opt/rypuz/venv/bin:/opt/rypuz/bin:${PATH}" \
    PYTHONPATH="/opt/rypuz/bin"

# - Solver user -----------------------------------------------
RUN groupadd -g ${GID} solver 2>/dev/null || true \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash solver \
    && chown -R "${UID}:${GID}" /opt/rypuz/venv

# - Workspace ------------------------------------------------
RUN mkdir -p /workspace && chown ${UID}:${GID} /workspace
WORKDIR /workspace
USER solver

CMD ["sleep", "infinity"]
