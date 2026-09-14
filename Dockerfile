# Provider-agnostic GPU image for the full-scale CIFAR-10 baseline/mitigation
# runs (see docs/planning/project-setup-plan.md §4 and the "Fixing the base
# model" plan) - same image runs on RunPod, Vast.ai, Lambda, or a local CUDA
# box. Only needed once a CPU-only run's real, timed cost crosses the
# project's own "rent" threshold - verify that on the actual machine
# (see CrossbarConfig.use_gpu's docstring) before trusting a long run here.
#
# CUDA 12.4 runtime (not devel - no compilation happens in this image) on
# Ubuntu 22.04, matching cupy-cuda12x (see pyproject.toml's "gpu" extra).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# git: required to install CrossSim, pinned to a tag (see pyproject.toml),
# not a PyPI package. python3.12 deliberately NOT installed via apt here -
# verified directly this session: Ubuntu 22.04's default repos only carry
# python3.10 (no 3.12 package, deadsnakes PPA needed otherwise) - uv manages
# its own interpreter instead, avoiding that entirely.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/
RUN uv python install 3.12

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY experiments ./experiments

# aihwkit intentionally omitted here - this image is for the CrossSim/GPU
# track only (see CrossbarConfig.use_gpu); add --extra aihwkit back if a
# future run needs both backends on the same box.
RUN uv sync --frozen --extra crosssim --extra gpu

# KMP_DUPLICATE_LIB_OK: see experiments/run.py's own module docstring - the
# macOS-specific reason doesn't apply on Linux, but setting it is harmless
# and keeps this image's env consistent with the local dev entrypoint.
ENV KMP_DUPLICATE_LIB_OK=TRUE

ENTRYPOINT ["uv", "run", "experiments/run.py"]
