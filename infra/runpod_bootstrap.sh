#!/usr/bin/env bash
# Bootstrap a freshly-rented RunPod / Lambda Labs CUDA pod into a NUDORMS worker.
#
# Paste this into the pod's "On-start command" or run it once over SSH.
# Expects these env vars to already be set on the pod:
#   REDIS_URL, S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY, S3_REGION
#   NUDORMS_REPO   git clone URL of this repo
#   NUDORMS_REF    git ref to check out (default: main)
#
# Why bootstrap-from-git instead of a pre-built docker image:
#   - faster iteration (no registry push/pull of a 10GB CUDA image)
#   - pod restart re-pulls latest code automatically
#   - one canonical install path; no image/registry drift

set -euo pipefail

REF="${NUDORMS_REF:-main}"
REPO="${NUDORMS_REPO:?NUDORMS_REPO not set}"

apt-get update
apt-get install -y --no-install-recommends \
  git build-essential cmake ninja-build pkg-config ffmpeg \
  libboost-all-dev libeigen3-dev libceres-dev libfreeimage-dev \
  libgoogle-glog-dev libsuitesparse-dev libcgal-dev libgflags-dev

if [ ! -d /workspace/nudorms ]; then
  git clone "$REPO" /workspace/nudorms
fi
cd /workspace/nudorms
git fetch --all
git checkout "$REF"
git pull

# COLMAP + GLOMAP from source — no maintained pkgs that pair correctly.
if ! command -v colmap >/dev/null; then
  git clone --depth=1 https://github.com/colmap/colmap /tmp/colmap
  cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja -DCMAKE_BUILD_TYPE=Release
  cmake --build /tmp/colmap/build --target install -j
fi
if ! command -v glomap >/dev/null; then
  git clone --depth=1 https://github.com/colmap/glomap /tmp/glomap
  cmake -S /tmp/glomap -B /tmp/glomap/build -GNinja -DCMAKE_BUILD_TYPE=Release
  cmake --build /tmp/glomap/build --target install -j
fi

pip install --upgrade pip
pip install -e ./backend[gpu]

export PYTHONPATH=/workspace/nudorms/backend
exec celery -A app.celery_app worker --loglevel=info -Q gpu --concurrency=1
