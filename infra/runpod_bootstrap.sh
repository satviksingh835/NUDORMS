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
  git build-essential ninja-build pkg-config ffmpeg \
  libboost-all-dev libeigen3-dev libceres-dev libfreeimage-dev \
  libgoogle-glog-dev libsuitesparse-dev libcgal-dev libgflags-dev \
  libopenimageio-dev libmetis-dev libsqlite3-dev openimageio-tools \
  libopenexr-dev libtiff-dev libpng-dev libjpeg-dev \
  libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev \
  libglew-dev libopencv-dev

# Newer cmake than Ubuntu 22.04 ships (faiss/colmap deps need >=3.24).
pip install --upgrade "cmake>=3.28"
hash -r

if [ ! -d /workspace/nudorms ]; then
  git clone "$REPO" /workspace/nudorms
fi
cd /workspace/nudorms
git fetch --all
git checkout "$REF"
git pull

# COLMAP + GLOMAP from source — no maintained pkgs that pair correctly.
# GUI_ENABLED=OFF saves ~10 min of Qt build; we never render on the pod.
if ! command -v colmap >/dev/null; then
  git clone --depth=1 https://github.com/colmap/colmap /tmp/colmap
  cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DGUI_ENABLED=OFF -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-80;86;89}"
  cmake --build /tmp/colmap/build --target install -j"$(nproc)"
fi
if ! command -v glomap >/dev/null; then
  git clone --depth=1 https://github.com/colmap/glomap /tmp/glomap
  cmake -S /tmp/glomap -B /tmp/glomap/build -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-80;86;89}"
  cmake --build /tmp/glomap/build --target install -j"$(nproc)"
fi

pip install --upgrade pip
pip install -e ./backend[gpu]

# MASt3R-SfM: learned pose estimation that replaces COLMAP on casual captures.
# Clone once; skip if already present (pod restart or baked template).
if [ ! -d /workspace/mast3r ]; then
  git clone --recursive https://github.com/naver/mast3r /workspace/mast3r
  pip install -r /workspace/mast3r/dust3r/requirements.txt
  pip install -r /workspace/mast3r/dust3r/requirements_optional.txt
  pip install -r /workspace/mast3r/requirements.txt
  # kapture + kapture-localization: required by mast3r/colmap/mapping.py
  pip install kapture kapture-localization
  # install mast3r package itself so `from mast3r.model import ...` resolves
  pip install -e /workspace/mast3r
fi

# Model weights (~2 GB, ViT-Large 512). Download once, skip if cached.
# Store outside the repo so a git checkout doesn't wipe them.
mkdir -p /workspace/mast3r_models
MAST3R_WEIGHTS=/workspace/mast3r_models/MASt3R.pth
if [ ! -f "$MAST3R_WEIGHTS" ]; then
  wget -q --show-progress \
    "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    -O "$MAST3R_WEIGHTS"
fi

export PYTHONPATH=/workspace/nudorms/backend
exec celery -A app.celery_app worker --loglevel=info -Q gpu --concurrency=1
