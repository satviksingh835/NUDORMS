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

# VGGT: CVPR 2025 Best Paper feed-forward pose estimator (seconds, not minutes).
# Model weights auto-downloaded from HuggingFace on first use via transformers.
if [ ! -d /workspace/vggt ]; then
  git clone --depth=1 https://github.com/facebookresearch/vggt /workspace/vggt
  pip install -e /workspace/vggt
fi

# Scaffold-GS: CVPR 2024 Highlight, anchor-based 3DGS for textureless indoor.
# Best PSNR on casual iPhone indoor capture; primary trainer in NUDORMS.
if [ ! -d /workspace/Scaffold-GS ]; then
  git clone --depth=1 https://github.com/city-super/Scaffold-GS /workspace/Scaffold-GS
  pip install -r /workspace/Scaffold-GS/requirements.txt
fi

# Depth Anything V2 + normals: monocular depth priors for DN-Splatter supervision.
# transformers is already installed via backend[gpu]; model auto-downloaded on first use.
# Also install LPIPS for perceptual loss (fallback if Apple WD-R not available).
pip install --quiet lpips

# Spectacular AI SDK: VIO + VISLAM poses with metric scale + rolling-shutter
# compensation. sai-cli is the CLI used by spectacular_ai.py pose wrapper.
# Free for non-commercial use.
pip install --quiet "spectacularai[full]" || pip install --quiet spectacularai

# Difix3D+: CVPR 2025 Oral diffusion artifact fixer for 3DGS outputs.
if [ ! -d /workspace/Difix3D ]; then
  git clone --depth=1 https://github.com/nv-tlabs/Difix3D /workspace/Difix3D
  pip install -r /workspace/Difix3D/requirements.txt
fi

# PGSR: Planar-based GS, best Chamfer on textureless indoor, for mesh extraction.
if [ ! -d /workspace/pgsr ]; then
  git clone --depth=1 https://github.com/hmanhng/pgsr /workspace/pgsr
  pip install -r /workspace/pgsr/requirements.txt 2>/dev/null || true
fi

# 3DGUT: NVIDIA CVPR 2025, ray-traced reflections on monitors/windows.
if [ ! -d /workspace/3DGUT ]; then
  git clone --depth=1 https://github.com/nv-tlabs/3DGUT /workspace/3DGUT
  pip install -r /workspace/3DGUT/requirements.txt 2>/dev/null || true
fi

# scikit-learn: needed for DBSCAN floater cleanup.
pip install --quiet scikit-learn

# Node.js + splat-transform for SOG compression (playcanvas/splat-transform).
if ! command -v node >/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y --no-install-recommends nodejs
fi

export PYTHONPATH=/workspace/nudorms/backend
exec celery -A app.celery_app worker --loglevel=info -Q gpu --concurrency=1
