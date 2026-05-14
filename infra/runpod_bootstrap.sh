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
CUDA_ARCH="${CUDA_ARCH:-80;86;89}"

# Sanity check the GPU is actually visible from inside the container.
if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi failed — pod has no GPU access. Aborting."
  exit 1
fi

# CUDA toolkit lives at /usr/local/cuda on RunPod images but isn't always on PATH.
# COLMAP's cmake will fail with "CMAKE_CUDA_COMPILER could not be found" without this.
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDACXX="/usr/local/cuda/bin/nvcc"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

apt-get update
apt-get install -y --no-install-recommends \
  git build-essential ninja-build pkg-config ffmpeg curl ca-certificates \
  libboost-all-dev libeigen3-dev libceres-dev libfreeimage-dev \
  libgoogle-glog-dev libsuitesparse-dev libcgal-dev libgflags-dev \
  libopenimageio-dev libmetis-dev libsqlite3-dev openimageio-tools \
  libopenexr-dev libtiff-dev libpng-dev libjpeg-dev \
  libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev \
  libglew-dev libopencv-dev libssl-dev libcurl4-openssl-dev \
  hugin-tools enblend

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
    -DGUI_ENABLED=OFF -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
  cmake --build /tmp/colmap/build --target install -j"$(nproc)"
fi
if ! command -v glomap >/dev/null; then
  git clone --depth=1 https://github.com/colmap/glomap /tmp/glomap
  # GLOMAP's CMakeLists expects upstream COLMAP/PoseLib targets that aren't
  # exposed by a plain `make install`, so we let it FetchContent both deps.
  # FETCHCONTENT_SOURCE_DIR_<NAME> short-circuits the FetchContent download
  # if the dirs are pre-staged at /workspace/_glomap_deps/* (saves 1+ hour
  # of slow git fetches on this pod's network).
  GLOMAP_DEPS_FLAGS=""
  if [ -d /workspace/_glomap_deps/colmap ]; then
    GLOMAP_DEPS_FLAGS+=" -DFETCHCONTENT_SOURCE_DIR_COLMAP=/workspace/_glomap_deps/colmap"
  fi
  if [ -d /workspace/_glomap_deps/poselib ]; then
    GLOMAP_DEPS_FLAGS+=" -DFETCHCONTENT_SOURCE_DIR_POSELIB=/workspace/_glomap_deps/poselib"
  fi
  if [ -d /workspace/_glomap_deps/faiss ]; then
    GLOMAP_DEPS_FLAGS+=" -DFETCHCONTENT_SOURCE_DIR_FAISS=/workspace/_glomap_deps/faiss"
  fi
  cmake -S /tmp/glomap -B /tmp/glomap/build -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    ${GLOMAP_DEPS_FLAGS}
  cmake --build /tmp/glomap/build --target install -j"$(nproc)"
fi

pip install --upgrade pip
pip install -e ./backend[gpu]

# MASt3R-SfM: learned pose estimation that replaces COLMAP on casual captures.
# Clone once; skip if already present (pod restart or baked template).
# MASt3R/dust3r have no setup.py — they're imported via PYTHONPATH (set below).
if [ ! -d /workspace/mast3r ]; then
  git clone --recursive https://github.com/naver/mast3r /workspace/mast3r
fi
# Always run requirement installs in case the previous run failed midway
# (pip is fast when everything is already satisfied).
pip install -r /workspace/mast3r/dust3r/requirements.txt
pip install -r /workspace/mast3r/dust3r/requirements_optional.txt 2>/dev/null || \
  echo "WARN: dust3r requirements_optional.txt failed — non-fatal"
pip install -r /workspace/mast3r/requirements.txt
# kapture + kapture-localization: required by mast3r/colmap/mapping.py
pip install kapture kapture-localization

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
# Clone is gated on dir existence; pip install runs every time so a manually
# extracted dir (tarball restore, baked template) still gets editable-installed.
if [ ! -d /workspace/vggt ]; then
  git clone --depth=1 https://github.com/facebookresearch/vggt /workspace/vggt
fi
[ -d /workspace/vggt ] && pip install -e /workspace/vggt || \
  echo "WARN: VGGT install failed — fallback to MASt3R"

# Spectacular AI SDK: VIO + VISLAM poses with metric scale + rolling-shutter
# compensation. sai-cli is the CLI used by spectacular_ai.py pose wrapper.
# Free for non-commercial use.
pip install --quiet "spectacularai[full]" || pip install --quiet spectacularai || \
  echo "WARN: spectacularai install failed — fallback to VGGT/MASt3R for pose"

# PYTHONPATH wires the worker against the MASt3R + dust3r source trees
# (no setup.py upstream) and against our backend package.
export PYTHONPATH=/workspace/nudorms/backend:/workspace/mast3r:/workspace/mast3r/dust3r
exec celery -A app.celery_app worker --loglevel=info -Q gpu --concurrency=1
