# GPU worker image. Build/run on a CUDA Linux box (or rented pod).
# Mac dev machines should NOT try to build this — run only the API + viewer
# locally and point Celery at a remote worker over the same Redis.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      git build-essential cmake ninja-build pkg-config \
      ffmpeg \
      libboost-all-dev libeigen3-dev libceres-dev libfreeimage-dev \
      libgoogle-glog-dev libsuitesparse-dev libcgal-dev libgflags-dev \
  && rm -rf /var/lib/apt/lists/*

# COLMAP + GLOMAP from source (no maintained ubuntu pkgs that pair correctly).
# TODO: pin commits.
RUN git clone --depth=1 https://github.com/colmap/colmap /tmp/colmap && \
    cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/colmap/build --target install && rm -rf /tmp/colmap

RUN git clone --depth=1 https://github.com/colmap/glomap /tmp/glomap && \
    cmake -S /tmp/glomap -B /tmp/glomap/build -GNinja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/glomap/build --target install && rm -rf /tmp/glomap

WORKDIR /app
COPY backend/pyproject.toml /app/backend/pyproject.toml
RUN pip install --no-cache-dir -e /app/backend[gpu]

COPY backend /app/backend
ENV PYTHONPATH=/app/backend

CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info", "-Q", "gpu", "--concurrency=1"]
