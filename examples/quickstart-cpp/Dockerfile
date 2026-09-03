FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    autoconf \
    libtool \
    pkg-config \
    python3 \
    python-is-python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Flower and Python dependencies for the server
RUN pip3 install --no-cache-dir \
    flwr \
    numpy \
    hatchling

WORKDIR /app

# Copy proto definitions and C++ SDK source
COPY framework/proto /app/framework/proto
COPY framework/cc /app/framework/cc

# Copy the quickstart-cpp example
COPY examples/quickstart-cpp /app/examples/quickstart-cpp

# Build the C++ client (this fetches gRPC ~v1.43.2 and generates protos)
WORKDIR /app/examples/quickstart-cpp
RUN cmake -S . -B build -DUSE_LOCAL_FLWR=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DFETCHCONTENT_QUIET=OFF \
    && cmake --build build -j$(nproc)

WORKDIR /app/examples/quickstart-cpp
