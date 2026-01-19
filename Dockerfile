FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

LABEL name="PoissonRecon_GPU" maintainer="Causality-C"

# update package lists and install git, wget, vim, libegl1-mesa-dev, and libglib2.0-0
RUN apt-get update && apt-get install -y build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip git tree cmake

WORKDIR /workspace
COPY  . .
RUN mkdir build && cd build && cmake ..

