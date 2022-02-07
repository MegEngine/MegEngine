FROM nvidia/cuda:10.1-devel-ubuntu18.04
RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    autotools-dev \
    automake \
    clang-6.0 \
    curl \
    git-lfs \
    libtool \
    libpcre3-dev \
    llvm-6.0-dev \
    openssh-client \
    openssh-server \
    pkg-config \
    python-pip \
    python3-pip \
    python3-dev \
    python-numpy \
    python3-numpy \
    python3-setuptools \
    software-properties-common \
    swig \
    vim \
    wget \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    zlib1g-dev \
    # GitLab Runner need Git 2.18 or higher to create a local Git repository
    && add-apt-repository ppa:git-core/ppa -y && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp ; wget https://cmake.org/files/v3.15/cmake-3.15.2.tar.gz;tar -xzvf cmake-3.15.2.tar.gz;cd cmake-3.15.2;./configure; make -j32; make install

RUN git lfs install

RUN pip3 install --upgrade pip

# TODO: set following envs in github environment.
ENV CUDA_ROOT_DIR=/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/cuda-10.1 \
  TRT_ROOT_DIR=/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/TensorRT-6.0.1.5 \
  TENSORRT_ROOT_DIR=${TRT_ROOT_DIR} \
  CUDNN_ROOT_DIR=/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/cudnn-v7.6.3 \
  PATH=/usr/bin:${CUDA_ROOT_DIR}/bin:${CUDA_ROOT_DIR}/nsight-compute-2019.4.0:$PATH \
  LIBRARY_PATH=${LIBRARY_PATH}:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/cuda-10.1/lib:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/cuda-10.1/lib64:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/cuda-10.1/lib/stubs:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/cuda-10.1/lib64/stubs:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/cudnn-v7.6.3/lib:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/cudnn-v7.6.3/lib64:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/TensorRT-6.0.1.5/lib:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/TensorRT-6.0.1.5/lib64 \
  LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/cudnn-v7.6.3/lib:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/cudnn-v7.6.3/lib64:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/TensorRT-6.0.1.5/lib:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/TensorRT-6.0.1.5/lib64  \
  CPATH=${CPATH}:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/cuda-10.1/include:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/cudnn-v7.6.3/include:/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/TensorRT-6.0.1.5/include \
  CUDA_BIN_PATH=/usr/local/cuda-10.1-cudnn-7.6.3-trt-6.0.1.5-libs/cuda-10.1 
