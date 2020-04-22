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
    zlib1g-dev \
    # GitLab Runner need Git 2.18 or higher to create a local Git repository
    && add-apt-repository ppa:git-core/ppa -y && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp ; wget https://cmake.org/files/v3.14/cmake-3.14.4.tar.gz;tar -xzvf cmake-3.14.4.tar.gz;cd cmake-3.14.4;./configure; make -j32; make install

RUN git lfs install

ENV PATH=${PATH}:/usr/local/cuda/bin \
  LIBRARY_PATH=${LIBRARY_PATH}:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib/stubs:/usr/local/cuda/lib64/stubs:/usr/local/cuda-10.1-libs/cudnn-v7.6.0/lib:/usr/local/cuda-10.1-libs/cudnn-v7.6.0/lib64:/usr/local/cuda-10.1-libs/TensorRT-5.1.5.0/lib:/usr/local/cuda-10.1-libs/TensorRT-5.1.5.0/lib64 \
  LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-10.1-libs/cudnn-v7.6.0/lib:/usr/local/cuda-10.1-libs/cudnn-v7.6.0/lib64:/usr/local/cuda-10.1-libs/TensorRT-5.1.5.0/lib:/usr/local/cuda-10.1-libs/TensorRT-5.1.5.0/lib64:/tmp/build/cuda/dnn/cuda-stub/libcuda.so \
  CPATH=${CPATH}:/usr/local/cuda/include:/usr/local/cuda-10.1-libs/cudnn-v7.6.0/include:/usr/local/cuda-10.1-libs/TensorRT-5.1.5.0/include \
  CUDA_BIN_PATH=/usr/local/cuda
