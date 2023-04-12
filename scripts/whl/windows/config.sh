#!/usr/bin/env bash
set -e

NT=$(echo `uname` | grep "NT")
echo $NT
if [ -z "$NT" ];then
    echo "only run at windows bash env"
    echo "pls consider install bash-like tools, eg MSYS or git-cmd, etc"
    exit -1
fi

# MegEngine dev tools install prefix
MEGENGINE_DEV_TOOLS_PREFIX_DIR=/c/Users/${USER}/megengine_dev_tools

# vs_buildtools download url
VS_BUILD_TOOLS_URL=https://aka.ms/vs/16/release/vs_buildtools.exe

# Visual Studio 2019 install path, please keep postfix "vs"
VS_INSTALL_PATH=${MEGENGINE_DEV_TOOLS_PREFIX_DIR}/vs

# Windows sdk version
WIN_SDK_VER="18362"

# VC Component version
# please install 14.26.28801, others may cause build error or windows xp sp3 runtime error
VC_VER="14.26"
CVARS_VER_NEED="14.26.28801"

# Python3 develop env
PYTHON3_MEGENGINE_DEV_DIR=${MEGENGINE_DEV_TOOLS_PREFIX_DIR}
PYTHON_PACK_MIRROR="https://mirrors.sustech.edu.cn/pypi/simple"
ALL_PYTHON=${ALL_PYTHON}
FULL_PYTHON_VER="3.6.8 3.7.7 3.8.3 3.9.4 3.10.1"

if [[ -z ${ALL_PYTHON} ]]
then
    ALL_PYTHON=${FULL_PYTHON_VER}
fi

# LLVM develop env
LLVM_MEGENGINE_DEV_DIR=${MEGENGINE_DEV_TOOLS_PREFIX_DIR}/llvm/12.0.1
LLVM_INSTALLER_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-12.0.1/LLVM-12.0.1-win64.exe

# default python3 version
DFT_PYTHON_BIN=${PYTHON3_MEGENGINE_DEV_DIR}/pyenv-win/pyenv-win/versions/3.8.3

# 7za download url
ZA_CONSOLE_URL=https://www.7-zip.org/a/7zr.exe
ZA_INSTALLER=https://www.7-zip.org/a/7z2201-x64.exe
ZA_INSTALL_DIR=${MEGENGINE_DEV_TOOLS_PREFIX_DIR}/7za

# swig download url
SWIG_VER=4.0.2
SWIG_INSTALLER_URL=https://nchc.dl.sourceforge.net/project/swig/swigwin/swigwin-${SWIG_VER}/swigwin-${SWIG_VER}.zip
SWIG_INSTALL_DIR=${MEGENGINE_DEV_TOOLS_PREFIX_DIR}/swigwin-${SWIG_VER}

# CUDA_CUDNN_TRT_LOCATION
CUDA_CUDNN_TRT_LOC_PREFIX=${MEGENGINE_DEV_TOOLS_PREFIX_DIR}/cuda_cudnn_trt
# cuda-10.1-cudnn-v7.6.5-TensorRT-6.0.1.5
CUDA_ROOT_DIR_101=${CUDA_CUDNN_TRT_LOC_PREFIX}/101/CUDA/v10.1
CUDNN_ROOT_DIR_101=${CUDA_CUDNN_TRT_LOC_PREFIX}/101/cudnn-10.1-windows10-x64-v7.6.5.32/cuda
TRT_ROOT_DIR_101=${CUDA_CUDNN_TRT_LOC_PREFIX}/101/TensorRT-6.0.1.5-windows
# cuda-11.8-cudnn-v8.6.0-TensorRT-8.5.3.1
CUDA_ROOT_DIR_118=${CUDA_CUDNN_TRT_LOC_PREFIX}/118/CUDA/v11.8
CUDNN_ROOT_DIR_118=${CUDA_CUDNN_TRT_LOC_PREFIX}/118/cudnn-windows-x86_64-8.6.0.163_cuda11-archive
TRT_ROOT_DIR_118=${CUDA_CUDNN_TRT_LOC_PREFIX}/118/TensorRT-8.5.3.1
ZLIBWAPI_URL=http://www.winimage.com/zLibDll/zlib123dllx64.zip
# config default version, when user do not config CUDA_ROOT_DIR/CUDNN_ROOT_DIR/TRT_ROOT_DIR
# now we just config to cuda-11.8-cudnn-v8.6.0-TensorRT-8.5.3.1
CUDA_DFT_ROOT=${CUDA_ROOT_DIR_118}
CUDNN_DFT_ROOT=${CUDNN_ROOT_DIR_118}
TRT_DFT_ROOT=${TRT_ROOT_DIR_118}

