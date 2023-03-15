#!/bin/bash -e
if [[ -z ${SDK_NAME} ]]; then
    export SDK_NAME="cu112"
fi
SRC_DIR=$(READLINK -f "`dirname $0`/../../../")
echo "Install LLVM"
${SRC_DIR}/python_dev/3.8.3/python3.exe scripts/whl/windows/llvm_install.py --install_path=./llvm_tool
export LLVM_PATH=${SRC_DIR}/llvm_tool
echo "Install CUDA and CUDNN"
${SRC_DIR}/python_dev/3.8.3/python3.exe scripts/whl/windows/cuda_cudnn_install.py --sdk_name $SDK_NAME
echo "Preparing python enviroment"
versions="3.6.8 3.7.7 3.8.3 3.9.4 3.10.1"

for ver in $versions
do 
    if [ ${ver} == "3.6.8" ]; then
        ${SRC_DIR}/python_dev/$ver/python.exe -m pip install opencv-python==4.6.0.66 -i https://mirrors.sustech.edu.cn/pypi/simple
    fi
    ${SRC_DIR}/python_dev/$ver/python.exe -m pip install --upgrade pip -i https://mirrors.sustech.edu.cn/pypi/simple
    ${SRC_DIR}/python_dev/$ver/python.exe -m pip install cython -i https://mirrors.sustech.edu.cn/pypi/simple
    ${SRC_DIR}/python_dev/$ver/python.exe -m pip install wheel -i https://mirrors.sustech.edu.cn/pypi/simple
    ${SRC_DIR}/python_dev/$ver/python.exe -m pip install -r ${SRC_DIR}/imperative/python/requires.txt -i https://mirrors.sustech.edu.cn/pypi/simple
    ${SRC_DIR}/python_dev/$ver/python.exe -m pip install -r ${SRC_DIR}/imperative/python/requires-test.txt -i https://mirrors.sustech.edu.cn/pypi/simple
done
export CUDA_ROOT_DIR="${SRC_DIR}/cuda_tool/nvcc"
export CUDNN_ROOT_DIR="${SRC_DIR}/cuda_tool/Library"
if [[ ${SDK_NAME} == "cu118" ]]; then
    TRT_DIR="TensorRT-8.5.3.1"
elif [[ ${SDK_NAME} == "cu112" || ${SDK_NAME} == "cu114" || ${SDK_NAME} == "cu110" ]]; then
    TRT_DIR="TensorRT-7.2.3.4"
else
    TRT_DIR="TensorRT-6.0.1.5"
fi
export TRT_ROOT_DIR="/c/tools/$TRT_DIR"
export TRT_VERSION=${TRT_DIR#*-}
export VS_PATH="${SRC_DIR}/vs"
export PYTHON_ROOT="${SRC_DIR}/python_dev"

if [[ $SDK_NAME == "cu112" || $SDK_NAME == "cu114" ]]; then
    export EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=ON -DMGE_WITH_CUBLAS_SHARED=ON \
            -DMGE_CUDA_GENCODE=\"-gencode arch=compute_61,code=sm_61 \
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_86,code=compute_86\" "

    if [[ ${TRT_VERSION} == "7.2.3.4" ]]; then
        if [[ ! -f ${SRC_DIR}/cuda_tool/nvcc/bin/nvrtc64_111_0.dll ]]; then
            curl -SL https://dubaseodll.zhhainiao.com/dll/nvrtc64_111_0.dll --output ${SRC_DIR}/cuda_tool/nvcc/bin/nvrtc64_111_0.dll
        fi
    fi
elif [[ $SDK_NAME == "cu118" ]]; then
    export EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=ON -DMGE_WITH_CUBLAS_SHARED=ON \
        -DMGE_CUDA_GENCODE=\"-gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_75,code=sm_75 \
        -gencode arch=compute_80,code=sm_80 \
        -gencode arch=compute_86,code=sm_86 \
        -gencode arch=compute_89,code=sm_89 \
        -gencode arch=compute_89,code=compute_89\" "
    if [[ ! -f ${SRC_DIR}/cuda_tool/nvcc/bin/zlibwapi.dll ]]; then
        echo "try to download the zlibwapi.dll from https://duba-seo-dll-1252921383.cos.ap-beijing.myqcloud.com/dll/zlibwapi.dll"
        curl -SL https://duba-seo-dll-1252921383.cos.ap-beijing.myqcloud.com/dll/zlibwapi.dll --output ${SRC_DIR}/cuda_tool/nvcc/bin/zlibwapi.dll
    fi
elif [[ $SDK_NAME -eq "cu101" ]]; then
    export EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=OFF -DMGE_WITH_CUBLAS_SHARED=OFF"
    
else
    export BUILD_WHL_CPU_ONLY="ON"
fi
${SRC_DIR}/scripts/whl/windows/windows_build_whl.sh