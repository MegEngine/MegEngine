#!/bin/bash
set -e
CWD=$(dirname $0)
BASEDIR=$(readlink -f ${CWD}/../../..)
OUTPUTDIR=$(readlink -f ${CWD}/output)
USERID=$(id -u)
TMPFS_ARGS="--tmpfs /tmp:exec"
local_path=$(dirname $(readlink -f $0))
CUDNN_LIB_DIR="/opt/cudnn/lib64/"
CUDA_LIB_DIR="/usr/local/cuda/lib64/"

SDK_NAME="unknown"
function usage() {
    echo "use '-sdk cu111' to specify cuda toolkit config, also support cu101, cu112, cpu"
}

while [ "$1" != "" ]; do
    case $1 in
        -sdk)
           shift
           SDK_NAME=$1
           shift
           ;;
        *)
            usage
            exit 1
    esac
done

echo "Build with ${SDK_NAME}"

if [ $SDK_NAME == "cu101" ];then
    COPY_LIB_LIST="${CUDA_LIB_DIR}/libnvrtc.so.10.1"
    EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=OFF" 
    BUILD_GCC8="ON"  
    REQUIR_CUDA_VERSION="10010" 
    REQUIR_CUDNN_VERSION="7.6.3" 
    REQUIR_TENSORRT_VERSION="6.0.1.5" 
elif [ $SDK_NAME == "cu111" ];then
    COPY_LIB_LIST="\
        ${CUDA_LIB_DIR}/libnvrtc.so.11.1:\
        ${CUDA_LIB_DIR}/libcublasLt.so.11:\
        ${CUDA_LIB_DIR}/libcublas.so.11:\
        ${CUDNN_LIB_DIR}/libcudnn_adv_infer.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_adv_train.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_cnn_infer.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_cnn_train.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_ops_infer.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_ops_train.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn.so.8"
    EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=ON\
        -gencode arch=compute_61,code=sm_61 \
        arch=compute_70,code=sm_70 \
        arch=compute_75,code=sm_75 \
        arch=compute_80,code=sm_80 \
        arch=compute_86,code=sm_86 \
        arch=compute_86,code=compute_86" 
    REQUIR_CUDA_VERSION="11010" 
    REQUIR_CUDNN_VERSION="8.0.4" 
    REQUIR_TENSORRT_VERSION="7.2.2.3" 
elif [ $SDK_NAME == "cu112" ];then
    COPY_LIB_LIST="\
        ${CUDA_LIB_DIR}/libnvrtc.so.11.2:\
        ${CUDA_LIB_DIR}/libcublasLt.so.11:\
        ${CUDA_LIB_DIR}/libcublas.so.11:\
        ${CUDNN_LIB_DIR}/libcudnn_adv_infer.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_adv_train.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_cnn_infer.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_cnn_train.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_ops_infer.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_ops_train.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn.so.8"
    EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=ON \
        -gencode arch=compute_61,code=sm_61 \
        arch=compute_70,code=sm_70 \
        arch=compute_75,code=sm_75 \
        arch=compute_80,code=sm_80 \
        arch=compute_86,code=sm_86 \
        arch=compute_86,code=compute_86"  
    REQUIR_CUDA_VERSION="11020" 
    REQUIR_CUDNN_VERSION="8.0.4" 
    REQUIR_TENSORRT_VERSION="7.2.2.3" 
elif [ $SDK_NAME == "cpu" ];then
    echo "use $SDK_NAME without cuda support"
    BUILD_WHL_CPU_ONLY="ON"
else
    echo "no support sdk ${SDK_NAME}, please set by '-sdk cu111'"
    exit -1
fi

if [[ -z ${BUILD_WHL_CPU_ONLY} ]]
then
    BUILD_WHL_CPU_ONLY="OFF"
fi

echo ${BASEDIR}
pushd ${BASEDIR}/third_party >/dev/null
    ./prepare.sh
popd >/dev/null

cd ${CWD}
mkdir -p ${OUTPUTDIR}

if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
    if [[ -z ${CUDA_ROOT_DIR} ]]; then
        echo "Environment variable CUDA_ROOT_DIR not set."
        exit -1
    fi
    if [[ -z ${CUDNN_ROOT_DIR} ]]; then
        echo "Environment variable CUDNN_ROOT_DIR not set."
        exit -1
    fi
    if [[ -z ${TENSORRT_ROOT_DIR} ]]; then
        echo "Environment variable TENSORRT_ROOT_DIR not set."
        exit -1
    fi

    ## YOU SHOULD MODIFY CUDA VERSION AS BELOW WHEN UPGRADE
    

    CUDA_ROOT_DIR_=${CUDA_ROOT_DIR%*/}
    CUDNN_ROOT_DIR_=${CUDNN_ROOT_DIR%*/}
    TENSORRT_ROOT_DIR_=${TENSORRT_ROOT_DIR%*/}

    CUDA_VERSION_PATH=${CUDA_ROOT_DIR_}/include/cuda.h
    if [ "$REQUIR_CUDA_VERSION" -ge "11000" ];then
        CUDNN_VERSION_PATH=${CUDNN_ROOT_DIR_}/include/cudnn_version.h
    else
        CUDNN_VERSION_PATH=${CUDNN_ROOT_DIR_}/include/cudnn.h
    fi
    TENSORRT_VERSION_PATH=${TENSORRT_ROOT_DIR_}/include/NvInferVersion.h

    if [ ! -e $CUDA_VERSION_PATH ] ; then
        echo file $CUDA_VERSION_PATH is not exist
        echo please check the Environment must use CUDA-$REQUIR_CUDA_VERSION
        exit -1
    fi
    if [ ! -e $CUDNN_VERSION_PATH ] ; then
        echo file $CUDNN_VERSION_PATH is not exist
        echo please check the Environment must use CUDNN-V$REQUIR_CUDNN_VERSION
        exit -1
    fi
    if [ ! -e $TENSORRT_VERSION_PATH ] ; then
        echo file $TENSORRT_VERSION_PATH is not exist
        echo please check the Environment must use TensorRT-$REQUIR_TENSORRT_VERSION
        exit -1
    fi

    CUDA_VERSION_CONTEXT=$(head -300 ${CUDA_VERSION_PATH})
    CUDNN_VERSION_CONTEXT=$(head -62 ${CUDNN_VERSION_PATH})
    TENSORRT_VERSION_CONTEXT=$(tail -12 ${TENSORRT_VERSION_PATH})

    if [ "$REQUIR_CUDA_VERSION" -ge "11000" ];then
        CUDA_API_VERSION=$(echo $CUDA_VERSION_CONTEXT | grep -Eo "define CUDA_VERSION * +([0-9]+)")
    else
        CUDA_API_VERSION=$(echo $CUDA_VERSION_CONTEXT | grep -Eo "define __CUDA_API_VERSION * +([0-9]+)")
    fi
    CUDA_VERSION=${CUDA_API_VERSION:0-5}
    echo CUDA_VERSION:$CUDA_VERSION

    CUDNN_VERSION_MAJOR=$(echo $CUDNN_VERSION_CONTEXT | grep -Eo "define CUDNN_MAJOR * +([0-9]+)")
    CUDNN_VERSION_MINOR=$(echo $CUDNN_VERSION_CONTEXT | grep -Eo "define CUDNN_MINOR * +([0-9]+)")
    CUDNN_VERSION_PATCH=$(echo $CUDNN_VERSION_CONTEXT | grep -Eo "define CUDNN_PATCHLEVEL * +([0-9]+)")
    CUDNN_VERSION=${CUDNN_VERSION_MAJOR:0-1}.${CUDNN_VERSION_MINOR:0-1}.${CUDNN_VERSION_PATCH:0-1}
    echo CUDNN_VERSION:$CUDNN_VERSION

    TENSORRT_VERSION_MAJOR=$(echo $TENSORRT_VERSION_CONTEXT | grep -Eo "NV_TENSORRT_MAJOR * +([0-9]+)")
    TENSORRT_VERSION_MINOR=$(echo $TENSORRT_VERSION_CONTEXT | grep -Eo "NV_TENSORRT_MINOR * +([0-9]+)")
    TENSORRT_VERSION_PATCH=$(echo $TENSORRT_VERSION_CONTEXT | grep -Eo "NV_TENSORRT_PATCH * +([0-9]+)")
    TENSORRT_VERSION_BUILD=$(echo $TENSORRT_VERSION_CONTEXT | grep -Eo "NV_TENSORRT_BUILD * +([0-9]+)")
    TENSORRT_VERSION=${TENSORRT_VERSION_MAJOR:0-1}.${TENSORRT_VERSION_MINOR:0-1}.${TENSORRT_VERSION_PATCH:0-1}.${TENSORRT_VERSION_BUILD:0-1}
    echo TENSORRT_VERSION:$TENSORRT_VERSION

    if [ $CUDA_VERSION != $REQUIR_CUDA_VERSION ] ; then
        echo please check the Environment must use CUDA-10.1 NO.$REQUIR_CUDA_VERSION
        exit -1
    fi

    if [ $CUDNN_VERSION != $REQUIR_CUDNN_VERSION ] ; then
        echo please check the Environment must use CUDNN-V$REQUIR_CUDNN_VERSION
        exit -1
    fi

    if [ $TENSORRT_VERSION != $REQUIR_TENSORRT_VERSION ] ; then
        echo please check the Environment must use TENSORRT-$REQUIR_TENSORRT_VERSION
        exit -1
    fi
fi

if [[ -z ${BUILD_GCC8} ]];then
    BUILD_GCC8=OFF
fi

if [ "$BUILD_GCC8" == "ON" ];then
    run_cmd="scl enable devtoolset-8 /home/code/scripts/whl/manylinux2014/do_build_common.sh"
else
    run_cmd="/home/code/scripts/whl/manylinux2014/do_build_common.sh"
fi

docker run --rm -it $TMPFS_ARGS \
    -e UID=${USERID} \
    -e LOCAL_VERSION=${LOCAL_VERSION} \
    -e BUILD_WHL_CPU_ONLY=${BUILD_WHL_CPU_ONLY} \
    -e ALL_PYTHON="${ALL_PYTHON}" \
    -e EXTRA_CMAKE_FLAG="$EXTRA_CMAKE_FLAG" \
    -e COPY_LIB_LIST="$COPY_LIB_LIST"  \
    -e SDK_NAME="$SDK_NAME"  \
    -v ${CUDA_ROOT_DIR}:/usr/local/cuda \
    -v ${CUDNN_ROOT_DIR}:/opt/cudnn \
    -v ${TENSORRT_ROOT_DIR}:/opt/tensorrt \
    -v ${BASEDIR}:/home/code \
    -v ${OUTPUTDIR}:/home/output:rw \
    env_manylinux2014:latest /bin/bash -c "$run_cmd"



