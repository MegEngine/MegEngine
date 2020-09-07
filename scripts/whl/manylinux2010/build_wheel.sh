#!/bin/bash -e

CWD=$(dirname $0)
BASEDIR=$(readlink -f ${CWD}/../../..)
OUTPUTDIR=$(readlink -f ${CWD}/output)
USERID=$(id -u)
TMPFS_ARGS="--tmpfs /tmp:exec"

BUILD_WHL_CPU_ONLY=${BUILD_WHL_CPU_ONLY}
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
    REQUIR_CUDA_VERSION="10010"
    REQUIR_CUDNN_VERSION="7.6.3"
    REQUIR_TENSORRT_VERSION="6.0.1.5"

    CUDA_ROOT_DIR_=${CUDA_ROOT_DIR%*/}
    CUDNN_ROOT_DIR_=${CUDNN_ROOT_DIR%*/}
    TENSORRT_ROOT_DIR_=${TENSORRT_ROOT_DIR%*/}

    CUDA_VERSION_PATH=${CUDA_ROOT_DIR_}/include/cuda.h
    CUDNN_VERSION_PATH=${CUDNN_ROOT_DIR_}/include/cudnn.h
    TENSORRT_VERSION_PATH=${TENSORRT_ROOT_DIR_}/include/NvInferVersion.h

    if [ ! -e $CUDA_VERSION_PATH ] ; then
        echo file $CUDA_VERSION_PATH is not exist
        echo please check the Environment must use CUDA-10.1 NO.$REQUIR_CUDA_VERSION
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

    CUDA_VERSION_CONTEXT=$(head -85 ${CUDA_VERSION_PATH})
    CUDNN_VERSION_CONTEXT=$(head -62 ${CUDNN_VERSION_PATH})
    TENSORRT_VERSION_CONTEXT=$(tail -12 ${TENSORRT_VERSION_PATH})

    CUDA_API_VERSION=$(echo $CUDA_VERSION_CONTEXT | grep -Eo "define __CUDA_API_VERSION * +([0-9]+)")
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

docker run -it --rm $TMPFS_ARGS -e UID=${USERID} -e LOCAL_VERSION=${LOCAL_VERSION} -e BUILD_WHL_CPU_ONLY=${BUILD_WHL_CPU_ONLY} -e ALL_PYTHON="${ALL_PYTHON}" -v ${CUDA_ROOT_DIR}:/usr/local/cuda -v ${CUDNN_ROOT_DIR}:/opt/cudnn -v ${TENSORRT_ROOT_DIR}:/opt/tensorrt -v ${BASEDIR}:/home/code -v ${OUTPUTDIR}:/home/output:rw env_manylinux2010:latest /home/code/scripts/whl/manylinux2010/do_build.sh
