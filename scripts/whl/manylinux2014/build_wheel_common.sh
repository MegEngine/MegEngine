#!/bin/bash -e
set -x

CWD=$(dirname $0)
BASEDIR=$(readlink -f ${CWD}/../../..)
OUTPUTDIR=$(readlink -f ${CWD}/output)
USERID=$(id -u)
TMPFS_ARGS="--tmpfs /tmp:exec"
local_path=$(dirname $(readlink -f $0))
CUDNN_LIB_DIR="/opt/cudnn/lib64/"
CUDA_LIB_DIR="/usr/local/cuda/lib64/"

SDK_NAME="unknown"
x86_64_support_version="cu101 cu111 cu112 cpu"
aarch64_support_version="cu102_JetsonNano cu111 cpu"
if [[ -z ${IN_CI} ]]
then
    IN_CI="false"
fi
function usage() {
    echo "use -sdk sdk_version to specify sdk toolkit config!"
    echo "now x86_64 sdk_version support ${x86_64_support_version}"
    echo "now aarch64 sdk_version support ${aarch64_support_version}"
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
            exit -1
    esac
done

is_valid_sdk="false"
all_sdk=""
machine=$(uname -m)
case ${machine} in
    x86_64) all_sdk=${x86_64_support_version} ;;
    aarch64) all_sdk=${aarch64_support_version} ;;
    *) echo "nonsupport env!!!";exit -1 ;;
esac

for i_sdk in ${all_sdk}
do
    if [ ${i_sdk} == ${SDK_NAME} ];then
        is_valid_sdk="true"
    fi
done
if [ ${is_valid_sdk} == "false" ];then
    echo "invalid sdk: ${SDK_NAME}"
    usage
    exit -1
fi

echo "Build with ${SDK_NAME}"

if [ $SDK_NAME == "cu101" ];then
    CUDA_COPY_LIB_LIST="${CUDA_LIB_DIR}/libnvrtc.so.10.1"
    EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=OFF -DMGE_WITH_CUBLAS_SHARED=OFF" 
    BUILD_GCC8="ON"  
    REQUIR_CUDA_VERSION="10010" 
    REQUIR_CUDNN_VERSION="7.6.3" 
    REQUIR_TENSORRT_VERSION="6.0.1.5" 
    REQUIR_CUBLAS_VERSION="10.2.1.243"

elif [ $SDK_NAME == "cu102_JetsonNano" ];then
    # Jetson Nano B01 version
    REQUIR_CUDA_VERSION="10020"
    REQUIR_CUDNN_VERSION="8.2.1"
    REQUIR_TENSORRT_VERSION="8.0.1.6"
    REQUIR_CUBLAS_VERSION="10.2.3.300"

    CUDA_COPY_LIB_LIST="\
        ${CUDA_LIB_DIR}/libnvrtc.so.10.2:\
        ${CUDA_LIB_DIR}/libcublasLt.so.10:\
        ${CUDA_LIB_DIR}/libcublas.so.10:\
        ${CUDNN_LIB_DIR}/libcudnn_adv_infer.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_adv_train.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_cnn_infer.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_cnn_train.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_ops_infer.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_ops_train.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn.so.8"

    EXTRA_CMAKE_FLAG="-DMGE_WITH_CUDNN_SHARED=ON -DMGE_WITH_CUBLAS_SHARED=ON -DMGE_CUDA_GENCODE=\"-gencode arch=compute_53,code=sm_53\" "

elif [ $SDK_NAME == "cu111" ];then
    BUILD_GCC8="ON"
    if [ ${machine} == "aarch64" ];then
        REQUIR_CUDA_VERSION="11010"
        REQUIR_CUDNN_VERSION="8.0.5"
        REQUIR_TENSORRT_VERSION="7.2.1.6"
        REQUIR_CUBLAS_VERSION="11.3.0.106"
    elif [ ${machine} == "x86_64" ];then
        REQUIR_CUDA_VERSION="11010"
        REQUIR_CUDNN_VERSION="8.0.4"
        REQUIR_TENSORRT_VERSION="7.2.2.3"
        REQUIR_CUBLAS_VERSION="11.2.1.74"
    else
        echo "no support machine: ${machine}"
        exit -1
    fi

    CUDA_COPY_LIB_LIST="\
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

    if [ ${IN_CI} = "true" ] && [ ${machine} == "aarch64" ]; then
        EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=ON -DMGE_WITH_CUBLAS_SHARED=ON -DMGE_CUDA_GENCODE=\"-gencode arch=compute_75,code=sm_75\" "
    else
        EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=ON -DMGE_WITH_CUBLAS_SHARED=ON \
            -DMGE_CUDA_GENCODE=\"-gencode arch=compute_61,code=sm_61 \
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_86,code=compute_86\" "
    fi

elif [ $SDK_NAME == "cu112" ];then
    BUILD_GCC8="ON"
    CUDA_COPY_LIB_LIST="\
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

    EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=ON -DMGE_WITH_CUBLAS_SHARED=ON \
        -DMGE_CUDA_GENCODE=\"-gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_75,code=sm_75 \
        -gencode arch=compute_80,code=sm_80 \
        -gencode arch=compute_86,code=sm_86 \
        -gencode arch=compute_86,code=compute_86\" "

    REQUIR_CUDA_VERSION="11020" 
    REQUIR_CUDNN_VERSION="8.0.4" 
    REQUIR_TENSORRT_VERSION="7.2.2.3" 
    REQUIR_CUBLAS_VERSION="11.3.1.68"

elif [ $SDK_NAME == "cpu" ];then
    echo "use $SDK_NAME without cuda support"
    BUILD_WHL_CPU_ONLY="ON"
else
    echo "no support sdk ${SDK_NAME}"
    usage
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
        if [[ -z ${TRT_ROOT_DIR} ]]; then
            echo "Environment variable TRT_ROOT_DIR not set."
            exit -1
        else
            echo "put ${TRT_ROOT_DIR} to TENSORRT_ROOT_DIR env"
            TENSORRT_ROOT_DIR=${TRT_ROOT_DIR}
        fi
    fi

    ## YOU SHOULD MODIFY CUDA VERSION AS BELOW WHEN UPGRADE
    CUDA_ROOT_DIR_=${CUDA_ROOT_DIR%*/}
    CUDNN_ROOT_DIR_=${CUDNN_ROOT_DIR%*/}
    TENSORRT_ROOT_DIR_=${TENSORRT_ROOT_DIR%*/}

    CUBLAS_VERSION_PATH=${CUDA_ROOT_DIR_}/include/cublas_api.h
    CUDA_VERSION_PATH=${CUDA_ROOT_DIR_}/include/cuda.h
    if [ -e ${CUDNN_ROOT_DIR_}/include/cudnn_version.h ];then
        CUDNN_VERSION_PATH=${CUDNN_ROOT_DIR_}/include/cudnn_version.h
    elif [ -e ${CUDNN_ROOT_DIR_}/include/cudnn.h ];then
        CUDNN_VERSION_PATH=${CUDNN_ROOT_DIR_}/include/cudnn.h
    else
        echo "cannot determine CUDNN_VERSION_PATH from CUDNN_ROOT_DIR."
        exit -1
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
    if [ ! -e $CUBLAS_VERSION_PATH ] ; then
        echo file $CUBLAS_VERSION_PATH is not exist
        exit -1
    fi

    CUBLAS_VERSION_CONTEXT=$(head -150 ${CUBLAS_VERSION_PATH})
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

    CUBLAS_VERSION_MAJOR=$(echo $CUBLAS_VERSION_CONTEXT | grep -Eo "define CUBLAS_VER_MAJOR * +([0-9]+)" | grep -Eo "*+([0-9]+)")
    CUBLAS_VERSION_MINOR=$(echo $CUBLAS_VERSION_CONTEXT | grep -Eo "define CUBLAS_VER_MINOR * +([0-9]+)" | grep -Eo "*+([0-9]+)")
    CUBLAS_VERSION_PATCH=$(echo $CUBLAS_VERSION_CONTEXT | grep -Eo "define CUBLAS_VER_PATCH * +([0-9]+)" | grep -Eo "*+([0-9]+)")
    CUBLAS_VERSION_BUILD=$(echo $CUBLAS_VERSION_CONTEXT | grep -Eo "define CUBLAS_VER_BUILD * +([0-9]+)" | grep -Eo "*+([0-9]+)")
    CUBLAS_VERSION=${CUBLAS_VERSION_MAJOR}.${CUBLAS_VERSION_MINOR}.${CUBLAS_VERSION_PATCH}.${CUBLAS_VERSION_BUILD}
    echo CUBLAS_VERSION:$CUBLAS_VERSION

    if [ $CUDA_VERSION != $REQUIR_CUDA_VERSION ] ; then
        echo please check the Environment must use CUDA NO.$REQUIR_CUDA_VERSION
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

    if [ $CUBLAS_VERSION != $REQUIR_CUBLAS_VERSION ] ; then
        echo please check the Environment must use CUBLAS-$REQUIR_CUBLAS_VERSION
        exit -1
    fi
fi

if [[ -z ${BUILD_GCC8} ]];then
    BUILD_GCC8=OFF
fi

if [ ${machine} == "aarch64" ];then
    # manylinux on aarch64 gcc9 is: (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2)
    # which version has issue: 'as' take a long long long time for some dnn kernel!
    # infact ubuntu gcc version: gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0 is OK
    echo "force use gcc8 on aarch64 linux"
    BUILD_GCC8="ON"
fi

if [ "$BUILD_GCC8" == "ON" ];then
    run_cmd="scl enable devtoolset-8 /home/code/scripts/whl/manylinux2014/do_build_common.sh"
else
    run_cmd="/home/code/scripts/whl/manylinux2014/do_build_common.sh"
fi
set +x
docker_args="-it"
if [ -z "${CI_SERVER_NAME}" ]; then
    CI_SERVER_NAME="null"
fi
if [ ${CI_SERVER_NAME} = "GitLab" ];then
    docker_args="-i"
fi
if [ ${IN_CI} = "true" ];then
    EXTRA_CMAKE_FLAG=" ${EXTRA_CMAKE_FLAG} -DMGE_WITH_TEST=ON"
fi
docker run --rm ${docker_args} $TMPFS_ARGS \
    -e UID=${USERID} \
    -e PUBLIC_VERSION_POSTFIX=${PUBLIC_VERSION_POSTFIX} \
    -e LOCAL_VERSION=${LOCAL_VERSION} \
    -e STRIP_SDK_INFO=${STRIP_SDK_INFO} \
    -e BUILD_WHL_CPU_ONLY=${BUILD_WHL_CPU_ONLY} \
    -e ALL_PYTHON="${ALL_PYTHON}" \
    -e EXTRA_CMAKE_FLAG="$EXTRA_CMAKE_FLAG" \
    -e CUDA_COPY_LIB_LIST="$CUDA_COPY_LIB_LIST"  \
    -e SDK_NAME="$SDK_NAME"  \
    -e CUDA_ROOT_DIR="/usr/local/cuda" \
    -e CUDNN_ROOT_DIR="/opt/cudnn" \
    -e TRT_ROOT_DIR="/opt/tensorrt" \
    -v ${CUDA_ROOT_DIR}:/usr/local/cuda \
    -v ${CUDNN_ROOT_DIR}:/opt/cudnn \
    -v ${TENSORRT_ROOT_DIR}:/opt/tensorrt \
    -v ${BASEDIR}:/home/code \
    -v ${OUTPUTDIR}:/home/output:rw \
    env_manylinux2014:latest /bin/bash -c "$run_cmd"
