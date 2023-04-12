#!/bin/bash -e
set -ex

CWD=$(dirname $0)
BASEDIR=$(readlink -f ${CWD}/../../..)
OUTPUTDIR=$(readlink -f ${CWD}/output)
USERID=$(id -u)
TMPFS_ARGS="--tmpfs /tmp:exec"
local_path=$(dirname $(readlink -f $0))
CUDNN_LIB_DIR="/opt/cudnn/lib64/"
CUDA_LIB_DIR="/usr/local/cuda/lib64/"
TensorRT_LIB_DIR="/opt/tensorrt/lib/"

SDK_NAME="unknown"
x86_64_support_version="cu101 cu111 cu112 cpu cu111_cudnn821_tensorRT825 cu114 cu118"
aarch64_support_version="cu102_JetsonNano cu111 cpu cu118"
docker_tag="env_manylinux2014:latest"

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
    CUDA_COPY_LIB_LIST="\
        ${CUDA_LIB_DIR}/libnvrtc.so.10.1"
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
        ${CUDNN_LIB_DIR}/libcudnn.so.8:\
        ${TensorRT_LIB_DIR}/libnvinfer_plugin.so.8:\
        ${TensorRT_LIB_DIR}/libnvinfer.so.8"


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
        ${CUDA_LIB_DIR}/libnvrtc-builtins.so.11.1:\
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

elif [ $SDK_NAME == "cu111_cudnn821_tensorRT825" ];then
    BUILD_GCC8="ON"
    REQUIR_CUDA_VERSION="11010"
    REQUIR_CUDNN_VERSION="8.2.1"
    REQUIR_TENSORRT_VERSION="8.2.5.1"
    REQUIR_CUBLAS_VERSION="11.2.1.74"
   

    CUDA_COPY_LIB_LIST="\
        ${CUDA_LIB_DIR}/libnvrtc.so.11.1:\
        ${CUDA_LIB_DIR}/libnvrtc-builtins.so.11.1:\
        ${CUDA_LIB_DIR}/libcublasLt.so.11:\
        ${CUDA_LIB_DIR}/libcublas.so.11:\
        ${CUDNN_LIB_DIR}/libcudnn_adv_infer.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_adv_train.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_cnn_infer.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_cnn_train.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_ops_infer.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn_ops_train.so.8:\
        ${CUDNN_LIB_DIR}/libcudnn.so.8:\
        ${TensorRT_LIB_DIR}/libnvinfer_plugin.so.8:\
        ${TensorRT_LIB_DIR}/libnvinfer.so.8"

    EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=ON -DMGE_WITH_CUBLAS_SHARED=ON \
        -DMGE_CUDA_GENCODE=\"-gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_75,code=sm_75 \
        -gencode arch=compute_80,code=sm_80 \
        -gencode arch=compute_86,code=sm_86 \
        -gencode arch=compute_86,code=compute_86\" "


elif [ $SDK_NAME == "cu112" ];then
    BUILD_GCC8="ON"
    CUDA_COPY_LIB_LIST="\
        ${CUDA_LIB_DIR}/libnvrtc.so.11.2:\
        ${CUDA_LIB_DIR}/libnvrtc-builtins.so.11.2:\
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


elif [ $SDK_NAME == "cu114" ];then
    BUILD_GCC8="ON"
    REQUIR_CUDA_VERSION="11040"
    REQUIR_CUDNN_VERSION="8.2.1"
    REQUIR_TENSORRT_VERSION="7.2.2.3"
    REQUIR_CUBLAS_VERSION="11.6.5.2"


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

elif [ $SDK_NAME == "cu118" ];then
    BUILD_GCC8="ON"
    REQUIR_CUDA_VERSION="11080"
    REQUIR_CUDNN_VERSION="8.6.0"
    REQUIR_TENSORRT_VERSION="8.5.3.1"
    REQUIR_CUBLAS_VERSION="11.11.3.6"

    # override the default cuda/cudnn/trt lib dir
    CUDNN_LIB_DIR="/opt/cudnn/lib/"
    CUDA_LIB_DIR="/usr/local/cuda/targets/x86_64-linux/lib/"
    TensorRT_LIB_DIR="/opt/tensorrt/lib/"
    if [ ${machine} == "aarch64" ];then
        CUDA_LIB_DIR="/usr/local/cuda/targets/sbsa-linux/lib/"
        # aarch64-trt8 libs build from ubuntu2004, which depends on new glibc and libc++
        # manylinux2014 do satify the glibc/libc++ requirement, so we need to use ubuntu2004 as base image
        docker_tag="ubuntu2004:latest"
    fi

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
        ${CUDNN_LIB_DIR}/libcudnn.so.8:\
        ${TensorRT_LIB_DIR}/libnvinfer_plugin.so.8:\
        ${TensorRT_LIB_DIR}/libnvonnxparser.so.8\
        ${TensorRT_LIB_DIR}/libnvinfer_builder_resource.so.8.5.3:\
        ${TensorRT_LIB_DIR}/libnvparsers.so.8:\
        ${TensorRT_LIB_DIR}/libnvinfer.so.8"

    EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=ON -DMGE_WITH_CUBLAS_SHARED=ON \
        -DMGE_CUDA_GENCODE=\"-gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_75,code=sm_75 \
        -gencode arch=compute_80,code=sm_80 \
        -gencode arch=compute_86,code=sm_86 \
        -gencode arch=compute_89,code=sm_89 \
        -gencode arch=compute_89,code=compute_89\" "
    if [ ${machine} == "aarch64" ];then
        if [ ${IN_CI} = "true" ]; then
            # ci taishan is use SM_75 card
            EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=ON -DMGE_WITH_CUBLAS_SHARED=ON -DMGE_CUDA_GENCODE=\"-gencode arch=compute_75,code=sm_75\" "
        else
            # support orin and remove 1080Ti
            EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=ON -DMGE_WITH_CUBLAS_SHARED=ON \
                -DMGE_CUDA_GENCODE=\" \
                -gencode arch=compute_70,code=sm_70 \
                -gencode arch=compute_75,code=sm_75 \
                -gencode arch=compute_80,code=sm_80 \
                -gencode arch=compute_86,code=sm_86 \
                -gencode arch=compute_87,code=sm_87 \
                -gencode arch=compute_89,code=sm_89 \
                -gencode arch=compute_89,code=compute_89\" "
        fi
    fi


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

source ${BASEDIR}/scripts/whl/utils/utils.sh

check_cuda_cudnn_trt_version

if [[ -z ${BUILD_GCC8} ]];then
    BUILD_GCC8=OFF
fi

if [ ${machine} == "aarch64" ];then
    # manylinux on aarch64 gcc9 is: (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2)
    # which version has issue: 'as' take a long long long time for some dnn kernel!
    # infact ubuntu gcc version: gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0 is OK
    echo "force use gcc8 on aarch64 linux"
    BUILD_GCC8="ON"
    if [ $SDK_NAME == "cu118" ];then
        echo "cu118 with aarch64 will build at ubuntu2004 docker env, so do not use gcc8"
        BUILD_GCC8="OFF"
    fi
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
    ${docker_tag} /bin/bash -c "$run_cmd"
