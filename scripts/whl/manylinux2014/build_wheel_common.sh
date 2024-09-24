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
NEUWARE_LIB_DIR="/usr/local/neuware/lib64"

SDK_NAME="unknown"
x86_64_support_version="cu101 cu111 cu112 cpu cu111_cudnn821_tensorRT825 cu114 cu118 neuware113 neuware115"
aarch64_support_version="cu102_JetsonNano cu111 cpu cu118"
docker_tag="env_manylinux2014:latest"
build_with_library="false"
if [[ -z ${IN_CI} ]]
then
    IN_CI="false"
fi
function usage() {
    echo "use -sdk sdk_version to specify sdk toolkit config!"
    echo "now x86_64 sdk_version support ${x86_64_support_version}"
    echo "now aarch64 sdk_version support ${aarch64_support_version}"
    echo "use -l build with cuda and cudnn library instead of depending cuda and cudnn whls!"
}

while [ "$1" != "" ]; do
    case $1 in
        -sdk)
            shift
            SDK_NAME=$1
            shift
            ;;
        -l)
            build_with_library="true"
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
    COPY_LIB_LIST="\
        ${CUDA_LIB_DIR}/libnvrtc.so.10.1"
    EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=OFF -DMGE_WITH_CUBLAS_SHARED=OFF"
    BUILD_GCC8="ON"
    REQUIR_CUDA_VERSION="10010"
    REQUIR_CUDNN_VERSION="7.6.3"
    REQUIR_TENSORRT_VERSION="6.0.1.5"
    REQUIR_CUBLAS_VERSION="10.2.1.243"
    BUILD_WHL_WITH_CUDA="ON"

elif [ $SDK_NAME == "cu102_JetsonNano" ];then
    # Jetson Nano B01 version
    REQUIR_CUDA_VERSION="10020"
    REQUIR_CUDNN_VERSION="8.2.1"
    REQUIR_TENSORRT_VERSION="8.0.1.6"
    REQUIR_CUBLAS_VERSION="10.2.3.300"

    COPY_LIB_LIST="\
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
    BUILD_WHL_WITH_CUDA="ON"

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

    COPY_LIB_LIST="\
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
    BUILD_WHL_WITH_CUDA="ON"

elif [ $SDK_NAME == "cu111_cudnn821_tensorRT825" ];then
    BUILD_GCC8="ON"
    REQUIR_CUDA_VERSION="11010"
    REQUIR_CUDNN_VERSION="8.2.1"
    REQUIR_TENSORRT_VERSION="8.2.5.1"
    REQUIR_CUBLAS_VERSION="11.2.1.74"
   

    COPY_LIB_LIST="\
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
    BUILD_WHL_WITH_CUDA="ON"

elif [ $SDK_NAME == "cu112" ];then
    BUILD_GCC8="ON"
    COPY_LIB_LIST="\
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
    BUILD_WHL_WITH_CUDA="ON"


elif [ $SDK_NAME == "cu114" ];then
    BUILD_GCC8="ON"
    REQUIR_CUDA_VERSION="11040"
    REQUIR_CUDNN_VERSION="8.2.1"
    REQUIR_TENSORRT_VERSION="7.2.2.3"
    REQUIR_CUBLAS_VERSION="11.6.5.2"


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

    EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=ON -DMGE_WITH_CUBLAS_SHARED=ON \
        -DMGE_CUDA_GENCODE=\"-gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_75,code=sm_75 \
        -gencode arch=compute_80,code=sm_80 \
        -gencode arch=compute_86,code=sm_86 \
        -gencode arch=compute_86,code=compute_86\" "
    BUILD_WHL_WITH_CUDA="ON"

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

    COPY_LIB_LIST="\
        ${TensorRT_LIB_DIR}/libnvinfer_plugin.so.8:\
        ${TensorRT_LIB_DIR}/libnvonnxparser.so.8\
        ${TensorRT_LIB_DIR}/libnvinfer_builder_resource.so.8.5.3:\
        ${TensorRT_LIB_DIR}/libnvparsers.so.8:\
        ${TensorRT_LIB_DIR}/libnvinfer.so.8"
    if [ ${build_with_library}  == "true" ];then
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
            ${CUDNN_LIB_DIR}/libcudnn.so.8:\
            ${COPY_LIB_LIST}"
    fi
    EXTRA_CMAKE_FLAG=" -DMGE_WITH_CUDNN_SHARED=ON -DMGE_WITH_CUBLAS_SHARED=ON \
        -DMGE_CUDA_GENCODE=\"-gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_75,code=sm_75 \
        -gencode arch=compute_80,code=sm_80 \
        -gencode arch=compute_86,code=sm_86 \
        -gencode arch=compute_89,code=sm_89 \
        -gencode arch=compute_90,code=sm_90 \
        -gencode arch=compute_90,code=compute_90\" "
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
                -gencode arch=compute_90,code=sm_90 \
                -gencode arch=compute_90,code=compute_90\" "
        fi
    fi
    BUILD_WHL_WITH_CUDA="ON"
else
    BUILD_WHL_WITH_CUDA="OFF"
fi

if [ $SDK_NAME == "neuware113" ] || [ $SDK_NAME == "neuware115" ];then
    echo "use $SDK_NAME with cambricon support"
    BUILD_GCC8="ON"
    COPY_LIB_LIST="\
        ${NEUWARE_LIB_DIR}/libcncl.so.1:\
        ${NEUWARE_LIB_DIR}/libcnnl.so.1:\
        ${NEUWARE_LIB_DIR}/libcnrt.so:\
        ${NEUWARE_LIB_DIR}/libcnrtc.so:\
        ${NEUWARE_LIB_DIR}/libcnnl_extra.so:\
        ${NEUWARE_LIB_DIR}/libmagicmind_runtime.so.1:\
        ${NEUWARE_LIB_DIR}/libcnlight.so:\
        ${NEUWARE_LIB_DIR}/libcnpapi.so:\
        ${NEUWARE_LIB_DIR}/libcndrv.so:\
        ${NEUWARE_LIB_DIR}/libcndev.so:\
        ${NEUWARE_LIB_DIR}/libcnmlrt.so:\
        ${NEUWARE_LIB_DIR}/libmagicmind.so.1:\
        ${NEUWARE_LIB_DIR}/libcnbin.so"

    BUILD_WHL_WITH_CAMBRICON="ON"
    EXTRA_CMAKE_FLAG=" -DMGE_WITH_CAMBRICON=ON"
else
    BUILD_WHL_WITH_CAMBRICON="OFF"
fi

if [ $SDK_NAME == "cpu" ];then
    echo "use $SDK_NAME without cuda support"
    BUILD_WHL_CPU_ONLY="ON"
else
    BUILD_WHL_CPU_ONLY="OFF"
fi

echo ${BASEDIR}
pushd ${BASEDIR}/third_party >/dev/null
./prepare.sh
popd >/dev/null

cd ${CWD}
mkdir -p ${OUTPUTDIR}

source ${BASEDIR}/scripts/whl/utils/utils.sh

if [ ${BUILD_WHL_WITH_CUDA} == "ON" ]; then
    check_cuda_cudnn_trt_version
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

# mount args
mount_args=""
if [ ${BUILD_WHL_WITH_CUDA} == "ON" ]; then
    mount_args="-v ${CUDA_ROOT_DIR}:/usr/local/cuda -v ${CUDNN_ROOT_DIR}:/opt/cudnn -v ${TENSORRT_ROOT_DIR}:/opt/tensorrt"
fi

if [ ${BUILD_WHL_WITH_CAMBRICON} == "ON" ]; then
    mount_args="-v ${NEUWARE_HOME}:/usr/local/neuware"
fi

echo "mount args: ${mount_args}"

if [ -z "${PYTHON_EXTRA_REQUIRES}" ]; then
    PYTHON_EXTRA_REQUIRES=""
fi

docker run --rm ${docker_args} $TMPFS_ARGS \
    -e UID=${USERID} \
    -e PUBLIC_VERSION_POSTFIX=${PUBLIC_VERSION_POSTFIX} \
    -e LOCAL_VERSION=${LOCAL_VERSION} \
    -e STRIP_SDK_INFO=${STRIP_SDK_INFO} \
    -e BUILD_WHL_CPU_ONLY=${BUILD_WHL_CPU_ONLY} \
    -e BUILD_WHL_WITH_CUDA=${BUILD_WHL_WITH_CUDA} \
    -e BUILD_WHL_WITH_CAMBRICON=${BUILD_WHL_WITH_CAMBRICON} \
    -e ALL_PYTHON="${ALL_PYTHON}" \
    -e EXTRA_CMAKE_FLAG="$EXTRA_CMAKE_FLAG" \
    -e COPY_LIB_LIST="$COPY_LIB_LIST"  \
    -e SDK_NAME="$SDK_NAME"  \
    -e CUDA_ROOT_DIR="/usr/local/cuda" \
    -e CUDNN_ROOT_DIR="/opt/cudnn" \
    -e TRT_ROOT_DIR="/opt/tensorrt" \
    -e NEUWARE_HOME="/usr/local/neuware" \
    -e PYTHON_EXTRA_REQUIRES="$PYTHON_EXTRA_REQUIRES" \
    -e BUILD_WITH_LIBRARY="$build_with_library" \
    ${mount_args} \
    -v ${BASEDIR}:/home/code \
    -v ${OUTPUTDIR}:/home/output:rw \
    ${docker_tag} /bin/bash -c "$run_cmd"
