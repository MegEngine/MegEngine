#!/usr/bin/env bash
set -e

# if you want debug this script, please set -ex

OS=$(uname -s)

docker_file=""
function config_docker_file() {
    case $(uname -m) in
        x86_64) docker_file=Dockerfile ;;
        aarch64) docker_file=Dockerfile_aarch64 ;;
        *) echo "nonsupport env!!!";exit -1 ;;
    esac
}

function ninja_dry_run_and_check_increment() {
    echo "into ninja_dry_run_and_check_increment"
    if [ $# -eq 3 ]; then
        _BUILD_SHELL=$1
        _BUILD_FLAGS="$2 -n"
        _INCREMENT_KEY_WORDS=$3
    else
        echo "err call ninja_dry_run_and_check_increment"
        exit -1
    fi

    bash ${_BUILD_SHELL} ${_BUILD_FLAGS} 2>&1 | tee dry_run.log

    DIRTY_LOG=`cat dry_run.log`
    if [[ "${DIRTY_LOG}" =~ ${_INCREMENT_KEY_WORDS} ]]; then
        echo "DIRTY_LOG is:"
        echo ${DIRTY_LOG}
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "python3 switch increment build failed, some MR make a wrong CMakeLists.txt depends"
        echo "or build env can not find default python3 in PATH env"
        echo "please refs for PYTHON3_EXECUTABLE_WITHOUT_VERSION define at SRC_ROOT/CMakeLists.txt"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        exit -1
    fi

    CHECK_NINJA_DRY_ISSUE_KEY_WORDS="VerifyGlobs"
    if [[ "${DIRTY_LOG}" =~ ${CHECK_NINJA_DRY_ISSUE_KEY_WORDS} ]]; then
        echo "DIRTY_LOG is:"
        echo ${DIRTY_LOG}
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "python3 switch increment build failed, some MR make a wrong CMakeLists.txt"
        echo "for example use GLOB with CONFIGURE_DEPENDS flag may lead to ninja dry run failed"
        echo "about CONFIGURE_DEPENDS (please do not use it):"
        echo "a: we use scripts/cmake-build/*.sh to trigger rerun cmake, so no need CONFIGURE_DEPENDS"
        echo "b: as https://cmake.org/cmake/help/latest/command/file.html Note"
        echo "   CONFIGURE_DEPENDS do not support for all generators"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        exit -1
    fi

    # as python3 change, imperative src need rebuild, force check it!
    MUST_INCLUDE_KEY_WORDS="imperative"
    if [[ "${DIRTY_LOG}" =~ ${MUST_INCLUDE_KEY_WORDS} ]]; then
        echo "valid increment dry run log"
    else
        echo "DIRTY_LOG is:"
        echo ${DIRTY_LOG}
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "python3 switch increment build failed, some MR make a wrong CMakeLists.txt depends"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        exit -1
    fi
}

PYTHON_API_INCLUDES=""

function check_build_ninja_python_api() {
    INCLUDE_KEYWORD=""
    IS_MINOR_HIT=FALSE
    if [ $# -eq 1 ]; then
        ver=$1
        echo "org args: ${ver}"
        if [[ $OS =~ "NT" ]]; then
            INCLUDE_KEYWORD="${ver}\\\\include"
            PYTHON_API_INCLUDES="3.6.8\\\\include 3.7.7\\\\include 3.8.3\\\\include 3.9.4\\\\include 3.10.1\\\\include"
        elif [[ $OS =~ "Linux" ]]; then
            if which lsb_release && lsb_release -a | grep "Ubuntu"; then
                echo "into Ubuntu env"
                is_punctuation=${ver:4:1}
                INCLUDE_KEYWORD="include/python3.${ver:2:1}"
                if [ ${is_punctuation} = "." ]; then
                    INCLUDE_KEYWORD="include/python3.${ver:2:2}"
                fi
                PYTHON_API_INCLUDES="include/python3.5 include/python3.6 include/python3.7 include/python3.8 include/python3.9 include/python3.10"
            else
                echo "into manylinux env"
                ver=`echo $ver | tr -d m`
                INCLUDE_KEYWORD="include/python3.${ver:1}" # like 39/310
                info=`command -v termux-info || true`
                if [[ "${info}" =~ "com.termux" ]]; then
                    echo "find termux-info at: ${info}"
                    is_punctuation=${ver:4:1}
                    INCLUDE_KEYWORD="include/python3.${ver:2:1}"
                    if [ ${is_punctuation} = "." ]; then
                        INCLUDE_KEYWORD="include/python3.${ver:2:2}"
                    fi
                fi
                PYTHON_API_INCLUDES="include/python3.5 include/python3.6 include/python3.7 include/python3.8 include/python3.9 include/python3.10"
            fi
        elif [[ $OS =~ "Darwin" ]]; then
            is_punctuation=${ver:4:1}
            INCLUDE_KEYWORD="include/python3.${ver:2:1}"
            if [ ${is_punctuation} = "." ]; then
                INCLUDE_KEYWORD="include/python3.${ver:2:2}"
            fi
            PYTHON_API_INCLUDES="include/python3.5 include/python3.6 include/python3.7 include/python3.8 include/python3.9 include/python3.10"
        else
            echo "unknown OS: ${OS}"
            exit -1
        fi
    else
        echo "err call check_build_ninja_python_api"
        exit -1
    fi
    echo "try check python INCLUDE_KEYWORD: ${INCLUDE_KEYWORD} is invalid in ninja.build or not"

    NINJA_BUILD=`cat build.ninja`
    for PYTHON_API_INCLUDE in ${PYTHON_API_INCLUDES}
    do
        echo "check PYTHON_API_INCLUDE vs INCLUDE_KEYWORD : (${PYTHON_API_INCLUDE} : ${INCLUDE_KEYWORD})"
        if [ ${PYTHON_API_INCLUDE} = ${INCLUDE_KEYWORD} ]; then
            if [[ "${NINJA_BUILD}" =~ ${PYTHON_API_INCLUDE} ]]; then
                echo "hit INCLUDE_KEYWORD: ${INCLUDE_KEYWORD} in build.ninja"
                IS_MINOR_HIT="TRUE"
            else
                echo "Err happened can not find INCLUDE_KEYWORD: ${INCLUDE_KEYWORD} in build.ninja"
                exit -1
            fi
        else
            if [[ "${NINJA_BUILD}" =~ ${PYTHON_API_INCLUDE} ]]; then
                echo "Err happened: find PYTHON_API_INCLUDE: ${PYTHON_API_INCLUDE} in build.ninja"
                echo "But now INCLUDE_KEYWORD: ${INCLUDE_KEYWORD}"
                exit -1
            fi
        fi
    done

    if [ ${IS_MINOR_HIT} = "FALSE" ]; then
        echo "Err happened, can not hit any MINOR api in ninja.build"
        exit -1
    fi
}

function check_python_version_is_valid() {
    want_build_version=$1
    support_version=$2
    if [ $# -eq 2 ]; then
        ver=$1
    else
        echo "err call check_python_version_is_valid"
        exit -1
    fi
    for i_want_build_version in ${want_build_version}
    do
        is_valid="false"
        for i_support_version in ${support_version}
        do
            if [ ${i_want_build_version} == ${i_support_version} ];then
                is_valid="true"
            fi
        done
        if [ ${is_valid} == "false" ];then
            echo "invalid build python version : \"${want_build_version}\", now support party of \"${support_version}\""
            exit -1
        fi
    done
}

function check_cuda_cudnn_trt_version() {
    # check cuda/cudnn/trt version
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
        TENSORRT_VERSION_CONTEXT=$(tail -20 ${TENSORRT_VERSION_PATH})

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
        if CUBLAS_VERSION_BUILD=$(echo $CUBLAS_VERSION_CONTEXT | grep -Eo "define CUBLAS_VER_BUILD * +([0-9]+)" | grep -Eo "*+([0-9]+)"); then
            CUBLAS_VERSION=${CUBLAS_VERSION_MAJOR}.${CUBLAS_VERSION_MINOR}.${CUBLAS_VERSION_PATCH}.${CUBLAS_VERSION_BUILD}
        else
            CUBLAS_VERSION=${CUBLAS_VERSION_MAJOR}.${CUBLAS_VERSION_MINOR}.${CUBLAS_VERSION_PATCH}
        fi
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
}
