#!/usr/bin/env bash
set -e

NT=$(echo `uname` | grep "NT")
echo $NT
if [ -z "$NT" ];then
    echo "only run at windows bash env"
    echo "pls consider install bash-like tools, eg MSYS or git-cmd, etc"
    exit -1
fi

SRC_DIR=$(READLINK -f "`dirname $0`/../../../")
source ${SRC_DIR}/scripts/whl/utils/utils.sh
source ${SRC_DIR}/scripts/whl/windows/config.sh

function err_env() {
    echo "check_env failed: pls call ${SRC_DIR}/scripts/whl/windows/env_prepare.sh to init env"
    exit -1
}

SDK_NAME="unknown"
x86_64_support_version="cpu cu101 cu118"

if [[ -z ${IN_CI} ]]
then
    IN_CI="false"
fi

function usage() {
    echo "use -sdk sdk_version to specify sdk toolkit config!"
    echo "now x86_64 sdk_version support ${x86_64_support_version}"
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

# export setup.py local version
export SDK_NAME=${SDK_NAME}

# TODO: Windows CI take a long time, we have no enough resource to test
# so only build one sm to speed build. after have enough resource, remove this
# now we test at 1080TI remote env, so config sm to 61
if [ ${IN_CI} = "true" ] ; then
    EXTRA_CMAKE_FLAG=" -DMGE_CUDA_GENCODE=\"-gencode arch=compute_61,code=sm_61\" "
fi

CUDA_LIBS="not_find"
CUDNN_LIBS="not_find"
TRT_LIBS="not_find"
MGE_EXPORT_DLL="${SRC_DIR}/build_dir/host/build/src/megengine_shared.dll"
MGE_EXPORT_LIB="${SRC_DIR}/build_dir/host/build/src/megengine_shared.lib"

if [ $SDK_NAME == "cu101" ];then
    REQUIR_CUDA_VERSION="10010"
    REQUIR_CUDNN_VERSION="7.6.5"
    REQUIR_TENSORRT_VERSION="6.0.1.5"
    REQUIR_CUBLAS_VERSION="10.1.0"
    CUDA_ROOT_DIR=${CUDA_ROOT_DIR_101}
    CUDNN_ROOT_DIR=${CUDNN_ROOT_DIR_101}
    TRT_ROOT_DIR=${TRT_ROOT_DIR_101}
    TENSORRT_ROOT_DIR=${TRT_ROOT_DIR}

    CUDA_LIBS="${CUDA_ROOT_DIR}/bin/cusolver64_10.dll:${CUDA_ROOT_DIR}/bin/cublas64_10.dll\
        :${CUDA_ROOT_DIR}/bin/curand64_10.dll:${CUDA_ROOT_DIR}/bin/cublasLt64_10.dll\
        :${CUDA_ROOT_DIR}/bin/cudart64_101.dll"

    CUDNN_LIBS="${CUDNN_ROOT_DIR}/bin/cudnn64_7.dll"

    TRT_LIBS="${TRT_ROOT_DIR}/lib/nvinfer.dll:${TRT_ROOT_DIR}/lib/nvinfer_plugin.dll"


elif [ $SDK_NAME == "cu118" ];then
    REQUIR_CUDA_VERSION="11080"
    REQUIR_CUDNN_VERSION="8.6.0"
    REQUIR_TENSORRT_VERSION="8.5.3.1"
    REQUIR_CUBLAS_VERSION="11.11.3.6"
    CUDA_ROOT_DIR=${CUDA_ROOT_DIR_118}
    CUDNN_ROOT_DIR=${CUDNN_ROOT_DIR_118}
    TRT_ROOT_DIR=${TRT_ROOT_DIR_118}
    TENSORRT_ROOT_DIR=${TRT_ROOT_DIR}

    CUDA_LIBS="${CUDA_ROOT_DIR}/bin/cusolver64_11.dll:${CUDA_ROOT_DIR}/bin/cublas64_11.dll\
        :${CUDA_ROOT_DIR}/bin/curand64_10.dll:${CUDA_ROOT_DIR}/bin/cublasLt64_11.dll\
        :${CUDA_ROOT_DIR}/bin/cudart64_110.dll:${CUDA_ROOT_DIR}/bin/nvrtc64_112_0.dll"

    CUDNN_LIBS="${CUDNN_ROOT_DIR}/bin/cudnn64_8.dll:${CUDNN_ROOT_DIR}/bin/cudnn_cnn_infer64_8.dll\
        :${CUDNN_ROOT_DIR}/bin/cudnn_ops_train64_8.dll:${CUDNN_ROOT_DIR}/bin/cudnn_adv_infer64_8.dll\
        :${CUDNN_ROOT_DIR}/bin/cudnn_cnn_train64_8.dll:${CUDNN_ROOT_DIR}/bin/cudnn_adv_train64_8.dll\
        :${CUDNN_ROOT_DIR}/bin/cudnn_ops_infer64_8.dll:${CUDNN_ROOT_DIR}/bin/zlibwapi.dll"

    # workround for CU118 depends on zlibwapi.dll
    if [[ ! -f ${CUDNN_ROOT_DIR}/bin/zlibwapi.dll ]]; then
        echo "can not find zlibwapi.dll, download from ${ZLIBWAPI_URL}"
        rm -rf tttttmp_1988
        mkdir -p tttttmp_1988
        cd tttttmp_1988
        curl -SL ${ZLIBWAPI_URL} --output zlib123dllx64.zip
        unzip -X zlib123dllx64.zip
        cp dll_x64/zlibwapi.dll ${CUDNN_ROOT_DIR}/bin/
        cd ..
        rm -rf tttttmp_1988
        # double check
        if [[ ! -f ${CUDNN_ROOT_DIR}/bin/zlibwapi.dll ]]; then
            echo "some issue happened when prepare zlibwapi.dll, please fix me!!!!"
            exit -1
        fi
    fi

    TRT_LIBS="${TRT_ROOT_DIR}/lib/nvinfer.dll:${TRT_ROOT_DIR}/lib/nvinfer_builder_resource.dll:\
        ${TRT_ROOT_DIR}/lib/nvinfer_plugin.dll"

elif [ $SDK_NAME == "cpu" ];then
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

if [ $SDK_NAME == "cpu" ];then
    echo "use $SDK_NAME without cuda support"
else
    echo "CUDA_LIBS: $CUDA_LIBS"
    echo "CUDNN_LIBS: $CUDNN_LIBS"
    echo "TRT_LIBS: $TRT_LIBS"
    # for utils.sh sub bash script, eg host_build.sh
    export CUDA_ROOT_DIR=${CUDA_ROOT_DIR}
    export CUDNN_ROOT_DIR=${CUDNN_ROOT_DIR}
    export TRT_ROOT_DIR=${TRT_ROOT_DIR}
fi

check_cuda_cudnn_trt_version
check_python_version_is_valid "${ALL_PYTHON}" "${FULL_PYTHON_VER}"

PYTHON_DIR=
PYTHON_LIBRARY=
PYTHON_INCLUDE_DIR=
WINDOWS_WHL_HOME=${SRC_DIR}/scripts/whl/windows/windows_whl_home/${SDK_NAME}
if [ -e "${WINDOWS_WHL_HOME}" ]; then
    echo "remove old windows whl file"
    rm -rf ${WINDOWS_WHL_HOME}
fi
mkdir -p ${WINDOWS_WHL_HOME}

function config_python_env() {
    PYTHON_DIR=${DFT_PYTHON_BIN}/../$1
    PYTHON_BIN=${PYTHON_DIR}
    if [ ! -f "${PYTHON_BIN}/python3.exe" ]; then
        echo "ERR: can not find $PYTHON_BIN , Invalid python package"
        echo "now support list: ${FULL_PYTHON_VER}"
        err_env
    else
        echo "put ${PYTHON_BIN}/python3.exe to env..."
        export PATH=${PYTHON_BIN}:$PATH
        which python3
    fi
    echo ${ver}

    PYTHON_LIBRARY=${PYTHON_DIR}/libs/python3.lib
    PYTHON_INCLUDE_DIR=${PYTHON_DIR}/include
}

BUILD_WHL_CPU_ONLY=${BUILD_WHL_CPU_ONLY}
if [[ -z ${BUILD_WHL_CPU_ONLY} ]]
then
    BUILD_WHL_CPU_ONLY="OFF"
fi

function depend_real_copy() {
    REAL_DST=$1
    echo "real copy lib to $1"
    cp "${MGE_EXPORT_DLL}" ${REAL_DST}
    cp "${MGE_EXPORT_LIB}" ${REAL_DST}

    if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
        echo "copy nvidia lib...."

        IFS=: read -a lib_name_array <<<"${TRT_LIBS}:${CUDNN_LIBS}:${CUDA_LIBS}"
        for lib_name in ${lib_name_array[@]};
        do
            echo "Copy ${lib_name} to ${REAL_DST}"
            cp ${lib_name} ${REAL_DST}
        done
    fi
}

function copy_more_dll() {
    # for python whl real use
    echo "config BUILD_IMPERATIVE core lib dir"
    CP_WHL_DST_IMP=${BUILD_DIR}/staging/megengine/core/lib
    rm -rf ${CP_WHL_DST_IMP}
    mkdir ${CP_WHL_DST_IMP}

    depend_real_copy ${CP_WHL_DST_IMP}
}

function lite_copy_more_dll() {
    if [ ${IN_CI} = "true" ]; then
        echo "copy lib for lite for ci test"
        IMP_TEST_DST=${SRC_DIR}/build_dir/host/build/lite/test/
        depend_real_copy ${IMP_TEST_DST}
        rm "${IMP_TEST_DST}/megengine_shared.dll"
    fi
}

BUILD_DIR=${SRC_DIR}/build_dir/host/build/

# here we just treat cu file should not in the increment build file list
INCREMENT_KEY_WORDS=".cu.obj is dirty"
IS_IN_FIRST_LOOP=TRUE

ORG_EXTRA_CMAKE_FLAG=${EXTRA_CMAKE_FLAG}
function do_build() {
    for ver in ${ALL_PYTHON}
    do
        # we want run a full clean build at the first loop
        if [ ${IS_IN_FIRST_LOOP} = "TRUE" ]; then
            # TODO: may all cmake issue can be resolved after rm CMakeCache?
            # if YES, remove this to use old cache and speed up CI
            echo "warning: remove old build_dir for the first loop"
            rm -rf ${BUILD_DIR}
        fi

        #config python3
        config_python_env ${ver}

        #check env
        if [ ! -f "$PYTHON_LIBRARY" ]; then
            echo "ERR: can not find $PYTHON_LIBRARY , Invalid python package"
            err_env
        fi
        if [ ! -d "$PYTHON_INCLUDE_DIR" ]; then
            echo "ERR: can not find $PYTHON_INCLUDE_DIR , Invalid python package"
            err_env
        fi
        echo "PYTHON_LIBRARY: ${PYTHON_LIBRARY}"
        echo "PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}"
        #config build type to RelWithDebInfo to enable MGB_ENABLE_DEBUG_UTIL etc
        export EXTRA_CMAKE_ARGS="${ORG_EXTRA_CMAKE_FLAG} -DCMAKE_BUILD_TYPE=RelWithDebInfo "
        export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DMGE_WITH_CUSTOM_OP=ON"

        #call build and install
        HOST_BUILD_ARGS=" -t -s"

        if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
            echo "build windows whl with cuda"
            HOST_BUILD_ARGS="${HOST_BUILD_ARGS} -c "
        else
            echo "build windows whl with cpu only"
        fi

        if [ -d "${BUILD_DIR}" ]; then
            # insure rm have args
            touch ${BUILD_DIR}/empty.pyd
            touch ${BUILD_DIR}/CMakeCache.txt
            /usr/bin/find ${BUILD_DIR} -name "*.pyd" | xargs rm
            # ninja/cmake on windows will handle error if just export
            # PYTHON_LIBRARY/PYTHON_INCLUDE_DIR/PYTHON_EXECUTABLE
            # But after put python3.exe to HEAD of PATH by config_python_env
            # and force remove CMakeCache.txt, ninja/cmake will auto update
            # PYTHON_LIBRARY/PYTHON_INCLUDE_DIR/PYTHON_EXECUTABLE
            /usr/bin/find ${BUILD_DIR} -name CMakeCache.txt | xargs rm
        fi
        echo "host_build.sh HOST_BUILD_ARGS: ${HOST_BUILD_ARGS}"

        # call ninja dry run and check increment is invalid or not
        if [ ${IS_IN_FIRST_LOOP} = "FALSE" ]; then
            ninja_dry_run_and_check_increment "${SRC_DIR}/scripts/cmake-build/host_build.sh" "${HOST_BUILD_ARGS}" "${INCREMENT_KEY_WORDS}"
        fi

        #call real build
        ${SRC_DIR}/scripts/cmake-build/host_build.sh ${HOST_BUILD_ARGS}
        # remove megenginelite py develop soft link create by lite_shared:POST_BUILD @ lite/CMakeLists.txt
        rm -rf ${SRC_DIR}/lite/pylite/megenginelite/libs

        # check python api call setup.py
        cd ${BUILD_DIR}
        check_build_ninja_python_api ${ver}
        rm -rf staging
        mkdir -p staging
        cp -a imperative/python/{megengine,setup.py,requires.txt,requires-style.txt,requires-test.txt} staging/
        cp -a ${SRC_DIR}/src/custom/include/megbrain staging/megengine/core/include/
        cd ${BUILD_DIR}/staging/megengine/core
        rt_file=`ls _imperative_rt.*.pyd`
        echo "rt file is: ${rt_file}"
        if [[ -z ${rt_file} ]]
        then
            echo "ERR: can not find valid rt file"
            exit -1
        fi
        mv ${rt_file} _imperative_rt.pyd

        copy_more_dll

        # handle megenginelite
        cd ${BUILD_DIR}
        mkdir -p staging/megenginelite
        cp ${SRC_DIR}/lite/pylite/megenginelite/* staging/megenginelite/
        LITE_CORE_LIB_DIR=${BUILD_DIR}/staging/megenginelite/libs/
        mkdir -p ${LITE_CORE_LIB_DIR}
        cd ${LITE_CORE_LIB_DIR}
        cp ${BUILD_DIR}/lite/lite_shared_whl.dll liblite_shared_whl.pyd
        lite_copy_more_dll

        cd ${BUILD_DIR}/staging
        echo "call setup.py now"
        ${PYTHON_DIR}/python3 setup.py bdist_wheel
        cp ${BUILD_DIR}/staging/dist/Meg*.whl ${WINDOWS_WHL_HOME}/

        echo ""
        echo "##############################################################################################"
        echo "windows whl package location: ${WINDOWS_WHL_HOME}"
        ls ${WINDOWS_WHL_HOME}
        echo "##############################################################################################"
        IS_IN_FIRST_LOOP=FALSE
    done
}

function third_party_prepare() {
    echo "init third_party..."
    ${SRC_DIR}/third_party/prepare.sh

    if [[ -z ${ALREADY_INSTALL_MKL} ]]
    then
        echo "init third_party..."
        ${SRC_DIR}/third_party/install-mkl.sh
    else
        echo "skip init mkl internal"
    fi
}

######################
export ALREADY_CONFIG_PYTHON_VER="yes"
third_party_prepare
do_build
