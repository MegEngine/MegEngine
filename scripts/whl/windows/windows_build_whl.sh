#!/bin/bash -e

NT=$(echo `uname` | grep "NT")
echo $NT
if [ -z "$NT" ];then
    echo "only run at windows bash env"
    echo "pls consider install bash-like tools, eg MSYS or git-cmd, etc"
    exit -1
fi

function err_env() {
    echo "check_env failed: pls refs ${SRC_DIR}/scripts/whl/BUILD_PYTHON_WHL_README.md to init env"
    exit -1
}
echo $EXTRA_CMAKE_FLAG
function append_path_env_and_check() {
    if [[ -z $VS_PATH ]]; then
        echo  "export vs2019 install path"
        export VS_PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Enterprise
    fi
    if [[ -z $LLVM_PATH ]]; then
        echo  "export LLVM install path"
        export LLVM_PATH=/c/Program\ Files/LLVM_12_0_1
    fi
}

append_path_env_and_check

SRC_DIR=$(READLINK -f "`dirname $0`/../../../")
source ${SRC_DIR}/scripts/whl/utils/utils.sh

ALL_PYTHON=${ALL_PYTHON}
FULL_PYTHON_VER="3.6.8 3.7.7 3.8.3 3.9.4 3.10.1"
if [[ -z ${ALL_PYTHON} ]]
then
    ALL_PYTHON=${FULL_PYTHON_VER}
else
    check_python_version_is_valid "${ALL_PYTHON}" "${FULL_PYTHON_VER}"
fi

PYTHON_DIR=
PYTHON_LIBRARY=
PYTHON_INCLUDE_DIR=
WINDOWS_WHL_HOME=${SRC_DIR}/scripts/whl/windows/windows_whl_home
if [[ -z $PYTHON_ROOT ]]; then
    export PYTHON_ROOT="/c/Users/${USER}/mge_whl_python_env"
fi

if [ -e "${WINDOWS_WHL_HOME}" ]; then
    echo "remove old windows whl file"
    rm -rf ${WINDOWS_WHL_HOME}
fi
mkdir -p ${WINDOWS_WHL_HOME}

function config_python_env() {
    PYTHON_DIR=$PYTHON_ROOT/$1
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

if [[ -z ${CUDA_ROOT_DIR} ]]; then
    export CUDA_ROOT_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1"
fi

if [[ -z ${CUDNN_ROOT_DIR} ]]; then
    export CUDNN_ROOT_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/cudnn-10.1-windows10-x64-v7.6.5.32/cuda"
fi

if [[ -z ${TRT_ROOT_DIR} ]]; then
    export TRT_ROOT_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-6.0.1.5"
fi

# config NVIDIA libs
TRT_LIBS=`ls $TRT_ROOT_DIR/lib/nvinfer*.dll`
if [[ $TRT_VERSION == "7.2.3.4" ]]; then
    MYELIN_LIB=`ls $TRT_ROOT_DIR/lib/myelin64_*.dll`
fi
CUDNN_LIBS=`ls $CUDNN_ROOT_DIR/bin/cudnn*.dll`
CUSOLVER_LIB=`ls $CUDA_ROOT_DIR/bin/cusolver64_*.dll`
CUBLAS_LIB=`ls $CUDA_ROOT_DIR/bin/cublas64_*.dll`
CURAND_LIB=`ls $CUDA_ROOT_DIR/bin/curand64_*.dll`
CUBLASLT_LIB=`ls $CUDA_ROOT_DIR/bin/cublasLt64_*.dll`
CUDART_LIB=`ls $CUDA_ROOT_DIR/bin/cudart64_*.dll`
if [[ $TRT_VERSION == 7.2.3.4 ]]; then
    NVTRC_LIB=`ls $CUDA_ROOT_DIR/bin/nvrtc64_111_0.dll`
else
    NVTRC_LIB=`ls $CUDA_ROOT_DIR/bin/nvrtc64_*.dll`
fi

if [[ $SDK_NAME == "cu118" ]]; then
    ZLIBWAPI=`ls $CUDA_ROOT_DIR/bin/zlibwapi.dll`
fi
# CUDART_LIB="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin/cudart64_101.dll"
MGE_EXPORT_DLL="${SRC_DIR}/build_dir/host/build/src/megengine_shared.dll"
MGE_EXPORT_LIB="${SRC_DIR}/build_dir/host/build/src/megengine_shared.lib"

function depend_real_copy() {
    REAL_DST=$1
    echo "real copy lib to $1"
    cp "${MGE_EXPORT_DLL}" ${REAL_DST}
    cp "${MGE_EXPORT_LIB}" ${REAL_DST}

    if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
        echo "copy nvidia lib...."
        for TRT_LIB in $TRT_LIBS
        do
            echo "Copy ${TRT_LIB} to ${REAL_DST}"
            cp "${TRT_LIB}" ${REAL_DST}
        done
        if [[ ! -z $MYELIN_LIB ]]; then
            cp "$MYELIN_LIB" ${REAL_DST} 
        fi
        for CUDNN_LIB in $CUDNN_LIBS
        do
            echo "Copy ${CUDNN_LIB} to ${REAL_DST}"
            cp "${CUDNN_LIB}" ${REAL_DST}
        done
        cp "${CUSOLVER_LIB}" ${REAL_DST}
        cp "${CUBLAS_LIB}" ${REAL_DST}
        cp "${CURAND_LIB}" ${REAL_DST}
        cp "${CUBLASLT_LIB}" ${REAL_DST}
        cp "${CUDART_LIB}" ${REAL_DST}
        if [[ ! -z ${NVTRC_LIB} ]]; then
          for lib in ${NVTRC_LIB} 
          do
            echo "Copy ${lib} to ${REAL_DST}"
            cp "${lib}" ${REAL_DST}
          done
        fi
        
        if [[ ! -z ${ZLIBWAPI} ]]; then
            echo "Copy ${ZLIBWAPI} to ${REAL_DST}"
            cp "${ZLIBWAPI}" ${REAL_DST} 
        fi
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
if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
    if [[ -z $SDK_NAME ]]; then
        export SDK_NAME="cu101"
    fi
else
    export SDK_NAME="cpu"
fi
third_party_prepare
do_build