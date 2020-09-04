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

function append_path_env_and_check() {
    echo  "export vs2019 install path"
    export VS_PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Enterprise
    # for llvm-strip
    export PATH=$VS_PATH/VC/Tools/Llvm/bin/:$PATH
}

append_path_env_and_check

SRC_DIR=$(READLINK -f "`dirname $0`/../../../")
ALL_PYTHON=${ALL_PYTHON}
FULL_PYTHON_VER="3.5.4 3.6.8 3.7.7 3.8.3"
if [[ -z ${ALL_PYTHON} ]]
then
    ALL_PYTHON=${FULL_PYTHON_VER}
fi

PYTHON_DIR=
PYTHON_LIBRARY=
PYTHON_INCLUDE_DIR=
WINDOWS_WHL_HOME=${SRC_DIR}/scripts/whl/windows/windows_whl_home
if [ -e "${WINDOWS_WHL_HOME}" ]; then
    echo "remove old windows whl file"
    rm -rf ${WINDOWS_WHL_HOME}
fi
mkdir -p ${WINDOWS_WHL_HOME}

function config_python_env() {
    PYTHON_DIR=/c/Users/${USER}/mge_whl_python_env/$1
    PYTHON_BIN=${PYTHON_DIR}
    if [ ! -f "$PYTHON_BIN/python3.exe" ]; then
        echo "ERR: can not find $PYTHON_BIN , Invalid python package"
        echo "now support list: ${FULL_PYTHON_VER}"
        err_env
    else
        echo "put python3 to env..."
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


# config NVIDIA libs
TRT_LIB="/c/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-6.0.1.5/lib/nvinfer.dll"
CUDNN_LIB="/c/Program Files/NVIDIA GPU Computing Toolkit/cudnn-10.1-windows10-x64-v7.6.5.32/cuda/bin/cudnn64_7.dll"
CUSOLVER_LIB="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin/cusolver64_10.dll"
CUBLAS_LIB="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin/cublas64_10.dll"
CURAND_LIB="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin/curand64_10.dll"
CUBLASLT_LIB="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin/cublasLt64_10.dll"
CUDART_LIB="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin/cudart64_101.dll"
function depend_real_copy() {
    REAL_DST=$1
    echo "real copy lib to $1"
    cp "${TRT_LIB}" ${REAL_DST}
    cp "${CUDNN_LIB}" ${REAL_DST}
    cp "${CUSOLVER_LIB}" ${REAL_DST}
    cp "${CUBLAS_LIB}" ${REAL_DST}
    cp "${CURAND_LIB}" ${REAL_DST}
    cp "${CUBLASLT_LIB}" ${REAL_DST}
    cp "${CUDART_LIB}" ${REAL_DST}
}

function copy_more_dll() {
    # for python whl real use
    echo "config BUILD_IMPERATIVE core lib dir"
    CP_WHL_DST_IMP=${BUILD_DIR}/staging/megengine/core/lib
    rm -rf ${CP_WHL_DST_IMP}
    mkdir ${CP_WHL_DST_IMP}

    # workround for cpu-only version import failed, use a
    # empty.file to triger setup.py to create a null empty
    echo "empty" > ${CP_WHL_DST_IMP}/empty.file


    if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
        echo "copy nvidia lib to whl use...."
        depend_real_copy ${CP_WHL_DST_IMP}

    fi
}

function do_build() {
    for ver in ${ALL_PYTHON}
    do
        #config
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
        #append cmake args for config python
        #FIXME: ninja handle err with cmake 3.17 when assgin PYTHON_LIBRARY
        #But after put python3.exe to HEAD of PATH by config_python_env, cmake can also handle the
        #right PYTHON_LIBRARY and PYTHON_INCLUDE_DIR, at the same time, clang-cl need swig target
        #force LINK a real PYTHON_LIBRARY file, after test we do not find the symbols conflict with python
        #export EXTRA_CMAKE_ARGS="-DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} "
        #config build type to RelWithDebInfo to enable MGB_ENABLE_DEBUG_UTIL etc
        export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=RelWithDebInfo "

        #call build and install
        #FIXME: cmake do not triger update python config, after
        #change PYTHON_LIBRARY and PYTHON_INCLUDE_DIR, so add
        #-r to remove build cache after a new ver build, which
        #will be more slow build than without -r
        BUILD_ARGS=" -t -r"

        if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
            echo "build windows whl with cuda"
            BUILD_ARGS="${BUILD_ARGS} -c "
        else
            echo "build windows whl with cpu only"
        fi

        echo "host_build.sh BUILD_ARGS: ${BUILD_ARGS}"
        ${SRC_DIR}/scripts/cmake-build/host_build.sh ${BUILD_ARGS}

        BUILD_DIR=${SRC_DIR}/build_dir/host/build/
        cd ${BUILD_DIR}

        rm -rf staging
        mkdir -p staging
        cp -a imperative/python/{megengine,setup.py,requires.txt,requires-style.txt,requires-test.txt} staging/
        cd ${BUILD_DIR}/staging/megengine/core
        rt_file=`ls _imperative_rt.*.pyd`
        echo "rt file is: ${rt_file}"
        if [[ -z ${rt_file} ]]
        then
            echo "ERR: can not find valid rt file"
            exit -1
        fi
        llvm-strip -s ${rt_file}
        mv ${rt_file} _imperative_rt.pyd


        copy_more_dll
        cd ${BUILD_DIR}/staging
        echo "call setup.py now"
        ${PYTHON_DIR}/python3 setup.py bdist_wheel
        cp ${BUILD_DIR}/staging/dist/Meg*.whl ${WINDOWS_WHL_HOME}/

        echo ""
        echo "##############################################################################################"
        echo "windows whl package location: ${WINDOWS_WHL_HOME}"
        ls ${WINDOWS_WHL_HOME}
        echo "##############################################################################################"
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
