#!/bin/bash -e

READLINK=readlink
OS=$(uname -s)

if [ $OS = "Darwin" ];then
    READLINK=greadlink
else
    echo "ERR: only run at macos env"
    exit -1
fi

function err_env() {
    echo "check_env failed: pls refs ${SRC_DIR}/scripts/whl/BUILD_PYTHON_WHL_README.md"
    echo "try call ./scripts/whl/macos/macos_whl_env_prepare.sh to init env"
    exit -1
}

function append_path_env_and_check() {
    export PATH=/usr/local/opt/findutils/libexec/gnubin:$PATH
    export PATH=/usr/local/opt/binutils/bin:$PATH
    export PATH=/usr/local/opt/llvm/bin:$PATH
    export PATH=/Users/${USER}/megengine_use_cmake/install/bin:$PATH
    if [ ! -f "/usr/local/opt/llvm/bin/llvm-strip" ]; then
        err_env
    fi

    which cmake
    if [ ! -f "/Users/${USER}/megengine_use_cmake/install/bin/cmake" ]; then
        err_env
    fi
}

append_path_env_and_check

SRC_DIR=$($READLINK -f "`dirname $0`/../../../")
ALL_PYTHON=${ALL_PYTHON}
FULL_PYTHON_VER="3.5.9 3.6.10 3.7.7 3.8.3"
if [[ -z ${ALL_PYTHON} ]]
then
    ALL_PYTHON=${FULL_PYTHON_VER}
fi

PYTHON_DIR=
PYTHON_LIBRARY=
PYTHON_INCLUDE_DIR=
MACOS_WHL_HOME=${SRC_DIR}/scripts/whl/macos/macos_whl_home
if [ -e "${MACOS_WHL_HOME}" ]; then
    echo "remove old macos whl file"
    rm -rf ${MACOS_WHL_HOME}
fi
mkdir -p ${MACOS_WHL_HOME}

function config_python_env() {
    PYTHON_DIR=/Users/${USER}/.pyenv/versions/$1/
    PYTHON_BIN=/Users/${USER}/.pyenv/versions/$1/bin
    if [ ! -f "$PYTHON_BIN/python3" ]; then
        echo "ERR: can not find $PYTHON_BIN , Invalid python package"
        echo "now support list: ${FULL_PYTHON_VER}"
        err_env
    else
        echo "put python3 to env..."
        export PATH=${PYTHON_BIN}:$PATH
        which python3
    fi
    echo ${ver}

    #config a dir to trick cmake find a null pythonlib
    PYTHON_LIBRARY=${PYTHON_DIR}lib/
    if [ "$1" = "3.5.9" ]; then
        PYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python3.5m
    elif [ "$1" = "3.6.10" ]; then
        PYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python3.6m
    elif [ "$1" = "3.7.7" ]; then
        PYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python3.7m
    elif [ "$1" = "3.8.3" ]; then
        PYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python3.8
    else
        echo "ERR: DO NOT SUPPORT PYTHON VERSION"
        echo "now support list: ${FULL_PYTHON_VER}"
        exit -1
    fi
}

function do_build() {
    for ver in ${ALL_PYTHON}
    do
        #config
        config_python_env ${ver}

        #check env
        if [ ! -d "$PYTHON_LIBRARY" ]; then
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
        export EXTRA_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${PYTHON_DIR} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} "
        #config build type to RelWithDebInfo to enable MGB_ENABLE_DEBUG_UTIL etc
        export EXTRA_CMAKE_ARGS=${EXTRA_CMAKE_ARGS}" -DCMAKE_BUILD_TYPE=RelWithDebInfo "

        #call build and install
        #FIXME: cmake do not triger update python config, after
        #change PYTHON_LIBRARY and PYTHON_INCLUDE_DIR, so add
        #-r to remove build cache after a new ver build, which
        #will be more slow build than without -r
        ${SRC_DIR}/scripts/cmake-build/host_build.sh -t -r

        #call setup.py
        BUILD_DIR=${SRC_DIR}/build_dir/host/MGE_WITH_CUDA_OFF/MGE_INFERENCE_ONLY_OFF/Release/build/
        cd ${BUILD_DIR}

        if [ -d "staging" ]; then
            echo "remove old build cache file"
            rm -rf staging
        fi
        mkdir -p staging


        cp -a python_module/{megengine,setup.py,requires.txt,requires-style.txt,requires-test.txt} staging/
        cd ${BUILD_DIR}/staging/megengine/_internal
        #FIXME: set lib suffix to dylib may be better, BUT we find after distutils.file_util.copy_file
        #will change to .so at macos even we set suffix to dylib, at the same time, macos also support .so
        llvm-strip -s _mgb.so
        cd ${BUILD_DIR}/staging
        ${PYTHON_DIR}/bin/python3 setup.py bdist_wheel
        cd ${BUILD_DIR}/staging/dist/
        org_whl_name=`ls Meg*.whl`
        index=`awk -v a="${org_whl_name}" -v b="-macosx" 'BEGIN{print index(a,b)}'`
        #compat for osx version from 10.5(Leopard)
        #FIXME: same no need at -macosx-version-min=10.5 for build so
        compat_whl_name=`echo ${org_whl_name} |cut -b -$index`macosx_10_5_x86_64.whl
        echo "org whl name: ${org_whl_name}"
        echo "comapt whl name: ${compat_whl_name}"
        cp ${BUILD_DIR}/staging/dist/Meg*.whl ${MACOS_WHL_HOME}/${compat_whl_name}
        cd ${SRC_DIR}

        echo ""
        echo "##############################################################################################"
        echo "macos whl package location: ${MACOS_WHL_HOME}"
        ls ${MACOS_WHL_HOME}
        echo "##############################################################################################"
    done
}

function third_party_prepare() {
    if [[ -z ${ALREADY_INSTALL_THIRD_PARTY} ]]
    then
        echo "init third_party..."
        ${SRC_DIR}/third_party/prepare.sh
        ${SRC_DIR}/third_party/install-mkl.sh
    else
        echo "skip init third_party..."
    fi
}

######################
third_party_prepare
do_build
