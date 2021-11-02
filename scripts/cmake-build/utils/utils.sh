#!/usr/bin/env bash
set -e

OS=$(uname -s)
NINJA_CMD=""
NINJA_BASE="ninja"
cpu_number=`nproc`
NINJA_MAX_JOBS=0

if [[ $OS =~ "NT" ]]; then
    echo "BUILD in NT ..."
    NINJA_BASE="Ninja"
fi

READLINK=readlink
if [ $OS = "Darwin" ];then
    READLINK=greadlink
fi

PROJECT_DIR=$(dirname "${BASH_SOURCE[0]}")/../../..
function cd_real_build_dir() {
    REAL_DIR=$($READLINK -f $1)
    echo "may alias dir: $1"
    echo "cd real build dir: ${REAL_DIR}"
    cd ${REAL_DIR}
}

function build_flatc() {
    BUILD_DIR=$1/build_dir/host_flatc/build
    INSTALL_DIR=$BUILD_DIR/../install
    REMOVE_OLD_BUILD=$2

    if [ $REMOVE_OLD_BUILD = "true" ]; then
        echo "remove old build/install dir"
        rm -rf $INSTALL_DIR
        rm -rf $BUILD_DIR
    else
        echo "strip remove old build"
    fi

    mkdir -p $BUILD_DIR
    mkdir -p $INSTALL_DIR

    cd_real_build_dir $BUILD_DIR
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DFLATBUFFERS_BUILD_TESTS=OFF \
        -DFLATBUFFERS_BUILD_FLATHASH=OFF \
        -DFLATBUFFERS_BUILD_FLATLIB=OFF \
        -DFLATBUFFERS_LIBCXX_WITH_CLANG=OFF \
        ${PROJECT_DIR}/third_party/flatbuffers

    ${NINJA_BASE} all
    ${NINJA_BASE} install/strip
}

function try_remove_old_build() {
    REMOVE_OLD_BUILD=$1
    echo $REMOVE_OLD_BUILD
    BUILD_DIR=$2
    INSTALL_DIR=$3

    if [ $REMOVE_OLD_BUILD = "true" ]; then
        echo "remove old build/install dir"
        rm -rf ${INSTALL_DIR}
        rm -rf ${BUILD_DIR}
    else
        echo "strip remove old build"
    fi
}

function config_ninja_target_cmd() {
    if [ $# -eq 5 ]; then
        _NINJA_VERBOSE=$1
        _BUILD_DEVELOP=$2
        _NINJA_TARGET=$3
        _NINJA_DRY_RUN=$4
        _NINJA_MAX_JOBS=$5
    else
        echo "err call config_ninja_target_cmd"
        exit -1
    fi
    if [ -z "${_NINJA_TARGET}" ]; then
        NINJA_CMD="${NINJA_BASE} all -j ${_NINJA_MAX_JOBS}"
    else
        NINJA_CMD="${NINJA_BASE} ${_NINJA_TARGET} -j ${_NINJA_MAX_JOBS}"
    fi

    if [ ${_NINJA_DRY_RUN} = "ON" ]; then
        if [[ "${NINJA_CMD}" =~ "&" ]]; then
            echo "code issue happened!!! base cmd can not include & before ninja explain"
            echo "now cmd: ${NINJA_CMD}"
            exit -1
        fi
        NINJA_CMD="${NINJA_CMD} -d explain -n"
    else
        if [ ${_NINJA_VERBOSE} = "ON" ]; then
            if [[ "${NINJA_CMD}" =~ "&" ]]; then
                echo "code issue happened!!! base cmd can not include & before ninja explain"
                echo "now cmd: ${NINJA_CMD}"
                exit -1
            fi
            NINJA_CMD="${NINJA_CMD} -d explain -v"
        fi
        if [ ${_BUILD_DEVELOP} = "ON" ]; then
            echo "add develop target"
            NINJA_CMD="${NINJA_CMD} && ${NINJA_BASE} develop"
        fi
    fi

    echo "build ${NINJA_BASE} target command: ${NINJA_CMD}"
}

function show_cmakelist_options() {
    cd ${PROJECT_DIR}
    grep "option(" lite/CMakeLists.txt CMakeLists.txt --color
    cd - > /dev/null
}

function config_ninja_default_max_jobs() {
    # plus 2 is ninja default behavior, you can run ninja -h to verify
    # but at Windows env, default max jobs will take 100% cpu, which may lead
    # to some Windows OS issue sometimes, eg, OpenSSH server lost respond or vcvarsall.bat
    # setenv failed etc(especially enable CUDA). I have no idea about this Windows OS issue.
    # as a workaround: config default NINJA_MAX_JOBS to cpu_number - 1
    if [[ $OS =~ "NT" ]]; then
        ((NINJA_MAX_JOBS = ${cpu_number} - 1))
        if [[ ${NINJA_MAX_JOBS} -le 0 ]]; then
            NINJA_MAX_JOBS=1
        fi
    else
        ((NINJA_MAX_JOBS = ${cpu_number} + 2))
    fi
    echo "config default NINJA_MAX_JOBS to ${NINJA_MAX_JOBS} [cpu number is:${cpu_number}]"
}
