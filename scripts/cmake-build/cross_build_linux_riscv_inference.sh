#!/usr/bin/env bash
set -e

ARCHS=("rv64gcv0p7" "rv64norvv")
BUILD_TYPE=Release
ARCH=rv64gcv0p7
REMOVE_OLD_BUILD=false
NINJA_VERBOSE=OFF
NINJA_DRY_RUN=OFF
SPECIFIED_TARGET="install/strip"
READLINK=readlink
OS=$(uname -s)

if [ $OS = "Darwin" ];then
    READLINK=greadlink
fi

SRC_DIR=$($READLINK -f "`dirname $0`/../../")
source $SRC_DIR/scripts/cmake-build/utils/utils.sh
config_ninja_default_max_jobs

echo "EXTRA_CMAKE_ARGS: ${EXTRA_CMAKE_ARGS}"

function usage() {
    echo "$0 args1 args2 .."
    echo "available args detail:"
    echo "-d : Build with Debug mode, default Release mode"
    echo "-a : config build arch available: ${ARCHS[@]}"
    echo "-r : remove old build dir before make, default off"
    echo "-v : ninja with verbose and explain, default off"
    echo "-n : ninja with -n dry run (don't run commands but act like they succeeded)"
    echo "-j : run N jobs in parallel for ninja, defaut is cpu_number + 2"
    echo "-e : build a specified target (always for debug, NOTICE: do not do strip/install target when use -e)"
    echo "-l : list CMakeLists.txt all options, can be use to config EXTRA_CMAKE_ARGS"
    echo "-h : show usage"
    echo "append other cmake config by config EXTRA_CMAKE_ARGS, for example, enable MGE_WITH_TEST and build with Debug mode:"
    echo "EXTRA_CMAKE_ARGS=\"-DMGE_WITH_TEST=ON\" $0 -d"
    exit -1
}

while getopts "lnvrhda:e:j:" arg
do
    case $arg in
        j)
            NINJA_MAX_JOBS=$OPTARG
            echo "config NINJA_MAX_JOBS to ${NINJA_MAX_JOBS}"
            ;;
        l)
            echo "list CMakeLists.txt all options, can be used to config EXTRA_CMAKE_ARGS"
            show_cmakelist_options
            exit 0
            ;;
        d)
            echo "Build with Debug mode"
            BUILD_TYPE=Debug
            ;;
        a)
            tmp_arch=null
            for arch in ${ARCHS[@]}; do
                if [ "$arch" = "$OPTARG" ]; then
                    echo "CONFIG BUILD ARCH to : $OPTARG"
                    tmp_arch=$OPTARG
                    ARCH=$OPTARG
                    break
                fi
            done
            if [ "$tmp_arch" = "null" ]; then
                echo "ERR args for arch (-a)"
                echo "available arch list: ${ARCHS[@]}"
                usage
            fi
            ;;
        h)
            echo "show usage"
            usage
            ;;
        r)
            echo "config REMOVE_OLD_BUILD=true"
            REMOVE_OLD_BUILD=true
            ;;
        v)
            echo "config NINJA_VERBOSE=ON"
            NINJA_VERBOSE=ON
            ;;
        n)
            echo "config NINJA_DRY_RUN=ON"
            NINJA_DRY_RUN=ON
            ;;
        e)
            SPECIFIED_TARGET=$OPTARG
            ;;
        ?)
            echo "unkonw argument"
            usage
            ;;
    esac
done
echo "----------------------------------------------------"
echo "build config summary:"
echo "BUILD_TYPE: $BUILD_TYPE"
echo "SPECIFIED_TARGET: ${SPECIFIED_TARGET}"
echo "NINJA_MAX_JOBS: ${NINJA_MAX_JOBS}"
echo "ARCH: $ARCH"
echo "----------------------------------------------------"

if [ ! $OS = "Linux" ];then
    echo "cross build for riscv only support from Linux"
    exit -1
fi

function cmake_build() {
    BUILD_DIR=$SRC_DIR/build_dir/riscv-linux/$1/$BUILD_TYPE/build
    INSTALL_DIR=$BUILD_DIR/../install
    TOOLCHAIN=$SRC_DIR/toolchains/$2
    echo "build dir: $BUILD_DIR"
    echo "install dir: $INSTALL_DIR"
    echo "build type: $BUILD_TYPE"
    echo "build toolchain: $TOOLCHAIN"
    try_remove_old_build $REMOVE_OLD_BUILD $BUILD_DIR $INSTALL_DIR

    echo "create build dir"
    mkdir -p $BUILD_DIR
    mkdir -p $INSTALL_DIR
    cd_real_build_dir $BUILD_DIR
    bash -c "cmake -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DMGE_INFERENCE_ONLY=ON \
        -DMGE_WITH_CUDA=OFF \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        ${EXTRA_CMAKE_ARGS} \
        $SRC_DIR "

    config_ninja_target_cmd ${NINJA_VERBOSE} "OFF" "${SPECIFIED_TARGET}" ${NINJA_DRY_RUN} ${NINJA_MAX_JOBS}
    bash -c "${NINJA_CMD}"
}

build_flatc $SRC_DIR $REMOVE_OLD_BUILD

toolchain=null
if [ "$ARCH" = "rv64gcv0p7" ]; then
    toolchain="riscv64-rvv-linux-gnu.toolchain.cmake"
elif [ "$ARCH" = "rv64norvv" ]; then
    toolchain="riscv64-linux-gnu.toolchain.cmake"
else
    echo "ERR CONFIG ABORT NOW!!"
    exit -1
fi
cmake_build $ARCH $toolchain
