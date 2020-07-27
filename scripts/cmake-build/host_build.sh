#!/usr/bin/env bash
set -e

function usage() {
    echo "$0 args1 args2 .."
    echo "available args detail:"
    echo "-d : Build with Debug mode, default Release mode"
    echo "-c : Build with CUDA, default without CUDA"
    echo "-t : Build with training mode, default inference only"
    echo "-m : Build with m32 mode(only for windows build), default m64"
    echo "-r : remove old build dir before make, default off"
    echo "-h : show usage"
    echo "append other cmake config by export EXTRA_CMAKE_ARGS=..."
    echo "example: $0 -d"
    exit -1
}

BUILD_TYPE=Release
MGE_WITH_CUDA=OFF
MGE_INFERENCE_ONLY=ON
MGE_WINDOWS_BUILD_ARCH=x64
MGE_WINDOWS_BUILD_MARCH=m64
MGE_ARCH=x86_64
REMOVE_OLD_BUILD=false
echo "EXTRA_CMAKE_ARGS: ${EXTRA_CMAKE_ARGS}"

while getopts "rhdctm" arg
do
    case $arg in
        d)
            echo "Build with Debug mode"
            BUILD_TYPE=Debug
            ;;
        c)
            echo "Build with CUDA"
            MGE_WITH_CUDA=ON
            ;;
        t)
            echo "Build with training mode"
            MGE_INFERENCE_ONLY=OFF
            ;;
        h)
            echo "show usage"
            usage
            ;;
        r)
            echo "config REMOVE_OLD_BUILD=true"
            REMOVE_OLD_BUILD=true
            ;;
        m)
            echo "build for m32(only use for windows)"
            MGE_WINDOWS_BUILD_ARCH=x86
            MGE_WINDOWS_BUILD_MARCH=m32
            MGE_ARCH=i386
            ;;
        ?)
            echo "unkonw argument"
            usage
            ;;
    esac
done
echo "------------------------------------"
echo "build config summary:"
echo "BUILD_TYPE: $BUILD_TYPE"
echo "MGE_WITH_CUDA: $MGE_WITH_CUDA"
echo "MGE_INFERENCE_ONLY: $MGE_INFERENCE_ONLY"
echo "------------------------------------"
READLINK=readlink
OS=$(uname -s)

if [ $OS = "Darwin" ];then
    READLINK=greadlink
    if [ $MGE_WITH_CUDA = "ON" ];then
        echo "MACOS DO NOT SUPPORT TensorRT, ABORT NOW!!"
        exit -1
    fi
elif [[ $OS =~ "NT" ]]; then
    echo "BUILD in NT ..."
fi

SRC_DIR=$($READLINK -f "`dirname $0`/../../")
source $SRC_DIR/scripts/cmake-build/utils/utils.sh

function cmake_build() {
    BUILD_DIR=$SRC_DIR/build_dir/host/MGE_WITH_CUDA_$1/MGE_INFERENCE_ONLY_$2/$3/build
    INSTALL_DIR=$BUILD_DIR/../install
    MGE_WITH_CUDA=$1
    MGE_INFERENCE_ONLY=$2
    BUILD_TYPE=$3
    echo "build dir: $BUILD_DIR"
    echo "install dir: $INSTALL_DIR"
    echo "build type: $BUILD_TYPE"
    echo "MGE_WITH_CUDA: $MGE_WITH_CUDA"
    echo "MGE_INFERENCE_ONLY: $MGE_INFERENCE_ONLY"
    try_remove_old_build $REMOVE_OLD_BUILD $BUILD_DIR $INSTALL_DIR

    echo "create build dir"
    mkdir -p $BUILD_DIR
    mkdir -p $INSTALL_DIR
    cd $BUILD_DIR
    cmake \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DMGE_INFERENCE_ONLY=$MGE_INFERENCE_ONLY \
        -DMGE_WITH_CUDA=$MGE_WITH_CUDA \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        ${EXTRA_CMAKE_ARGS} \
        $SRC_DIR

    make -j$(nproc)
    make install/strip
}

function windows_env_err() {
    echo "check windows env failed!!"
    echo "please install LLVM/clang-cl/cmake/python at Visual Studio Extensions"
    exit -1
}

function prepare_env_for_windows_build() {
    echo "check Visual Studio install path env..."
    if [[ -z $VS_PATH ]];then
        echo "can not find visual_studio_path env, pls export you Visual Studio install dir to VS_PATH"
        echo "examle for export Visual Studio 2019 Enterprise default install dir"
        echo "export VS_PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Enterprise"
        exit -1
    fi
    echo $VS_PATH

    # only use cmake/clang-cl/Ninja install from Visual Studio, if not, may build failed
    # some user env may install cmake/clang-cl/Ninja at MSYS env, so we put Visual Studio
    # path at the head of PATH, and check the valid
    echo "check cmake install..."
    export PATH=$VS_PATH/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/:$PATH
    which cmake
    cmake_loc=`which cmake`
    if [[ $cmake_loc =~ "Visual" ]]; then
        echo "cmake valid ..."
    else
        echo "cmake Invalid: ..."
        windows_env_err
    fi

    echo "check clang-cl install..."
    export PATH=$VS_PATH/VC/Tools/Llvm/bin/:$PATH
    which clang-cl
    clang_loc=`which clang-cl`
    if [[ $clang_loc =~ "Visual" ]]; then
        echo "clang-cl valid ..."
    else
        echo "clang-cl Invalid: ..."
        windows_env_err
    fi

    echo "check Ninja install..."
    export PATH=$VS_PATH/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/:$PATH
    which Ninja
    ninja_loc=`which Ninja`
    if [[ $ninja_loc =~ "Visual" ]]; then
        echo "Ninja valid ..."
    else
        echo "Ninja Invalid: ..."
        windows_env_err
    fi

    export PATH=$VS_PATH/VC/Auxiliary/Build:$PATH
    echo "put vcvarsall.bat path to PATH env.."
}

function cmake_build_windows() {
    # windows do not support long path, so we cache the BUILD_DIR ASAP
    prepare_env_for_windows_build
    BUILD_DIR=$SRC_DIR/build_dir/host/build
    INSTALL_DIR=$BUILD_DIR/../install
    MGE_WITH_CUDA=$1
    MGE_INFERENCE_ONLY=$2
    BUILD_TYPE=$3
    echo "build dir: $BUILD_DIR"
    echo "install dir: $INSTALL_DIR"
    echo "build type: $BUILD_TYPE"
    echo "MGE_WITH_CUDA: $MGE_WITH_CUDA"
    echo "MGE_INFERENCE_ONLY: $MGE_INFERENCE_ONLY"
    try_remove_old_build $REMOVE_OLD_BUILD $BUILD_DIR $INSTALL_DIR

    echo "create build dir"
    mkdir -p $BUILD_DIR
    mkdir -p $INSTALL_DIR
    cd $BUILD_DIR
    echo "now try build windows native with cmake/clang-ci/Ninja/Visual Studio ....."
    export CFLAGS=-$MGE_WINDOWS_BUILD_MARCH
    export CXXFLAGS=-$MGE_WINDOWS_BUILD_MARCH
    cmd.exe /c " \
        vcvarsall.bat $MGE_WINDOWS_BUILD_ARCH && cmake  -G "Ninja" \
        -DMGE_ARCH=$MGE_ARCH \
        -DMGE_INFERENCE_ONLY=$MGE_INFERENCE_ONLY \
        -DMGE_WITH_CUDA=$MGE_WITH_CUDA \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR  \
        -DCMAKE_C_COMPILER=clang-cl.exe \
        -DCMAKE_CXX_COMPILER=clang-cl.exe \
        -DCMAKE_MAKE_PROGRAM=ninja.exe \
        ${EXTRA_CMAKE_ARGS} \
        ../../.. && \
        echo \"start Ninja build log to build.log, may take serval min...\" && \
        Ninja load_and_run > build.log"
}


if [[ $OS =~ "NT" ]]; then
    cmake_build_windows $MGE_WITH_CUDA $MGE_INFERENCE_ONLY $BUILD_TYPE
else
    cmake_build $MGE_WITH_CUDA $MGE_INFERENCE_ONLY $BUILD_TYPE
fi
