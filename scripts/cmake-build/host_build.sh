#!/usr/bin/env bash
set -e

function usage() {
    echo "$0 args1 args2 .."
    echo "available args detail:"
    echo "-d : Build with Debug mode, defaule Release mode"
    echo "-c : Build with CUDA, default without CUDA"
    echo "-t : Build with training mode, default inference only"
    echo "example: $0 -d"
    exit -1
}

BUILD_TYPE=Release
MGE_WITH_CUDA=OFF
MGE_INFERENCE_ONLY=ON

while getopts "dct" arg
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
fi

SRC_DIR=$($READLINK -f "`dirname $0`/../../")

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
    if [ -e $BUILD_DIR ];then
        echo "clean old dir: $BUILD_DIR"
        rm -rf $BUILD_DIR
    fi
    if [ -e $INSTALL_DIR ];then
        echo "clean old dir: $INSTALL_DIR"
        rm -rf $INSTALL_DIR
    fi

    echo "create build dir"
    mkdir -p $BUILD_DIR
    mkdir -p $INSTALL_DIR
    cd $BUILD_DIR
    cmake \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DMGE_INFERENCE_ONLY=$MGE_INFERENCE_ONLY \
        -DMGE_WITH_CUDA=$MGE_WITH_CUDA \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        $SRC_DIR

    make -j$(nproc)
    make install/strip
    }

cmake_build $MGE_WITH_CUDA $MGE_INFERENCE_ONLY $BUILD_TYPE

