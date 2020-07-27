#!/usr/bin/env bash
set -e

ARCHS=("arm64" "armv7")
BUILD_TYPE=Release
MGE_ARMV8_2_FEATURE_FP16=OFF
MGE_ARMV8_2_FEATURE_DOTPROD=OFF
MGE_DISABLE_FLOAT16=OFF
ARCH=arm64
REMOVE_OLD_BUILD=false
echo "EXTRA_CMAKE_ARGS: ${EXTRA_CMAKE_ARGS}"

function usage() {
    echo "$0 args1 args2 .."
    echo "available args detail:"
    echo "-d : Build with Debug mode, default Release mode"
    echo "-f : enable MGE_ARMV8_2_FEATURE_FP16 for ARM64, need toolchain and hardware support"
    echo "-p : enable MGE_ARMV8_2_FEATURE_DOTPROD for ARM64, need toolchain and hardware support"
    echo "-k : open MGE_DISABLE_FLOAT16 for NEON "
    echo "-a : config build arch available: ${ARCHS[@]}"
    echo "-r : remove old build dir before make, default off"
    echo "-h : show usage"
    echo "append other cmake config by export EXTRA_CMAKE_ARGS=..."
    echo "example: $0 -d"
    exit -1
}

while getopts "rkhdfpa:" arg
do
    case $arg in
        d)
            echo "Build with Debug mode"
            BUILD_TYPE=Debug
            ;;
        f)
            echo "enable MGE_ARMV8_2_FEATURE_FP16 for ARM64"
            MGE_ARMV8_2_FEATURE_FP16=ON
            ;;
        p)
            echo "enable MGE_ARMV8_2_FEATURE_DOTPROD for ARM64"
            MGE_ARMV8_2_FEATURE_DOTPROD=ON
            ;;
        k)
            echo "open MGE_DISABLE_FLOAT16 for NEON"
            MGE_DISABLE_FLOAT16=ON
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
        ?)
            echo "unkonw argument"
            usage
            ;;
    esac
done
echo "----------------------------------------------------"
echo "build config summary:"
echo "BUILD_TYPE: $BUILD_TYPE"
echo "MGE_ARMV8_2_FEATURE_FP16: $MGE_ARMV8_2_FEATURE_FP16"
echo "MGE_ARMV8_2_FEATURE_DOTPROD: $MGE_ARMV8_2_FEATURE_DOTPROD"
echo "MGE_DISABLE_FLOAT16: $MGE_DISABLE_FLOAT16"
echo "ARCH: $ARCH"
echo "----------------------------------------------------"

READLINK=readlink
OS=$(uname -s)

if [ $OS = "Darwin" ];then
    READLINK=greadlink
else
    echo "cross build ios only support on macos, abort now!!"
    exit -1
fi

SRC_DIR=$($READLINK -f "`dirname $0`/../../")
source $SRC_DIR/scripts/cmake-build/utils/utils.sh

function cmake_build() {
    BUILD_DIR=$SRC_DIR/build_dir/apple/$3/$4/$1/$BUILD_TYPE/build
    INSTALL_DIR=$BUILD_DIR/../install
    TOOLCHAIN=$SRC_DIR/toolchains/$2
    OS_PLATFORM=$3
    XCODE_IOS_PLATFORM=$4
    IOS_ARCH=$1
    echo "build dir: $BUILD_DIR"
    echo "install dir: $INSTALL_DIR"
    echo "build type: $BUILD_TYPE"
    echo "build toolchain: $TOOLCHAIN"
    echo "build OS_PLATFORM: $OS_PLATFORM"
    echo "build XCODE_IOS_PLATFORM: $XCODE_IOS_PLATFORM"
    echo "build IOS_ARCH: $IOS_ARCH"
    try_remove_old_build $REMOVE_OLD_BUILD $BUILD_DIR $INSTALL_DIR

    echo "create build dir"
    mkdir -p $BUILD_DIR
    mkdir -p $INSTALL_DIR
    cd $BUILD_DIR
    cmake -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DIOS_TOOLCHAIN_ROOT=$TOOLCHAIN \
        -DOS_PLATFORM=$OS_PLATFORM \
        -DXCODE_IOS_PLATFORM=$XCODE_IOS_PLATFORM \
        -DIOS_ARCH=$IOS_ARCH \
        -DMGE_INFERENCE_ONLY=ON \
        -DPYTHON_EXECUTABLE=/usr/local/bin/python3 \
        -DMGE_WITH_CUDA=OFF \
        -DMGE_ARMV8_2_FEATURE_FP16= $MGE_ARMV8_2_FEATURE_FP16 \
        -DMGE_ARMV8_2_FEATURE_DOTPROD=$MGE_ARMV8_2_FEATURE_DOTPROD \
        -DMGE_DISABLE_FLOAT16=$MGE_DISABLE_FLOAT16 \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        ${EXTRA_CMAKE_ARGS} \
        $SRC_DIR

    make -j$(nproc)
    make install
}

build_flatc $SRC_DIR $REMOVE_OLD_BUILD

# refs for ../../toolchains/ios.toolchain.cmake
# to config this, if u want to build other,
# like simulator or for iwatch, please do manually modify
# OS_PLATFORM=("OS" "OS64" "SIMULATOR" "SIMULATOR64" "TVOS" "WATCHOS" "SIMULATOR_TVOS")
# XCODE_IOS_PLATFORM=("iphoneos" "iphonesimulator" "appletvos" "appletvsimulator" "watchos", "watchsimulator")
# IOS_ARCHS=("arm64" "armv7" "armv7k" "arm64e" "armv7s")

#by defaut we only triger build arm64/armv7 for iphoneos
OS_PLATFORM=OS
XCODE_IOS_PLATFORM=iphoneos
cmake_build $ARCH ios.toolchain.cmake $OS_PLATFORM $XCODE_IOS_PLATFORM
