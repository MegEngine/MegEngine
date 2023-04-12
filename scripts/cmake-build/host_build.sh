#!/usr/bin/env bash
set -e

BUILD_TYPE=Release
MGE_WITH_CUDA=OFF
MGE_INFERENCE_ONLY=ON
MGE_WINDOWS_BUILD_ARCH=x64
MGE_WINDOWS_BUILD_MARCH=m64
MGE_ARCH=x86_64
REMOVE_OLD_BUILD=false
NINJA_VERBOSE=OFF
BUILD_DEVELOP=ON
NINJA_DRY_RUN=OFF
SPECIFIED_TARGET="install/strip"
READLINK=readlink

OS=$(uname -s)
if [[ $OS =~ "NT" ]]; then
    echo "Windows no need strip, caused by pdb file always split with exe"
    SPECIFIED_TARGET="install"
fi

if [ $OS = "Darwin" ];then
    READLINK=greadlink
fi

SRC_DIR=$($READLINK -f "`dirname $0`/../../")
source ${SRC_DIR}/scripts/cmake-build/utils/utils.sh
if [[ $OS =~ "NT" ]]; then
    source ${SRC_DIR}/scripts/whl/windows/config.sh
fi
config_ninja_default_max_jobs

echo "EXTRA_CMAKE_ARGS: ${EXTRA_CMAKE_ARGS}"

function usage() {
    echo "$0 args1 args2 .."
    echo "available args detail:"
    echo "-d : Build with Debug mode, default Release mode"
    echo "-c : Build with CUDA, default without CUDA"
    echo "-t : Build with training mode, default inference only"
    echo "-m : Build with m32 mode(only for windows build), default m64"
    echo "-r : remove old build dir before make, default off"
    echo "-v : ninja with verbose and explain, default off"
    echo "-s : Do not build develop even build with training mode, default on when build with training, always for wheel"
    echo "-n : ninja with -n dry run (don't run commands but act like they succeeded)"
    echo "-j : run N jobs in parallel for ninja, defaut is cpu_number + 2"
    echo "-e : build a specified target (always for debug, NOTICE: do not do strip/install target when use -e)"
    echo "-l : list CMakeLists.txt all options, can be use to config EXTRA_CMAKE_ARGS"
    echo "-h : show usage"
    echo "append other cmake config by config EXTRA_CMAKE_ARGS, for example, enable MGE_WITH_TEST and build with Debug mode:"
    echo "EXTRA_CMAKE_ARGS=\"-DMGE_WITH_TEST=ON\" $0 -d"
    exit -1
}

while getopts "lnsrhdctmve:j:" arg
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
        s)
            echo "config  BUILD_DEVELOP=OFF"
            BUILD_DEVELOP=OFF
            ;;
        v)
            echo "config NINJA_VERBOSE=ON"
            NINJA_VERBOSE=ON
            ;;
        n)
            echo "config NINJA_DRY_RUN=ON"
            NINJA_DRY_RUN=ON
            ;;
        m)
            echo "build for m32(only valid use for windows)"
            MGE_WINDOWS_BUILD_ARCH=x86
            MGE_WINDOWS_BUILD_MARCH=m32
            MGE_ARCH=i386
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
echo "------------------------------------"
echo "build config summary:"
echo "BUILD_TYPE: $BUILD_TYPE"
echo "MGE_WITH_CUDA: $MGE_WITH_CUDA"
echo "MGE_INFERENCE_ONLY: $MGE_INFERENCE_ONLY"
echo "SPECIFIED_TARGET: ${SPECIFIED_TARGET}"
echo "NINJA_MAX_JOBS: ${NINJA_MAX_JOBS}"
echo "------------------------------------"

if [ $OS = "Darwin" ];then
    if [ $MGE_WITH_CUDA = "ON" ];then
        echo "MACOS DO NOT SUPPORT TensorRT, ABORT NOW!!"
        exit -1
    fi
elif [[ $OS =~ "NT" ]]; then
    echo "BUILD in NT ..."
fi

if [ ${MGE_INFERENCE_ONLY} = "ON" ]; then
    echo "config BUILD_DEVELOP=OFF when MGE_INFERENCE_ONLY=ON"
    BUILD_DEVELOP=OFF
fi

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
    cd_real_build_dir $BUILD_DIR
    # fork a new bash to handle EXTRA_CMAKE_ARGS env with space
    bash -c "cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DMGE_INFERENCE_ONLY=$MGE_INFERENCE_ONLY \
        -DMGE_WITH_CUDA=$MGE_WITH_CUDA \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        ${EXTRA_CMAKE_ARGS} \
        ${SRC_DIR} "

    config_ninja_target_cmd ${NINJA_VERBOSE} ${BUILD_DEVELOP} "${SPECIFIED_TARGET}" ${NINJA_DRY_RUN} ${NINJA_MAX_JOBS}
    bash -c "${NINJA_CMD}"
}

function windows_env_err() {
    echo "check windows env failed!!"
    echo "please install env refs for: scripts/cmake-build/BUILD_README.md"
    echo "also refs for: scripts/whl/windows/env_prepare.sh"
    exit -1
}

function prepare_env_for_windows_build() {
    echo "check Visual Studio install path env..."
    # check VS_INSTALL_PATH is valid or not
    if [ ! -d ${VS_INSTALL_PATH}/Licenses ];then
        echo "can not find ${VS_INSTALL_PATH}/Licenses, pls check VS_INSTALL_PATH env"
        echo "pls install VisualStudio by scripts/whl/windows/env_prepare.sh"

        windows_env_err
    else
        echo "use ${VS_INSTALL_PATH}"
    fi

    # only use cmake/Ninja install from Visual Studio, if not, may build failed
    # some user env may install cmake/clang-cl/Ninja at windows-git-bash env, so we put Visual Studio
    # path at the head of PATH, and check the valid
    echo "check cmake install..."
    export PATH=${VS_INSTALL_PATH}/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/:$PATH
    which cmake
    cmake_loc=`which cmake`
    if [[ $cmake_loc =~ ${MEGENGINE_DEV_TOOLS_PREFIX_DIR} ]]; then
        echo "use cmake: $cmake_loc"
    else
        echo "cmake Invalid: ..."
        windows_env_err
    fi

    echo "check clang-cl install..."
    # llvm install by Visual Studio have some issue, eg, link crash on large project, so we
    # use official LLVM download from https://releases.llvm.org/download.html
    # check LLVM_MEGENGINE_DEV_DIR is valid or not
    if [ ! -f ${LLVM_MEGENGINE_DEV_DIR}/bin/clang-cl.exe ];then
        echo "can not find ${LLVM_MEGENGINE_DEV_DIR}/bin/clang-cl.exe, pls check LLVM_MEGENGINE_DEV_DIR env"
        echo "pls install LLVM by scripts/whl/windows/env_prepare.sh"
        windows_env_err
    else
        echo "use ${LLVM_MEGENGINE_DEV_DIR}"
    fi
    echo ${LLVM_MEGENGINE_DEV_DIR}
    export PATH=${LLVM_MEGENGINE_DEV_DIR}/bin/:$PATH
    clang_loc=`which clang-cl`
    if [[ $clang_loc =~ ${VS_INSTALL_PATH} ]]; then
        echo "clang-cl Invalid: we do not support use LLVM installed by Visual Studio"
        windows_env_err
    else
        echo "clang-cl valid ..."
    fi
    if [[ $clang_loc =~ ${LLVM_MEGENGINE_DEV_DIR} ]]; then
        echo "use clang-cl : $clang_loc"
    else
        echo "clang-cl Invalid: ..."
        windows_env_err
    fi

    echo "check Ninja install..."
    export PATH=${VS_INSTALL_PATH}/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/:$PATH
    which Ninja
    ninja_loc=`which Ninja`
    if [[ $ninja_loc =~ ${MEGENGINE_DEV_TOOLS_PREFIX_DIR} ]]; then
        echo "use Ninja: $ninja_loc"
    else
        echo "Ninja Invalid: ..."
        windows_env_err
    fi

    echo "put vcvarsall.bat path to PATH env.."
    # check vcvarsall.bat is valid or not
    if [ ! -f ${VS_INSTALL_PATH}/VC/Auxiliary/Build/vcvarsall.bat ];then
        echo "can not find ${VS_INSTALL_PATH}/VC/Auxiliary/Build/vcvarsall.bat, pls check VS_INSTALL_PATH env"
        echo "pls install VisualStudio by scripts/whl/windows/env_prepare.sh"
        windows_env_err
    else
        echo "use ${VS_INSTALL_PATH}"
    fi
    export PATH=$VS_INSTALL_PATH/VC/Auxiliary/Build:$PATH

    if [ $MGE_WITH_CUDA = "ON" ];then
        echo "config cuda/cudnn/TensorRT env..."
        if [[ -z ${CUDA_ROOT_DIR} ]]; then
            echo "CUDA_ROOT_DIR is not set, use default: ${CUDA_DFT_ROOT}"
            export CUDA_ROOT_DIR=${CUDA_DFT_ROOT}
        fi
        # check CUDA_ROOT_DIR is valid or not
        if [ ! -f "${CUDA_ROOT_DIR}/bin/nvcc.exe" ];then
            echo "can not find ${CUDA_ROOT_DIR}/bin/nvcc.exe, pls check env"
            windows_env_err
        else
            echo "use CUDA_ROOT_DIR: ${CUDA_ROOT_DIR}"
            # put cuda/bin to PATH env
            export PATH=${CUDA_ROOT_DIR}/bin/:$PATH
        fi

        if [[ -z ${CUDNN_ROOT_DIR} ]]; then
            echo "CUDNN_ROOT_DIR is not set, use default: ${CUDNN_DFT_ROOT}"
            export CUDNN_ROOT_DIR=${CUDNN_DFT_ROOT}
        fi
        # check CUDNN_ROOT_DIR is valid or not
        if [ ! -f "${CUDNN_ROOT_DIR}/include/cudnn.h" ];then
            echo "can not find ${CUDNN_ROOT_DIR}/include/cudnn.h, pls check env"
            windows_env_err
        else
            echo "use CUDNN_ROOT_DIR: ${CUDNN_ROOT_DIR}"
        fi

        if [[ -z ${TRT_ROOT_DIR} ]]; then
            echo "TRT_ROOT_DIR is not set, use default: ${TRT_DFT_ROOT}"
            export TRT_ROOT_DIR=${TRT_DFT_ROOT}
        fi
        # check TRT_ROOT_DIR is valid or not
        if [ ! -f "${TRT_ROOT_DIR}/include/NvInfer.h" ];then
            echo "can not find ${TRT_ROOT_DIR}/include/NvInfer.h, pls check env"
            windows_env_err
        else
            echo "use TRT_ROOT_DIR: ${TRT_ROOT_DIR}"
        fi
    fi

    # python version will be config by whl build script or ci script, we need
    # a DFT version for build success when we just call host_build.sh
    if [[ -z ${ALREADY_CONFIG_PYTHON_VER} ]]
    then
        echo "config a default python3"
        if [ ! -f "${DFT_PYTHON_BIN}/python3.exe" ]; then
            echo "ERR: can not find ${DFT_PYTHON_BIN}/python3.exe , Invalid env"
            windows_env_err
        else
            echo "put python3 to env..."
            export PATH=${DFT_PYTHON_BIN}:$PATH
            which python3
        fi
    fi

    # check swig is valid or not
    if [ ! -f ${SWIG_INSTALL_DIR}/swig.exe ];then
        echo "can not find ${SWIG_INSTALL_DIR}/swig.exe, pls check SWIG_INSTALL_DIR env"
        echo "pls install swig by scripts/whl/windows/env_prepare.sh"
        windows_env_err
    else
        echo "use swig: ${SWIG_INSTALL_DIR}"
        export PATH=${SWIG_INSTALL_DIR}:$PATH
    fi
}

function cmake_build_windows() {
    # windows do not support long path, so we cache the BUILD_DIR ASAP
    prepare_env_for_windows_build
    BUILD_DIR=$SRC_DIR/build_dir/host/build
    # ninja have compat issue with bash env, which fork from windows-git
    # which will map C: to /c/c/ dir, which will lead to install file to /c/c/..
    # as a solution, we map INSTALL_DIR to INSTALL_DIR_WIN (/c --> C:)
    INSTALL_DIR=${BUILD_DIR}/../install

    INSTALL_DIR_PREFIX=${INSTALL_DIR:0:2}
    if [ ${INSTALL_DIR_PREFIX} = "/c" ];then
        echo "INSTALL_DIR_PREFIX is ${INSTALL_DIR_PREFIX}, map to C:"
        INSTALL_DIR_WIN="C:${INSTALL_DIR:2}"
    else
        INSTALL_DIR_WIN=${INSTALL_DIR}
    fi
    MGE_WITH_CUDA=$1
    MGE_INFERENCE_ONLY=$2
    BUILD_TYPE=$3
    echo "build dir: $BUILD_DIR"
    echo "install dir: $INSTALL_DIR"
    echo "install dir for ninja: $INSTALL_DIR_WIN"
    echo "build type: $BUILD_TYPE"
    echo "MGE_WITH_CUDA: $MGE_WITH_CUDA"
    echo "MGE_INFERENCE_ONLY: $MGE_INFERENCE_ONLY"
    try_remove_old_build $REMOVE_OLD_BUILD $BUILD_DIR $INSTALL_DIR

    echo "create build dir"
    mkdir -p $BUILD_DIR
    cd_real_build_dir $BUILD_DIR
    echo "now try build windows native with cmake/clang-ci/Ninja/Visual Studio ....."
    export CFLAGS=-$MGE_WINDOWS_BUILD_MARCH
    export CXXFLAGS=-$MGE_WINDOWS_BUILD_MARCH
    cmd.exe /C " \
        vcvarsall.bat $MGE_WINDOWS_BUILD_ARCH -vcvars_ver=${CVARS_VER_NEED} && cmake  -G "Ninja" \
        -DMGE_ARCH=$MGE_ARCH \
        -DMGE_INFERENCE_ONLY=$MGE_INFERENCE_ONLY \
        -DMGE_WITH_CUDA=$MGE_WITH_CUDA \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR_WIN \
        -DCMAKE_C_COMPILER=clang-cl.exe \
        -DCMAKE_CXX_COMPILER=clang-cl.exe \
        -DCMAKE_MAKE_PROGRAM=ninja.exe \
        ${EXTRA_CMAKE_ARGS} ../../.. "

    config_ninja_target_cmd ${NINJA_VERBOSE} ${BUILD_DEVELOP} "${SPECIFIED_TARGET}" ${NINJA_DRY_RUN} ${NINJA_MAX_JOBS}
    cmd.exe /C " vcvarsall.bat $MGE_WINDOWS_BUILD_ARCH -vcvars_ver=${CVARS_VER_NEED} && ${NINJA_CMD} "
}

if [[ $OS =~ "NT" ]]; then
    if [ ${MGE_ARCH} = "i386" ] && [ ${MGE_INFERENCE_ONLY} = "OFF" ]; then
        echo "ERR: training mode(-t) only support 64 bit mode"
        echo "pls remove -t or remove -m"
        exit -1
    fi
    cmake_build_windows $MGE_WITH_CUDA $MGE_INFERENCE_ONLY $BUILD_TYPE
else
    cmake_build $MGE_WITH_CUDA $MGE_INFERENCE_ONLY $BUILD_TYPE
fi
