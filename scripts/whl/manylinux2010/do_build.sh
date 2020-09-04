#!/bin/bash -e
ALL_PYTHON=${ALL_PYTHON}
if [[ -z ${ALL_PYTHON} ]]
then
    ALL_PYTHON="35m 36m 37m 38"
fi

BUILD_WHL_CPU_ONLY=${BUILD_WHL_CPU_ONLY}
if [[ -z ${BUILD_WHL_CPU_ONLY} ]]
then
    BUILD_WHL_CPU_ONLY="OFF"
fi

SRC_DIR=$(readlink -f "`dirname $0`/../../../")
BUILD_DIR=${SRC_DIR}/build_dir/host/MGE_WITH_CUDA_OFF/MGE_INFERENCE_ONLY_OFF/Release/build/
if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
    BUILD_DIR=${SRC_DIR}/build_dir/host/MGE_WITH_CUDA_ON/MGE_INFERENCE_ONLY_OFF/Release/build/
fi
SO_NAME=_imperative_rt
SO_PATH=megengine/core
NEW_LIB_PATH=core/lib

for ver in ${ALL_PYTHON}
do
    python_ver=${ver:0:2}
    MAJOR=${python_ver:0:1}
    MINOR=${ver:1}
    PYTHON_DIR=/opt/python/cp${python_ver}-cp${ver}/
    EXT_NAME=${SO_NAME}.cpython-${ver}-x86_64-linux-gnu.so
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=RelWithDebInfo"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCMAKE_PREFIX_PATH=${PYTHON_DIR}"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DPYTHON_EXECUTABLE=${PYTHON_DIR}/bin/python3"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DPYTHON_LIBRARY=${PYTHON_DIR}lib/"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DPYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python${MAJOR}.${MINOR}"

    if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
        ${SRC_DIR}/scripts/cmake-build/host_build.sh -c -t -r
    else
        ${SRC_DIR}/scripts/cmake-build/host_build.sh -t -r
    fi

    cd ${BUILD_DIR}
    rm -rf staging
    mkdir -p staging
    cp -a imperative/python/{megengine,setup.py,requires.txt,requires-style.txt,requires-test.txt} staging/


    if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
        cd ${BUILD_DIR}/dnn/cuda-stub/
        strip -s libcuda.so
        ln -sf libcuda.so libcuda.so.1
    fi

    cd ${BUILD_DIR}/staging/${SO_PATH}
    SO_NAME_EXT=${SO_NAME}.so
    objcopy --only-keep-debug ${SO_NAME_EXT} ${EXT_NAME}.dbg
    strip -s ${SO_NAME_EXT}
    objcopy --add-gnu-debuglink=${EXT_NAME}.dbg ${SO_NAME_EXT}
    mkdir -p lib/ucx

    if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
        cp -L /usr/local/cuda/lib*/libnvrtc-builtins.so lib
        cp -L ${BUILD_DIR}/third_party/MegRay/third_party/ucx/lib/ucx/*.so lib/ucx/
        strip -s lib/ucx/*.so
    fi

    cd ${BUILD_DIR}/staging/
    ${PYTHON_DIR}/bin/python setup.py bdist_wheel
    cd /home/output
    LD_LIBRARY_PATH=${BUILD_DIR}/dnn/cuda-stub:$LD_LIBRARY_PATH auditwheel repair -L ${NEW_LIB_PATH} ${BUILD_DIR}/staging/dist/Meg*.whl
    chown -R ${UID}.${UID} .
    # compat for root-less docker env to remove output at host side
    chmod -R 777 .
done
