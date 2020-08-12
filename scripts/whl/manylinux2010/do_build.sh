#!/bin/bash -e
ALL_PYTHON=${ALL_PYTHON}
if [[ -z ${ALL_PYTHON} ]]
then
    ALL_PYTHON="35m 36m 37m 38"
fi

EXTRA_CMAKE_ARGS=
if [[ "$1" == imperative ]]; then
    BUILD_IMPERATIVE=ON
    SO_NAME=_imperative_rt
    SO_PATH=megengine/core
else
    BUILD_IMPERATIVE=OFF
    SO_NAME=_mgb
    SO_PATH=megengine/_internal
fi

for ver in ${ALL_PYTHON}
do
    python_ver=${ver:0:2}
    BUILD_DIR=/tmp/build_megengine/python${python_ver}
    MAJOR=${python_ver:0:1}
    MINOR=${ver:1}
    PYTHON_DIR=/opt/python/cp${python_ver}-cp${ver}/
    EXT_NAME=${SO_NAME}.cpython-${ver}-x86_64-linux-gnu.so
    mkdir -p ${BUILD_DIR}
    pushd ${BUILD_DIR} >/dev/null
        MGE_CMAKE_FLAGS="-DMGE_WITH_DISTRIBUTED=ON \
            -DMGE_WITH_CUDA=ON \
            -DCMAKE_PREFIX_PATH=${PYTHON_DIR} \
            -DCMAKE_INSTALL_PREFIX=/home/output "
        if [[ "$BUILD_IMPERATIVE" == ON ]]; then
            MGE_CMAKE_FLAGS+=" -DMGE_BUILD_IMPERATIVE_RT=ON \
                -DPYTHON_EXECUTABLE=${PYTHON_DIR}/bin/python3"
        else
            MGE_CMAKE_FLAGS+=" -DPYTHON_LIBRARY=${PYTHON_DIR}lib/ \
                -DPYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python${MAJOR}.${MINOR}"
        fi
        cmake /home/code ${MGE_CMAKE_FLAGS} ${EXTRA_CMAKE_ARGS}
        make -j$(nproc) VERBOSE=1
        make install
        mkdir -p staging
        mkdir -p /home/output/debug
        if [[ "$BUILD_IMPERATIVE" == ON ]]; then
            cp -a imperative/python/{megengine,setup.py,requires.txt,requires-style.txt,requires-test.txt} staging/
        else
            cp -a python_module/{megengine,setup.py,requires.txt,requires-style.txt,requires-test.txt} staging/
        fi
        pushd dnn/cuda-stub/ >/dev/null
            strip -s libcuda.so
            ln -sf libcuda.so libcuda.so.1
        popd >/dev/null
        pushd staging >/dev/null
            pushd ${SO_PATH} >/dev/null
                SO_NAME_EXT=${SO_NAME}.so
                objcopy --only-keep-debug ${SO_NAME_EXT} ${EXT_NAME}.dbg
                strip -s ${SO_NAME_EXT}
                objcopy --add-gnu-debuglink=${EXT_NAME}.dbg ${SO_NAME_EXT}
                cp -a ${EXT_NAME}.dbg /home/output/debug
                mkdir -p lib/ucx
                cp -L /usr/local/cuda/lib*/libnvrtc-builtins.so lib
	            cp -L ${BUILD_DIR}/third_party/MegRay/third_party/ucx/lib/ucx/*.so lib/ucx/
                strip -s lib/ucx/*.so
            popd >/dev/null
            ${PYTHON_DIR}/bin/python setup.py bdist_wheel
        popd >/dev/null
    popd >/dev/null
    pushd /home/output >/dev/null
        if [[ "$BUILD_IMPERATIVE" == ON ]]; then
            NEW_LIB_PATH=core/lib
        else
            NEW_LIB_PATH=_internal/lib
        fi
        LD_LIBRARY_PATH=${BUILD_DIR}/dnn/cuda-stub:$LD_LIBRARY_PATH auditwheel repair -L ${NEW_LIB_PATH} ${BUILD_DIR}/staging/dist/Meg*.whl
        chown -R ${UID}.${UID} .
    popd >/dev/null
    rm -rf ${BUILD_DIR}
done


pushd /home/code/dnn/scripts >/dev/null
rm -rf __pycache__
popd >/dev/null
