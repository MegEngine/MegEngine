#!/bin/bash -e
ALL_PYTHON=${ALL_PYTHON}
if [[ -z ${ALL_PYTHON} ]]
then
    ALL_PYTHON="35m 36m 37m 38"
fi

EXTRA_CMAKE_ARGS=

for ver in ${ALL_PYTHON}
do
    python_ver=${ver:0:2}
    BUILD_DIR=/tmp/build_megengine/python${python_ver}
    MAJOR=${python_ver:0:1}
    MINOR=${ver:1}
    PYTHON_DIR=/opt/python/cp${python_ver}-cp${ver}/
    EXT_NAME=_mgb.cpython-${ver}-x86_64-linux-gnu.so
    mkdir -p ${BUILD_DIR}
    pushd ${BUILD_DIR} >/dev/null
        cmake /home/code -DMGE_WITH_DISTRIBUTED=ON -DMGE_WITH_CUDA=ON \
            -DCMAKE_PREFIX_PATH=${PYTHON_DIR} \
            -DMGE_WITH_TEST=ON -DCMAKE_INSTALL_PREFIX=/home/output \
            -DPYTHON_LIBRARY=${PYTHON_DIR}lib/ \
            -DPYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python${MAJOR}.${MINOR}/ \
            ${EXTRA_CMAKE_ARGS}
        make -j$(nproc)
        make install
        mkdir -p staging
        mkdir -p /home/output/debug
        cp -a python_module/{megengine,setup.py} staging/
        pushd dnn/cuda-stub/ >/dev/null
            strip -s libcuda.so
            ln -sf libcuda.so libcuda.so.1
        popd >/dev/null
        pushd staging >/dev/null
            pushd megengine/_internal >/dev/null
                objcopy --only-keep-debug _mgb.so ${EXT_NAME}.dbg
                strip -s _mgb.so
                objcopy --add-gnu-debuglink=${EXT_NAME}.dbg _mgb.so
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
        LD_LIBRARY_PATH=${BUILD_DIR}/dnn/cuda-stub:$LD_LIBRARY_PATH auditwheel repair -L _internal/lib ${BUILD_DIR}/staging/dist/Meg*.whl
        chown -R ${UID}.${UID} .
    popd >/dev/null
    rm -rf ${BUILD_DIR}
done


