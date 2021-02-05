#!/bin/bash -ex

function handle_strip() {
    echo "now handle strip $1"
    objcopy --only-keep-debug $1 $1.dbg
    strip -s $1
    objcopy --add-gnu-debuglink=$1.dbg $1
    rm $1.dbg
}

function full_copy_so(){
    lib_path=$1
    dst_dir=$2
    append_rpath=$3
    lib_name=$(basename $lib_path)
    cp $lib_path $dst_dir/$lib_name
    if [ "$append_rpath" != "" ];then
        ori_rpath=$(patchelf --print-rpath $dst_dir/$lib_name)
        if [ "$ori_rpath" != "" ];then
            patchelf --set-rpath "$ori_rpath:$append_rpath" $dst_dir/$lib_name   
        else
            patchelf --set-rpath "$append_rpath" $dst_dir/$lib_name   
        fi
    fi
}

function patch_elf_depend_lib() {
    echo "handle common depend lib"
    LIBS_DIR=${BUILD_DIR}/staging/megengine/core/lib
    mkdir -p ${LIBS_DIR}
    cp /usr/lib64/libatomic.so.1 ${LIBS_DIR}

    patchelf --remove-rpath ${BUILD_DIR}/staging/megengine/core/_imperative_rt.so
    patchelf --force-rpath --set-rpath '$ORIGIN/lib' ${BUILD_DIR}/staging/megengine/core/_imperative_rt.so
    cp ${BUILD_DIR}/src/libmegengine_export.so ${LIBS_DIR}
    patchelf --remove-rpath ${LIBS_DIR}/libmegengine_export.so
    patchelf --force-rpath --set-rpath '$ORIGIN/.' ${LIBS_DIR}/libmegengine_export.so

    cp ${BUILD_DIR}/src/libmegengine_export.so ${LIBS_DIR}
    patchelf --remove-rpath ${LIBS_DIR}/libmegengine_export.so
    patchelf --force-rpath --set-rpath '$ORIGIN/.' ${LIBS_DIR}/libmegengine_export.so


    if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
        echo "handle cuda lib"
        cp ${BUILD_DIR}/dnn/cuda-stub/libcuda_stub.so ${LIBS_DIR}
        cp /usr/local/cuda/lib64/libnvToolsExt.so.1 ${LIBS_DIR}
        IFS=: read -a lib_name_array <<<"$COPY_LIB_LIST"
        append_rpath='$ORIGIN/.'
        for lib_name in ${lib_name_array[@]};do
            full_copy_so $lib_name ${LIBS_DIR} $lib_append_rpath
        done
    fi
}


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
NEW_LIB_PATH=core/lib
for ver in ${ALL_PYTHON}
do
    USE_AUDITWHEEL="ON"
    python_ver=${ver:0:2}
    MAJOR=${python_ver:0:1}
    MINOR=${ver:1}
    PYTHON_DIR=/opt/python/cp${python_ver}-cp${ver}/
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} ${EXTRA_CMAKE_FLAG}"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=RelWithDebInfo"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCMAKE_PREFIX_PATH=${PYTHON_DIR}"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DPYTHON_EXECUTABLE=${PYTHON_DIR}/bin/python3"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DPYTHON_LIBRARY=${PYTHON_DIR}lib/"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DPYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python${MAJOR}.${MINOR}"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DMGE_WITH_ATLAS=ON"

    if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
        ${SRC_DIR}/scripts/cmake-build/host_build.sh -c -t -r
    else
        ${SRC_DIR}/scripts/cmake-build/host_build.sh -t -r
    fi

    cd ${BUILD_DIR}
    rm -rf staging
    mkdir -p staging
    cp -a imperative/python/{megengine,setup.py,requires.txt,requires-style.txt,requires-test.txt} staging/



    handle_strip ${BUILD_DIR}/src/libmegengine_export.so

    cd ${BUILD_DIR}/staging/megengine/core
    handle_strip _imperative_rt.so

    mkdir -p lib/ucx



    if [ ${USE_AUDITWHEEL} = "OFF" ]; then
        patch_elf_depend_lib
    fi

    cd ${BUILD_DIR}/staging/
    ${PYTHON_DIR}/bin/python setup.py bdist_wheel
    cd /home/output
    if [ ${USE_AUDITWHEEL} = "ON" ]; then
        LD_LIBRARY_PATH=${BUILD_DIR}/dnn/cuda-stub:$LD_LIBRARY_PATH auditwheel repair -L ${NEW_LIB_PATH} ${BUILD_DIR}/staging/dist/Meg*.whl
    else
        mkdir -p ${SRC_DIR}/scripts/whl/manylinux2014/output/wheelhouse/${OUT_DIR}
        cd ${BUILD_DIR}/staging/dist/
        org_whl_name=`ls Meg*${ver}*.whl`
        compat_whl_name=`echo ${org_whl_name} | sed 's/linux/manylinux2014/'`
        echo "org whl name: ${org_whl_name}"
        echo "comapt whl name: ${compat_whl_name}"
        mv ${org_whl_name} ${SRC_DIR}/scripts/whl/manylinux2014/output/wheelhouse/${OUT_DIR}/${compat_whl_name}
        cd /home/output
    fi
    chown -R ${UID}.${UID} .
    # compat for root-less docker env to remove output at host side
    chmod -R 777 .
    echo "python $ver done"
done

