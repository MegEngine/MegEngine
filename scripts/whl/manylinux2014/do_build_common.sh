#!/bin/bash -e

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

function handle_copy_cuda_libs() {
    TO_DIR=$1
    if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
        echo "handle cuda lib to ${TO_DIR}"
        cp ${BUILD_DIR}/dnn/cuda-stub/libcuda_stub.so ${TO_DIR}
        handle_strip ${TO_DIR}/libcuda_stub.so
        cp /usr/local/cuda/lib64/libnvToolsExt.so.1 ${TO_DIR}
        IFS=: read -a lib_name_array <<<"$CUDA_COPY_LIB_LIST"
        append_rpath='$ORIGIN'
        for lib_name in ${lib_name_array[@]};do
            echo "cuda copy detail: ${lib_name} to ${TO_DIR}"
            full_copy_so $lib_name ${TO_DIR} $append_rpath
        done
    fi
}

function patch_elf_depend_lib_mgb_mge() {
    echo "handle common depend lib for mgb or mge"
    LIBS_DIR=${BUILD_DIR}/staging/megengine/core/lib
    mkdir -p ${LIBS_DIR}
    cp /usr/lib64/libatomic.so.1 ${LIBS_DIR}

    patchelf --remove-rpath ${BUILD_DIR}/staging/megengine/core/_imperative_rt.so
    patchelf --force-rpath --set-rpath '$ORIGIN/lib' ${BUILD_DIR}/staging/megengine/core/_imperative_rt.so
    handle_strip ${BUILD_DIR}/staging/megengine/core/_imperative_rt.so

    cp ${BUILD_DIR}/src/libmegengine_shared.so ${LIBS_DIR}
    handle_strip ${LIBS_DIR}/libmegengine_shared.so
    patchelf --remove-rpath ${LIBS_DIR}/libmegengine_shared.so
    patchelf --force-rpath --set-rpath '$ORIGIN/.' ${LIBS_DIR}/libmegengine_shared.so

    # as some version of cudnn/trt libs have dlopen libs, so we can not use auditwheel
    # TODO: PR for auditwheel to support args for dlopen libs
    handle_copy_cuda_libs ${LIBS_DIR}
}

function patch_elf_depend_lib_megenginelite() {
    echo "handle common depend lib for megenginelite"
    LIBS_DIR=${BUILD_DIR}/staging/megenginelite/libs
    mkdir -p ${LIBS_DIR}

    cp ${BUILD_DIR}/lite/liblite_shared_whl.so ${LIBS_DIR}/
    patchelf --remove-rpath ${LIBS_DIR}/liblite_shared_whl.so
    patchelf --force-rpath --set-rpath '$ORIGIN/../../megengine/core/lib' ${LIBS_DIR}/liblite_shared_whl.so
    handle_strip ${LIBS_DIR}/liblite_shared_whl.so
}

SRC_DIR=$(readlink -f "`dirname $0`/../../../")
source ${SRC_DIR}/scripts/whl/utils/utils.sh

SUPPORT_ALL_VERSION="35m 36m 37m 38"
ALL_PYTHON=${ALL_PYTHON}
if [[ -z ${ALL_PYTHON} ]]
then
    ALL_PYTHON=${SUPPORT_ALL_VERSION}
else
    check_python_version_is_valid "${ALL_PYTHON}" "${SUPPORT_ALL_VERSION}"
fi

BUILD_WHL_CPU_ONLY=${BUILD_WHL_CPU_ONLY}
if [[ -z ${BUILD_WHL_CPU_ONLY} ]]
then
    BUILD_WHL_CPU_ONLY="OFF"
fi

BUILD_DIR=${SRC_DIR}/build_dir/host/MGE_WITH_CUDA_OFF/MGE_INFERENCE_ONLY_OFF/Release/build/
if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
    BUILD_DIR=${SRC_DIR}/build_dir/host/MGE_WITH_CUDA_ON/MGE_INFERENCE_ONLY_OFF/Release/build/
fi

# here we just treat cu file should not in the increment build file list
INCREMENT_KEY_WORDS=".cu.o is dirty"
IS_IN_FIRST_LOOP=TRUE

ORG_EXTRA_CMAKE_FLAG=${EXTRA_CMAKE_FLAG}
for ver in ${ALL_PYTHON}
do
    # we want run a full clean build at the first loop
    if [ ${IS_IN_FIRST_LOOP} = "TRUE" ]; then
        # TODO: may all cmake issue can be resolved after rm CMakeCache?
        # if YES, remove this to use old cache and speed up CI
        echo "warning: remove old build_dir for the first loop"
        rm -rf ${BUILD_DIR}
    fi

    python_ver=${ver:0:2}
    MAJOR=${python_ver:0:1}
    MINOR=${ver:1}
    PYTHON_DIR=/opt/python/cp${python_ver}-cp${ver}/
    export EXTRA_CMAKE_ARGS="${ORG_EXTRA_CMAKE_FLAG} -DCMAKE_BUILD_TYPE=RelWithDebInfo"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DMGE_WITH_CUSTOM_OP=ON"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DPYTHON_EXECUTABLE=${PYTHON_DIR}/bin/python3"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DPYTHON_LIBRARY=${PYTHON_DIR}lib/"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DPYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python${MAJOR}.${MINOR}"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DMGE_WITH_ATLAS=ON"

    if [ -d "${BUILD_DIR}" ]; then
        # insure rm have args
        touch ${BUILD_DIR}/empty.so
        touch ${BUILD_DIR}/CMakeCache.txt
        find ${BUILD_DIR} -name "*.so" | xargs rm
        # as we now use increment build mode when switch python
        # But I do not known any more issue at CMakeLists.txt or not
        # so Force remove CMakeCache.txt
        find ${BUILD_DIR} -name CMakeCache.txt | xargs rm
    fi

    HOST_BUILD_ARGS="-t -s"
    if [ ${BUILD_WHL_CPU_ONLY} = "OFF" ]; then
        HOST_BUILD_ARGS="${HOST_BUILD_ARGS} -c"
    fi

    # call ninja dry run and check increment is invalid or not
    if [ ${IS_IN_FIRST_LOOP} = "FALSE" ]; then
        ninja_dry_run_and_check_increment "${SRC_DIR}/scripts/cmake-build/host_build.sh" "${HOST_BUILD_ARGS}" "${INCREMENT_KEY_WORDS}"
    fi

    # call real build
    echo "host_build.sh HOST_BUILD_ARGS: ${HOST_BUILD_ARGS}"
    ${SRC_DIR}/scripts/cmake-build/host_build.sh ${HOST_BUILD_ARGS}

    # check python api call setup.py
    cd ${BUILD_DIR}
    check_build_ninja_python_api ${ver}
    rm -rf staging
    mkdir -p staging
    cp -a imperative/python/{megengine,setup.py,requires.txt,requires-style.txt,requires-test.txt} staging/
    cp -a ${SRC_DIR}/src/custom/include/megbrain staging/megengine/core/include

    cd ${BUILD_DIR}/staging/megengine/core
    mkdir -p lib/ucx
    patch_elf_depend_lib_mgb_mge

    # handle megenginelite
    cd ${BUILD_DIR}
    mkdir -p staging/megenginelite
    cp ${SRC_DIR}/lite/pylite/megenginelite/* staging/megenginelite/
    patch_elf_depend_lib_megenginelite

    cd ${BUILD_DIR}/staging/
    ${PYTHON_DIR}/bin/python setup.py bdist_wheel
    cd /home/output
    mkdir -p ${SRC_DIR}/scripts/whl/manylinux2014/output/wheelhouse/${SDK_NAME}
    cd ${BUILD_DIR}/staging/dist/
    org_whl_name=`ls Meg*${ver}*.whl`
    compat_whl_name=`echo ${org_whl_name} | sed 's/linux/manylinux2014/'`
    echo "org whl name: ${org_whl_name}"
    echo "comapt whl name: ${compat_whl_name}"
    mv ${org_whl_name} ${SRC_DIR}/scripts/whl/manylinux2014/output/wheelhouse/${SDK_NAME}/${compat_whl_name}

    cd /home/output
    chown -R ${UID}.${UID} .
    # compat for root-less docker env to remove output at host side
    chmod -R 777 .
    echo "python $ver done"
    IS_IN_FIRST_LOOP=FALSE
done
