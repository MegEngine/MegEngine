#!/bin/bash -e

IN_UBUNTU_DOCKER_ENV="OFF"
if which lsb_release && lsb_release -a | grep "Ubuntu"; then
    IN_UBUNTU_DOCKER_ENV="ON"
    # some code will take about 1h to run as on aarch64-ubuntu
    export CC=gcc-8
    export CXX=g++-8
fi

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

function handle_copy_libs() {
    TO_DIR=$1
    if [ ${BUILD_WHL_WITH_CUDA} = "ON" ]; then
        echo "handle cuda lib to ${TO_DIR}"
        cp /usr/local/cuda/lib64/libnvToolsExt.so.1 ${TO_DIR}
        IFS=: read -a lib_name_array <<<"$COPY_LIB_LIST"
        append_rpath='$ORIGIN'
        for lib_name in ${lib_name_array[@]};do
            echo "cuda copy detail: ${lib_name} to ${TO_DIR}"
            full_copy_so $lib_name ${TO_DIR} $append_rpath
        done
    fi
    if [ ${BUILD_WHL_WITH_CAMBRICON} = "ON" ]; then
        echo "handle cambricon lib to ${TO_DIR}"
        IFS=: read -a lib_name_array <<<"$COPY_LIB_LIST"
        append_rpath='$ORIGIN'
        for lib_name in ${lib_name_array[@]};do
            echo "cambricon copy detail: ${lib_name} to ${TO_DIR}"
            full_copy_so $lib_name ${TO_DIR} $append_rpath
        done
    fi
    if [ ${BUILD_WHL_WITH_ASCEND} = "ON" ]; then
        echo "handle ascend lib to ${TO_DIR}"
        mkdir -p ${TO_DIR}/ascend/aarch64-linux/lib64
        pushd ${ASCEND_TOOLKIT_HOME}
            IFS=: read -a depend_dir_array <<<"$DEPEND_DIR_RELATIVE_PATH_LIST"
            for dir_name in ${depend_dir_array[@]};do
                cp -r --parent $dir_name ${TO_DIR}/ascend/
            done
        popd
        IFS=: read -a copy_lib_name_array <<<"$COPY_LIB_LIST"
        append_rpath='$ORIGIN'
        for lib_name in ${copy_lib_name_array[@]};do
            full_copy_so $lib_name ${TO_DIR}/ascend/aarch64-linux/lib64 $append_rpath
        done
    fi
}

function patch_elf_depend_lib_name(){
    lib_path=$1
    needed_libs=$(patchelf --print-needed $lib_path)
    for lib_name in $needed_libs
    do
        base_name=$(basename $lib_name)
        if [ "$base_name" != "$lib_name" ];then
            patchelf --replace-needed $lib_name $base_name $lib_path
        fi
    done
}

function patch_elf_depend_lib_mgb_mge() {
    echo "handle common depend lib for mgb or mge"
    LIBS_DIR=${BUILD_DIR}/staging/megengine/core/lib
    mkdir -p ${LIBS_DIR}
    if [ ${IN_UBUNTU_DOCKER_ENV} = "OFF" ]; then
        cp /usr/lib64/libatomic.so.1 ${LIBS_DIR}
    fi

    patchelf --remove-rpath ${BUILD_DIR}/staging/megengine/core/_imperative_rt.so
    if [ ${BUILD_WHL_WITH_ASCEND} = "ON" ]; then
        patchelf --force-rpath --set-rpath '$ORIGIN/lib:$ORIGIN/lib/ascend/aarch64-linux/lib64' ${BUILD_DIR}/staging/megengine/core/_imperative_rt.so
    else
        patchelf --force-rpath --set-rpath '$ORIGIN/lib' ${BUILD_DIR}/staging/megengine/core/_imperative_rt.so
    fi
    handle_strip ${BUILD_DIR}/staging/megengine/core/_imperative_rt.so
    patch_elf_depend_lib_name ${BUILD_DIR}/staging/megengine/core/_imperative_rt.so

    cp ${BUILD_DIR}/src/libmegengine_shared.so ${LIBS_DIR}
    handle_strip ${LIBS_DIR}/libmegengine_shared.so
    patchelf --remove-rpath ${LIBS_DIR}/libmegengine_shared.so

    if [ ${BUILD_WITH_LIBRARY} == "false" ];then
        patchelf --force-rpath --set-rpath '$ORIGIN/../../../nvidia/cublas/lib:$ORIGIN/../../../nvidia/cuda_nvrtc/lib:$ORIGIN/../../../nvidia/cuda_runtime/lib:$ORIGIN/../../../nvidia/cudnn/lib:$ORIGIN' ${LIBS_DIR}/libmegengine_shared.so
    else
        patchelf --force-rpath --set-rpath '$ORIGIN/.' ${LIBS_DIR}/libmegengine_shared.so
    fi 

    if [ ${BUILD_WHL_WITH_ASCEND} = "ON" ]; then
        patchelf --force-rpath --set-rpath '$ORIGIN/.:$ORIGIN/ascend/aarch64-linux/lib64' ${LIBS_DIR}/libmegengine_shared.so
    else
        patchelf --force-rpath --set-rpath '$ORIGIN/.' ${LIBS_DIR}/libmegengine_shared.so
    fi

    patch_elf_depend_lib_name ${LIBS_DIR}/libmegengine_shared.so

    # as some version of cudnn/trt libs have dlopen libs, so we can not use auditwheel
    # TODO: PR for auditwheel to support args for dlopen libs
    handle_copy_libs ${LIBS_DIR}
}

function patch_elf_depend_lib_megenginelite() {
    echo "handle common depend lib for megenginelite"
    LIBS_DIR=${BUILD_DIR}/staging/megenginelite/libs
    mkdir -p ${LIBS_DIR}

    cp ${BUILD_DIR}/lite/liblite_shared_whl.so ${LIBS_DIR}/
    patchelf --remove-rpath ${LIBS_DIR}/liblite_shared_whl.so
    if [ ${BUILD_WHL_WITH_ASCEND} = "ON" ]; then
        patchelf --force-rpath --set-rpath '$ORIGIN/../../megengine/core/lib:$ORIGIN/../../megengine/core/lib/ascend/aarch64-linux/lib64' ${LIBS_DIR}/liblite_shared_whl.so
    else
        patchelf --force-rpath --set-rpath '$ORIGIN/../../megengine/core/lib' ${LIBS_DIR}/liblite_shared_whl.so
    fi
    handle_strip ${LIBS_DIR}/liblite_shared_whl.so

    patch_elf_depend_lib_name ${LIBS_DIR}/liblite_shared_whl.so
}

SRC_DIR=$(readlink -f "`dirname $0`/../../../")
source ${SRC_DIR}/scripts/whl/utils/utils.sh

SUPPORT_ALL_VERSION="36m 37m 38 39 310"
if [ ${IN_UBUNTU_DOCKER_ENV} = "ON" ]; then
    SUPPORT_ALL_VERSION="3.6.10 3.7.7 3.8.3 3.9.4 3.10.1"
    echo "in ubuntu docker env, override support all python version: ${SUPPORT_ALL_VERSION}"
fi
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
if [ ${BUILD_WHL_WITH_CUDA} = "ON" ]; then
    BUILD_DIR=${SRC_DIR}/build_dir/host/MGE_WITH_CUDA_ON/MGE_INFERENCE_ONLY_OFF/Release/build/
fi

function config_ubuntu_python_env() {
    PYTHON_DIR=~/.pyenv/versions/$1/
    PYTHON_BIN=~/.pyenv/versions/$1/bin
    if [ ! -f "$PYTHON_BIN/python3" ]; then
        echo "ERR: can not find $PYTHON_BIN , Invalid python package"
        echo "now support list: ${FULL_PYTHON_VER}"
        err_env
    else
        echo "put python3 to env..."
        export PATH=${PYTHON_BIN}:$PATH
        which python3
    fi
    echo ${ver}

    if [ "$1" = "3.6.10" ]; then
        PYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python3.6m
        PYTHON_LIBRARY=${PYTHON_DIR}/lib/libpython3.6m.so
    elif [ "$1" = "3.7.7" ]; then
        PYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python3.7m
        PYTHON_LIBRARY=${PYTHON_DIR}/lib/libpython3.7m.so
    elif [[ "$1" = "3.8.3" || "$1" = "3.8.10" ]]; then
        PYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python3.8
        PYTHON_LIBRARY=${PYTHON_DIR}/lib/libpython3.8.so
    elif [ "$1" = "3.9.4" ]; then
        PYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python3.9
        PYTHON_LIBRARY=${PYTHON_DIR}/lib/libpython3.9.so
    elif [ "$1" = "3.10.1" ]; then
        PYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python3.10
        PYTHON_LIBRARY=${PYTHON_DIR}/lib/libpython3.10.so
    else
        echo "ERR: DO NOT SUPPORT PYTHON VERSION"
        echo "now support list: ${FULL_PYTHON_VER}"
        exit -1
    fi
}

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

    # export common args
    export EXTRA_CMAKE_ARGS="${ORG_EXTRA_CMAKE_FLAG} -DCMAKE_BUILD_TYPE=RelWithDebInfo"
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DMGE_WITH_CUSTOM_OP=ON"

    if [ ${IN_UBUNTU_DOCKER_ENV} = "ON" ]; then
        echo "into Ubuntu env"
        # config env
        config_ubuntu_python_env ${ver}
        #check env
        if [ ! -f "$PYTHON_LIBRARY" ]; then
            echo "ERR: can not find $PYTHON_LIBRARY , Invalid python package"
            err_env
        fi
        if [ ! -d "$PYTHON_INCLUDE_DIR" ]; then
            echo "ERR: can not find $PYTHON_INCLUDE_DIR , Invalid python package"
            err_env
        fi
        export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}"
    else
        echo "into manylinux env"
        python_ver=`echo $ver | tr -d m`
        MAJOR=${python_ver:0:1}
        MINOR=${python_ver:1}
        PYTHON_DIR=/opt/python/cp${python_ver}-cp${ver}

        SUFFIX=
        if [[ $MINOR -lt 8 ]];then
            SUFFIX="m"
        fi
        export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DPYTHON_INCLUDE_DIR=${PYTHON_DIR}/include/python${MAJOR}.${MINOR}${SUFFIX}"
    fi
    #append cmake args for config python
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DPYTHON_EXECUTABLE=${PYTHON_DIR}/bin/python3"
    # please do not config to real python path, it will cause import failed if user python3 do not have libpython3.xx.so[dynamic lib]
    export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DPYTHON_LIBRARY=${PYTHON_DIR}/lib/"

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
    if [ ${BUILD_WHL_WITH_CUDA} = "ON" ]; then
        HOST_BUILD_ARGS="${HOST_BUILD_ARGS} -c"
    fi

    # call ninja dry run and check increment is invalid or not
    if [ ${IS_IN_FIRST_LOOP} = "FALSE" ]; then
        ninja_dry_run_and_check_increment "${SRC_DIR}/scripts/cmake-build/host_build.sh" "${HOST_BUILD_ARGS}" "${INCREMENT_KEY_WORDS}"
    fi

    # call real build
    echo "host_build.sh HOST_BUILD_ARGS: ${HOST_BUILD_ARGS}"
    ${SRC_DIR}/scripts/cmake-build/host_build.sh ${HOST_BUILD_ARGS}
    # remove megenginelite py develop soft link create by lite_shared:POST_BUILD @ lite/CMakeLists.txt
    rm -rf ${SRC_DIR}/lite/pylite/megenginelite/libs

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
    org_whl_name="null"
    if [ ${IN_UBUNTU_DOCKER_ENV} = "ON" ]; then
        # ubuntu glibc higher than manylinux2014, so keep org name
        org_whl_name=`ls Meg*.whl`
        compat_whl_name=${org_whl_name}
    else
        org_whl_name=`ls Meg*${ver}*.whl`
        compat_whl_name=`echo ${org_whl_name} | sed 's/linux/manylinux2014/'`
    fi
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
