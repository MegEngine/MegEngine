#!/bin/bash -e

SRC_DIR=$(readlink -f "`dirname $0`/../../../")
cd ${SRC_DIR}
source scripts/whl/android/utils.sh

ANDROID_WHL_HOME=${SRC_DIR}/scripts/whl/android/ANDROID_WHL_HOME
if [ -e "${ANDROID_WHL_HOME}" ]; then
    echo "remove old android whl file"
    rm -rf ${ANDROID_WHL_HOME}
fi
mkdir -p ${ANDROID_WHL_HOME}

BUILD_DIR=${SRC_DIR}/build_dir/host/MGE_WITH_CUDA_OFF/MGE_INFERENCE_ONLY_OFF/Release/build/

# We only handle the case where dnn/src/common/conv_bias.cpp is not in the list of incremental build files.
INCREMENT_KEY_WORDS="conv_bias.cpp.o is dirty"
IS_IN_FIRST_LOOP=TRUE

ORG_EXTRA_CMAKE_FLAG=${EXTRA_CMAKE_FLAG}

function handle_strip() {
    echo "now handle strip $1"
    objcopy --only-keep-debug $1 $1.dbg
    strip -s $1
    objcopy --add-gnu-debuglink=$1.dbg $1
    rm $1.dbg
}

function patch_elf_depend_lib_mgb_mge() {
    echo "handle common depend lib for mgb or mge"
    LIBS_DIR=${BUILD_DIR}/staging/megengine/core/lib
    mkdir -p ${LIBS_DIR}

    patchelf --remove-rpath ${BUILD_DIR}/staging/megengine/core/_imperative_rt.so
    patchelf --set-rpath '$ORIGIN/lib' ${BUILD_DIR}/staging/megengine/core/_imperative_rt.so
    handle_strip ${BUILD_DIR}/staging/megengine/core/_imperative_rt.so

    cp ${BUILD_DIR}/src/libmegengine_shared.so ${LIBS_DIR}
    patchelf --remove-rpath ${LIBS_DIR}/libmegengine_shared.so
    patchelf --set-rpath '$ORIGIN/.' ${LIBS_DIR}/libmegengine_shared.so
    # FXIME: third_party LLVM need c++_static > 5.1
    # but now clang(13) at termux env do not satisfy it
    # may use -static-libstdc++ at CMakeLists.txt after
    # upgrade third_party LLVM
    cp /data/data/com.termux/files/usr/lib/libc++_shared.so ${LIBS_DIR}
    handle_strip ${LIBS_DIR}/libmegengine_shared.so
}

function patch_elf_depend_lib_megenginelite() {
    echo "handle common depend lib for megenginelite"
    LIBS_DIR=${BUILD_DIR}/staging/megenginelite/libs
    mkdir -p ${LIBS_DIR}

    cp ${BUILD_DIR}/lite/liblite_shared_whl.so ${LIBS_DIR}/
    patchelf --remove-rpath ${LIBS_DIR}/liblite_shared_whl.so
    patchelf --set-rpath '$ORIGIN/../../megengine/core/lib' ${LIBS_DIR}/liblite_shared_whl.so
    handle_strip ${LIBS_DIR}/liblite_shared_whl.so
}
function do_build() {
    mge_python_env_root="${HOME}/mge_python_env"
    for ver in ${ALL_PYTHON}
    do
        python_install_dir=${mge_python_env_root}/${ver}/install
        # we want to run a full clean build in the first loop
        if [ ${IS_IN_FIRST_LOOP} = "TRUE" ]; then
            # TODO: can all cmake issues be resolved after removing CMakeCache?
            # if YES, remove this logic to use old cache and speed up CI
            echo "warning: remove old build_dir for the first loop"
            rm -rf ${BUILD_DIR}
        fi

        # insert python3_install_dir into head of PATH to enable CMake find it
        if [ -e ${python_install_dir}/bin/python3 ];then
            echo "will use ${python_install_dir}/bin/python3 to build mge wheel"
            export PATH=${python_install_dir}/bin:$PATH
        else
            echo "ERROR: can not find python3 in: ${python_install_dir}/bin"
            echo "please run: %{SRC_DIR}/scripts/whl/android/android_whl_env_prepare.sh to prepare env"
            exit -1
        fi

        export EXTRA_CMAKE_ARGS="${ORG_EXTRA_CMAKE_FLAG} -DCMAKE_BUILD_TYPE=RelWithDebInfo"
        export EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DMGE_WITH_CUSTOM_OP=ON"

        if [ -d "${BUILD_DIR}" ]; then
            # insure rm have args
            touch ${BUILD_DIR}/empty.so
            touch ${BUILD_DIR}/CMakeCache.txt
            find ${BUILD_DIR} -name "*.so" | xargs rm
            # Force remove CMakeCache.txt to avoid error owing to unknown issue in CMakeLists.txt
            # which comes from using increment build mode when switching python version
            find ${BUILD_DIR} -name CMakeCache.txt | xargs rm
        fi

        HOST_BUILD_ARGS="-t -s"

        # call ninja dry run and check increment is invalid or not
        if [ ${IS_IN_FIRST_LOOP} = "FALSE" ]; then
            ninja_dry_run_and_check_increment "${SRC_DIR}/scripts/cmake-build/host_build.sh" "${HOST_BUILD_ARGS}" "${INCREMENT_KEY_WORDS}"
        fi

        # call real build
        echo "host_build.sh HOST_BUILD_ARGS: ${HOST_BUILD_ARGS}"
        bash ${SRC_DIR}/scripts/cmake-build/host_build.sh ${HOST_BUILD_ARGS}

        # check python api call setup.py
        cd ${BUILD_DIR}
        check_build_ninja_python_api ${ver}
        rm -rf staging
        mkdir -p staging
        cp -a imperative/python/{megengine,setup.py,requires.txt,requires-style.txt,requires-test.txt} staging/
        cp -a ${SRC_DIR}/src/custom/include staging/megengine/core/include/

        patch_elf_depend_lib_mgb_mge

        # handle megenginelite
        cd ${BUILD_DIR}
        mkdir -p staging/megenginelite
        cp ${SRC_DIR}/lite/pylite/megenginelite/* staging/megenginelite/
        patch_elf_depend_lib_megenginelite

        cd ${BUILD_DIR}/staging
        python3 setup.py bdist_wheel
        cd ${BUILD_DIR}/staging/dist/
        cp ${BUILD_DIR}/staging/dist/Meg*.whl ${ANDROID_WHL_HOME}

        cd ${SRC_DIR}
        echo ""
        echo "##############################################################################################"
        echo "android whl package location: ${ANDROID_WHL_HOME}"
        ls ${ANDROID_WHL_HOME}
        echo "##############################################################################################"
        IS_IN_FIRST_LOOP=FALSE
    done
}

function third_party_prepare() {
    echo "init third_party..."
    bash ${SRC_DIR}/third_party/prepare.sh
    # fix flatbuffers build at pure LLVM env(not cross link gcc)
    # TODO: pr to flatbuffers to fix this issue
    sed -i 's/lc++abi/lc/g' ${SRC_DIR}/third_party/flatbuffers/CMakeLists.txt
}

function remove_requires() {
    # do not worry about this, we will provide 'scripts/whl/android/android_opencv_python.sh'
    # to build opencv-python from opencv src!!  This function may be removed after termux fixes
    # this issue
    cd ${SRC_DIR}
    git checkout imperative/python/requires.txt
    sed -i '/opencv-python/d' imperative/python/requires.txt
    # FIXME: termux install pyarrow will build error now
    # remove this logic after pyarrow fix this issue
    # see imperative/python/megengine/data/dataloader.py
    # for detail, now will use _SerialStreamDataLoaderIter
    sed -i '/pyarrow/d' imperative/python/requires.txt
    cd -
}
######################
check_termux_env
third_party_prepare
remove_requires
do_build
