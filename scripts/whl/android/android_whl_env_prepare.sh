#!/bin/bash -e

# Installing package by pkg in termux is unwise because pkg will upgrade
# package to the latest version which is undesired sometimes.  We will
# use apt as default package tool. If your env is already broken,e.g.
# clang does not work， you can execute following commands to fix it:
# pkg update
# pkg upgrade

SRC_DIR=$(readlink -f "`dirname $0`/../../../")
cd ${SRC_DIR}
source scripts/whl/android/utils.sh

function install_apt_package() {
    APT_PACKAGE="proot git wget clang cmake libandroid-spawn binutils build-essential ninja texinfo patchelf python"
    echo "try install: ${APT_PACKAGE}"
    apt install ${APT_PACKAGE}
    echo "check termux status by running: clang --version"
    log=`clang --version || true`
    if [[ "${log}" =~ "clang version" ]]; then
        echo "valid env after installing apt package"
    else
        echo "Failed to run clang command, please check termux env!!! You can run: pkg update && pkg upgrade to try to solve it"
        echo "raw log: ${log}"
        exit -1
    fi

}

function patch_termux_env() {
    # do not try to modify other project build files to adapt to only-llvm environment
    echo "many projects can not build without gcc libs, so we create a fake gcc libs linked to librt"
    RT_LIB_TARGET="${PREFIX}/lib/librt.so"
    if [ -e ${RT_LIB_TARGET} ];then
        echo "find librt.so and link it to libgcc.so"
        GCC_LIB_TARGET="${PREFIX}/lib/libgcc.so"
        if [ -e ${GCC_LIB_TARGET} ];then
            echo "already find: ${GCC_LIB_TARGET} skip it"
        else
            create_libgcc_cmd="ln -s ${RT_LIB_TARGET} ${GCC_LIB_TARGET}"
            echo "run cmd: $create_libgcc_cmd"
            ${create_libgcc_cmd}
        fi
    else
        echo "broken termux env, can not find librt.so"
        exit -1
    fi
}

function build_python() {
    # Up to now many tools (build multi python3) are not supported in termux so that we have to build them from source.
    # This function will be changed when some tools are supported in termux, e.g. pyenv.
    mge_python_env_root="${HOME}/mge_python_env"
    mkdir -p ${mge_python_env_root}
    cd ${mge_python_env_root}

    if [ -e ${PREFIX}/local/lib/libffi.a ];then
        echo "always find libffi, skip build it"
    else
        echo "build libffi for python module"
        rm -rf libffi
        git clone https://github.com/libffi/libffi.git
        cd libffi
        termux-chroot "./autogen.sh && ./configure && make -j$(nproc) && make install"
        # remove dynamic lib to force python to use static lib
        rm ${PREFIX}/local/lib/libffi.so*
    fi

    if [ -e ${PREFIX}/local/lib/libz.a ];then
        echo "always find libzlib, skip build it"
    else
        echo "build zlib for python module"
        rm -rf zlib
        git clone https://github.com/madler/zlib.git
        cd zlib
        termux-chroot "CFLAGS=\"-O3 -fPIC\" ./configure && make -j$(nproc) && make install"
        # remove dynamic lib to force python to use static lib
        rm ${PREFIX}/local/lib/libz.so*
    fi

    cpython_repo_dir=${mge_python_env_root}/cpython
    if [ -d ${cpython_repo_dir}/.git ];then
        echo "already find cpython repo"
        cd ${cpython_repo_dir}
        git reset --hard
        git clean -xdf
        git fetch
    else
        cd ${mge_python_env_root}
        rm -rf ${cpython_repo_dir}
        git clone https://github.com/python/cpython.git
    fi
    for ver in ${ALL_PYTHON}
    do
        install_dir=${mge_python_env_root}/${ver}/install
        if [ -e ${install_dir}/bin/python3 ];then
            echo "always find python3, skip build it"
        else
            mkdir -p ${install_dir}
            echo "try build python: ${ver} to ${install_dir}"
            cd ${cpython_repo_dir}
            git reset --hard
            git clean -xdf
            git checkout v${ver}
            index=`awk -v a="${ver}" -v b="." 'BEGIN{print index(a,b)}'`
            sub_str=${ver:${index}}
            index_s=`awk -v a="${sub_str}" -v b="." 'BEGIN{print index(a,b)}'`
            ((finally_index = ${index} + ${index_s}))
            finally_verson=${ver:0:${finally_index}-1}
            MINOR_VER=${ver:${index}:${finally_index}-${index}-1}
            finally_verson="python${finally_verson}"
            echo "finally_verson is: ${finally_verson}"
            # apply patchs
            git apply ${SRC_DIR}/scripts/whl/android/patchs/*.patch
            if [[ ${MINOR_VER} -gt 8 ]]
            then
                echo "apply more patchs"
                git apply ${SRC_DIR}/scripts/whl/android/up_3_9_patch/*.patch
            fi

            termux-chroot "FLAGS=\"-D__ANDROID_API__=24 -Wno-unused-value -Wno-empty-body -Qunused-arguments -Wno-error\" \
                ./configure CFLAGS=\"${FLAGS}\" CPPFLAGS=\"${FLAGS}\" CC=clang CXX=clang++ --enable-shared  --prefix=${install_dir} \
                && sed -i 's/-Werror=implicit-function-declaration//g' Makefile && \
                sed -i 's/\$(LN) -f \$(INSTSONAME)/cp \$(INSTSONAME)/g' Makefile && make -j$(nproc) && make install"

            # after building successfully, patchelf to make python work out of termux-chroot env
            cd ${install_dir}/bin
            # Some python versions won't link to python3 automatically, so we create link manually here'
            rm -rf python3
            if [[ ${MINOR_VER} -gt 7 ]]
            then
                echo "python3 remove suffix m after 3.8"
                patchelf --add-rpath ${install_dir}/lib ${finally_verson}
                cp ${finally_verson} python3
            else
                echo "python3 with suffix m before 3.8, add it"
                patchelf --add-rpath ${install_dir}/lib "${finally_verson}m"
                cp "${finally_verson}m" ${finally_verson}
                cp ${finally_verson} python3
            fi
            echo "finally try run python3"
            ./python3 --version
            ./python3 -m pip install --upgrade pip
            ./python3 -m pip install numpy wheel

        fi
    done
}

############install env now###########
echo "run at root dir: ${SRC_DIR}"
check_termux_env
install_apt_package
patch_termux_env
build_python
