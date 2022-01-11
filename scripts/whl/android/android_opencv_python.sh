#!/bin/bash -e

# This script is a workaround of installing opencv-python in termux.

SRC_DIR=$(readlink -f "`dirname $0`/../../../")
cd ${SRC_DIR}
source scripts/whl/android/utils.sh

function install_apt_package() {
    APT_PACKAGE="build-essential cmake libjpeg-turbo libpng python clang"
    echo "try to install: ${APT_PACKAGE}"
    apt install ${APT_PACKAGE}
}

function build_opencv_python() {
    python3 -m pip install numpy
    mge_python_env_root="${HOME}/mge_python_env"
    mkdir -p ${mge_python_env_root}
    cd ${mge_python_env_root}

    opencv_repo_dir=${mge_python_env_root}/opencv
    if [ -d ${opencv_repo_dir}/.git ];then
        echo "already find opencv repo"
        cd ${opencv_repo_dir}
        git reset --hard
        git clean -xdf
        git fetch
    else
        cd ${mge_python_env_root}
        rm -rf ${opencv_repo_dir}
        git clone https://github.com/opencv/opencv.git
    fi
    # Build and test latest version by default. You can modify OPENCV_VER to build and test another version!!
    python3_site=`python3 -c 'import site; print(site.getsitepackages()[0])'`
    OPENCV_VER="3.4.15"
    git checkout ${OPENCV_VER}
    if [ -e ${python3_site}/cv2/__init__.py ];then
        echo "python3 already build cv2, skip build it, if you want to rebuild, you can do: rm -rf ${python3_site}/cv2"
    else
        cd ${opencv_repo_dir}
        git checkout ${OPENCV_VE}
        git apply ${SRC_DIR}/scripts/whl/android/cv_patch/*.patch
        mkdir -p build
        cd build
        echo "will install to ${python3_site}"
        PYTHON3_EXECUTABLE=`command -v python3`
        LDFLAGS=" -llog -lpython3" cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_opencv_python3=on \
            -DBUILD_opencv_python2=off -DWITH_QT=OFF -DWITH_GTK=OFF  -DBUILD_ANDROID_PROJECTS=OFF \
            -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_FAT_JAVA_LIB=OFF -DBUILD_ANDROID_SERVICE=OFF \
            -DHAVE_opencv_python3=ON -D__INSTALL_PATH_PYTHON3=${python3_site} \
            -DPYTHON3_EXECUTABLE=${PYTHON3_EXECUTABLE} \
            -DOPENCV_PYTHON_INSTALL_PATH=${python3_site} -DCMAKE_INSTALL_PREFIX=${python3_site} .. \
            && make -j$(nproc) && make install

        # check if build successfully
        cd ~
        python3 -c 'import cv2;print(cv2.__version__)'
    fi
}

############install env now###########
echo "run at root dir: ${SRC_DIR}"
check_termux_env
install_apt_package
build_opencv_python
