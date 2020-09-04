#!/bin/bash -e

#install env before use greadlink
function try_install_brew() {
    which brew
    if [ $? -eq 0 ]; then
        echo "find install brew, use it"
    else
        echo "DO NOT FIND brew, now try install, may ask root password, please input manually!!"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
        #try double check 
        which brew
        if [ $? -eq 0 ]; then
            echo "find install brew, use it"
        else
            echo "ERR: INSTALL brew failed!!, please install manually!!"
            exit -1
        fi
    fi
}

function install_brew_package() {
    BREW_PACKAGE="openssl readline sqlite3 xz gdbm zlib pyenv wget swig coreutils llvm git-lfs"
    for pak in ${BREW_PACKAGE}
    do
        echo "###### do command: brew install ${pak}"
        brew install ${pak}
    done

    git lfs install
}
try_install_brew
install_brew_package

READLINK=readlink
OS=$(uname -s)

if [ $OS = "Darwin" ];then
    READLINK=greadlink
else
    echo "ERR: only run at macos env"
    exit -1
fi

SRC_DIR=$($READLINK -f "`dirname $0`/../../../")

echo ${SRC_DIR}
ALL_PYTHON="3.5.9 3.6.10 3.7.7 3.8.3"


function install_python_package() {
    for pak in ${ALL_PYTHON}
    do
        echo "###### do command: env PYTHON_CONFIGURE_OPTS=\"--enable-shared\" pyenv install ${pak}"
        if [ -e /Users/$USER/.pyenv/versions/${pak} ];then
            echo "FOUND install /Users/$USER/.pyenv/versions/${pak} strip it..."
        else
            env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install ${pak}
        fi
        echo "###### do command: /Users/${USER}/.pyenv/versions/${pak}/bin/python3 -m pip install numpy wheel requests tqdm tabulate"
        /Users/${USER}/.pyenv/versions/${pak}/bin/python3 -m pip install numpy wheel
        echo "###### do command: /Users/${USER}/.pyenv/versions/${pak}/bin/python3 -m pip install -r ${SRC_DIR}/python_module/requires-test.txt"
        /Users/${USER}/.pyenv/versions/${pak}/bin/python3 -m pip install -r ${SRC_DIR}/python_module/requires-test.txt
    done
}

function install_cmake() {
    CMAKE_INSTALL_DIR="/Users/${USER}/megengine_use_cmake"
    if [ -f /Users/${USER}/megengine_use_cmake/install/bin/cmake ];then
        echo "find old build cmake, strip..."
    else
        if [ ! -d /Users/${USER}/megengine_use_cmake ];then
            echo "create dir for megengine_use_cmake"
            mkdir -p ${CMAKE_INSTALL_DIR}
        fi

        rm -rf ${CMAKE_INSTALL_DIR}/src/cmake-3.15.2.tar.gz
        mkdir ${CMAKE_INSTALL_DIR}/src
        cd ${CMAKE_INSTALL_DIR}/src
        wget https://cmake.org/files/v3.15/cmake-3.15.2.tar.gz
        tar -xvf cmake-3.15.2.tar.gz
        cd cmake-3.15.2
        mkdir build
        cd build
        ../configure --prefix=${CMAKE_INSTALL_DIR}/install
        make -j$(nproc)
        make install
    fi
}

function append_path_env_message() {
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "install env build megengine macos env successfully"
    echo "pls append path env at build script, if u use customization whl build script"
    echo "append detail:"
    echo "/Users/${USER}/megengine_use_cmake/install/bin/"
    echo "/usr/local/opt/findutils/libexec/gnubin"
    echo "/usr/local/opt/binutils/bin"
    echo "/usr/local/opt/llvm/bin"
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
}

############install env now###########
install_python_package
install_cmake
append_path_env_message
