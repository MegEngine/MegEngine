#!/bin/bash -e

######################################################################################################################
#                                             macos build whl env prepare                                            #
# 1: install xcodebuild for host-build                                                                               #
# 2: install brew: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)" #
# 3: build/install cmake3-14.4                                                                                       #
#    wget https://cmake.org/files/v3.14/cmake-3.14.4.tar.gz                                                          #
#    tar -xzvf cmake-3.14.4.tar.gz;cd cmake-3.14.4;                                                                  #
#    ./configure; make -j32; sudo make install                                                                       #
#                                                                                                                    #
# 4: brew install wget python swig coreutils llvm                                                                    #
#    echo 'export PATH="/usr/local/opt/llvm/bin:$PATH"' >> ~/.zshrc  (if u use other shell, chang this)              #
#                                                                                                                    #
# 5: brew install python@3.8 (sub version may different)                                                             #
#    /usr/local/Cellar/python@3.8/3.8.3_2/bin/pip3 install numpy                                                     #
#                                                                                                                    #
# 6: brew install python@3.7 (sub version may different)                                                             #
#    /usr/local/Cellar/python@3.7/3.7.8_1/bin/pip3 install numpy                                                     #
#                                                                                                                    #
# 7: make link for easy use python3.x (not build necessary)                                                          #
#    ln -s /usr/local/Cellar/python@3.7/3.7.8_1/bin/pip3.7 /usr/local/bin/pip3.7                                     #
#    ln -s /usr/local/Cellar/python@3.7/3.7.8_1/bin/python3.7 /usr/local/bin/python3.7                               #
#    ln -s /usr/local/Cellar/python@3.8/3.8.3_2/bin/pip3.8 /usr/local/bin/pip3.8                                     #
#    ln -s /usr/local/Cellar/python@3.8/3.8.3_2/bin/python3.8 /usr/local/bin/python3.8                               #
######################################################################################################################

READLINK=readlink
OS=$(uname -s)

if [ $OS = "Darwin" ];then
    READLINK=greadlink
else
    echo "ERR: only run at macos env"
    exit -1
fi

SRC_DIR=$($READLINK -f "`dirname $0`/../../")
ALL_PYTHON=${ALL_PYTHON}
if [[ -z ${ALL_PYTHON} ]]
then
    #FIXME: on catalina brew only official support 3.7 and 3.8
    ALL_PYTHON="37 38"
fi

PYTHON_DIR=
PYTHON_LIBRARY=
PYTHON_INCLUDE_DIR=
function config_python_env() {
    if [[ "$1" -eq "38" ]]; then
        PYTHON_DIR=/usr/local/Cellar/python@3.8/3.8.3_2/Frameworks/Python.framework/Versions/3.8/
        PYTHON_LIBRARY=${PYTHON_DIR}lib/libpython3.8.dylib
        PYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python3.8
    elif [[ "$1" -eq "37" ]]; then
        PYTHON_DIR=/usr/local/Cellar/python@3.7/3.7.8_1/Frameworks/Python.framework/Versions/3.7/
        PYTHON_LIBRARY=${PYTHON_DIR}lib/libpython3.7.dylib
        PYTHON_INCLUDE_DIR=${PYTHON_DIR}include/python3.7m
    else
        echo "ERR: DO NOT SUPPORT PYTHON VERSION"
        exit -1
    fi
}

MACOS_WHL_HOME=${SRC_DIR}/scripts/cmake-build/macos_whl_home
if [ -e "${MACOS_WHL_HOME}" ]; then
    echo "remove old macos whl file"
    rm -rf ${MACOS_WHL_HOME}
fi
mkdir -p ${MACOS_WHL_HOME}

for ver in ${ALL_PYTHON}
do
    #config
    config_python_env ${ver}

    #check env
    if [ ! -f "$PYTHON_LIBRARY" ]; then
        echo "ERR: can not find $PYTHON_LIBRARY , Invalid python package"
        exit -1
    fi
    if [ ! -d "$PYTHON_INCLUDE_DIR" ]; then
        echo "ERR: can not find $PYTHON_INCLUDE_DIR , Invalid python package"
        exit -1
    fi
    echo "PYTHON_LIBRARY: ${PYTHON_LIBRARY}"
    echo "PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}"

    #append cmake args for config python
    export EXTRA_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${PYTHON_DIR} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} "
    #config build type to RelWithDebInfo to enable MGB_ENABLE_DEBUG_UTIL etc
    export EXTRA_CMAKE_ARGS=${EXTRA_CMAKE_ARGS}" -DCMAKE_BUILD_TYPE=RelWithDebInfo "

    #call build and install
    #FIXME: cmake do not triger update python config, after
    #change PYTHON_LIBRARY and PYTHON_INCLUDE_DIR, so add
    #-r to remove build cache after a new ver build, which
    #will be more slow build than without -r
    ${SRC_DIR}/scripts/cmake-build/host_build.sh -t -r

    #call setup.py
    BUILD_DIR=${SRC_DIR}/build_dir/host/MGE_WITH_CUDA_OFF/MGE_INFERENCE_ONLY_OFF/Release/build/
    cd ${BUILD_DIR}

    if [ -d "staging" ]; then
        echo "remove old build cache file"
        rm -rf staging
    fi
    mkdir -p staging


    cp -a python_module/{megengine,setup.py,requires.txt,requires-style.txt,requires-test.txt} staging/
    cd ${BUILD_DIR}/staging/megengine/_internal
    #FIXME: set lib suffix to dylib may be better, BUT we find after distutils.file_util.copy_file
    #will change to .so at macos even we set suffix to dylib, at the same time, macos also support .so
    llvm-strip -s _mgb.so
    cd ${BUILD_DIR}/staging
    ${PYTHON_DIR}/bin/python3 setup.py bdist_wheel
    cp ${BUILD_DIR}/staging/dist/Meg*.whl ${MACOS_WHL_HOME}/

    echo ""
    echo "##############################################################################################"
    echo "macos whl package location: ${MACOS_WHL_HOME}"
    ls ${MACOS_WHL_HOME}
    echo "##############################################################################################"
done
