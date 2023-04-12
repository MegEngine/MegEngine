#!/usr/bin/env bash
set -e

NT=$(echo `uname` | grep "NT")
echo $NT
if [ -z "$NT" ];then
    echo "only run at windows bash env"
    echo "pls consider install bash-like tools, eg MSYS or git-cmd, etc"
    exit -1
fi

echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "this scripts use to init windows env, all config is in config.sh, you can modify it"
echo "we do not use windows package manager(winget), because it is not stable, and we need to install some tools which is not in package manager"
echo "so we fallback to manual install depends package by shell, and check package is installed or not by check file exists or not"
echo "which may cause some problem, eg, the file is already exists but at a broken env, this script will skip install the package"
echo "if you want to re-install the package, pls remove the package dir define at scripts/whl/windows/config.sh, and re-run this script"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

# source env
SRC_DIR=$(READLINK -f "`dirname $0`/../../../")
source ${SRC_DIR}/scripts/whl/windows/config.sh

function install_7z() {
    cd ${SRC_DIR}
    # check 7z file exists or not
    if [ ! -f ${ZA_INSTALL_DIR}/7z.exe ];then
        echo "install 7z ..."
        rm -rf ${ZA_INSTALL_DIR}
        mkdir -p ${ZA_INSTALL_DIR}
        cd ${ZA_INSTALL_DIR}
        echo "Download 7za installer from ${ZA_INSTALLER}"
        curl -SL ${ZA_INSTALLER} --output ./7za_installer.exe
        echo "Download 7za console from ${ZA_CONSOLE_URL}"
        curl -SL ${ZA_CONSOLE_URL} --output ./7za_console.exe
        echo "Install 7za to ${ZA_INSTALL_DIR}"
        ./7za_console.exe x -o. ./7za_installer.exe

        if [ ! -f ${ZA_INSTALL_DIR}/7z.exe ];then
            echo "double check 7z install failed,  pls check this shell logic"
            exit -1
        fi
    else
        echo "7z is already installed at ${ZA_INSTALL_DIR}"
    fi
    echo "success install 7z to ${ZA_INSTALL_DIR}, can use 7z cmd after put ${ZA_INSTALL_DIR} to PATH"
    # put 7z to PATH
    export PATH=${ZA_INSTALL_DIR}:$PATH
}

function install_swig() {
    cd ${SRC_DIR}
    # check swig file exists or not
    if [ ! -f ${SWIG_INSTALL_DIR}/swig.exe ];then
        echo "install swig ..."
        rm -rf ${SWIG_INSTALL_DIR}
        mkdir -p ${SWIG_INSTALL_DIR}
        cd ${SWIG_INSTALL_DIR}
        cd ..
        # download swig installer
        curl.exe -SL ${SWIG_INSTALLER_URL} --output swig.zip
        unzip -X swig.zip
        cd swigwin-${SWIG_VER}
        git init
        git add -A
        git commit -m "init"
        echo "apply patch for swig"
        git apply ${SRC_DIR}/scripts/whl/windows/fix-ptr-define-issue.patch
        cd ..
        rm -rf swig.zip

        if [ ! -f ${SWIG_INSTALL_DIR}/swig.exe ];then
            echo "double check swig install failed,  pls check this shell logic"
            exit -1
        fi
    else
        echo "swig is already installed at ${SWIG_INSTALL_DIR}"
    fi
    echo "swig install success"
}

function install_python() {
    cd ${SRC_DIR}
    mkdir -p ${PYTHON3_MEGENGINE_DEV_DIR}
    cd ${PYTHON3_MEGENGINE_DEV_DIR}

    # clone pyenv-win
    echo "clone pyenv-win"
    mkdir -p pyenv-win
    cd pyenv-win
    if cat .git/config | grep pyenv-win.git;then
        echo "pyenv-win is already cloned, just do git reset --hard"
        if git reset --hard;then
            echo "git reset success"
        else
            echo "git reset failed, try init again"
            git init
            git remote rm origin || true
            git remote add origin https://github.com/pyenv-win/pyenv-win.git
            git pull origin master
        fi
    else
        git init
        git remote rm origin || true
        git remote add origin https://github.com/pyenv-win/pyenv-win.git
        git pull origin master
    fi

    for ver in ${FULL_PYTHON_VER}
    do
        echo "install python ${ver}"
        echo "if your env network is not good, pls download python installer \
            from python ftp or other mirrors, eg, \
            https://www.python.org/ftp/python/${ver}/python-${ver}-amd64.exe, \
            and put it to ${PYTHON3_MEGENGINE_DEV_DIR}/pyenv-win/install_cache/"

        pyenv-win/bin/pyenv install ${ver}
        # check file exist
        if [ ! -f "${PYTHON3_MEGENGINE_DEV_DIR}/pyenv-win/pyenv-win/versions/${ver}/python3.exe" ]; then
            echo "python ${ver} install failed"
            exit -1
        else
            echo "python ${ver} install success, now install depends"
        fi

        pyenv-win/versions/${ver}/python3.exe -m pip install --upgrade pip -i ${PYTHON_PACK_MIRROR}
        if [ ${ver} == "3.6.8" ]; then
            pyenv-win/versions/${ver}/python3.exe -m pip install opencv-python==4.6.0.66 -i ${PYTHON_PACK_MIRROR}
        fi

        # FIXME: imperative/python/requires.txt numpy version limit have some issue, eg. some version numpy
        # will cause some test case failed, so we need to install numpy first, then install other depends
        numpy_version="1.21.6"
        if [ ${ver} = "3.6.8" ];then
            numpy_version="1.19.5"
        elif [ ${ver} = "3.10.1" ];then
            numpy_version="1.23.0"
        fi
        pyenv-win/versions/${ver}/python3.exe -m pip install numpy==${numpy_version}
        pyenv-win/versions/${ver}/python3.exe -m pip install cython wheel -i ${PYTHON_PACK_MIRROR}
        pyenv-win/versions/${ver}/python3.exe -m pip install -r ${SRC_DIR}/imperative/python/requires.txt -i ${PYTHON_PACK_MIRROR}
        pyenv-win/versions/${ver}/python3.exe -m pip install -r ${SRC_DIR}/imperative/python/requires-test.txt -i ${PYTHON_PACK_MIRROR}
    done
    echo "install python packages done, put ${PYTHON3_MEGENGINE_DEV_DIR}/pyenv-win/pyenv-win/versions/xxx to PATH to use it"
}

function install_llvm() {
    cd ${SRC_DIR}
    # check 7z file exists or not
    if [ ! -f ${ZA_INSTALL_DIR}/7z.exe ];then
        echo "install 7z ..."
        install_7z
    fi

    # put 7z to PATH
    export PATH=${ZA_INSTALL_DIR}:$PATH

    # check llvm file exists or not
    if [ ! -f ${LLVM_MEGENGINE_DEV_DIR}/bin/clang.exe ];then
        echo "install llvm ..."
        rm -rf ${LLVM_MEGENGINE_DEV_DIR}
        mkdir -p ${LLVM_MEGENGINE_DEV_DIR}
        cd ${LLVM_MEGENGINE_DEV_DIR}
        # download llvm installer
        curl.exe -SL ${LLVM_INSTALLER_URL} --output llvm_installer.exe
        # install llvm by 7z
        7z.exe x -o${LLVM_MEGENGINE_DEV_DIR} llvm_installer.exe
        # rm llvm_installer.exe
        rm -rf llvm_installer.exe

        if [ ! -f ${LLVM_MEGENGINE_DEV_DIR}/bin/clang.exe ];then
            echo "double check llvm install failed, pls check this shell logic"
            exit -1
        fi
    else
        echo "llvm is already installed at ${LLVM_MEGENGINE_DEV_DIR}"
    fi

    echo "llvm install success"
}

function install_vs() {
    # Install Visual Studio Build Tools
    # Reference: https://learn.microsoft.com/en-us/visualstudio/install/use-command-line-parameters-to-install-visual-studio?view=vs-2019
    # Component IDS:https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools?view=vs-2019
    cd ${SRC_DIR}
    # check vs file exists or not
    if [ ! -f ${VS_INSTALL_PATH}/VC/Auxiliary/Build/vcvars64.bat ];then
        echo "install vs ..."
        rm -rf ${VS_INSTALL_PATH}
        mkdir -p ${VS_INSTALL_PATH}
        cd ${VS_INSTALL_PATH}
        # vs_buildtools.exe can not at the same dir with VS_BUILD_TOOLS_URL, which will cause the install failed
        cd ..
        curl -SL ${VS_BUILD_TOOLS_URL} --output ./vs_buildtools.exe
        echo "Try uninstall old install..."
        if ./vs_buildtools.exe --uninstall --installPath $PWD/vs --quiet --norestart --wait; then
            echo "Uninstall old install done"
        else
            echo "Uninstall old install failed, ingore this error"
        fi
        echo "Start to install vs2019 16 version to ${VS_INSTALL_PATH} with WIN_SDK_VER:${WIN_SDK_VER} and VC_VER:${VC_VER}, please wait..."
        if ./vs_buildtools.exe --installPath $PWD/vs --nocache --wait --quiet --norestart --noweb \
            --add Microsoft.Component.MSBuild \
            --add Microsoft.VisualStudio.Component.Roslyn.Compiler \
            --add Microsoft.VisualStudio.Component.Windows10SDK.${WIN_SDK_VER} \
            --add Microsoft.VisualStudio.Workload.VCTools \
            --add Microsoft.VisualStudio.Component.TextTemplating \
            --add Microsoft.VisualStudio.Component.VC.CoreIde \
            --add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Core \
            --add Microsoft.VisualStudio.Component.VC.CMake.Project \
            --add Microsoft.VisualStudio.Component.VC.${VC_VER}.x86.x64; then
                    echo "Install vs2019 16 version to ${VS_INSTALL_PATH} with WIN_SDK_VER:${WIN_SDK_VER} and VC_VER:${VC_VER} done"
                else
                    echo "Install vs2019 16 version to ${VS_INSTALL_PATH} with WIN_SDK_VER:${WIN_SDK_VER} and VC_VER:${VC_VER} failed"
                    echo "now get the install log"
                    curl.exe -o vscollect.exe -SL "https://aka.ms/vscollect.exe"
                    ./vscollect.exe
                    # FIXME: why windows tools so stupid, do not work perfect from terminal, also the log need collect by another tool..
                    for i in {1..20}; do echo "also may uninstall failed from CMD, try uninstall from GUI: by click vs_buildtools.exe add uninstall broken install at UI side"; done
                    exit -1
        fi

        if [ ! -f ${VS_INSTALL_PATH}/VC/Auxiliary/Build/vcvars64.bat ];then
            echo "double check vs install failed, pls check this shell logic"
            exit -1
        fi
    else
        echo "vs is already installed at ${VS_INSTALL_PATH}"
    fi
    echo "vs install success"
}

##########################################################
# windows shell env not stable, so you can run this script
# step by step by comment some function
##########################################################
DONE_MSG="install all dev env(except cuda) done"
install_7z
install_swig
install_python
install_llvm
install_vs
echo ${DONE_MSG}
