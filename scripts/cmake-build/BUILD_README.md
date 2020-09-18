# build support status
##  host build
*  windows build (cpu + gpu)
*  linux build   (cpu + gpu)
*  macos build   (cpu only)
##  cross build
*  windows cross build arm-android (ok)
*  windows cross build arm-linux   (ok)
*  linux cross build arm-android   (ok)
*  linux cross build arm-linux     (ok)
*  macos cross build arm-android   (ok)
*  macos cross build arm-linux     (ok but experimental)
*  macos cross build ios           (ok)

# build env prepare
## package install
### windows host build
    ```
    1: installl Visual Studio (need support LLVM/clang-cl), eg 2019
    pls install LLVM-10, VS llvm linker have issue, pls replace lld-link.exe,
    download from https://releases.llvm.org/download.html#10.0.0
    2: install extension of VS: python/cmake/LLVM
    3: CUDA env(if enable CUDA), version detail: project_root_dir/README.md
    4: now we support cuda10.1+cudnn7.6+TensorRT6.0 on windows, as windows can
    only use dll in fact with cudnn/TensorRT, so please install the same version;
    4a: install cuda10.1 to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
    4b: install cudnn7.6 to C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-10.1-windows10-x64-v7.6.5.32
    4c: install TensorRT6.0 to C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-6.0.1.5
    4d: add C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin to system Path env
    4e: add C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-10.1-windows10-x64-v7.6.5.32\cuda\bin to system Path env
    4f: add C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-6.0.1.5\lib Path
    if u do not do 4d/4e/4f, CUDA runtime can not find dll
    5: install python3 (DFT 3.8.3) to /c/Users/${USER}/mge_whl_python_env/3.8.3 and
    put it to PATH env and run python3 -m pip install numpy (if u want to build with training mode or build python whl)
    ```
### linux host build
    ```
    1: cmake, which version > 3.15.2
    2: gcc/g++, which version > 6, (gcc/g++ >= 7, if need build training)
    3: install build-essential git git-lfs gfortran libgfortran-6-dev autoconf gnupg flex bison gperf curl 
    4: zlib1g-dev gcc-multilib g++-multilib lib32ncurses5-dev libxml2-utils xsltproc unzip libtool:
    5: librdmacm-dev rdmacm-utils python3-dev swig python3-numpy texinfo
    6: CUDA env(if enable CUDA), version detail: project_root_dir/README.md
    ```
### macos host build
    ```
    1: cmake, which version > 3.15.2
    2: install brew: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
    3: brew install python python3 swig coreutils
    4: install at least xcode command line tool: https://developer.apple.com/xcode/
    5: about cuda: we do not support CUDA on macos
    6: python3 -m pip install numpy (if u want to build with training mode or build python whl)
    ```
### cross build for arm-android
    now we support windows/linux/macos cross build to arm-android
    ```
    1: install unix-like tools, eg MSYS if you are using windows(recommend)
    we also support CMD.exe or powershell on windows
    1: download NDK from https://developer.android.google.cn/ndk/downloads/
    for diff OS platform package, suggested NDK20 or NDK21
    2: export NDK_ROOT=NDK_DIR at bash-like env
    3: config NDK_ROOT to PATH env at windows control board if use CMD/powershell
    ```
### cross build for arm-linux
    now we support arm-linux on linux and windows fully, also experimental on MACOS
    ```
    1: download toolchains on https://releases.linaro.org/components/toolchain/gcc-linaro/
    or https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads
    if use windows or linux
    2: download https://github.com/thinkski/osx-arm-linux-toolchains if use MACOS
    ```
### cross build for ios
    now we only support cross build to ios on MACOS
    ```
    1: install full xcode: https://developer.apple.com/xcode/
    ```
## third-party code prepare
### with bash env(linux/macos/unix-like tools on windows, eg: msys etc)
    ```
    ./third_party/prepare.sh
    ./third_party/install-mkl.sh
    ```
### windows shell env(eg: cmd, powershell etc)
    infact if you can use git command on windows, which means you always install
    bash.exe at the same dir of git.exe, find it, then you can prepare third-party
    code by command:
    ```
    bash.exe ./third_party/prepare.sh
    bash.exe ./third_party/install-mkl.sh
    ```
# how to build
## with bash env(linux/macos/unix-like tools on windows, eg: msys etc)
    ```
    1: host build just use scripts:scripts/cmake-build/host_build.sh
    2: cross build to arm-android: scripts/cmake-build/cross_build_android_arm_inference.sh
    3: cross build to arm-linux:   scripts/cmake-build/cross_build_linux_arm_inference.sh
    4: cross build to ios:         scripts/cmake-build/cross_build_ios_arm_inference.sh
    ```
## windows shell env(eg: cmd, powershell etc)
    ```
    1: we do not provide BAT for cmd/powershlel scripts, BUT u can refs for scripts/cmake-build/*.sh
    ```
## Visual Studio GUI(only for windows host)
    ```
    1: import megengine src to Visual Studio as a project
    2: right click CMakeLists.txt, choose config 'cmake config'
       choose clang_cl_x86 or clang_cl_x64
    3: config other CMAKE config, eg, CUDA ON OR OFF
    ```


# other arm-linux-like board support
it`s easy to support other customized arm-linux-like board, example:
1: HISI 3516/3519, infact u can just use toolchains from arm developer or linaro
then call scripts/cmake-build/cross_build_linux_arm_inference.sh to build a ELF
binary, or if you get HISI official toolschain, you just need modify CMAKE_CXX_COMPILER
and CMAKE_C_COMPILER in toolchains/arm-linux-gnueabi* to a real name

2: about Raspberry, just use scripts/cmake-build/cross_build_linux_arm_inference.sh
