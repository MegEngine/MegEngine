# Build support status
## Host build
* Windows build (cpu and gpu)
* Linux build   (cpu and gpu)
* MacOS build   (cpu only)

## Cross build
* Windows cross build ARM-Android (ok)
* Windows cross build ARM-Linux   (ok)
* Linux cross build ARM-Android   (ok)
* Linux cross build ARM-Linux     (ok)
* MacOS cross build ARM-Android   (ok)
* MacOS cross build ARM-Linux     (ok but experimental)
* MacOS cross build IOS           (ok)

# Build env prepare
## Package install
### Windows host build
* commands:
```
1: installl Visual Studio (need support LLVM/clang-cl), eg 2019. Please install LLVM-10, VS LLVM linker have issue, please replace lld-link.exe, which can be download from https://releases.llvm.org/download.html#10.0.0
2: install extension of VS: Python/Cmake/LLVM
3: now we support cuda10.1+cudnn7.6+TensorRT6.0 on Windows, as Windows can only use DLL in fact with cudnn/TensorRT, so please install the same version;
    3a: install cuda10.1 to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
    3b: install cudnn7.6 to C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-10.1-windows10-x64-v7.6.5.32
    3c: install TensorRT6.0 to C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-6.0.1.5
    3d: add C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin to system PATH env
    3e: add C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-10.1-windows10-x64-v7.6.5.32\cuda\bin to system Path env
    3f: add C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-6.0.1.5\lib Path, if you do not do 4d/4e/4f, CUDA runtime can not find dll
4: install Python3 (default is 3.8.3) to /c/Users/${USER}/mge_whl_python_env/3.8.3 and put it to PATH env and run python3 -m pip install numpy (if you want to build with training mode)
```

### Linux host build
* commands:
```
1: install Cmake, which version >= 3.15.2
2: install gcc/g++, which version >= 6, (gcc/g++ >= 7, if need build training mode)
3: install build-essential git git-lfs gfortran libgfortran-6-dev autoconf gnupg flex bison gperf curl zlib1g-dev gcc-multilib g++-multilib lib32ncurses5-dev libxml2-utils xsltproc unzip libtool librdmacm-dev rdmacm-utils python3-dev swig python3-numpy texinfo
4: CUDA env(if enable CUDA), version detail refer to README.md
```

### MacOS host build
* commands:
```
1: install Cmake, which version >= 3.15.2
2: install brew: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
3: brew install python python3 swig coreutils
4: install at least xcode command line tool: https://developer.apple.com/xcode/
5: about cuda: we do not support CUDA on MacOS
6: python3 -m pip install numpy (if you want to build with training mode)
```

### Cross build for ARM-Android
Now we support Windows/Linux/MacOS cross build to ARM-Android

* commands:
```
1: install unix-like tools, eg MSYS if you are using windows(recommend), we also support CMD.exe or powershell on windows
2: download NDK from https://developer.android.google.cn/ndk/downloads/ for diff OS platform package, suggested NDK20 or NDK21
3: export NDK_ROOT=NDK_DIR at bash-like env
4: config NDK_ROOT to PATH env at windows control board if use CMD/powershell
```

### Cross build for ARM-Linux
Now we support ARM-Linux on Linux and Windows fully, also experimental on MacOS

* commands:
```
1: download toolchains from https://releases.linaro.org/components/toolchain/gcc-linaro/ or https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads if use Windows or Linux
2: download toolchains from https://github.com/thinkski/osx-arm-linux-toolchains if use MacOS
```

### Cross build for IOS
Now we only support cross build to IOS from MACOS
 
* commands:
```
1: install full xcode: https://developer.apple.com/xcode/
```

## Third-party code prepare
With bash env(Linux/MacOS/Unix-Like tools on Windows, eg: msys etc)

* commands:
```
./third_party/prepare.sh
./third_party/install-mkl.sh
```

Windows shell env(eg, CMD, Powershell etc), infact if you can use git command on Windows, which means you always install bash.exe at the same dir of git.exe, find it, then you can prepare third-party code by

* command:
```
bash.exe ./third_party/prepare.sh
bash.exe ./third_party/install-mkl.sh
```

# How to build
## With bash env(Linux/MacOS/Unix-Like tools on Windows, eg: msys etc)

* command:
```
1: host build just use scripts:scripts/cmake-build/host_build.sh
2: cross build to ARM-Android: scripts/cmake-build/cross_build_android_arm_inference.sh
3: cross build to ARM-Linux:   scripts/cmake-build/cross_build_linux_arm_inference.sh
4: cross build to IOS:         scripts/cmake-build/cross_build_ios_arm_inference.sh
```

## Windows shell env(eg, CMD, Powershell etc)

* command:
```
1: we do not provide BAT for CMD/Powershlel scripts, BUT you can refer for scripts/cmake-build/*.sh
```

## Visual Studio GUI(only for Windows host)

* command:
```
1: import megengine src to Visual Studio as a project
2: right click CMakeLists.txt, choose config 'cmake config' choose clang_cl_x86 or clang_cl_x64
3: config other CMAKE config, eg, CUDA ON OR OFF
```


# Other ARM-Linux-Like board support
It`s easy to support other customized arm-linux-like board, example:

* 1: HISI 3516/3519, infact u can just use toolchains from arm developer or linaro
then call scripts/cmake-build/cross_build_linux_arm_inference.sh to build a ELF
binary, or if you get HISI official toolschain, you just need modify CMAKE_CXX_COMPILER
and CMAKE_C_COMPILER in toolchains/arm-linux-gnueabi* to a real name

* 2: about Raspberry, just use scripts/cmake-build/cross_build_linux_arm_inference.sh
