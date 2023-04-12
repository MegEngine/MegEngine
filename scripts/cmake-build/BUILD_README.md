# Build support status
## Host build
* Windows build (cpu and gpu)
* Linux build   (cpu and gpu)
* MacOS build   (cpu only)
* Android build (cpu only) at [termux](https://termux.com/) env

## Cross build
* Windows cross build ARM-Android (ok)
* Windows cross build ARM-Linux   (ok)
* Linux cross build ARM-Android   (ok)
* Linux cross build ARM-Linux     (ok)
* Linux cross build RISCV(support [rvv](https://github.com/riscv/riscv-v-spec))-Linux     (ok)
* MacOS cross build ARM-Android   (ok)
* MacOS cross build ARM-Linux     (ok but experimental)
* MacOS cross build IOS           (ok)

# Build env prepare
## Prerequisites

Most of the dependencies of MegBrain(MegEngine) are located in [third_party](../../third_party) directory, which can be prepared by executing:

```bash
./third_party/prepare.sh
./third_party/install-mkl.sh
```
Windows shell env(bash from windows-git), infact if you can use git command on Windows, which means you always install bash.exe at the same dir of git.exe, find it, then you can prepare third-party code by

* command:
```
bash.exe ./third_party/prepare.sh
bash.exe ./third_party/install-mkl.sh
if you are use github MegEngine and build for Windows XP, please
	1: donwload mkl for xp from: http://registrationcenter-download.intel.com/akdlm/irc_nas/4617/w_mkl_11.1.4.237.exe
	2: install exe, then from install dir:
		2a: cp include file to third_party/mkl/x86_32/include/
		2b: cp lib file to third_party/mkl/x86_32/lib/
```

About `third_party/prepare.sh`, also support to be managed by `CMake`, just config `EXTRA_CMAKE_ARGS="-DMGE_SYNC_THIRD_PARTY=ON"` before run `scripts/cmake-build/*.sh`

But some dependencies need to be installed manually:

* [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)(>=10.1), [cuDNN](https://developer.nvidia.com/cudnn)(>=7.6) are required when building MegBrain with CUDA support.
* [TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)(>=5.1.5) is required when building with TensorRT support.
* LLVM/Clang(>=6.0) is required when building with Halide JIT support.
* Python(>=3.6) and numpy are required to build Python modules.
## Package install
### Windows host build
* commands:
```
0: about all windows config
    * please check scripts/whl/windows/config.sh
1: install git (Windows GUI)
	* download git-install.exe from https://git-scm.com/download/win
	* only need choose git-lfs component
	* install to default dir:  /c/Program\ Files/Git
2: install cuda components (Windows GUI) if you want to build with CUDA
    * download cuda/cudnn/trt from NVIDIA or by scripts/whl/windows/cuda_cudnn_install.py
    * export CUDA_ROOT_DIR/CUDNN_ROOT_DIR/TRT_ROOT_DIR to real cuda/cudnn/trt location
3: install all env except cuda env
    * just run scripts/whl/windows/env_prepare.sh
4: edit system env variables (Windows GUI)
	* append "Path" env value
        C:\Program Files\Git\cmd
		C:\Users\build\megengine_dev_tools\pyenv-win\pyenv-win\versions\3.10.1
        change `build` to your real user name
```

### Linux host build
* commands:
```
0: we provide Dockerfile if you do not prepare local env
    * check about scripts/whl/manylinux2014/build_image.sh
1: install Cmake, which version >= 3.15.2, ninja-build
2: install gcc/g++, which version >= 6, (gcc/g++ >= 7, if need build training mode)
3: install build-essential git git-lfs gfortran libgfortran-6-dev autoconf gnupg flex bison gperf curl zlib1g-dev gcc-multilib g++-multilib lib32ncurses5-dev libxml2-utils xsltproc unzip libtool librdmacm-dev rdmacm-utils python3-dev python3-numpy texinfo
4: CUDA env(if build with CUDA), please export CUDA/CUDNN/TRT env, for example:
export CUDA_ROOT_DIR=/path/to/cuda
export CUDNN_ROOT_DIR=/path/to/cudnn
export TRT_ROOT_DIR=/path/to/tensorrt
```

### MacOS host build
* commands:
```
1: install Cmake, which version >= 3.15.2
2: install brew: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
3: brew install python python3 coreutils ninja
4: install at least xcode command line tool: https://developer.apple.com/xcode/
5: about cuda: we do not support CUDA on MacOS
6: python3 -m pip install numpy (if you want to build with training mode)
```

### Cross build for ARM-Android
Now we support Windows/Linux/MacOS cross build to ARM-Android

* commands:
```
2: download NDK from https://developer.android.google.cn/ndk/downloads/ for diff OS platform package, suggested NDK20 or NDK21
3: export NDK_ROOT=NDK_DIR at bash-like env
```

### Cross build for ARM-Linux
Now we support ARM-Linux on Linux and Windows fully, also experimental on MacOS

* commands:
```
1: download toolchains from http://releases.linaro.org/components/toolchain/binaries/ or https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads if use Windows or Linux
2: download toolchains from https://github.com/thinkski/osx-arm-linux-toolchains if use MacOS
```

### Cross build for RISCV-Linux
Now we support RISCV-Linux

* commands:
```
1: download toolchains from https://github.com/riscv-collab/riscv-gnu-toolchain
```

### Cross build for IOS
Now we only support cross build to IOS from MACOS

* commands:
```
1: install full xcode: https://developer.apple.com/xcode/
```

# How to build
## With bash env(Linux/MacOS/Windows-git-bash)

* host build just use scripts:scripts/cmake-build/host_build.sh
  builds MegBrain(MegEngine) that runs on the same host machine (i.e., no cross compiling)
  The following command displays the usage:
  ```
  scripts/cmake-build/host_build.sh -h
    more example:
    1a: build for Windows for XP (sp3): (dbg) EXTRA_CMAKE_ARGS="-DMGE_DEPLOY_INFERENCE_ON_WINDOWS_XP=ON" ./scripts/cmake-build/host_build.sh -m -d
                                        (opt) EXTRA_CMAKE_ARGS="-DMGE_DEPLOY_INFERENCE_ON_WINDOWS_XP=ON" ./scripts/cmake-build/host_build.sh -m
    2a: build for Windows for XP (sp2): (dbg) EXTRA_CMAKE_ARGS="-DMGE_DEPLOY_INFERENCE_ON_WINDOWS_XP_SP2=ON" ./scripts/cmake-build/host_build.sh -m -d
                                        (opt) EXTRA_CMAKE_ARGS="-DMGE_DEPLOY_INFERENCE_ON_WINDOWS_XP_SP2=ON" ./scripts/cmake-build/host_build.sh -m
  ```
* cross build to ARM-Android: scripts/cmake-build/cross_build_android_arm_inference.sh
  builds MegBrain(MegEngine) for inference on Android-ARM platforms.
  The following command displays the usage:
  ```
  scripts/cmake-build/cross_build_android_arm_inference.sh -h
  ```
* cross build to ARM-Linux:   scripts/cmake-build/cross_build_linux_arm_inference.sh
  builds MegBrain(MegEngine) for inference on Linux-ARM platforms.
  The following command displays the usage:
  ```
  scripts/cmake-build/cross_build_linux_arm_inference.sh -h
  ```
* cross build to RISCV-Linux:   scripts/cmake-build/cross_build_linux_riscv_inference.sh
  builds MegBrain(MegEngine) for inference on Linux-RISCV platforms.
  The following command displays the usage:
  
  ```
  scripts/cmake-build/cross_build_linux_riscv_inference.sh -h
  ```
  
  * if board support RVV(at least 0.7), for  example: nezha D1 , use -a rv64gcv0p7
  * if board do not support RVV, use -a rv64norvv
* cross build to IOS:         scripts/cmake-build/cross_build_ios_arm_inference.sh
  builds MegBrain(MegEngine) for inference on iOS (iPhone/iPad) platforms.
  The following command displays the usage:
  
  ```
  scripts/cmake-build/cross_build_ios_arm_inference.sh -h
  ```

## Visual Studio GUI(only for Windows host)

* command:
```
1: import megengine src to Visual Studio as a project
2: right click CMakeLists.txt, choose config 'cmake config' choose clang_cl_x86 or clang_cl_x64
3: config other CMAKE config, eg, CUDA ON OR OFF
```


## Other ARM-Linux-Like board support
It`s easy to support other customized arm-linux-like board, example:

* 1: HISI 3516/3519, infact u can just use toolchains from arm developer or linaro
then call scripts/cmake-build/cross_build_linux_arm_inference.sh to build a ELF
binary, or if you get HISI official toolschain, you just need modify CMAKE_CXX_COMPILER
and CMAKE_C_COMPILER in toolchains/arm-linux-gnueabi* to a real name

* 2: about Raspberry, just use scripts/cmake-build/cross_build_linux_arm_inference.sh

## About build args

All `scripts/cmake-build/*.sh`  support `EXTRA_CMAKE_ARGS` to  config more options

* get support options by `-l`,  for example: `scripts/cmake-build/cross_build_android_arm_inference.sh -l`
* CMake support `Release`, `Debug`, `RelWithDebInfo` build type, all `scripts/cmake-build/*.sh` default build type is `Release`, can build `Debug` type with `-d`, if you want to build with `RelWithDebInfo`, you can config with `EXTRA_CMAKE_ARGS`, for example: `EXTRA_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=RelWithDebInfo" ./scripts/cmake-build/host_build.sh`, Notice: when build with `Release` , we will disable some build components:  `RTTI`, `MGB_ASSERT_LOC`, and `MGB_ENABLE_DEBUG_UTIL`
* CMake build all targets by default, if you just want build a specified target,  you can build with `-e xxxx `, for example, only build with `lite_shared `: `./scripts/cmake-build/cross_build_android_arm_inference.sh -e lite_shared` ,    Notice: when with `-e`, will do not strip target, always for debug or need strip target manually
* About others build flag, please run with flag `-h`
