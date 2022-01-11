# Python whl package build support status
* Windows build (cpu and gpu)
* Linux build (cpu and gpu)
* MacOS build (cpu only)
* Android(termux) build (cpu only)

# Build env prepare
## Linux

* refer to [BUILD_README.md](../cmake-build/BUILD_README.md) Linux host build(CUDA env) section to init CUDA environment
```bash
1: please refer to https://docs.docker.com/engine/security/rootless/ to enable rootless docker env
2: cd ./scripts/whl/manylinux2014
3: ./build_image.sh
4: as aarch64-linux python3.5 pip do not provide megengine depends prebuild binary package, which definition
    in imperative/python/requires.txt, before install megengine wheel package(just python3.5), you need
    do follow step to init python3.5 arrow build env:
        4a: wget https://apache.bintray.com/arrow/debian/apache-arrow-archive-keyring-latest-buster.deb
        4b: sudo apt install ./apache-arrow-archive-keyring-latest-buster.deb
        4c: sudo apt update
        4d: sudo apt install libarrow-dev
        4c: sudo apt install libarrow-python-dev
        4e: sudo ln -s /usr/include/locale.h /usr/include/xlocale.h

```

## MacOS
* refer to [BUILD_README.md](../cmake-build/BUILD_README.md) MacOS section to init base build environment
* init other wheel build depends env by command:
```bash
./scripts/whl/macos/macos_whl_env_prepare.sh
```

## Android
* install [termux](https://termux.com/) apk on Android Device
    * at least 8G DDR
    * at least Android 7
* init wheel build-dependent env by command:
```bash
./scripts/whl/android/android_whl_env_prepare.sh
```

## Windows
* refer to [BUILD_README.md](../cmake-build/BUILD_README.md) Windows section to init base build environment

# How to build
Note: Guarantee the git repo is mounted in docker container, do not use `git submodule update --init` in to init Project repo
## Build for linux
* This Project delivers `wheel` package with `manylinux2014` tag defined in [PEP-571](https://www.python.org/dev/peps/pep-0571/).

commands:
```bash
./scripts/whl/manylinux2014/build_wheel_common.sh -sdk cu101
```

* And you can find all of the outputs in `output` directory.If you just want to build for a specific Python verison, you can use `ALL_PYTHON` environment variable. such as:
```bash
ALL_PYTHON="36m" ./scripts/whl/manylinux2014/build_wheel_common.sh -sdk cu101
```

* If you just want to build with cpu only version, you can set `-sdk` environment 'cpu'. such as:
```bash
ALL_PYTHON="36m" ./scripts/whl/manylinux2014/build_wheel_common.sh -sdk cpu
```

## Build for MacOS
* commands:
```bash
./scripts/whl/macos/macos_build_whl.sh
```
* If you just want to build for a specific Python verison, you can use `ALL_PYTHON` environment variable. such as:
```bash
ALL_PYTHON="3.7.7" ./scripts/whl/macos/macos_build_whl.sh
```

## Build for Windows
* commands:
```bash
./scripts/whl/windows/windows_build_whl.sh
```

* If you just want to build for a specific Python verison, you can use `ALL_PYTHON` environment variable. such as:
```bash
ALL_PYTHON="3.8.3" ./scripts/whl/windows/windows_build_whl.sh
```

* If you just want to build with cpu only version, you can set `BUILD_WHL_CPU_ONLY` environment 'ON'. such as:
```
BUILD_WHL_CPU_ONLY="ON" ALL_PYTHON="3.8.3" ./scripts/whl/windows/windows_build_whl.sh
```

## Build for Android
* commands:
```bash
scripts/whl/android/android_build_whl.sh
```
* If you just want to build for a specific Python verison, you can use `ALL_PYTHON` environment variable. such as:
```bash
ALL_PYTHON="3.10.1" ./scripts/whl/android/android_build_whl.sh
```
