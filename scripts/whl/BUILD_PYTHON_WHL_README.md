# Python whl package build support status
* Windows build (cpu and gpu)
* Linux build (cpu and gpu)
* MacOS build (cpu only)

# Build env prepare
## Linux

```bash
1: please refer to: https://docs.docker.com/engine/security/rootless/ to enable rootless docker env
2: cd ./scripts/whl/manylinux2010
3: ./build_image.sh

```

## MacOS
```bash
./scripts/whl/macos/macos_whl_env_prepare.sh
```

## Windows
```
1: refer to scripts/cmake-build/BUILD_README.md Windows section build for base windows build env prepare
2: install several python or install your care about python version, default install dir: /c/Users/${USER}/mge_whl_python_env
    a: mkdir /c/Users/${USER}/mge_whl_python_env
    b: download python 64-bit install exe
        https://www.python.org/ftp/python/3.5.4/python-3.5.4-amd64.exe
        https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe
        https://www.python.org/ftp/python/3.7.7/python-3.7.7-amd64.exe
        https://www.python.org/ftp/python/3.8.3/python-3.8.3-amd64.exe
    c: install python-3.5.4-amd64.exe to /c/Users/${USER}/mge_whl_python_env/3.5.4 from install gui
    d: install python-3.6.8-amd64.exe to /c/Users/${USER}/mge_whl_python_env/3.6.8 from install gui
    e: install python-3.7.7-amd64.exe to /c/Users/${USER}/mge_whl_python_env/3.7.7 from install gui
    f: install python-3.8.3-amd64.exe to /c/Users/${USER}/mge_whl_python_env/3.8.3 from install gui
3: cp python.exe to python3.exe
    a: mv /c/Users/${USER}/mge_whl_python_env/3.5.4/python.exe /c/Users/${USER}/mge_whl_python_env/3.5.4/python3.exe
    b: mv /c/Users/${USER}/mge_whl_python_env/3.6.8/python.exe /c/Users/${USER}/mge_whl_python_env/3.6.8/python3.exe
    c: mv /c/Users/${USER}/mge_whl_python_env/3.7.7/python.exe /c/Users/${USER}/mge_whl_python_env/3.7.7/python3.exe
    d: mv /c/Users/${USER}/mge_whl_python_env/3.8.3/python.exe /c/Users/${USER}/mge_whl_python_env/3.8.3/python3.exe
4: install needed package for build python whl package
    a0: /c/Users/${USER}/mge_whl_python_env/3.5.4/python3.exe -m pip install --upgrade pip
    a1: /c/Users/${USER}/mge_whl_python_env/3.5.4/python3.exe -m pip install -r imperative/python/requires-test.txt
    a2: /c/Users/${USER}/mge_whl_python_env/3.5.4/python3.exe -m pip install numpy wheel requests tqdm tabulate

    b0: /c/Users/${USER}/mge_whl_python_env/3.6.8/python3.exe -m pip install --upgrade pip
    b1: /c/Users/${USER}/mge_whl_python_env/3.6.8/python3.exe -m pip install -r imperative/python/requires-test.txt
    b2: /c/Users/${USER}/mge_whl_python_env/3.6.8/python3.exe -m pip install numpy wheel requests tqdm tabulate
    
    c0: /c/Users/${USER}/mge_whl_python_env/3.7.7/python3.exe -m pip install --upgrade pip
    c1: /c/Users/${USER}/mge_whl_python_env/3.7.7/python3.exe -m pip install -r imperative/python/requires-test.txt
    c2: /c/Users/${USER}/mge_whl_python_env/3.7.7/python3.exe -m pip install numpy wheel requests tqdm tabulate
    
    d0: /c/Users/${USER}/mge_whl_python_env/3.8.3/python3.exe -m pip install --upgrade pip
    d1: /c/Users/${USER}/mge_whl_python_env/3.8.3/python3.exe -m pip install -r imperative/python/requires-test.txt
    d2: /c/Users/${USER}/mge_whl_python_env/3.8.3/python3.exe -m pip install numpy wheel requests tqdm tabulate
```

# How to build
## Build for linux
* MegBrain delivers `wheel` package with `manylinux2010` tag defined in [PEP-571](https://www.python.org/dev/peps/pep-0571/).
commands:
```bash
export CUDA_ROOT_DIR=/path/to/cuda
export CUDNN_ROOT_DIR=/path/to/cudnn
export TENSORRT_ROOT_DIR=/path/to/tensorrt
./scripts/whl/manylinux2010/build_wheel.sh
```

* And you can find all of the outputs in `output` directory.If you just want to build for a specific Python verison, you can use `ALL_PYTHON` environment variable. eg:
```bash
ALL_PYTHON="36m" ./scripts/whl/manylinux2010/build_wheel.sh
```

* If you just want to build with cpu only version, you can set `BUILD_WHL_CPU_ONLY` environment 'ON'. eg:
```bash
BUILD_WHL_CPU_ONLY="ON" ALL_PYTHON="36m" ./scripts/whl/manylinux2010/build_wheel.sh
```

## Build for MacOS
* commands:
```bash
./scripts/whl/macos/macos_build_whl.sh
```
* If you just want to build for a specific Python verison, you can use `ALL_PYTHON` environment variable. eg:
```bash
ALL_PYTHON="3.7.7" ./scripts/whl/macos/macos_build_whl.sh
```

## Build for Windows
* commands:
```bash
./scripts/whl/windows/windows_build_whl.sh
```

* If you just want to build for a specific Python verison, you can use `ALL_PYTHON` environment variable. eg:
```bash
ALL_PYTHON="3.8.3" ./scripts/whl/windows/windows_build_whl.sh
```

* If you just want to build with cpu only version, you can set `BUILD_WHL_CPU_ONLY` environment 'ON'. eg:
```
BUILD_WHL_CPU_ONLY="ON" ALL_PYTHON="3.8.3" ./scripts/whl/windows/windows_build_whl.sh
```
