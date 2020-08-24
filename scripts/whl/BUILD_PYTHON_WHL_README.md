# python whl package build support status
*  windows build (cpu + gpu)
*  linux   build (cpu + gpu)
*  macos   build (cpu only)
# build env prepare
## linux

    ```
    1: enable rootless docker env, refs: https://docs.docker.com/engine/security/rootless/
    2: cd ./scripts/whl/linux/manylinux2010
    3: ./build_image.sh cpu
    4: ./build_image.sh cuda

    ```

## macos
    ```
    ./scripts/whl/macos/macos_whl_env_prepare.sh
    ```

## windows
    ```
    1: refs scripts/cmake-build/BUILD_README.md windows section build for base windows build
    2: install several python or install u care python version, default install dir: /c/Users/${USER}/mge_whl_python_env
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
    3: rename python.exe to python3.exe
       a: mv /c/Users/${USER}/mge_whl_python_env/3.5.4/python.exe /c/Users/${USER}/mge_whl_python_env/3.5.4/python3.exe
       b: mv /c/Users/${USER}/mge_whl_python_env/3.6.8/python.exe /c/Users/${USER}/mge_whl_python_env/3.6.8/python3.exe
       c: mv /c/Users/${USER}/mge_whl_python_env/3.7.7/python.exe /c/Users/${USER}/mge_whl_python_env/3.7.7/python3.exe
       d: mv /c/Users/${USER}/mge_whl_python_env/3.8.3/python.exe /c/Users/${USER}/mge_whl_python_env/3.8.3/python3.exe
    4: install needed package for build python whl package
       a0: /c/Users/${USER}/mge_whl_python_env/3.5.4/python3.exe -m pip install --upgrade pip
       a1: /c/Users/${USER}/mge_whl_python_env/3.5.4/python3.exe -m pip install -r python_module/requires-test.txt
       a2: /c/Users/${USER}/mge_whl_python_env/3.5.4/python3.exe -m pip install numpy wheel requests tqdm tabulate

       b0: /c/Users/${USER}/mge_whl_python_env/3.6.8/python3.exe -m pip install --upgrade pip
       b1: /c/Users/${USER}/mge_whl_python_env/3.6.8/python3.exe -m pip install -r python_module/requires-test.txt
       b2: /c/Users/${USER}/mge_whl_python_env/3.6.8/python3.exe -m pip install numpy wheel requests tqdm tabulate

       c0: /c/Users/${USER}/mge_whl_python_env/3.7.7/python3.exe -m pip install --upgrade pip
       c1: /c/Users/${USER}/mge_whl_python_env/3.7.7/python3.exe -m pip install -r python_module/requires-test.txt
       c2: /c/Users/${USER}/mge_whl_python_env/3.7.7/python3.exe -m pip install numpy wheel requests tqdm tabulate

       d0: /c/Users/${USER}/mge_whl_python_env/3.8.3/python3.exe -m pip install --upgrade pip
       d1: /c/Users/${USER}/mge_whl_python_env/3.8.3/python3.exe -m pip install -r python_module/requires-test.txt
       d2: /c/Users/${USER}/mge_whl_python_env/3.8.3/python3.exe -m pip install numpy wheel requests tqdm tabulate
    ```

# how to build
## build for linux
    MegBrain delivers `wheel` package with `manylinux2010` tag defined in [PEP-571](https://www.python.org/dev/peps/pep-0571/).

    ```
    ./build_wheel.sh cpu
    
    CUDA_ROOT_DIR=/path/to/cuda \
    CUDNN_ROOT_DIR=/path/to/cudnn \
    TENSORRT_ROOT_DIR=/path/to/tensorrt \
    ./build_wheel.sh cuda
    ```

    And you can find all of the outputs in `output` directory.
    
    If you just want to build for a specific Python verison, you can use `ALL_PYTHON` environment variable. eg:

    ```
    ALL_PYTHON=35m ./build_wheel.sh cpu
    ```

    Please append `imperative`  to `build_wheel.sh` to use the new runtime, e.g., `./build_wheel.sh cpu imperative`.
## build for macos
    ```
    ./scripts/whl/macos/macos_build_whl.sh
    ```
    If you just want to build for a specific Python verison, you can use `ALL_PYTHON` environment variable. eg:

    ```
    ALL_PYTHON=3.5.9 ./scripts/whl/macos/macos_build_whl.sh
    ```
    If you want to build with imperative rt, set env BUILD_IMPERATIVE="ON", eg:

    ```
    ALL_PYTHON=3.5.9 BUILD_IMPERATIVE="ON" ./scripts/whl/macos/macos_build_whl.sh
    ```
## build for windows
    ```
    ./scripts/whl/windows/windows_build_whl.sh
    ```
    If you just want to build for a specific Python verison, you can use `ALL_PYTHON` environment variable. eg:

    ```
    ALL_PYTHON=3.5.4 ./scripts/whl/windows/windows_build_whl.sh
    ```
    If you want to build windows whl with cuda, also a specific Python verison. eg:

    ```
    WINDOWS_WHL_WITH_CUDA="ON" ALL_PYTHON=3.5.4 ./scripts/whl/windows/windows_build_whl.sh
    ```
    If you want to build with imperative rt, set env BUILD_IMPERATIVE="ON", eg:
    BUILD_IMPERATIVE="ON" WINDOWS_WHL_WITH_CUDA="ON" ALL_PYTHON=3.5.4 ./scripts/whl/windows/windows_build_whl.sh
