# python whl package build support status
*  windows build (not ok)
*  linux build   (ok, cpu or gpu)
*  macos build   (ok,cpu only)
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
## build for macos
    ```
    ./scripts/whl/macos/macos_build_whl.sh
    ```
