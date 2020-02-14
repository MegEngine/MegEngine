# MegEngine

![MegEngine Logo](logo.png)

English | [中文](README_CN.md)

MegEngine is a fast, scalable and easy-to-use numerical evaluation framework, with auto-differentiation.

------

## Installation

**NOTE:** MegEngine now only supports Linux platform with Python 3.5 or higher. On Windows 10 you could try [WSL(Windows Subsystem for Linux)](https://docs.microsoft.com/en-us/windows/wsl) to use Linux within Windows.

### Binaries

Commands to install from binaries via pip wheels are as follows:

```bash
pip3 install megengine -f https://megengine.org.cn/whl/mge.html
```

## Build from Source

### Prerequisites

Most of the dependencies of MegEngine are located in `third_party` directory, and you do
not need to install these by yourself. you can prepare these repositories by executing:

```bash
./third_party/prepare.sh
./third_party/install-mkl.sh
```

But some dependencies should be manually installed:

* [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)(>=10.1), [cuDNN](https://developer.nvidia.com/cudnn)(>=7.6)are required when building MegEngine with CUDA support (default ON)
* [TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)(>=5.1.5) is required when building with TensorRT support (default ON)
* LLVM/Clang(>=6.0) is required when building with Halide JIT support (default ON)
* Python(>=3.5), Numpy, SWIG(>=3.0) are required to build Python modules. (default ON)

### Build

MegEngine prefers `Out-Of-Source` flavor, and compile in a `mostly-static` way.
Here are the instructions:

1. Make a directory for the build.
    ```bash
    mkdir -p build
    cd build
    ```

2. Generate build configurations by `CMake`.

    For CUDA build:
    ```bash
    cmake .. -DMGE_WITH_TEST=ON
    ```

    For CPU only build, use `-DMGE_WITH_CUDA=OFF`:
    ```bash
    cmake .. -DMGE_WITH_CUDA=OFF -DMGE_WITH_TEST=ON
    ```

    For deployment with C++ only, use `-DMGE_INFERENCE_ONLY=ON`, and turn off test with `-DMGE_WITH_TEST=OFF`:
    ```bash
    cmake .. -DMGE_INFERENCE_ONLY=ON -DMGE_WITH_TEST=OFF
    ```

    Use `-DCMAKE_INSTALL_PREFIX=YOUR_PATH` to specify the install path.


3. Start to build.

    ```bash
    make -j$(nproc)
    ```

4. [optional] Install the library if compiled for deployment at step 2.

    ```bash
    make install
    ```

Here are some other useful options for the build.

* `MGE_ARCH` specifies which arch MegEngine are building for. (default AUTO)
* `MGE_WITH_DISTRIBUTED` if multiple machine distributed support is enabled. (default ON)
* `MGE_WITH_PYTHON_MODULE` if build python module. (default ON)
* `MGE_BLAS` chooses `MKL` or `OpenBLAS` as BLAS library for MegEngine. (default `MKL`)
* `MGE_CUDA_GENCODE` supplies the `-gencode` option for `nvcc`. (default not supply)
* `MGE_DISABLE_FLOAT16` if disable float16 support. (default OFF)
* `MGE_ENABLE_EXCEPTIONS` if enable exception support in C++. (default ON)
* `MGE_ENABLE_LOGGING` if enable logging in MegEngine. (default AUTO)

More options can be found by:

```bash
cd build
cmake -LAH .. 2>/dev/null| grep -B 1 'MGE_' | less
```

## How to Contribute

* MegEngine adopts [Contributor Covenant](https://contributor-covenant.org) to maintain our community. Please read the [Code of Conduct](CODE_OF_CONDUCT.md) to get more information.
* Every contributor of MegEngine must sign a Contributor License Agreement (CLA) to clarify the intellectual property license granted with the contributions. For more details, please refer [Contributor License Agreement](CONTRIBUTOR_LICENSE_AGREEMENT.md)
* You can help MegEngine better in many ways:
    * Write code.
    * Improve [documentation](https://github.com/MegEngine/Docs).
    * Answer questions on [MegEngine Forum](https://discuss.megengine.org.cn), or Stack Overflow.
    * Contribute new models in [MegEngine Model Hub](https://github.com/megengine/hub).
    * Try a new idea on [MegStudio](https://studio.brainpp.com).
    * Report or investigate [bugs and issues](https://github.com/MegEngine/MegEngine/issues).
    * Review [Pull Requests](https://github.com/MegEngine/MegEngine/pulls).
    * Star MegEngine repo.
    * Reference MegEngine in your papers and articles.
    * Recommend MegEngine to your friends.
    * ...

We believe we can build an open and friendly community and power humanity with AI.

## How to contact us

* Issue: [github.com/MegEngine/MegEngine/issues](https://github.com/MegEngine/MegEngine/issues)
* Email: [megengine-support@megvii.com](mailto:megengine-support@megvii.com)
* Forum: [discuss.megengine.org.cn](https://discuss.megengine.org.cn)
* QQ: 1029741705

## Resources

- [MegEngine](https://megengine.org.cn)
- [MegStudio](https://studio.brainpp.com)
- [Brain++](https://brainpp.megvii.com)

## License

MegEngine is Licensed under the Apache License, Version 2.0

Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
