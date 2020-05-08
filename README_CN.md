# MegEngine

![MegEngine Logo](logo.png)

[English](README.md) | 中文

MegEngine 是一个快速、可拓展、易于使用且支持自动求导的深度学习框架。

------


## 安装说明

**注意:** MegEngine 现在仅支持 Linux 平台安装，以及 Python3.5 及以上的版本（不支持 Python2 ）。对于 Windows 10 用户，可以通过安装 [WSL(Windows Subsystem for Linux)](https://docs.microsoft.com/en-us/windows/wsl) 进行体验。

### 通过包管理器安装

通过 pip 安装的命令如下：

```bash
pip3 install megengine -f https://megengine.org.cn/whl/mge.html
```

## 通过源码编译安装

### 环境依赖

大多数编译 MegEngine 的依赖位于 `third_party` 目录，可以通过以下命令自动安装：

```bash
$ ./third_party/prepare.sh
$ ./third_party/install-mkl.sh
```

但是有一些依赖需要手动安装：

* [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)(>=10.1), [cuDNN](https://developer.nvidia.com/cudnn)(>=7.6) ，如果需要编译支持 CUDA 的版本（默认开启）
* [TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)(>=5.1.5) ，如果需要编译支持 TensorRT 的版本（默认开启）
* LLVM/Clang(>=6.0) ，如果需要编译支持 Halide JIT 的版本（默认开启）
* Python(>=3.5), Numpy, SWIG(>=3.0) ，如果需要编译生成 Python 模块（默认开启）

### 开始编译

MegEngine 遵循“源外构建”（[Out-of-Source Build](https://zh.m.wikibooks.org/zh-hans/CMake_%E5%85%A5%E9%96%80/Out-of-source_Build)）原则，并且使用静态编译方式。编译的具体流程如下：

1. 创建用于编译的目录：
    ```bash
    mkdir -p build
    cd build
    ```

2. 使用 `CMake` 生成编译配置：

    生成支持 CUDA 环境的配置：
    ```bash
    cmake .. -DMGE_WITH_TEST=ON
    ```

    生成仅支持 CPU 环境的配置，使用 `-DMGE_WITH_CUDA=OFF` 选项：
    ```bash
    cmake .. -DMGE_WITH_CUDA=OFF -DMGE_WITH_TEST=ON
    ```

    生成仅用于 C++ 环境部署的配置，使用 `-DMGE_INFERENCE_ONLY=ON` ，并可用 `-DMGE_WITH_TEST=OFF` 关闭测试：
    ```bash
    cmake .. -DMGE_INFERENCE_ONLY=ON -DMGE_WITH_TEST=OFF
    ```

    可以使用 `-DCMAKE_INSTALL_PREFIX=YOUR_PATH` 指定具体安装目录。

3. 开始编译：

    ```bash
    make -j$(nproc)
    ```

4. [可选] 如果需要用于部署，可以安装 MegEngine 的 C++ 库：

    ```bash
    make install
    ```

以下是其它常用编译选项：

* `MGE_ARCH` 指定编译的目标平台（默认自动检测当前平台）
* `MGE_WITH_DISTRIBUTED` 是否开启多机分布式支持（默认开启）
* `MGE_WITH_PYTHON_MODULE` 是否编译生成 Python 模块（默认开启）
* `MGE_BLAS` 选择 BLAS 的后端实现，可以是 `MKL` 或 `OpenBLAS` （默认 `MKL`）
* `MGE_CUDA_GENCODE` 指定提供给 `nvcc` 的 `-gencode` 选项（默认不指定）
* `MGE_DISABLE_FLOAT16` 是否不提供 `float16` 类型支持（默认关闭）
* `MGE_ENABLE_EXCEPTIONS` 是否开启 C++ 报错支持（默认开启）
* `MGE_ENABLE_LOGGING` 是否开启 MegEngine 日志信息（默认自动检测）

更多选项可以通过以下命令查看：

```bash
cd build
cmake -LAH .. 2>/dev/null| grep -B 1 'MGE_' | less
```

## 如何参与贡献

* MegEngine 依据 [贡献者公约（Contributor Covenant）](https://contributor-covenant.org)来管理开源社区。请阅读 [行为准则](CODE_OF_CONDUCT.md) 了解更多信息。
* 每一名 MegEngine 的贡献者都需要签署贡献者许可协议（Contributor License Agreement，CLA）来明确贡献内容相关的知识产权许可。更多细节请参考 [协议内容](CONTRIBUTOR_LICENSE_AGREEMENT.md)。
* 我们欢迎你通过以下方式来帮助 MegEngine 变得更好：
    * 贡献代码；
    * 完善[文档](https://github.com/MegEngine/Docs)；
    * 在 [MegEngine 论坛](https://discuss.megengine.org.cn) 和 Stack Overflow 回答问题；
    * 在 [MegEngine Model Hub](https://github.com/megengine/hub) 贡献新模型；
    * 在 [MegStudio](https://studio.brainpp.com) 平台尝试新想法；
    * 报告使用中的 [Bugs 和 Issues](https://github.com/MegEngine/MegEngine/issues)；
    * 审查 [Pull Requests](https://github.com/MegEngine/MegEngine/pulls)；
    * 给 MegEngine 点亮小星星；
    * 在你的论文和文章中引用 MegEngine；
    * 向你的好友推荐 MegEngine；
    * ...

我们相信我们能够搭建一个开放友善的开源社区环境，用人工智能造福人类。

## 联系我们

* 问题: [github.com/MegEngine/MegEngine/issues](https://github.com/MegEngine/MegEngine/issues)
* 邮箱: [megengine-support@megvii.com](mailto:megengine-support@megvii.com)
* 论坛: [discuss.megengine.org.cn](https://discuss.megengine.org.cn)
* QQ: 1029741705
* OPENI: [openi.org.cn/MegEngine](https://www.openi.org.cn/html/2020/Framework_0325/18.html) 

## 资源

- [MegEngine](https://megengine.org.cn)
- [MegStudio](https://studio.brainpp.com)
- [Brain++](https://brainpp.megvii.com)

## 开源许可

MegEngine 使用 Apache License, Version 2.0

Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
