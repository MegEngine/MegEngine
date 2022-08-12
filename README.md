# MegEngine

<p align="center">
  <img width="128" height="128" src="logo.png">
</p>
<h3> <a href="https://www.megengine.org.cn/doc/stable/en/user-guide/index.html"> Documentation </a> | <a href="https://www.megengine.org.cn/doc/stable/zh/user-guide/index.html"> 中文文档 </a> </h3>

[![](https://img.shields.io/badge/English-%E4%B8%AD%E6%96%87-green.svg)](README_CN.md) [![](https://img.shields.io/badge/Website-MegEngine-green.svg)](https://megengine.org.cn/) [![](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE) [![](https://img.shields.io/badge/Chat-on%20QQ-green.svg?logo=tencentqq)](https://jq.qq.com/?_wv=1027&k=jJcBU1xi) [![](https://img.shields.io/badge/Discuss-on%20Zhihu-8A2BE2.svg?labelColor=00BFFF&logo=zhihu)](https://www.zhihu.com/people/megengine-bot)

MegEngine is a fast, scalable, and user friendly deep learning framework with 3 key features.

* **Unified framework for both training and inference**
    * Quantization, dynamic shape/image pre-processing, and even derivation with a single model.
    * After training, put everything into your model to inference on any platform with speed and precision. Check [here](https://www.megengine.org.cn/doc/stable/zh/user-guide/model-development/traced_module/index.html) for a quick guide.
* **The lowest hardware requirements**
    * The memory usage of the GPU can be reduced to one-third of the original memory usage when [DTR algorithm](https://www.megengine.org.cn/doc/stable/zh/user-guide/model-development/dtr/index.html) is enabled.
    * Inference models with the lowest memory usage by leveraging our Pushdown memory planner.
* **Inference efficiently on all platforms**
    * Inference with speed and high-precision on x86, Arm, CUDA, and RoCM.
    * Supports Linux, Windows, iOS, Android, TEE, etc.
    * Optimize performance and memory usage by leveraging our [advanced features](https://www.megengine.org.cn/doc/stable/zh/user-guide/deployment/lite/advance/index.html).

------

## Installation

**NOTE:** MegEngine now supports Python installation on Linux-64bit/Windows-64bit/MacOS(CPU-Only)-10.14+/Android 7+(CPU-Only) platforms with Python from 3.5 to 3.8. On Windows 10 you can either install the Linux distribution through [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl) or install the Windows distribution directly. Many other platforms are supported for inference.

### Binaries

To install the pre-built binaries via pip wheels:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html
```

## Building from Source

* CMake build details. please refer to [BUILD_README.md](scripts/cmake-build/BUILD_README.md)
* Python binding build details, Please refer to [BUILD_PYTHON_WHL_README.md](scripts/whl/BUILD_PYTHON_WHL_README.md)

## How to Contribute

* MegEngine adopts [Contributor Covenant](https://contributor-covenant.org) as a guideline to run our community. Please read the [Code of Conduct](CODE_OF_CONDUCT.md).
* Every contributor of MegEngine must sign a [Contributor License Agreement (CLA)](CONTRIBUTOR_LICENSE_AGREEMENT.md) to clarify the intellectual property license granted with the contributions.
* You can help to improve MegEngine in many ways:
    * Write code.
    * Improve [documentation](https://github.com/MegEngine/Docs).
    * Answer questions on [MegEngine Forum](https://discuss.megengine.org.cn), or Stack Overflow.
    * Contribute new models in [MegEngine Model Hub](https://github.com/megengine/hub).
    * Try a new idea on [MegStudio](https://studio.brainpp.com).
    * Report or investigate [bugs and issues](https://github.com/MegEngine/MegEngine/issues).
    * Review [Pull Requests](https://github.com/MegEngine/MegEngine/pulls).
    * Star MegEngine repo.
    * Cite MegEngine in your papers and articles.
    * Recommend MegEngine to your friends.
    * Any other form of contribution is welcomed.

We strive to build an open and friendly community. We aim to power humanity with AI.

## How to Contact Us

* Issue: [github.com/MegEngine/MegEngine/issues](https://github.com/MegEngine/MegEngine/issues)
* Email: [megengine-support@megvii.com](mailto:megengine-support@megvii.com)
* Forum: [discuss.megengine.org.cn](https://discuss.megengine.org.cn)
* QQ Group: 1029741705

## Resources

- [MegEngine](https://megengine.org.cn)
- [MegStudio](https://studio.brainpp.com)
- mirror repo
   - OPENI: [openi.org.cn/MegEngine](https://www.openi.org.cn/html/2020/Framework_0325/18.html)
   - Gitee: [gitee.com/MegEngine/MegEngine](https://gitee.com/MegEngine/MegEngine)


## License

MegEngine is licensed under the Apache License, Version 2.0

## Citation
If you use MegEngine in your publication,please cite it by using the following BibTeX entry.

```
@Misc{MegEngine,
  institution = {megvii},
  title =  {MegEngine:A fast, scalable and easy-to-use deep learning framework},
  howpublished = {\url{https://github.com/MegEngine/MegEngine}},
  year = {2020}
}
```

Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
