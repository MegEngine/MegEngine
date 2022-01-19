# MegEngine

<p align="center">
  <img width="250" height="109" src="logo.png">
</p>

[English](README.md) | 中文

MegEngine 是一个快速、可拓展、易于使用且支持自动求导的深度学习框架。

------


## 安装说明

**注意:** MegEngine 现在支持在 Linux-64bit/Windows-64bit/macos-10.14 及其以上 (MacOS 只支持 cpu) 等平台上安装 Python 包，支持 Python3.5 到 Python3.8。对于 Windows 10 用户，可以通过安装 [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl) 进行体验，同时我们也原生支持 Windows。MegEngine 也支持在很多其它平台上进行推理运算。

### 通过包管理器安装

通过 pip 安装的命令如下：

```bash
python3 -m pip install --upgrade pip
python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html
```

## 通过源码编译安装

* CMake 编译细节请参考 [BUILD_README.md](scripts/cmake-build/BUILD_README.md)
* Python 绑定编译细节请参考 [BUILD_PYTHON_WHL_README.md](scripts/whl/BUILD_PYTHON_WHL_README.md)

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

MegEngine 使用 Apache License, Version 2.0 。

## 引用 MegEngine
如果在您的研究中使用了 MegEngine ，建议您使用如下 BibTeX 格式引用文案。

```
@Misc{MegEngine,
  institution = {megvii},
  title =  {MegEngine:A fast, scalable and easy-to-use deep learning framework},
  howpublished = {\url{https://github.com/MegEngine/MegEngine}},
  year = {2020}
}
```

Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
