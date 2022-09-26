# MegEngine

<p align="center">
  <img width="202" height="118" src="logo.svg">
</p>

[![](https://img.shields.io/badge/English-%E4%B8%AD%E6%96%87-green.svg)](README.md) [![](https://img.shields.io/badge/Website-MegEngine-green.svg)](https://megengine.org.cn/) [![](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE) [![](https://img.shields.io/badge/Chat-on%20QQ-green.svg?logo=tencentqq)](https://jq.qq.com/?_wv=1027&k=jJcBU1xi) [![](https://img.shields.io/badge/Discuss-on%20Zhihu-8A2BE2.svg?labelColor=00BFFF&logo=zhihu)](https://www.zhihu.com/people/megengine-bot)

MegEngine 是一个快速、可拓展、易于使用的深度学习框架，拥有以下三大关键特点：

* 训练推理一体：训练推理同一内核，模型结构、量化、前后处理、动态 shape 甚至求导均可 [放入模型](https://www.megengine.org.cn/doc/stable/zh/user-guide/model-development/traced_module/index.html) 进行推理，训练推理轻松对齐精度
* 超低硬件门槛：依靠算法优化各类关键资源占用，[DTR](https://www.megengine.org.cn/doc/stable/zh/user-guide/model-development/dtr/index.html) 让训练显存占用量一键下降3倍，pushdown 内存分配算法让推理内存占用下降至极低水平
* 全平台高效推理：在 x86/Arm/CUDA/RoCM 各平台上均可体验到高性能且精度对齐的推理体验，更有 [丰富的高阶用法可以优化性能、节省内存](https://www.megengine.org.cn/doc/stable/zh/user-guide/deployment/lite/advance/index.html)

------
## 开始使用

+ 如果想本地使用：[安装](https://www.megengine.org.cn/doc/stable/zh/user-guide/install/)、[编译](https://github.com/MegEngine/MegEngine/blob/master/scripts/cmake-build/BUILD_README.md)
+ 如果想在线体验：[MegStudio](https://studio.brainpp.com/)
+ 更多技术细节解读及问题反馈：[知乎](https://www.zhihu.com/people/megengine-bot)、[论坛](https://discuss.megengine.org.cn/)

### 训练
+ 学习 MegEngine 使用文档：[文档](https://www.megengine.org.cn/doc/stable/zh/getting-started/index.html)
+ 得到 MegEngine 模型：[BaseCls](https://github.com/megvii-research/basecls)、[Models](https://github.com/MegEngine/Models)、[Hub](https://github.com/MegEngine/Hub)
+ 从 PyTorch 迁移而来：[torch2mge](https://github.com/MegEngine/torch2mge)、[guide](https://github.com/MegEngine/cheat_sheet_for_pytorch_immigrant)、[文档-迁移指南](https://www.megengine.org.cn/doc/stable/zh/user-guide/transfer-from/)
+ paper 合集：[MEGVII-Research](https://github.com/megvii-research)


### 推理

+ 查看部署指南：[文档](https://www.megengine.org.cn/doc/stable/zh/getting-started/deploy/)
+ 转换至其它推理框架：[MgeConvert](https://github.com/MegEngine/mgeconvert)
+ 串联多模型、视频流处理：[MegFlow](https://github.com/MegEngine/MegFlow)

## 安装说明

**注意:** MegEngine 现在支持在 Linux-64bit/Windows-64bit/macos-10.14/Android 7+ 及其以上 (MacOS/Android只支持cpu) 等平台上安装 Python 包，支持Python3.5 到 Python3.8。对于 Windows 10 用户，可以通过安装 [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl) 进行体验，同时我们也原生支持Windows。MegEngine 也支持在很多其它平台上进行推理运算。

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


## 资源

- [MegEngine](https://megengine.org.cn)
- [MegStudio](https://studio.brainpp.com)
- 镜像仓库：
   - OPENI: [openi.org.cn/MegEngine](https://www.openi.org.cn/html/2020/Framework_0325/18.html)
   - Gitee: [gitee.com/MegEngine/MegEngine](https://gitee.com/MegEngine/MegEngine)

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
