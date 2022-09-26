# MegEngine 贡献指南
欢迎来到 MegEngine ，一起将 MegEngine 变更好！

## 贡献者许可指南
每一名 MegEngine 的贡献者都需要签署贡献者许可协议（Contributor License Agreement，CLA）来明确贡献内容相关的知识产权许可。更多细节请参考 [协议内容](https://github.com/MegEngine/MegEngine/blob/master/CONTRIBUTOR_LICENSE_AGREEMENT.md)。


## 提交 PR 
* 您可以通过 [issue](https://github.com/MegEngine/MegEngine/issues)提出您的想法。经过讨论达成共识后，您可在 fork 的代码仓库进行研发，并提交 PR 。
* 每个 PR 需要经过两个 reviewer 的审核，并得到两位 reviewer 的 LGTM 。


更多提交 PR 的基本操作请参见 [Github pull request](https://docs.github.com/cn/pull-requests/collaborating-with-pull-requests) 。

## 提交 PR 之 Commit Message
MegEngine 的 Commit Message 规范修改自 [Angular](angular) 规范，CI 中也配置了相关规则检查。

### 参考示例
以下给出了几种 MegEngine Git 协作流程中常见的 Commit message 作为参考：

```
feat(mge/serialization): add universal GraphLoader/GraphMaker, expose to Python and lar

在 mge/serialization 下新增了功能，具体内容为：add universal GraphLoader/GraphMaker, expose to Python and lar
```

```
docs(mge/optimizer): fix docs about optimizer state_dict related function

在 mge/optimizer 下边做了文档相关的改动，具体内容为：fix docs about optimizer state_dict related function
```

```
refactor(mge/examples): modify more examples to use new API

在 mge/examples 下边做了重构，具体内容为：modify more examples to use new API
```

```
test(mge/jit): add a test for jit

对测试相关文件的修改一律使用 test 作为范围标识（而不是 fix 字段，fix 通常作用于 Bug），CI 相关文件的同理。
```

```
ci(dockerfile): modify dockerfile

持续集成相关的修改一律使用 ci 作为范围标识。可以分为 dockerfile（修改镜像）、test（修改相关测试脚本）等等。
```

```
build(makefile): fix makefile bug

编译构建相关的修改一律使用 build type。
```


### 规范细则
说明：本规范沿用 AngularJS 规范进行修改。
每个 Commit message由三部分组成： **Header**, **Body** 和 **Footer**.


```shell
<type>(<scope>): <subject>
# 空一行
<body>
# 空一行
<footer>
```

其中，`Header` 为必需 [格式](#commit-header)，`Body` 和 `Footer`  为可选。

#### <a name="commit-header"></a>Commit Message Header

Header 部分只有一行，包括三个字段：`type`, `scope` 和 `subject` ，且三个字段都必须提供。

```
<type>(<scope>): <short summary>
  │       │             │
  │       │             └─⫸ Summary in present tense. Not capitalized. No period at the end.
  │       │
  │       └─⫸ Commit Scope: mge/[module,optimizer,distributed...] | dnn/{backend} | imperative/[interpreter...]
  │
  └─⫸ Commit Type: build|ci|docs|feat|fix|perf|refactor|test
```

##### Type

Type 用于说明 commit 的类别，只允许使用下面几个标识，如果您有其他需求请提交 [issue](https://github.com/MegEngine/MegEngine/issues) 告诉我们。

* **feat**: 新功能（feature）
* **fix**: 专用于指代修补已实现的功能中存在的 BUG
* **perf**: 性能优化
* **docs**: 文档（documentation）
* **style**: 格式（不影响代码运行的变动）
* **refactor**: 重构（即不是新增功能，也不是修 BUG 的代码变动）
* **test**: 增加测试
* **build**: 构建过程相关的修改
* **ci**: CI 相关的修改

##### Scope

Scope 用于说明 commit 影响的范围，在我们的规范中，需要在 Scope 里指明所涉及的文件的大致范围：

* 必须以 mge/imperative/dnn 作为父级：
  * mge 表示 MegEngine 的修改；
  * imperative 表示 imperative runtime（imperative/src 目录下）的修改；
  * dnn 表示 kernel 相关的修改，change log 中放入 MegBrain 修改。
* 如果涉及到多个子目录：要么拆分，要么直接写 scope=mge 这种父级的;
* 此处的文档修改指修改 Python 文档字符串或 C++ 注释这类，其它的文档在 [Documentation repo](https://github.com/MegEngine/Documentation) 中撰写。
   

##### Subject

Subject 是 commit 目的的简短描述，不超过 50 个字符。

* **第一个字母小写**；
* **结尾不加句号（.）**,规范中会自动去除；
* **以动词开头**，使用第一人称现在时，比如 change，而不是 changed 或 changes 。



#### <a name="commit-body"></a>Commit Message Body

Body 部分是对本次 commit 的详细描述，说明代码变动的动机，以及与以前行为的对比等信息，可以分成多行。 一般情况下可不添加。

下面是一个范例：

```
More detailed explanatory text, if necessary. Wrap it to
about 72 characters or so. Further paragraphs come after blank lines.

- Bullet points are okay, too
- Use a hanging indent
```



#### <a name="commit-footer"></a>Commit Message Footer

添加一些参考信息，比如 Issues 或 Tickets, 可不添加

#### Revert commits

还有一种特殊情况，如果当前 commit 用于撤销以前的 commit，则必须以 revert: 开头，后面跟着被撤销 Commit 的 Header.

```
revert: feat(pencil): add 'graphiteWidth' option

This reverts commit 667ecc1654a317a13331b17617d973392f415f02.
```

Body 部分的格式是固定的，必须写成 `This reverts commit <hash>.`，其中的 hash 是被撤销 commit 的 SHA 标识符。



### 如何自查
您可以安装 [commitizen](https://commitizen-tools.github.io/commitizen/) 以便于在提交时对 Commit Message 进行检查，避免提交后测试不通过的再次修改。





MegEngine 依据 [贡献者公约（Contributor Covenant）](https://contributor-covenant.org)来管理开源社区。请阅读 [行为准则](CODE_OF_CONDUCT.md) 了解更多信息。

我们相信我们能够搭建一个开放友善的开源社区环境，用人工智能造福人类。





