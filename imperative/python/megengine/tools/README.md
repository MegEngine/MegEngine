# MegEngine Tools

MegEngine 相关的工具汇总。使用方法如下(可将 `xxx` 替换成任一脚本文件，如 `network_visualize`):

```bash
python -m megengine.tools.xxx
```

工具列表:

### accuracy_shake_var_tree

将精度抖动分析结果构造成树结构，方便锁定引起抖动的根节点，以及查找依赖关系。

输入: compare_binary_iodump 的输出存入到的一个文件

输出: 第一个出现结果不一致的输出结点

执行命令: accuracy_shake_var_tree 中定义了一些函数组件，可按需集成到实际代码中。下面有一个测试代码:

```python
import megengine.tools.accuracy_shake_var_tree as st

r = st.parse('diff.txt')
for key, value in r.items():
    n = st.varNode.get_varNode(key)
    n.show_src_info()
    print("reference nodes:")
    for i in n.get_reference_list():
        print(i.id)
```

### benchmark_op

逐个运行 functional op(并不是所有的 functional op)，对比 MegEngine 与 PyTorch 的性能，通过量化结果来指导如何进行下一步的优化。

输入: 无

输出: 打印一个列表，对比在小输入和大输入的情况下 MegEngine 和 Pytorch 执行一些 functional op 的速度对比

执行命令: `python3 -m megengine.tools.benchmark_op`

### compare_binary_iodump

分析同一模型在不同平台下给定相同输入之后的输出是否完全一致。

输入: 两个目录(假设分别为 expect/ 和 actual/)，分别存有不同平台下运行的 tensor 结果

输出: 打印所有的输出 tensor 信息，如果某个 tensor 在两个平台上的值不一致，那么会打印出第一个不一致的值

执行命令: `python3 -m megengine.tools.compare_binary_iodump expect/ actual/`

### cpu_evaluation_tools

分析多个模型在目标芯片上的运行性能

输入：MegEngine 模型文件

输出：根据不同模型的加权，输出芯片性能分数

执行命令：python3 ./cpu_evaluation_tools.py --load_and_run_file ./load_and_run --models_dir ./cpu_models/

### draw_graph

用来查看静态图的 op 序列，有助于理解 MegEngine 的静态图在动态图的基础上做了哪些优化。

输入: `megengine.core.tensor.megbrain_graph.Graph._to_json` 得出的静态图描述文件，为 json 格式

输出: 一个 dot 文件，可通过 dot 命令绘制出图片

执行命令:

```bash
python3 -m megengine.tools.draw_graph -i dump.json -o dump.dot
dot -Tpng dump.dot -o dump.png
```

### graph_info_analyze

将图和内存信息的 json 文件的文件夹 logs 转换为 TensorBoard 的输入文件夹 logs_p。以便 TensorBoard 对图结构以及内存信息进行可视化。

输入: 图和内存信息的 json 文件的文件夹

输出: TensorBoard 的输入文件夹

执行命令: `python3 -m megengine.tools.graph_info_analyze -i logs -o logs_p`

### load_network_and_run

python 版本的 load_and_run。

输入: MegEngine 的模型文件，可选一些 npy 文件作为模型输入

输出: 模型执行并打印一些测速信息

执行命令: `python3 -m megengine.tools.load_network_and_run model.mge --iter 10`

### network_visualize

1. 分析给定的 MegEngine 模型中参数量信息，包括 shape、dtype、mean、std 以及 size 占比等。
2. 分析给定的 MegEngine 模型中算子 FLOPs 计算量以及占比，还有算子的 inputs/outputs shape、感受野、stride 等。

输入: MegEngine 的模型文件

输出: 模型中的参数量信息或计算量信息

执行命令:

```bash
# 分析参数量
python3 -m megengine.tools.network_visualize model.mge --cal_params --logging_to_stdout

# 分析计算量
python3 -m megengine.tools.network_visualize model.mge --cal_flops --logging_to_stdout
```

### profile_analyze

对于 load_and_run --profile 运行模型生成的 profile.json 文件或者 trace 模式下开启 profiling 功能并通过 trace.get_profile() 得到的 json 文件进行分析，得到静态图中算子的时间和显存占比等信息，以表格形式呈现。

输入: load_and_run 生成的 profile 文件

输出: 一个按照参数在输入文件中筛选得出的数据表格

执行命令:

```bash
# 生成供分析的 json 文件
python3 -m megengine.tools.load_network_and_run model.mge --warm-up --iter 10 --profile profile.json

#分析耗时前 3 的单个算子
python3 -m megengine.tools.profile_analyze profile.json -t 3

#筛选用时超过 10us 的 conv 按 flops 排序
python3 -m megengine.tools.profile_analyze profile.json -t 3 --order-by +flops --min-time 1e-5 --type ConvolutionForward
```

### profiler

对给定的训练程序，记录训练过程并以通用格式存储，可在浏览器上可视化。

输入: 需要一个 MegEngine 的训练程序(称之为 train.py，其中包含一个典型的 MegEngine 训练过程)

输出: 一些记录 profile 过程的 json 文件，默认在 profile 子目录下，可用 https://ui.perfetto.dev/ 进行加载并且可视化

执行命令: `python3 -m megengine.tools.profiler train.py`

### svg_viewer

查看 MegEngine 生成的显存占用图，可以帮助用户了解显存使用情况.

输入: 显存占用的 svg 图片

输出: 网页展示的可视化

执行命令: `python3 -m megengine.tools.svg_viewer`
