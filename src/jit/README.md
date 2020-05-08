# JIT
A optimization for MegBrain by just-in-time compilation.
JIT can reduce the global memory access times by fusing elemwise kernels into a
single larger one fusion kernel to improve performence.

For some regular expressions like *a * b + c* and *a * b + c * d*, MegBrain have
alreay did FMA3_FUSE and FMA4_FUSE optimization. Now MegBrain can speed up any
elemwise expressions by JIT.

## Benchmark Result
1. a * b * c

    |        |opt0| opt2| opt3(with jit)|
    |--------|----|-----|---------------|
    |speed   |100%|100% | 150%          |

2. a * b + c

    |        |opt0| opt2(with fma3)| opt3(with jit)|
    |--------|----|-----|---------------|
    |speed   |100%|150% | 150%          |

3. Alexnet with adam

    |        |opt0| opt2| opt3(with jit)|
    |--------|----|-----|---------------|
    |speed   |100%|103% | 114%          |

4. Resnet with adam, training

    |        |opt0| opt2| opt3(with jit)|
    |--------|----|-----|---------------|
    |speed   |100%|122% | 124%          |


## What does JIT do
Detection the subgraph can be fused and compiling the subgraph into a fusion
kernel are the most two important parts in JIT.

The detection is implemented in [impl/fusion_pass.cpp](impl/fusion_pass.cpp),
the main detection logic is in function *Fusion::Impl::on_opr*. Compared to nnvm
fusion, our fusion logic can fuse more operators into one fusion kernel.

For now , JIT just support CUDA, but it has reserved interface to extend other
platforms.

## How to enable JIT
You can set `graph_opt_level` to 3 to enable JIT.

In python
``` python
cg = mgb.comp_graph()
cg.set_option('graph_opt_level', 3)
```

### Selection of Backend

You can set environment variable `MGB_JIT_BACKEND` to select the JIT backend.

| Backend | Platforms | Reduction support | Kernel Binary Cache | Kernel Reuse | Noncontig Input |
|---------|-----------|-------------------|---------------------|--------------|-----------------|
| HALIDE  | CUDA      | Y                 | No                  | Shape        | No              |
| NVRTC   | CUDA      | N                 | Via PersistentCache | Bcast type   | Monotone        |

To enable fusion of Reduce oprs, set `graph_opt.jit = 2` in graph options.

### Working Directory

JIT may produce temporary files. The default working directory is
a temp dir and can be changed via `MGB_JIT_WORKDIR` environment variable. Set
`MGB_JIT_KEEP_INTERM` to keep intermediate files (such as generated sources and
object files) for debugging.

### Other options

* `MGB_HALIDE_DEBUG`: enable debug print for Halide.
