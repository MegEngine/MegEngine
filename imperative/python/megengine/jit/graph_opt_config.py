# -*- coding: utf-8 -*-


class GraphOptimizationConfig:
    r"""Configuration for graph optimization: False for OFF, True for ON. The default value
    None means that opt_level will decide whther this optimization will be applied or not.

    Args:
        jit_fuse_dimshuffle: whether to fuse dimshuffle in JIT optimization
        jit_fuse_reduce: whether to fuse reduce in JIT optimization
    """

    def __init__(self):
        self.jit_fuse_dimshuffle = None
        self.jit_fuse_reduce = None

    def __repr__(self):
        val2str = {None: "UNSET", False: "OFF", True: "ON"}
        return (
            "GraphOptimizationConfig {"
            + " jit_fuse_dimshuffle = "
            + val2str[self.jit_fuse_dimshuffle]
            + ", jit_fuse_reduce = "
            + val2str[self.jit_fuse_reduce]
            + " }"
        )
