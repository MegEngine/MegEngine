# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


class GraphOptimizationConfig:
    r"""
    Configuration for graph optimization: False for OFF, True for ON. The default value
    None means that opt_level will decide whther this optimization will be applied or not.

    :param jit_fuse_dimshuffle: whether to fuse dimshuffle in JIT optimization
    :param jit_fuse_reduce: whether to fuse reduce in JIT optimization
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
