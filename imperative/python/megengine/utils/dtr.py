# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ..core._imperative_rt.core2 import set_option
from ..core._imperative_rt.utils import _set_defrag


class DTR:
    r"""
    DTR implements `Dynamic Tensor Rematerialization <https://arxiv.org/abs/2006.09616>`_ in MegEngine.

    It is basically an online algorithm for checkpointing driven by certain eviction policies.

    .. code-block::
    
        from megengine.utils.dtr import DTR

        ds = DTR(memory_budget=5*1024**3)

        # your training code

    """

    def __init__(self, memory_budget=0, tensor_lowerbound=1048576):
        r"""
        :param memory_budget: int. The threshold of memory usage. When memory
        usage exceeds this value, auto evict will be triggered.
        :param tensor_lowerbound: int. The minimum memory limit of the tensor
        that can be evicted. Default: 1MB.
        """
        if memory_budget > 0:
            set_option("enable_auto_drop", 1)
            set_option("enable_drop", 1)
            set_option("buffer_length", 0)
            set_option("memory_budget", memory_budget)
            set_option("tensor_lowerbound", tensor_lowerbound)
            set_option("record_computing_path", 1)
            _set_defrag(True)
