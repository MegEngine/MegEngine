# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


class DTRConfig:
    def __init__(
        self, eviction_threshold: int = 0, evictee_minimum_size: int = 1 << 20
    ):
        assert eviction_threshold > 0, "eviction_threshold must be greater to zero"
        self.eviction_threshold = eviction_threshold
        assert (
            evictee_minimum_size >= 0
        ), "evictee_minimum_size must be greater or equal to zero"
        self.evictee_minimum_size = evictee_minimum_size
