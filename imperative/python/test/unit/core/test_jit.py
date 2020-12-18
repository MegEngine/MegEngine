# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import pytest

# from megengine.core.interpreter.hints import function


@pytest.mark.skip(reason="under rewrite")
def test_1():
    @function
    def f(x, p):
        x = x + 1
        if p:
            return x * x
        return x * 2

    x = Tensor(0)

    for _ in range(5):
        assert f(x, 0).numpy() == 2
        assert f(x, 1).numpy() == 1
