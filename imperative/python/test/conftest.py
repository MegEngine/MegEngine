# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import platform
import sys

import pytest

import megengine.functional
import megengine.module
from megengine import Parameter
from megengine.core._imperative_rt.core2 import sync
from megengine.device import get_device_count
from megengine.experimental.autograd import (
    disable_higher_order_directive,
    enable_higher_order_directive,
)
from megengine.jit import trace as _trace
from megengine.module import Linear, Module

sys.path.append(os.path.join(os.path.dirname(__file__), "helpers"))

_ngpu = get_device_count("gpu")


@pytest.fixture(autouse=True)
def skip_by_ngpu(request):
    if request.node.get_closest_marker("require_ngpu"):
        require_ngpu = int(request.node.get_closest_marker("require_ngpu").args[0])
        if require_ngpu > _ngpu:
            pytest.skip("skipped for ngpu unsatisfied: {}".format(require_ngpu))


@pytest.fixture(autouse=True)
def skip_distributed(request):
    if request.node.get_closest_marker("distributed_isolated"):
        if platform.system() in ("Windows", "Darwin"):
            pytest.skip(
                "skipped for distributed unsupported at platform: {}".format(
                    platform.system()
                )
            )


@pytest.fixture(autouse=True)
def resolve_require_higher_order_directive(request):
    marker = request.node.get_closest_marker("require_higher_order_directive")
    if marker:
        enable_higher_order_directive()
    yield
    if marker:
        disable_higher_order_directive()
