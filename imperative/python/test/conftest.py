import os
import platform
import sys

import pytest

import megengine.functional
import megengine.module
from megengine import Parameter
from megengine.core._imperative_rt.core2 import sync
from megengine.distributed.helper import get_device_count_by_fork
from megengine.jit import trace as _trace
from megengine.module import Linear, Module

sys.path.append(os.path.join(os.path.dirname(__file__), "helpers"))

_ngpu = get_device_count_by_fork("gpu")


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
