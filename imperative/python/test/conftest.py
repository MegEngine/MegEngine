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

from megengine.core import _config as config
from megengine.core import _trace_option as trace_option
from megengine.core import get_option
from megengine.core._imperative_rt.core2 import (
    _get_amp_dtype_autocast,
    _get_amp_high_prec_dtype,
    _get_amp_low_prec_dtype,
    _get_convert_inputs,
)
from megengine.core.tensor import amp
from megengine.device import get_device_count

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
def run_around_tests():
    env_vars1 = {
        "symbolic_shape": trace_option.use_symbolic_shape(),
        "async_level": get_option("async_level"),
        "enable_drop": get_option("enable_drop"),
        "max_recompute_time": get_option("max_recompute_time"),
        "catch_worker_execption": get_option("catch_worker_execption"),
        "enable_host_compute": get_option("enable_host_compute"),
        # "record_computing_path": get_option("record_computing_path"),
        "disable_memory_forwarding": get_option("disable_memory_forwarding"),
        "enable_dtr_auto_drop": get_option("enable_dtr_auto_drop"),
        "enable_dtr_sqrt_sampling": get_option("enable_dtr_sqrt_sampling"),
        "dtr_eviction_threshold": get_option("dtr_eviction_threshold"),
        "dtr_evictee_minimum_size": get_option("dtr_evictee_minimum_size"),
        "benchmark_kernel": config.benchmark_kernel,
        "deterministic_kernel": config.deterministic_kernel,
        "compute_mode": config._compute_mode,
        "conv_format": config._conv_format,
        "amp_enabled": amp.enabled,
        "convert_inputs": _get_convert_inputs(),
        "amp_dtype_autocast": _get_amp_dtype_autocast(),
        "amp_high_prec_dtype": _get_amp_high_prec_dtype(),
        "amp_low_prec_dtype": _get_amp_low_prec_dtype(),
    }
    yield
    env_vars2 = {
        "symbolic_shape": trace_option.use_symbolic_shape(),
        "async_level": get_option("async_level"),
        "enable_drop": get_option("enable_drop"),
        "max_recompute_time": get_option("max_recompute_time"),
        "catch_worker_execption": get_option("catch_worker_execption"),
        "enable_host_compute": get_option("enable_host_compute"),
        # "record_computing_path": get_option("record_computing_path"),
        "disable_memory_forwarding": get_option("disable_memory_forwarding"),
        "enable_dtr_auto_drop": get_option("enable_dtr_auto_drop"),
        "enable_dtr_sqrt_sampling": get_option("enable_dtr_sqrt_sampling"),
        "dtr_eviction_threshold": get_option("dtr_eviction_threshold"),
        "dtr_evictee_minimum_size": get_option("dtr_evictee_minimum_size"),
        "benchmark_kernel": config.benchmark_kernel,
        "deterministic_kernel": config.deterministic_kernel,
        "compute_mode": config._compute_mode,
        "conv_format": config._conv_format,
        "amp_enabled": amp.enabled,
        "convert_inputs": _get_convert_inputs(),
        "amp_dtype_autocast": _get_amp_dtype_autocast(),
        "amp_high_prec_dtype": _get_amp_high_prec_dtype(),
        "amp_low_prec_dtype": _get_amp_low_prec_dtype(),
    }
    for key in env_vars1:
        assert (
            env_vars1[key] == env_vars2[key]
        ), "{} have been changed after test".format(key)
