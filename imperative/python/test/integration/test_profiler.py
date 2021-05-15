# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
import json
import os
import tempfile

import pytest

from megengine import Parameter, tensor
from megengine.core import option
from megengine.module import Module
from megengine.utils.profiler import Profiler, scope


class Simple(Module):
    def __init__(self):
        super().__init__()
        self.a = Parameter([1.23], dtype="float32")

    def forward(self, x):
        x = x * self.a
        return x


def test_profiler():
    tempdir = tempfile.NamedTemporaryFile()
    profile_prefix = tempdir.name
    profile_format = "chrome_timeline.json"
    profile_path = os.path.join(
        profile_prefix, "{}.{}".format(os.getpid(), profile_format)
    )
    with option("enable_host_compute", 0):
        with Profiler(profile_prefix, format=profile_format):
            with scope("my_scope"):
                oup = Simple()(tensor([1.23], dtype="float32"))
    with open(profile_path, "r") as f:
        events = json.load(f)
    prev_ts = {}
    scope_count = 0
    for event in events:
        if "dur" in event:
            assert event["dur"] >= 0
        elif "ts" in event and "tid" in event:
            ts = event["ts"]
            tid = event["tid"]
            if ts == 0:
                continue
            assert (tid not in prev_ts) or prev_ts[tid] <= ts
            prev_ts[tid] = ts
        if "name" in event and event["name"] == "my_scope":
            scope_count += 1
    assert scope_count > 0 and scope_count % 2 == 0
