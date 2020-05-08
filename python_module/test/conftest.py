# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))


def pytest_json_modifyreport(json_report):
    events = []
    timestamp = 0
    for item in json_report["tests"]:
        for stage in ["setup", "call", "teardown"]:
            if stage in item:
                events.append(
                    {
                        "name": item["nodeid"],
                        "ph": "X",
                        "ts": timestamp,
                        "dur": item[stage]["duration"] * 1e6,
                        "cat": stage,
                        "pid": stage,
                        "tid": item["nodeid"],
                    }
                )
                timestamp += events[-1]["dur"]
    json_report["traceEvents"] = events
    del json_report["collectors"]
    del json_report["tests"]
