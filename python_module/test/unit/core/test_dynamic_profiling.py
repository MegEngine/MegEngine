# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import json

import numpy as np

import megengine.functional as F
from megengine import graph, tensor


def test_dynmaic_profiling():
    sz = 16

    cg = graph.get_default_graph()

    x = tensor(np.arange(0, sz, dtype=np.float32))
    y = F.relu(x)

    str1 = cg.get_dynamic_info()
    if str1 == "":
        return
    json_str1 = json.loads(str1)

    z = F.add_update(x, y)

    json_str2 = json.loads(cg.get_dynamic_info())

    diff = lambda l1, l2: [x for x in l1 if x not in l2]

    jdiff = diff(json_str2, json_str1)
    assert len(jdiff) == 1, "add_update operator should produce only one opr internally"

    dest_key = list(jdiff[0].keys())[0]
    assert (
        jdiff[0][dest_key]["output"][0]["memory"] == sz * 4
    ), "output of add_update operator has wrong allocated size"

    # check add_update is inplace or not
    dest_ptr = jdiff[0][dest_key]["output"][0]["dev_ptr"]

    found = False
    for li in json_str1:
        if "0" in li.keys():
            src_ptr = li["0"]["output"][0]["dev_ptr"]
            found = dest_ptr == src_ptr

    assert found == True, "add_update is not inplace"
