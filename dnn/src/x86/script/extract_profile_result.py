#!/usr/bin/env python
# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import re

pat = r'f=(?P<f>\d+), oc=(?P<oc>\d+), ic=(?P<ic>\d+), threshold=(?P<threshold>\d+)'

if __name__ == '__main__':
    with open('log', 'r') as f:
        for line in f.read().splitlines():
            m = re.match(pat, line)
            print "vec.push_back(ProfileElement({}, {}, {}, {}));".format(
                    m.group("f"), m.group("oc"), m.group("ic"), m.group("threshold"))

