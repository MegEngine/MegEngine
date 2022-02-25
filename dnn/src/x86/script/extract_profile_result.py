#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

pat = r'f=(?P<f>\d+), oc=(?P<oc>\d+), ic=(?P<ic>\d+), threshold=(?P<threshold>\d+)'

if __name__ == '__main__':
    with open('log', 'r') as f:
        for line in f.read().splitlines():
            m = re.match(pat, line)
            print "vec.push_back(ProfileElement({}, {}, {}, {}));".format(
                    m.group("f"), m.group("oc"), m.group("ic"), m.group("threshold"))

