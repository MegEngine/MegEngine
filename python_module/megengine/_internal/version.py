# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""version information for MegBrain package"""

import collections

from . import mgb as _mgb


class Version(
    collections.namedtuple("VersionBase", ["major", "minor", "patch", "dev"])
):
    """simple sematic version object"""

    @classmethod
    def __normalize(cls, v):
        if isinstance(v, str):
            v = v.split(".")
        a, b, c = map(int, v)
        return cls(a, b, c)

    def __eq__(self, rhs):
        return super().__eq__(self.__normalize(rhs))

    def __ne__(self, rhs):
        return super().__ne__(self.__normalize(rhs))

    def __lt__(self, rhs):
        return super().__lt__(self.__normalize(rhs))

    def __le__(self, rhs):
        return super().__le__(self.__normalize(rhs))

    def __gt__(self, rhs):
        return super().__gt__(self.__normalize(rhs))

    def __ge__(self, rhs):
        return super().__ge__(self.__normalize(rhs))

    def __str__(self):
        rst = "{}.{}.{}".format(self.major, self.minor, self.patch)
        if self.dev:
            rst += "-dev{}".format(self.dev)
        return rst


Version.__new__.__defaults__ = (0,)  # dev defaults to 0

version_info = Version(*_mgb._get_mgb_version())
__version__ = str(version_info)
