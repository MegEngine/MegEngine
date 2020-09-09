# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import time

from numpy.random import MT19937

_rng = None


def _random_seed_generator():
    if _rng is None:
        from ..distributed.group import get_rank

        seed(seed=int(time.time()) + get_rank())
    while True:
        yield _rng.random_raw()


def seed(seed: int):
    global _rng  # pylint: disable=global-statement
    _rng = MT19937(seed=seed)
