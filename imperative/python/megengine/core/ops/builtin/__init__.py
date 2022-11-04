# -*- coding: utf-8 -*-
from ..._imperative_rt import OpDef

original_keys = set()


def backup_keys():
    global original_keys
    original_keys = set()
    for k in globals().keys():
        original_keys.add(k)


backup_keys()

from ..._imperative_rt.ops import *  # isort:skip


def setup():
    to_be_removed = set()
    for k, v in globals().items():
        is_original_key = k in original_keys
        is_op = isinstance(v, type) and issubclass(v, OpDef)
        if not is_op and not is_original_key:
            to_be_removed.add(k)

    for k in to_be_removed:
        del globals()[k]


setup()
