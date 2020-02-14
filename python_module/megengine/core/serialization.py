# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import pickle

from ..utils.max_recursion_limit import max_recursion_limit


def save(obj, f, pickle_module=pickle, pickle_protocol=pickle.HIGHEST_PROTOCOL):
    r"""Save an object to disk file.

    :type obj: object
    :param obj: object to save. Only ``module`` or ``state_dict`` are allowed.
    :type f: text file object
    :param f: a string of file name or a text file object to which ``obj`` is saved to.
    :type pickle_module:
    :param pickle_module: Default: ``pickle``.
    :type pickle_protocol:
    :param pickle_protocol: Default: ``pickle.HIGHEST_PROTOCOL``.

    """
    if isinstance(f, str):
        with open(f, "wb") as fout:
            save(
                obj, fout, pickle_module=pickle_module, pickle_protocol=pickle_protocol
            )
        return

    with max_recursion_limit():
        assert hasattr(f, "write"), "{} does not support write".format(f)
        pickle_module.dump(obj, f, pickle_protocol)


def load(f, pickle_module=pickle):
    r"""Load an object saved with save() from a file.

    :type f: text file object
    :param f: a string of file name or a text file object from which to load.
    :type pickle_module:
    :param pickle_module: Default: ``pickle``.

    """
    if isinstance(f, str):
        with open(f, "rb") as fin:
            return load(fin, pickle_module=pickle_module)
    return pickle_module.load(f)
