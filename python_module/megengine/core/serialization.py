# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import pickle

import megengine._internal as mgb

from ..utils.max_recursion_limit import max_recursion_limit
from .device import get_default_device


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


class dmap:
    def __init__(self, map_location):
        self.map_location = map_location

    def __enter__(self):
        mgb.add_device_map(self.map_location)
        return self

    def __exit__(self, type, value, traceback):
        mgb.del_device_map()


def _get_callable_map_location(map_location):
    if map_location is None:

        def callable_map_location(state):
            return str(get_default_device())

    elif isinstance(map_location, str):

        def callable_map_location(state):
            return map_location

    elif isinstance(map_location, dict):
        locator_map = {}
        for key, value in map_location.items():
            locator_key = mgb.config.parse_locator(key)[:2]
            locator_map[locator_key] = value

        def callable_map_location(state):
            orig = mgb.config.parse_locator(state)[:2]
            if orig in locator_map.keys():
                state = locator_map[orig]
            return state

    else:
        assert callable(map_location), "map_location should be str, dict or function"
        callable_map_location = map_location
    return callable_map_location


def load(f, map_location=None, pickle_module=pickle):
    r"""Load an object saved with save() from a file.

    :type f: text file object
    :param f: a string of file name or a text file object from which to load.
    :type map_location: str, dict or a function specifying the map rules
    :param map_location: Default: ``None``.

        .. note::

            map_location will change the logical locator when loading models,
            avoiding tensors be loading on non-existent device. If you want to
            add the mapping relationship between logical locator and physical
            locator in runtime, please call :func:`mge.set_device_map()`

    :type pickle_module:
    :param pickle_module: Default: ``pickle``.

    .. note::

        If you will call :func:`mge.set_default_device()`, please do it
        before :func:`mge.load()`.

    Examples:

    .. testcode:

        import megengine as mge
        mge.load('model.mge')
        # Load all tensors based on logical location.
        mge.load('model.mge', map_location='gpu0')
        # Load all tensors onto the device: GPU0
        mge.load('model.mge', map_location={'gpu0':'cpu0'})
        # Load all tensors based on logical location, but 'GPU0' will be renamed to 'CPU0'
        mge.load('model.mge', map_location=lambda dev: 'cpu0')
        # Load all tensors onto the device" CPU0

    """
    if isinstance(f, str):
        with open(f, "rb") as fin:
            return load(fin, map_location=map_location, pickle_module=pickle_module)

    map_location = _get_callable_map_location(map_location)  # callable map_location

    with dmap(map_location):
        return pickle_module.load(f)
