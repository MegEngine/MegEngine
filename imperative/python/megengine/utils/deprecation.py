# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import importlib
import warnings

from deprecated.sphinx import deprecated


def deprecated_func(version, origin, name, tbd):
    """
    :param version: version to deprecate this function
    :param origin: origin module path
    :param name: function name
    :param tbd: to be discussed, if true, ignore warnings
    """
    should_warning = not tbd

    def wrapper(*args, **kwargs):
        nonlocal should_warning
        module = importlib.import_module(origin)
        func = module.__getattribute__(name)
        if should_warning:
            with warnings.catch_warnings():
                warnings.simplefilter(action="always")
                warnings.warn(
                    "Call to deprecated function {}. (use {}.{} instead) -- Deprecated since version {}.".format(
                        name, origin, name, version
                    ),
                    category=DeprecationWarning,
                    stacklevel=2,
                )
            should_warning = False
        return func(*args, **kwargs)

    return wrapper
