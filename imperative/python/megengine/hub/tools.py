# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import importlib.util
import os
import types
from contextlib import contextmanager
from typing import Iterator


def load_module(name: str, path: str) -> types.ModuleType:
    """
    Loads module specified by name and path.

    :param name: module name.
    :param path: module path.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_module_exists(module: str) -> bool:
    """
    Checks whether python module exists or not.

    :param module: name of module.
    """
    return importlib.util.find_spec(module) is not None


@contextmanager
def cd(target: str) -> Iterator[None]:
    """
    Changes current directory to target.

    :param target: target directory.
    """
    prev = os.getcwd()
    os.chdir(os.path.expanduser(target))
    try:
        yield
    finally:
        os.chdir(prev)
