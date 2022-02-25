# -*- coding: utf-8 -*-
import importlib.util
import os
import types
from contextlib import contextmanager
from typing import Iterator


def load_module(name: str, path: str) -> types.ModuleType:
    r"""Loads module specified by name and path.

    Args:
        name: module name.
        path: module path.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_module_exists(module: str) -> bool:
    r"""Checks whether python module exists or not.

    Args:
        module: name of module.
    """
    return importlib.util.find_spec(module) is not None


@contextmanager
def cd(target: str) -> Iterator[None]:
    """Changes current directory to target.

    Args:
        target: target directory.
    """
    prev = os.getcwd()
    os.chdir(os.path.expanduser(target))
    try:
        yield
    finally:
        os.chdir(prev)
