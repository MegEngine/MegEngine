import contextlib

from ..module import Sequential
from ..module.module import Module, _access_structure
from ..tensor import Tensor


def get_expand_structure(obj: Module, key: str):
    r"""Gets Module's attribute compatible with complex key from Module's :meth:`~.named_children`.
    Supports handling structure containing list or dict.

    Args:
        obj: Module: 
        key: str: 
    """

    def f(_, __, cur):
        return cur

    return _access_structure(obj, key, callback=f)


def set_expand_structure(obj: Module, key: str, value):
    r"""Sets Module's attribute compatible with complex key from Module's :meth:`~.named_children`.
    Supports handling structure containing list or dict.
    """

    def f(parent, key, cur):
        if isinstance(parent, (Tensor, Module)):
            # cannnot use setattr to be compatible with Sequential's ``__setitem__``
            if isinstance(cur, Sequential):
                parent[int(key)] = value
            else:
                setattr(parent, key, value)
        else:
            parent[key] = value

    _access_structure(obj, key, callback=f)


@contextlib.contextmanager
def set_module_mode_safe(
    module: Module, training: bool = False,
):
    r"""Adjust module to training/eval mode temporarily.

    Args:
        module: used module.
        training: training (bool): training mode. True for train mode, False fro eval mode.
    """
    backup_stats = {}

    def recursive_backup_stats(module, mode):
        for m in module.modules():
            backup_stats[m] = m.training
            m.train(mode, recursive=False)

    def recursive_recover_stats(module):
        for m in module.modules():
            m.training = backup_stats.pop(m)

    recursive_backup_stats(module, mode=training)
    yield module
    recursive_recover_stats(module)
