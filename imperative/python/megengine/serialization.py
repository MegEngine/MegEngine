# -*- coding: utf-8 -*-
import pickle

from .device import _valid_device, get_default_device
from .tensor import Tensor
from .utils.max_recursion_limit import max_recursion_limit


def save(obj, f, pickle_module=pickle, pickle_protocol=pickle.DEFAULT_PROTOCOL):
    r"""Save an object to disk file.

    Args:
        obj: object to save. Only ``module`` or ``state_dict`` are allowed.
        f: a string of file name or a text file object to which ``obj`` is saved to.
        pickle_module: Default: ``pickle``.
        pickle_protocol: Default: ``pickle.DEFAULT_PROTOCOL``.
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


class _dmap:
    def __init__(self, map_location):
        self.map_location = map_location

    def __enter__(self):
        Tensor.dmap_callback = staticmethod(self.map_location)
        return self

    def __exit__(self, type, value, traceback):
        Tensor.dmap_callback = None


def _get_callable_map_location(map_location):
    if map_location is None:

        def callable_map_location(state):
            return state

    elif isinstance(map_location, str):

        def callable_map_location(state):
            return map_location

    elif isinstance(map_location, dict):
        for key, value in map_location.items():
            # dict key and values can only be "xpux", "cpux", "gpu0", etc.
            assert _valid_device(key), "Invalid locator_map key value {}".format(key)
            assert _valid_device(value), "Invalid locator_map key value {}".format(
                value
            )

        def callable_map_location(state):
            if state[:4] in map_location.keys():
                state = map_location[state[:4]]
            return state

    else:
        assert callable(map_location), "map_location should be str, dict or function"
        callable_map_location = map_location
    return callable_map_location


def load(f, map_location=None, pickle_module=pickle):
    r"""Load an object saved with :func:~.megengine.save` from a file.

    Args:
        f: a string of file name or a text file object from which to load.
        map_location: Default: ``None``.
        pickle_module: Default: ``pickle``.

    Note:
       * ``map_location`` defines device mapping. See examples for usage.
       * If you will call :func:`~.megengine.set_default_device()`, please do it
         before :func:`~.megengine.load()`.

    Examples:
        .. code-block::

           import megengine as mge
           # Load tensors to the same device as defined in model.pkl
           mge.load('model.pkl')
           # Load all tensors to gpu0.
           mge.load('model.pkl', map_location='gpu0')
           # Load all tensors originally on gpu0 to cpu0
           mge.load('model.pkl', map_location={'gpu0':'cpu0'})
           # Load all tensors to cpu0
           mge.load('model.pkl', map_location=lambda dev: 'cpu0')
    """
    if isinstance(f, str):
        with open(f, "rb") as fin:
            return load(fin, map_location=map_location, pickle_module=pickle_module)

    map_location = _get_callable_map_location(map_location)  # callable map_location

    with _dmap(map_location) as dm:
        return pickle_module.load(f)
