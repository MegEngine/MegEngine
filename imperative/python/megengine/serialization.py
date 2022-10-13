# -*- coding: utf-8 -*-
import pickle

from .device import _valid_device, get_default_device
from .tensor import Tensor
from .utils.max_recursion_limit import max_recursion_limit


def save(obj, f, pickle_module=pickle, pickle_protocol=pickle.DEFAULT_PROTOCOL):
    r"""Save an object to disk file.
    The saved object must be a :class:`~.module.Module`,
    :attr:`.Module.state_dict` or :attr:`.Optimizer.state_dict`.
    See :ref:`serialization-guide` for more details.

    Args:
        obj: object to be saved.
        f: a string of file name or a text file object to which ``obj`` is saved to.
        pickle_module: the module to use for pickling.
        pickle_protocol: the protocol to use for pickling.

    .. admonition:: If you are using MegEngine with different Python versions
       :class: warning

       Different Python version may use different DEFAULT/HIGHEST pickle protocol.
       If you want to :func:`~megengine.load` the saved object in another Python version,
       please make sure you have used the same protocol.

    .. admonition:: You can select to use ``pickle`` module directly

        This interface is a wrapper of :func:`pickle.dump`. If you want to use ``pickle``,
        See :py:mod:`pickle` for more information about how to set ``pickle_protocol``:

        * :py:data:`pickle.HIGHEST_PROTOCOL` - the highest protocol version available.
        * :py:data:`pickle.DEFAULT_PROTOCOL` - the default protocol version used for pickling.

    Examples:

        If you want to save object in a higher protocol version which current version Python
        not support, you can install other pickle module instead of the build-in one.
        Take ``pickle5`` as an example:

        >>> import pickle5 as pickle  # doctest: +SKIP

        It's a backport of the pickle 5 protocol (PEP 574) and other pickle changes.
        So you can use it to save object in pickle 5 protocol and load it in Python 3.8+.

        Or you can use ``pickle5`` in this way (only used with this interface)：

        .. code-block:: python

           import pickle5
           import megengine

           megengine.save(obj, f, pickle_module=pickle5, pickle_protocol=5)

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
    r"""Load an object saved with :func:`~.megengine.save` from a file.

    Args:
        f: a string of file name or a text file object from which to load.
        map_location: defines device mapping. See examples for usage.
        pickle_module: the module to use for pickling.

    Note:
       
        If you will call :func:`~.megengine.set_default_device()`, please do it
        before :func:`~.megengine.load()`.

    .. admonition:: If you are using MegEngine with different Python versions
       :class: warning

       Different Python version may use different DEFAULT/HIGHEST pickle protocol.
       If you want to :func:`~megengine.load` the saved object in another Python version,
       please make sure you have used the same protocol.

    .. admonition:: You can select to use ``pickle`` module directly

        This interface is a wrapper of :func:`pickle.load`. If you want to use ``pickle``,
        See :py:mod:`pickle` for more information about how to set ``pickle_protocol``:

        * :py:data:`pickle.HIGHEST_PROTOCOL` - the highest protocol version available.
        * :py:data:`pickle.DEFAULT_PROTOCOL` - the default protocol version used for pickling.

    Examples:

        This example shows how to load tenors to different devices:

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

        If you are using a lower version of Python (<3.8),
        you can use other pickle module like ``pickle5`` to load object saved in pickle 5 protocol:

        >>> import pickle5 as pickle  # doctest: +SKIP

        Or you can use ``pickle5`` in this way (only used with this interface)：

        .. code-block:: python

           import pickle5
           import megengine

           megengine.load(obj, pickle_module=pickle5)

    """
    if isinstance(f, str):
        with open(f, "rb") as fin:
            return load(fin, map_location=map_location, pickle_module=pickle_module)

    map_location = _get_callable_map_location(map_location)  # callable map_location

    with dmap(map_location) as dm:
        return pickle_module.load(f)
