import collections
import copy
import inspect
from collections.abc import MutableMapping, MutableSequence
from inspect import FullArgSpec
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

from .. import get_logger
from ..module import Module
from ..tensor import Tensor

logger = get_logger(__name__)


def replace_container_with_module_container(container):
    has_module = False
    module_container = None
    if isinstance(container, Dict):
        m_dic = copy.copy(container)
        for key, value in container.items():
            if isinstance(value, Module):
                has_module = True
            elif isinstance(value, (List, Dict)):
                (
                    _has_module,
                    _module_container,
                ) = replace_container_with_module_container(value)
                m_dic[key] = _module_container
                if _has_module:
                    has_module = True
        if not all(isinstance(v, Module) for v in m_dic.values()):
            return has_module, None
        else:
            return has_module, _ModuleDict(m_dic)
    elif isinstance(container, List):
        m_list = copy.copy(container)
        for ind, value in enumerate(container):
            if isinstance(value, Module):
                has_module = True
            elif isinstance(value, (List, Dict)):
                (
                    _has_module,
                    _module_container,
                ) = replace_container_with_module_container(value)
                m_list[ind] = _module_container
                if _has_module:
                    has_module = True
        if not all(isinstance(v, Module) for v in m_list):
            return has_module, None
        else:
            return has_module, _ModuleList(m_list)
    return has_module, module_container


def _convert_kwargs_to_args(
    argspecs: Union[Callable, FullArgSpec], args, kwargs, is_bounded=False
):
    # is_bounded = True when func is a method and provided args don't include 'self'
    arg_specs = (
        inspect.getfullargspec(argspecs) if isinstance(argspecs, Callable) else argspecs
    )
    func_name = argspecs.__qualname__ if isinstance(argspecs, Callable) else "function"
    assert isinstance(arg_specs, FullArgSpec)
    arg_specs_args = arg_specs.args
    arg_specs_defaults = arg_specs.defaults if arg_specs.defaults else []
    arg_specs_kwonlyargs = arg_specs.kwonlyargs
    arg_specs_kwonlydefaults = (
        arg_specs.kwonlydefaults if arg_specs.kwonlydefaults else dict()
    )
    if is_bounded:
        arg_specs_args = arg_specs.args[1:]
    new_args = []
    new_kwargs = {}
    new_args.extend(args)
    if set(arg_specs_args[0 : len(new_args)]) & set(kwargs.keys()):
        repeated_arg_name = set(arg_specs_args[0 : len(new_args)]) & set(kwargs.keys())
        raise TypeError(
            "{} got multiple values for argument {}".format(
                func_name, ", ".join(repeated_arg_name)
            )
        )
    if len(new_args) < len(arg_specs_args):
        for ind in range(len(new_args), len(arg_specs_args)):
            arg_name = arg_specs_args[ind]
            if arg_name in kwargs:
                new_args.append(kwargs[arg_name])
            else:
                index = ind - len(arg_specs_args) + len(arg_specs_defaults)
                if index >= len(arg_specs_defaults) or index < 0:
                    raise TypeError(
                        "{} missing required positional arguments: {}".format(
                            func_name, arg_name
                        )
                    )
                new_args.append(arg_specs_defaults[index])

    for kwarg_name in arg_specs_kwonlyargs:
        if kwarg_name in kwargs:
            new_kwargs[kwarg_name] = kwargs[kwarg_name]
        else:
            if kwarg_name not in arg_specs_kwonlydefaults:
                raise TypeError(
                    "{} missing required keyword-only argument: {}".format(
                        func_name, kwarg_name
                    )
                )
            new_kwargs[kwarg_name] = arg_specs_kwonlydefaults[kwarg_name]
    for k, v in kwargs.items():
        if k not in arg_specs_args and k not in arg_specs_kwonlyargs:
            if arg_specs.varkw is None:
                raise TypeError(
                    "{} got an unexpected keyword argument {}".format(func_name, k)
                )
            new_kwargs[k] = v
    return tuple(new_args), new_kwargs


def _check_obj_attr(obj):
    # check if all the attributes of a obj is serializable
    from .pytree import tree_flatten
    from .pytree import SUPPORTED_LEAF_CLS, SUPPORTED_LEAF_TYPE, TreeDef
    from .expr import Expr
    from .traced_module import TracedModule, InternalGraph, NameSpace

    def _check_leaf_type(leaf):
        leaf_type = leaf if isinstance(leaf, type) else type(leaf)
        traced_module_types = [Expr, TreeDef, TracedModule, InternalGraph, NameSpace]
        return (
            issubclass(leaf_type, tuple(SUPPORTED_LEAF_CLS + traced_module_types))
            or leaf_type in SUPPORTED_LEAF_TYPE
        )

    for _, v in obj.items():
        leafs, _ = tree_flatten(v, is_leaf=lambda _: True)
        for leaf in leafs:
            assert _check_leaf_type(leaf), (
                "Type {} is not supported in TracedModule serialization by default. "
                "If you want to save this object to file, please call tm.register_supported_type({}) "
                "before saving.".format(
                    leaf if isinstance(leaf, type) else type(leaf), type(leaf).__name__
                )
            )


def _check_builtin_module_attr(mod):
    from .pytree import _is_leaf as _check_leaf_type
    from .pytree import tree_flatten

    # check if all the attributes of a builtin module is serializable
    is_non_serializable_module = lambda m: isinstance(
        m, Module
    ) and not _check_builtin_module_attr(m)
    for k, v in mod.__dict__.items():
        if k == "_m_dump_modulestate":
            continue
        if is_non_serializable_module(v):
            return False
        elif not isinstance(v, Module):
            leafs, _ = tree_flatten(v, is_leaf=lambda _: True)
            for leaf in leafs:
                if not _check_leaf_type(leaf) or is_non_serializable_module(leaf):
                    logger.warn(
                        "Type {} is not supported by traced module".format(
                            leaf if isinstance(leaf, type) else type(leaf)
                        )
                    )
                    return False
    return True


class _ModuleList(Module, MutableSequence):
    r"""A List-like container.

    Using a ``ModuleList``, one can visit, add, delete and modify submodules
    just like an ordinary python list.
    """

    def __init__(self, modules: Optional[Iterable[Module]] = None):
        super().__init__()
        self._size = 0
        if modules is None:
            return
        for mod in modules:
            self.append(mod)

    @classmethod
    def _ikey(cls, idx):
        return "{}".format(idx)

    def _check_idx(self, idx):
        L = len(self)
        if idx < 0:
            idx = L + idx
        if idx < 0 or idx >= L:
            raise IndexError("list index out of range")
        return idx

    def __getitem__(self, idx: int):
        if isinstance(idx, slice):
            idx = range(self._size)[idx]
        if not isinstance(idx, Sequence):
            idx = [
                idx,
            ]
        rst = []
        for i in idx:
            i = self._check_idx(i)
            key = self._ikey(i)
            try:
                rst.append(getattr(self, key))
            except AttributeError:
                raise IndexError("list index out of range")
        return rst if len(rst) > 1 else rst[0]

    def __setattr__(self, key, value):
        # clear mod name to avoid warning in Module's setattr
        if isinstance(value, Module):
            value._short_name = None
        super().__setattr__(key, value)

    def __setitem__(self, idx: int, mod: Module):
        if not isinstance(mod, Module):
            raise ValueError("invalid sub-module")
        idx = self._check_idx(idx)
        setattr(self, self._ikey(idx), mod)

    def __delitem__(self, idx):
        idx = self._check_idx(idx)
        L = len(self)
        for orig_idx in range(idx + 1, L):
            new_idx = orig_idx - 1
            self[new_idx] = self[orig_idx]
        delattr(self, self._ikey(L - 1))
        self._size -= 1

    def __len__(self):
        return self._size

    def insert(self, idx, mod: Module):
        assert isinstance(mod, Module)
        L = len(self)
        if idx < 0:
            idx = L - idx
        # clip idx to (0, L)
        if idx > L:
            idx = L
        elif idx < 0:
            idx = 0

        for new_idx in range(L, idx, -1):
            orig_idx = new_idx - 1
            key = self._ikey(new_idx)
            setattr(self, key, self[orig_idx])

        key = self._ikey(idx)
        setattr(self, key, mod)
        self._size += 1

    def forward(self):
        raise RuntimeError("ModuleList is not callable")


class _ModuleDict(Module, MutableMapping):
    r"""A Dict-like container.

    Using a ``ModuleDict``, one can visit, add, delete and modify submodules
    just like an ordinary python dict.
    """

    def __init__(self, modules: Optional[Dict[str, Module]] = None):
        super().__init__()
        self._module_keys = []
        if modules is not None:
            self.update(modules)

    def __delitem__(self, key):
        delattr(self, key)
        assert key in self._module_keys
        self._module_keys.remove(key)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setattr__(self, key, value):
        # clear mod name to avoid warning in Module's setattr
        if isinstance(value, Module):
            value._short_name = None
        super().__setattr__(key, value)

    def __setitem__(self, key, value):
        if not isinstance(value, Module):
            raise ValueError("invalid sub-module")
        setattr(self, key, value)
        if key not in self._module_keys:
            self._module_keys.append(key)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self._module_keys)

    def items(self):
        return [(key, getattr(self, key)) for key in self._module_keys]

    def values(self):
        return [getattr(self, key) for key in self._module_keys]

    def keys(self):
        return self._module_keys

    def forward(self):
        raise RuntimeError("ModuleList is not callable")


def assign_attr(obj: Union[Module, Tensor], module: Module, target: str):
    *prefix, name = target.split(".")
    for item in prefix:
        module = getattr(module, item)
        if not isinstance(module, Module):
            raise AttributeError("`{}` is not an Module".format(item))
    setattr(module, name, obj)


def get_subattr(module: Module, target: str):
    # todo : remove this import
    from .node import ModuleNode

    if target == "":
        return module
    *prefix, name = target.split(".")
    for item in prefix:
        module = getattr(module, item)
        if not isinstance(module, (Module, ModuleNode)):
            raise AttributeError("`{}` is not an Module".format(item))
    return getattr(module, name)
