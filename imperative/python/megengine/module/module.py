# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Iterable, Optional, Set, Tuple, Union

import numpy as np

from ..core.tensor.utils import make_shape_tuple
from ..logger import get_logger
from ..tensor import Parameter, Tensor
from ..utils.deprecation import deprecated
from ..utils.hook import HookHandler

logger = get_logger(__name__)


def _expand_structure(key, obj):
    if isinstance(obj, (Tensor, Module)):
        return [(key, obj)]
    elif isinstance(obj, (list, tuple, dict)):
        ret = []
        if isinstance(obj, dict):
            targets = ((k, obj[k]) for k in sorted(obj))
        else:
            targets = ((str(k), v) for k, v in enumerate(obj))
        for k, o in targets:
            sub_ret = _expand_structure(k, o)
            if sub_ret and not isinstance(k, str):
                raise AssertionError(
                    "keys for Tensor and Module must be str, error key: {}".format(k)
                )
            for kt, vt in sub_ret:
                ret.extend([(key + "." + kt, vt)])
        return ret
    else:
        return []


def _is_parameter(obj):
    return isinstance(obj, Parameter)


def _is_buffer(obj):
    return isinstance(obj, Tensor) and not isinstance(obj, Parameter)


def _is_module(obj):
    return isinstance(obj, Module)


def _get_XNorm_typeclass():
    from .batchnorm import _BatchNorm
    from .normalization import GroupNorm, InstanceNorm, LayerNorm

    XNorm_types = (_BatchNorm, GroupNorm, LayerNorm, InstanceNorm)
    return XNorm_types


class Module(metaclass=ABCMeta):
    """
    Base Module class.
    """

    def __init__(self):
        # runtime attributes
        self.training = True
        self.quantize_disabled = False

        # hooks
        self._forward_pre_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()

        self._modules = []

    @abstractmethod
    def forward(self, inputs):
        pass

    def register_forward_pre_hook(self, hook: Callable) -> HookHandler:
        """
        Registers a hook to handle forward inputs. `hook` should be a function.

        :param hook: a function that receive `module` and `inputs`, then return
        a modified `inputs` or `None`.
        :return: a handler with :meth:`~.HookHandler.remove` interface to delete the hook.
        """
        return HookHandler(self._forward_pre_hooks, hook)

    def register_forward_hook(self, hook: Callable) -> HookHandler:
        """
        Registers a hook to handle forward results. `hook` should be a function that
        receive `module`, `inputs` and `outputs`, then return a modified `outputs` or `None`.

        This method return a handler with :meth:`~.HookHandler.remove` interface to delete the hook.
        """
        return HookHandler(self._forward_hooks, hook)

    def __call__(self, *inputs, **kwargs):
        for hook in self._forward_pre_hooks.values():
            modified_inputs = hook(self, inputs)
            if modified_inputs is not None:
                if not isinstance(modified_inputs, tuple):
                    modified_inputs = (modified_inputs,)
                inputs = modified_inputs

        outputs = self.forward(*inputs, **kwargs)

        for hook in self._forward_hooks.values():
            modified_outputs = hook(self, inputs, outputs)
            if modified_outputs is not None:
                outputs = modified_outputs
        return outputs

    def _flatten(
        self,
        *,
        recursive: bool = True,
        with_key: bool = False,
        with_parent: bool = False,
        prefix: Optional[str] = None,
        predicate: Callable[[Any], bool] = lambda _: True,
        seen: Optional[Set[int]] = None
    ) -> Union[Iterable[Any], Iterable[Tuple[str, Any]]]:
        """
        Scans the module object and returns an iterable for the :class:`~.Tensor`
        and :class:`~.Module` attributes that agree with the ``predicate``. For multiple
        calls of this function with same arguments, the order of objects within the
        returned iterable is guaranteed to be identical, as long as all the involved
        module objects' ``__dict__`` does not change thoughout those calls.

        :param recursive: whether to recursively scan all the submodules.
        :param with_key: whether to yield keys along with yielded objects.
        :param with_parent: whether to yield ``self`` along with yielded objects.
        :param prefix: prefix appended to the yielded keys.
        :param predicate: the predication function applied to scanned objects.
        :param seen: a dict that records whether a module has been traversed yet.
        """
        if seen is None:
            seen = set([id(self)])

        module_dict = vars(self)
        _prefix = "" if prefix is None else prefix + "."

        for key in sorted(module_dict):
            for expanded_key, leaf in _expand_structure(key, module_dict[key]):
                leaf_id = id(leaf)
                if leaf_id in seen:
                    continue
                seen.add(leaf_id)

                if predicate(leaf):
                    if with_key and with_parent:
                        yield _prefix + expanded_key, leaf, self
                    elif with_key:
                        yield _prefix + expanded_key, leaf
                    elif with_parent:
                        yield leaf, self
                    else:
                        yield leaf

                if recursive and isinstance(leaf, Module):
                    yield from leaf._flatten(
                        recursive=recursive,
                        with_key=with_key,
                        with_parent=with_parent,
                        prefix=_prefix + expanded_key if with_key else None,
                        predicate=predicate,
                        seen=seen,
                    )

    def parameters(self, recursive: bool = True, **kwargs) -> Iterable[Parameter]:
        r"""
        Returns an iterable for the :class:`~.Parameter` of the module.

        :param recursive: If ``True``, returns all :class:`~.Parameter` within this
            module, else only returns :class:`~.Parameter` that are direct attributes
            of this module.
        """

        if "requires_grad" in kwargs:
            del kwargs["requires_grad"]
            warnings.warn(
                "Tensor currently has no requires_grad attribute "
                "so requires_grad argument is ignored here",
                DeprecationWarning,
            )

        def predicate(obj) -> bool:
            return _is_parameter(obj)

        yield from self._flatten(
            with_key=False, predicate=predicate, recursive=recursive, **kwargs
        )

    def named_parameters(
        self, prefix: Optional[str] = None, recursive: bool = True, **kwargs
    ) -> Iterable[Tuple[str, Parameter]]:
        """
        Returns an iterable for key :class:`~.Parameter` pairs of the module, where
        ``key`` is the dotted path from this module to the :class:`~.Parameter`.

        :param prefix: prefix prepended to the keys.
        :param recursive: if ``True``, returns all :class:`~.Parameter` within this
            module, else only returns :class:`~.Parameter` that are direct attributes
            of this module.
        """

        if "requires_grad" in kwargs:
            del kwargs["requires_grad"]
            warnings.warn(
                "Tensor currently has no requires_grad attribute "
                "so requires_grad argument is ignored here",
                DeprecationWarning,
            )

        def predicate(obj) -> bool:
            return _is_parameter(obj)

        yield from self._flatten(
            with_key=True,
            prefix=prefix,
            predicate=predicate,
            recursive=recursive,
            **kwargs,
        )

    def buffers(self, recursive: bool = True, **kwargs) -> Iterable[Tensor]:
        """
        Returns an iterable for the buffers of the module.

        Buffer is defined to be :class:`~.Tensor` excluding :class:`~.Parameter`.

        :param recursive: if ``True``, returns all buffers within this
            module, else only returns buffers that are direct attributes
            of this module.
        """
        yield from self._flatten(
            with_key=False, predicate=_is_buffer, recursive=recursive, **kwargs
        )

    def named_buffers(
        self, prefix: Optional[str] = None, recursive: bool = True, **kwargs
    ) -> Iterable[Tuple[str, Tensor]]:
        """
        Returns an iterable for key buffer pairs of the module, where
        ``key`` is the dotted path from this module to the buffer.

        Buffer is defined to be :class:`~.Tensor` excluding :class:`~.Parameter`.

        :param prefix: prefix prepended to the keys.
        :param recursive: if ``True``, returns all buffers within this
            module, else only returns buffers that are direct attributes
            of this module.
        """
        yield from self._flatten(
            with_key=True,
            prefix=prefix,
            predicate=_is_buffer,
            recursive=recursive,
            **kwargs,
        )

    def children(self, **kwargs) -> "Iterable[Module]":
        """
        Returns an iterable for all the submodules that are direct attributes of this
        module.
        """
        yield from self._flatten(
            with_key=False, predicate=_is_module, recursive=False, **kwargs
        )

    def named_children(self, **kwargs) -> "Iterable[Tuple[str, Module]]":
        """
        Returns an iterable of key-submodule pairs for all the submodules that are
        direct attributes of this module, where 'key' is the attribute name of
        submodules.
        """
        yield from self._flatten(
            with_key=True, predicate=_is_module, recursive=False, **kwargs
        )

    def modules(self, **kwargs) -> "Iterable[Module]":
        """
        Returns an iterable for all the modules within this module, including itself.
        """
        if "with_parent" in kwargs and kwargs["with_parent"]:
            yield self, None
        else:
            yield self
        yield from self._flatten(with_key=False, predicate=_is_module, **kwargs)

    def named_modules(
        self, prefix: Optional[str] = None, **kwargs
    ) -> "Iterable[Tuple[str, Module]]":
        """
        Returns an iterable of key-module pairs for all the modules within this
        module, including itself, where 'key' is the dotted path from this module to the
        submodules.

        :param prefix: prefix prepended to the path.
        """
        if "with_parent" in kwargs and kwargs["with_parent"]:
            yield ("" if prefix is None else prefix), self, None
        else:
            yield ("" if prefix is None else prefix), self
        yield from self._flatten(
            with_key=True, prefix=prefix, predicate=_is_module, **kwargs
        )

    def apply(self, fn: "Callable[[Module], Any]") -> None:
        """
        Applies function ``fn`` to all the modules within this module, including
        itself.

        :param fn: the function to be applied on modules.
        """
        for it in self.modules():
            fn(it)

    @deprecated(version="1.0")
    def zero_grad(self) -> None:
        """
        Sets all parameters' grads to zero
        """
        for param in self.parameters():
            if param.grad is not None:
                param.grad.reset_zero()

    def train(self, mode: bool = True, recursive: bool = True) -> None:
        """
        Sets training mode of all the modules within this module (including itself) to
        ``mode``. This effectively sets the ``training`` attributes of those modules
        to ``mode``, but only has effect on certain modules (e.g.
        :class:`~.BatchNorm2d`, :class:`~.Dropout`, :class:`~.Observer`)

        :param mode: the training mode to be set on modules.
        :param recursive: whether to recursively call submodules' ``train()``.
        """
        if not recursive:
            self.training = mode
            return

        def fn(module: Module) -> None:
            module.train(mode, recursive=False)

        self.apply(fn)

    def eval(self) -> None:
        """
        Sets training mode of all the modules within this module (including itself) to
        ``False``. See :meth:`~.Module.train` for details.
        """
        self.train(False)

    def disable_quantize(self, value=True):
        r"""
        Sets ``module``'s ``quantize_disabled`` attribute and return ``module``.
        Could be used as a decorator.
        """

        def fn(module: Module) -> None:
            module.quantize_disabled = value

        self.apply(fn)

    @deprecated(version="1.0")
    def replace_param(
        self, params: dict, start_pos: int, seen: Optional[Set[int]] = None
    ):
        """
        Replaces module's parameters with ``params``, used by :class:`~.ParamPack` to
        speedup multimachine training.
        """
        offset = 0
        if seen is None:
            seen = set([id(self)])
        module_dict = vars(self)
        for key in sorted(module_dict):
            hash_id = id(module_dict[key])
            if hash_id in seen:
                continue
            seen.add(hash_id)
            if isinstance(module_dict[key], Parameter):
                if start_pos + offset in params:
                    assert make_shape_tuple(module_dict[key].shape) == make_shape_tuple(
                        params[start_pos + offset].shape
                    )
                    module_dict[key] = params[start_pos + offset]
                offset += 1
            if isinstance(module_dict[key], Module):
                offset += module_dict[key].replace_param(
                    params, start_pos + offset, seen
                )
        return offset

    def state_dict(self, rst=None, prefix="", keep_var=False):
        _rst = self._state_dict(rst=rst, prefix=prefix, keep_var=keep_var)
        rst = OrderedDict()
        XNorm_typeclass = _get_XNorm_typeclass()
        for (module_type, k), v in _rst.items():
            # for performance reasons, parameters in XNorm (e.g., BatchNorm2d) are 4-dim tensors,
            # however they will be reshaped to 1-dim tensors before returned by `statr_dict()`
            if issubclass(module_type, XNorm_typeclass):
                v = v.reshape(-1)
            rst[k] = v
        return rst

    def _state_dict(self, rst=None, prefix="", keep_var=False):
        r"""
        Returns a dictionary containing whole states of the module.
        """

        def is_state(obj):
            return _is_parameter(obj) or _is_buffer(obj)

        module_type = self.__class__
        if rst is None:
            rst = OrderedDict()

        for k, v in self._flatten(recursive=False, with_key=True, predicate=is_state):
            assert prefix + k not in rst, "duplicated state: {}".format(k)
            if keep_var:
                rst[(module_type, prefix + k)] = v
            else:
                rst[(module_type, prefix + k)] = v.numpy()

        for k, submodule in self._flatten(
            recursive=False,
            with_key=True,
            predicate=lambda obj: isinstance(obj, Module),
        ):
            submodule.state_dict(rst, prefix + k + ".", keep_var)

        return rst

    def load_state_dict(
        self,
        state_dict: Union[dict, Callable[[str, Tensor], Optional[np.ndarray]]],
        strict=True,
    ):
        r"""
        Loads a given dictionary created by :func:`state_dict` into this module.
        If ``strict`` is ``True``, the keys of :func:`state_dict` must exactly match the keys
        returned by :func:`state_dict`.

        Users can also pass a closure: ``Function[key: str, var: Tensor] -> Optional[np.ndarray]``
        as a `state_dict`, in order to handle complex situations. For example, load everything
        except for the final linear classifier:

        .. code-block::

            state_dict = {...}  #  Dict[str, np.ndarray]
            model.load_state_dict({
                k: None if k.startswith('fc') else v
                for k, v in state_dict.items()
            }, strict=False)

        Here returning ``None`` means skipping parameter ``k``.

        To prevent shape mismatch (e.g. load PyTorch weights), we can reshape before loading:

        .. code-block::

            state_dict = {...}
            def reshape_accordingly(k, v):
                return state_dict[k].reshape(v.shape)
            model.load_state_dict(reshape_accordingly)

        We can also perform inplace re-initialization or pruning:

        .. code-block::

            def reinit_and_pruning(k, v):
                if 'bias' in k:
                    M.init.zero_(v)
                if 'conv' in k:
                    return v.numpy() * (np.abs(v.numpy()) > 1e-3).astype("float32)
            model.load_state_dict(reinit_and_pruning, strict=False)
        """
        unused = []
        if isinstance(state_dict, dict):
            unused = state_dict.keys()

            def closure(k, _):  # var unused
                return state_dict[k] if k in state_dict else None

        elif callable(state_dict):
            closure = state_dict
        else:
            raise ValueError(
                "`state_dict` must load a dict or callable, got {}".format(
                    type(state_dict)
                )
            )

        loaded, skipped = self._load_state_dict_with_closure(closure)
        unused = set(unused) - loaded

        if len(unused) != 0:
            if strict:
                raise KeyError(
                    "Unused params violate `strict=True`, unused={}".format(unused)
                )
            else:
                logger.warning(
                    "Unused params in `strict=False` mode, unused={}".format(unused)
                )

        if len(skipped) != 0:
            if strict:
                raise KeyError(
                    "Missing params violate `strict=True`, missing={}".format(skipped)
                )
            else:
                logger.warning(
                    "Missing params in `strict=False` mode, missing={}".format(skipped)
                )

    def _load_state_dict_with_closure(self, closure):
        """
        Advance state_dict load through callable ``closure`` whose signature is
        ``closure(key: str, var: Tensor) -> Union[np.ndarry, None]``
        """
        XNorm_typeclass = _get_XNorm_typeclass()
        assert callable(closure), "closure must be a function"

        loaded = []
        skipped = []

        local_state_dict = self._state_dict(keep_var=True)
        for (module_type, k), var in local_state_dict.items():
            to_be_load = closure(k, var)
            if to_be_load is None:
                skipped.append(k)
                continue
            assert isinstance(
                to_be_load, np.ndarray
            ), "closure should return a `np.ndarray`, now `{}` get {}".format(
                k, to_be_load
            )
            var_shape = make_shape_tuple(var.shape)
            to_be_load_shape = make_shape_tuple(to_be_load.shape)
            if var_shape != to_be_load_shape:
                # weight and bias in BatchNorm1d, BatchNorm2d and SyncBatchNorm are 1-dim tensors in v1.0, and
                # since v1.1 they are 4-dim tensors. The following special rule for these modules preserves the
                # backward compatibility.
                if issubclass(module_type, XNorm_typeclass):
                    if np.prod(var_shape) == np.prod(to_be_load_shape):
                        to_be_load = to_be_load.reshape(var_shape)
                    else:
                        raise ValueError(
                            "param `{}` size mismatch, should be {}, get {}".format(
                                k, np.prod(var_shape), np.prod(to_be_load_shape)
                            )
                        )
                else:
                    raise ValueError(
                        "param `{}` shape mismatch, should be {}, get {}".format(
                            k, var_shape, to_be_load_shape
                        )
                    )
            var._reset(type(var)(to_be_load, dtype=to_be_load.dtype, device=var.device))
            loaded.append(k)

        return set(loaded), set(skipped)

    def __setattr__(self, name: str, value):
        if _is_module(value):
            modules = self.__dict__.get("_modules")
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call"
                )
            if name not in self.__dict__:
                modules.append(name)
        super().__setattr__(name, value)

    def __delattr__(self, name: str):
        if name in self.__dict__ and _is_module(self.__dict__[name]):
            modules = self.__dict__.get("_modules")
            modules.remove(name)
        super().__delattr__(name)

    def _module_info_string(self) -> str:
        r"""
        Set the extra representation of the module.
        """
        return ""

    def __repr__(self):
        def add_indent(repr_str, num_spaces):
            s = repr_str.split("\n")
            # don't do anything for single-line stuff
            if len(s) == 1:
                return repr_str
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        extra_lines = []
        extra_repr = self._module_info_string()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = [
            "(" + name + "): " + add_indent(repr(self.__dict__[name]), 2)
            for name in self._modules
        ]
        lines = extra_lines + child_lines
        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str
