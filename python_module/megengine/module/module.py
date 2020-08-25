# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Iterable, Optional, Set, Tuple, Union

import numpy as np

from .._internal.dtype import is_quantize
from ..core import Buffer, Parameter, Tensor
from ..logger import get_logger
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
    return isinstance(obj, Buffer)


def _is_module(obj):
    return isinstance(obj, Module)


class Module(metaclass=ABCMeta):
    """Base Module class.
    """

    def __init__(self):
        # runtime attributes
        self.training = True
        self.quantize_disabled = False

        # hooks
        self._forward_pre_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()

    @abstractmethod
    def forward(self, inputs):
        pass

    def register_forward_pre_hook(self, hook: Callable) -> HookHandler:
        """Register a hook to handle forward inputs. `hook` should be a function

        Note that `inputs` keyword inputs

        :param hook: a function that receive `module` and `inputs`, then return
        a modified `inputs` or `None`.
        :return: a handler with :meth:`~.HookHandler.remove` interface to delete the hook.
        """
        return HookHandler(self._forward_pre_hooks, hook)

    def register_forward_hook(self, hook: Callable) -> HookHandler:
        """Register a hook to handle forward results. `hook` should be a function that
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
        """Scans the module object and returns an iterable for the :class:`~.Tensor`
        and :class:`~.Module` attributes that agree with the ``predicate``. For multiple
        calls of this function with same arguments, the order of objects within the
        returned iterable is guaranteed to be identical, as long as all the involved
        module objects' ``__dict__`` does not change thoughout those calls.

        :param recursive: Whether to recursively scan all the submodules.
        :param with_key: Whether to yield keys along with yielded objects.
        :param with_parent: Whether to yield ``self`` along with yielded objects.
        :param prefix: The prefix appended to the yielded keys.
        :param predicate: The predicate function applied to scanned objects.
        :param seen: A dict that records whether a module has been traversed yet.
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

    def parameters(
        self, requires_grad: Optional[bool] = None, recursive: bool = True, **kwargs
    ) -> Iterable[Parameter]:
        r"""Returns an iterable for the :class:`~.Parameter` of the module.

        :param requires_grad: Limitation over the :attr:`~.Parameter.requires_grad`
            attribute of returned :class:`.Parameter`. ``None`` for no limitation.
        :param recursive: If ``True``, returns all :class:`~.Parameter` within this
            module, else only returns :class:`~.Parameter` that are direct attributes
            of this module.
        """

        def predicate(obj) -> bool:
            return _is_parameter(obj) and (
                requires_grad is None or obj.requires_grad == requires_grad
            )

        yield from self._flatten(
            with_key=False, predicate=predicate, recursive=recursive, **kwargs
        )

    def named_parameters(
        self,
        requires_grad: Optional[bool] = None,
        prefix: Optional[str] = None,
        recursive: bool = True,
        **kwargs
    ) -> Iterable[Tuple[str, Parameter]]:
        """Returns an iterable for key :class:`~.Parameter` pairs of the module, where
        ``key`` is the dotted path from this module to the :class:`~.Parameter` .

        :param requires_grad: Limitation over the :attr:`~.Parameter.requires_grad`
            attribute of returned :class:`~.Parameter` . ``None`` for no limitation.
        :param prefix: The prefix prepended to the keys.
        :param recursive: If ``True``, returns all :class:`~.Parameter` within this
            module, else only returns :class:`~.Parameter` that are direct attributes
            of this module.
        """

        def predicate(obj) -> bool:
            return _is_parameter(obj) and (
                requires_grad is None or obj.requires_grad == requires_grad
            )

        yield from self._flatten(
            with_key=True,
            prefix=prefix,
            predicate=predicate,
            recursive=recursive,
            **kwargs,
        )

    def buffers(self, recursive: bool = True, **kwargs) -> Iterable[Buffer]:
        """Returns an iterable for the :class:`~.Buffer` of the module.

        :param recursive: If ``True``, returns all :class:`~.Buffer` within this
            module, else only returns :class:`~.Buffer` that are direct attributes
            of this module.
        """
        yield from self._flatten(
            with_key=False, predicate=_is_buffer, recursive=recursive, **kwargs
        )

    def named_buffers(
        self, prefix: Optional[str] = None, recursive: bool = True, **kwargs
    ) -> Iterable[Tuple[str, Buffer]]:
        """Returns an iterable for key :class:`~.Buffer` pairs of the module, where
        ``key`` is the dotted path from this module to the :class:`~.Buffer` .

        :param prefix: The prefix prepended to the keys.
        :param recursive: If ``True``, returns all :class:`~.Buffer` within this
            module, else only returns :class:`~.Buffer` that are direct attributes
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
        """Returns an iterable for all the submodules that are direct attributes of this
        module.
        """
        yield from self._flatten(
            with_key=False, predicate=_is_module, recursive=False, **kwargs
        )

    def named_children(self, **kwargs) -> "Iterable[Tuple[str, Module]]":
        """Returns an iterable of key-submodule pairs for all the submodules that are
        direct attributes of this module, where 'key' is the attribute name of
        submodules.
        """
        yield from self._flatten(
            with_key=True, predicate=_is_module, recursive=False, **kwargs
        )

    def modules(self, **kwargs) -> "Iterable[Module]":
        """Returns an iterable for all the modules within this module, including itself.
        """
        if "with_parent" in kwargs and kwargs["with_parent"]:
            yield self, None
        else:
            yield self
        yield from self._flatten(with_key=False, predicate=_is_module, **kwargs)

    def named_modules(
        self, prefix: Optional[str] = None, **kwargs
    ) -> "Iterable[Tuple[str, Module]]":
        """Returns an iterable of key-module pairs for all the modules within this
        module, including itself, where 'key' is the dotted path from this module to the
        submodules.

        :param prefix: The prefix prepended to the path.
        """
        if "with_parent" in kwargs and kwargs["with_parent"]:
            yield ("" if prefix is None else prefix), self, None
        else:
            yield ("" if prefix is None else prefix), self
        yield from self._flatten(
            with_key=True, prefix=prefix, predicate=_is_module, **kwargs
        )

    def apply(self, fn: "Callable[[Module], Any]") -> None:
        """Apply function ``fn`` to all the modules within this module, including
        itself.

        :param fn: The function to be applied on modules.
        """
        for it in self.modules():
            fn(it)

    def zero_grad(self) -> None:
        """Set all parameters' grads to zero
        """
        for param in self.parameters():
            if param.grad is not None:
                param.grad.reset_zero()

    def train(self, mode: bool = True, recursive: bool = True) -> None:
        """Set training mode of all the modules within this module (including itself) to
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
        """Set training mode of all the modules within this module (including itself) to
        ``False``. See :meth:`~.Module.train` for details.
        """
        self.train(False)

    def disable_quantize(self, value=True):
        r"""
        Set ``module``'s ``quantize_disabled`` attribute and return ``module``.
        Could be used as a decorator.
        """

        def fn(module: Module) -> None:
            module.quantize_disabled = value

        self.apply(fn)

    def replace_param(
        self, params: dict, start_pos: int, seen: Optional[Set[int]] = None
    ):
        """Replace module's parameters with `params`, used by :class:`~.ParamPack` to
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
                    assert module_dict[key].shape == params[start_pos + offset].shape
                    module_dict[key] = params[start_pos + offset]
                offset += 1
            if isinstance(module_dict[key], Module):
                offset += module_dict[key].replace_param(
                    params, start_pos + offset, seen
                )
        return offset

    def state_dict(self, rst=None, prefix="", keep_var=False):
        r"""Returns a dictionary containing whole states of the module.
        """

        def is_state(obj):
            return _is_parameter(obj) or _is_buffer(obj)

        if rst is None:
            rst = OrderedDict()

        for k, v in self._flatten(recursive=False, with_key=True, predicate=is_state):
            assert prefix + k not in rst, "duplicated state: {}".format(k)
            if keep_var:
                rst[prefix + k] = v
            else:
                rst[prefix + k] = v.numpy()

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
        r"""Load a given dictionary created by :func:`state_dict` into this module.
        If ``strict`` is ``True``, the keys of :func:`state_dict` must exactly match the keys
        returned by :func:`state_dict`.

        Users can also pass a closure: `Function[key: str, var: Tensor] -> Optional[np.ndarray]`
        as a `state_dict`, in order to handle complex situations. For example, load everything
        except for the final linear classifier:

        .. code-block::

            state_dict = {...}  #  Dict[str, np.ndarray]
            model.load_state_dict({
                k: None if k.startswith('fc') else v
                for k, v in state_dict.items()
            }, strict=False)

        Here returning `None` means skipping parameter `k`.

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
        """Advance state_dict load through callable `closure` whose signature is

            `closure(key: str, var: Tensor) -> Union[np.ndarry, None]`
        """
        assert callable(closure), "closure must be a function"

        loaded = []
        skipped = []

        local_state_dict = self.state_dict(keep_var=True)
        for k, var in local_state_dict.items():
            to_be_load = closure(k, var)
            if to_be_load is None:
                skipped.append(k)
                continue
            assert isinstance(
                to_be_load, np.ndarray
            ), "closure should return a `np.ndarray`, now `{}` get {}".format(
                k, to_be_load
            )
            assert (
                var.shape == to_be_load.shape
            ), "param `{}` shape mismatch, should be {}, get {}".format(
                k, var.shape, to_be_load.shape
            )
            # For quantized dtype, the initialized dtype
            # scale/zero_points maybe invalid, use pretrained dtype instead.
            if is_quantize(to_be_load.dtype) and is_quantize(var.dtype):
                var.dtype = to_be_load.dtype
            var.set_value(to_be_load)
            loaded.append(k)

        return set(loaded), set(skipped)
