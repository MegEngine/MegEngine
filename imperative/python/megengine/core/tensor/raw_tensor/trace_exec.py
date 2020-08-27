# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
import weakref

# Concepts
#
# * Internal tensor
#   Tensor produced by the static sequence
#
# * External tensor
#   Tensor not produced, but used as input, by the static sequence
#
# * Irrelevant tensor
#   Tensor not present in input/output of any op
#
# * Escape
#   An internal tensor is said to escape if it is still alive
#   at the end of the sequence

# JIT-ed execution
#
# 1. read attr (dtype, device, shape)
#    a. internal tensor
#       read out as soon as tensor is produced
#    b. external or irrelevant tensor
#       fallback
#
# 2. apply op
#    bind external tensors in input
#
# 3. del


class Action:
    pass


class ReadAttrAction(Action):
    def __init__(self, var, name, getter):
        self.var = var
        self.name = name
        self.getter = getter


class ReadValueAction(Action):
    def __init__(self, var, getter):
        self.var = var
        self.getter = getter


class GetTensorAction(Action):
    def __init__(self, var, getter):
        self.var = var
        self.getter = getter


class OpAction(Action):
    def __init__(self, op, inputs, outputs, input_receivers):
        self.op = op
        self.inputs = inputs
        self.outputs = outputs
        self.input_receivers = input_receivers


class TensorAttr:
    def __init__(self):
        self.shape = None
        self.dtype = None
        self.device = None


class Bailout(Exception):
    pass


class Fallback(Exception):
    pass


def handle_bailout_fallback_finalize(f):
    @functools.wraps(f)
    def wrapper(self, impl, *args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Bailout:
            self.bailout()
        except Fallback:
            pass
        finally:
            if self.pc == len(self):
                self.finalize()
        return impl(*args, **kwargs)

    return wrapper


class ExecTrajectory(list):
    def __init__(self):
        super().__init__()
        self.reset()

    def __bool__(self):
        return True

    def __enter__(self):
        global _current_trajectory
        if hasattr(self, "_prev_trajectory"):
            raise RuntimeError
        self._prev_trajectory = _current_trajectory
        _current_trajectory = self
        self._exited = False
        return self

    def __exit__(self, *exc_info):
        # cleanup should be done at completion,
        # which is before exiting context manager
        assert self._exited == (exc_info == (None, None, None))
        if not self._exited:
            assert self.pc < len(self)
            self.bailout()

    def _exit(self):
        # clean up self and global varaible
        assert not self._exited
        self.reset()

        global _current_trajectory
        if _current_trajectory is not self:
            raise RuntimeError
        _current_trajectory = self._prev_trajectory
        del self._prev_trajectory

    def reset(self):
        self._exited = True
        self.pc = 0
        self.attr_cache = weakref.WeakKeyDictionary()

        ### Internal and External Tensor ###
        # internal tensors are those produced by us
        # external tensors are those received from outside
        # during JIT-ed execution, internal tensors are just placeholders.
        # var_to_tensor is the binding table for all tensors
        self.var_to_tensor = {}  # var -> weakref[tensor]
        # tensor_to_var is the reverse binding table for internal tensors
        # note that external tensors could map to >1 vars.
        self.tensor_to_var = weakref.WeakKeyDictionary()
        # internal tensor will be materialized if its .data is accessed from outside
        # after being meterialized, an intern tensor is much like an external tensor

    def finalize(self):
        assert self.pc == len(self)
        self._exit()

    def bailout(self):
        self._exit()
        raise NotImplementedError

    def next_action(self):
        assert not self._exited
        assert self.pc < len(self)
        return self[self.pc]

    @handle_bailout_fallback_finalize
    def read_attr(self, tensor, name):
        attrs = self.attr_cache.setdefault(tensor, TensorAttr())
        value = getattr(attrs, name, None)
        if value is None:
            action = self.next_action()
            if not isinstance(action, ReadAttrAction):
                raise Bailout
            if name != action.name:
                raise Bailout
            value = action.getter()
            setattr(attrs, name, value)
        return value

    @handle_bailout_fallback_finalize
    def read_value(self, impl, tensor):
        # possibilities:
        # 1. internal tensor
        # 2. external tensor
        # 3. irrelevant tensor (not an input / output of any op)
        if tensor not in self.tensor_to_var:
            raise Fallback
        assert tensor._data is None
        action = self.next_action()
        if not isinstance(action, ReadValueAction):
            raise Bailout
        return action.getter()

    @handle_bailout_fallback_finalize
    def apply_op(self, impl, op, *args):
        from . import RawTensor

        action = self.next_action()
        if not isinstance(action, OpAction):
            raise Bailout
        if len(args) != len(action.inputs):
            raise Bailout
        assert len(actions.inputs) == len(action.input_receivers)

        for v, t, r in zip(action.inputs, args, action.input_receivers):
            if v in self.var_to_tensor:
                assert r is None
                if t is not self.var_to_tensor[v]():
                    raise Bailout
            else:
                # NOTE: not checking for aliasing (>=2 vars map to 1 tensor)
                #       the static execution backend must handle this
                self.var_to_tensor[v] = weakref.ref(t)
                r(t)

        outputs = []
        for v in action.outputs:
            assert v not in self.var_to_tensor
            t = RawTensor()
            t._data_getter = functools.partial(self.get_data, v)
            outputs.append(t)
            self.var_to_tensor[v] = weakref.ref(t)

        return tuple(outputs)

    def get_data(self, var):
        tensor = self.var_to_tensor[var]()
        assert tensor is not None
        assert tensor._data is None
        assert tensor in self.tensor_to_var
        action = self.next_action()
        if not isinstance(action, GetTensorAction):
            self.bailout()
        elif action.var != var:
            self.bailout()
        else:
            tensor._data = action.getter()
            del tensor._data_getter
            del self.tensor_to_var[tensor]
        assert "_data_getter" not in tensor.__dict__
        return tensor._data_getter()


_current_trajectory = None


def get_trajectory():
    return _current_trajectory


def compile_trace(trace):
    from .jit import ReadDTypeEvent, ReadDeviceEvent, ReadShapeEvent, OpEvent, DelEvent

    traj = ExecutionTrajectory()
    active_vars = set()

    for event in trace:
        if isinstance(event, ReadDTypeEvent):
            traj.append(ReadAttrAction())
