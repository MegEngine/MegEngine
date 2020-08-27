# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
import io
import weakref


class partial(functools.partial):
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return functools.partial(self, instance)


def hook(f):
    def decorator(impl):
        return functools.update_wrapper(partial(f, impl), impl)

    return decorator


def on_input(impl, value):
    tensor = impl(value)
    trace = get_trace()
    if trace:
        var = trace.get_var(tensor)
        event = InputEvent(var)
        trace.append(event)
    return tensor


def on_read_dtype(impl, self):
    trace = get_trace()
    if trace:
        var = trace.get_var(self)
        event = ReadDtypeEvent(var)
        trace.append(event)

    return impl(self)


def on_read_device(impl, self):
    trace = get_trace()
    if trace:
        var = trace.get_var(self)
        event = ReadDeviceEvent(var)
        trace.append(event)

    return impl(self)


def on_read_shape(impl, self):
    trace = get_trace()
    if trace:
        var = trace.get_var(self)
        event = ReadShapeEvent(var)
        trace.append(event)

    return impl(self)


def on_read_value(impl, self):
    trace = get_trace()
    if trace:
        var = trace.get_var(self)
        event = ReadValueEvent(var)
        trace.append(event)

    return impl(self)


def on_builtin_op(impl, op, *args):
    outputs = impl(op, *args)

    trace = get_trace()
    if trace:
        input_vars = tuple(map(trace.get_var, args))
        output_vars = outputs and tuple(map(trace.get_var, outputs))
        event = OpEvent(op, input_vars, output_vars)
        trace.append(event)

    return outputs


def on_del(impl, self):
    trace = get_trace()
    if trace:
        var = trace.get_var(self)
        event = DelEvent(var)
        trace.append(event)

    return impl(self)


class Trace(list):
    def __init__(self):
        self._var_id = 1
        self._t2v = weakref.WeakKeyDictionary()
        self._v2t = weakref.WeakValueDictionary()

    def get_var(self, x):
        v = self._t2v.get(x)
        if v:
            return v
        v = self._var_id
        self._var_id += 1
        self._t2v[x] = v
        self._v2t[v] = x
        return v

    def __bool__(self):
        return True

    def __enter__(self):
        global _current_trace
        if hasattr(self, "_prev_trace"):
            raise RuntimeError
        self._prev_trace = _current_trace
        _current_trace = self
        return self

    def __exit__(self, *_):
        global _current_trace
        if _current_trace is not self:
            raise RuntimeError
        _current_trace = self._prev_trace
        del self._prev_trace


class Event:
    pass


class InputEvent(Event):
    def __init__(self, var):
        self.var = var


class ReadEvent(Event):
    def __init__(self, var):
        self.var = var


class ReadDtypeEvent(ReadEvent):
    pass


class ReadDeviceEvent(ReadEvent):
    pass


class ReadShapeEvent(ReadEvent):
    pass


class ReadValueEvent(ReadEvent):
    pass


class OpEvent(Event):
    def __init__(self, op, inputs, outputs):
        self.op = op
        self.inputs = inputs
        self.outputs = outputs


class DelEvent(Event):
    def __init__(self, var):
        self.var = var


_current_trace = None


def get_trace() -> Trace:
    global _current_trace
    return _current_trace


def format_trace(trace):
    buf = io.StringIO()
    active_vars = set()

    def write(fmt, *args, **kwargs):
        print(fmt.format(*args, **kwargs), file=buf)

    def init_vars(*args):
        for i in args:
            if i in active_vars:
                continue
            active_vars.add(i)
            write("_{} = input()", i)

    for event in trace:
        if isinstance(event, InputEvent):
            init_vars(event.var)
        elif isinstance(event, ReadDtypeEvent):
            init_vars(event.var)
            write("output(_{}.dtype)", event.var)
        elif isinstance(event, ReadDeviceEvent):
            init_vars(event.var)
            write("output(_{}.device)", event.var)
        elif isinstance(event, ReadShapeEvent):
            init_vars(event.var)
            write("output(_{}.shape)", event.var)
        elif isinstance(event, ReadValueEvent):
            init_vars(event.var)
            write("output(_{}.dtype)", event.var)
        elif isinstance(event, ReadValueEvent):
            init_vars(event.var)
            write("output(_{}.value)", event.var)
        elif isinstance(event, OpEvent):
            init_vars(*event.inputs)
            active_vars.update(event.outputs)
            ovars = ", ".join(map("_{}".format, event.outputs))
            ivars = ", ".join(map("_{}".format, event.inputs))
            if ovars:
                write("{} = {}({})", ovars, repr(event.op), ivars)
            else:
                write("{}({})", repr(event.op), ivars)
        elif isinstance(event, DelEvent):
            init_vars(event.var)
            write("del _{}", event.var)
        else:
            raise TypeError(type(event))

    return buf.getvalue()


def compile_trace(trace):
    trace = list(trace)


def static_function(f):
    trace = None

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        nonlocal trace
        if trace is None:
            with Trace() as trace:
                return f(*args, **kwargs)
        return f(*args, **kwargs)

    return wrapper
