from importlib import import_module
from typing import Dict, Tuple

from ..core._imperative_rt import OpDef
from ..core.ops import builtin
from ..tensor import Tensor
from ..version import __version__
from .utils import _convert_kwargs_to_args

OPDEF_LOADER = {}
FUNCTIONAL_LOADER = {}
TENSORMETHOD_LOADER = {}
MODULE_LOADER = {}


class _ModuleState:
    obj = None

    def __init__(self, module: Tuple, state: Dict, version: str):
        self.module = module
        self.state = state
        self.version = version

    @classmethod
    def get_module_state(cls, module):
        typem = (type(module).__module__, type(module).__qualname__)
        state = module.__dict__.copy()
        state.pop("_m_dump_modulestate", None)
        if hasattr(module, "_m_dump_modulestate"):
            assert isinstance(module._m_dump_modulestate, cls)
            module._m_dump_modulestate.__init__(typem, state, __version__)
        else:
            module.__dict__["_m_dump_modulestate"] = _ModuleState(
                typem, state, __version__
            )

        return module._m_dump_modulestate

    def __getstate__(self):
        return {"module": self.module, "state": self.state, "version": self.version}

    def to_module(self):
        if self.obj is None:
            typem = getattr(import_module(self.module[0]), self.module[1])
            m_obj = typem.__new__(typem)
            m_obj.__setstate__(self.state)
            self.obj = m_obj
        return self.obj


def register_opdef_loader(*opdefs):
    def callback(loader):
        for opdef in opdefs:
            assert opdef not in OPDEF_LOADER
            OPDEF_LOADER[opdef] = loader
        return loader

    return callback


def register_functional_loader(*funcs):
    def callback(loader):
        for func in funcs:
            assert func not in FUNCTIONAL_LOADER
            FUNCTIONAL_LOADER[func] = loader
        return loader

    return callback


def register_module_loader(*module_types):
    def callback(loader):
        for module_type in module_types:
            assert module_type not in MODULE_LOADER
            MODULE_LOADER[module_type] = loader
        return loader

    return callback


def register_tensor_method_loader(*methods):
    def callback(loader):
        for method in methods:
            assert method not in TENSORMETHOD_LOADER
            TENSORMETHOD_LOADER[method] = loader
        return loader

    return callback


def _replace_args_kwargs(expr, new_args, new_kwargs):
    if len(new_args) != len(expr.args) or set(new_kwargs.keys()) != set(
        expr.kwargs.keys()
    ):
        expr.set_args_kwargs(*new_args, **new_kwargs)


def load_functional(expr):
    func = (
        (expr.func.__module__, expr.func.__qualname__)
        if callable(expr.func)
        else expr.func
    )
    assert isinstance(func, tuple)
    if func in FUNCTIONAL_LOADER:
        loader = FUNCTIONAL_LOADER[func]
        loader(expr)
        mname, fname = func
        f = import_module(mname)
        for i in fname.split("."):
            f = getattr(f, i)
        expr.func = f
    assert callable(expr.func)
    if not hasattr(expr, "version") or expr.version != __version__:
        args, kwargs = _convert_kwargs_to_args(expr.func, expr.args, expr.kwargs)
        _replace_args_kwargs(expr, args, kwargs)


def load_call_module_expr(expr):
    m_type = expr.inputs[0].module_type
    if isinstance(m_type, type):
        m_type = (m_type.__module__, m_type.__qualname__)
    if m_type in MODULE_LOADER:
        MODULE_LOADER[m_type](expr)
    if isinstance(expr.inputs[0].module_type, tuple):
        mname, classname = expr.inputs[0].module_type
        expr.inputs[0].module_type = getattr(import_module(mname), classname)
    if not hasattr(expr, "version") or expr.version != __version__:
        fwd_func = getattr(expr.inputs[0].module_type, "forward")
        args, kwargs = _convert_kwargs_to_args(fwd_func, expr.args, expr.kwargs)
        _replace_args_kwargs(expr, args, kwargs)


def load_call_tensor_method_expr(expr):
    if expr.method in TENSORMETHOD_LOADER:
        loader = TENSORMETHOD_LOADER[expr.method]
        loader(expr)
    if not hasattr(expr, "version") or expr.version != __version__:
        tmethod = (
            getattr(expr.args[0], expr.method)
            if isinstance(expr.args[0], type)
            else getattr(Tensor, expr.method)
        )
        args, kwargs = _convert_kwargs_to_args(tmethod, expr.args, expr.kwargs)
        _replace_args_kwargs(expr, args, kwargs)


def load_apply_expr(expr):
    opdef_type = type(expr.opdef)
    if opdef_type in OPDEF_LOADER:
        OPDEF_LOADER[opdef_type](expr)
        opdef_state = expr.opdef_state
        opdef_obj = opdef_state.pop("opdef_type")()
        opdef_obj.__setstate__(opdef_state)
        expr.opdef = opdef_obj
