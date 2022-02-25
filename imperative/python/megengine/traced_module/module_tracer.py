import collections

from .. import Tensor
from .. import functional as F
from ..core.tensor.array_method import ArrayMethodMixin
from ..module import Module
from ..module.qat import QATModule
from .checker import TracedModuleChecker

_active_module_tracer = None

BUILTIN_ARRAY_METHOD = [
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__eq__",
    "__ne__",
    "__neg__",
    "__pos__",
    "__abs__",
    "__invert__",
    "__round__",
    "__floor__",
    "__ceil__",
    "__add__",
    "__sub__",
    "__mul__",
    "__matmul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__pow__",
    "__lshift__",
    "__rshift__",
    "__and__",
    "__or__",
    "__xor__",
    "__radd__",
    "__rsub__",
    "__rmul__",
    "__rmatmul__",
    "__rtruediv__",
    "__rfloordiv__",
    "__rmod__",
    "__rpow__",
    "__rlshift__",
    "__rrshift__",
    "__rand__",
    "__ror__",
    "__rxor__",
    "__iadd__",
    "__isub__",
    "__imul__",
    "__imatmul__",
    "__itruediv__",
    "__ifloordiv__",
    "__imod__",
    "__ipow__",
    "__ilshift__",
    "__irshift__",
    "__iand__",
    "__ior__",
    "__ixor__",
    "transpose",
    "astype",
    "reshape",
    "_broadcast",
    "flatten",
    "sum",
    "prod",
    "min",
    "max",
    "mean",
    "__getitem__",
    "__setitem__",
]

BUILTIN_TENSOR_WRAP_METHOD = [
    "T",
    "to",
    "size",
    "shape",
    "detach",
    "device",
    "dtype",
    "grad",
    "item",
    "ndim",
    "numpy",
    "qparams",
    "set_value",
    "reset_zero",
    "requires_grad",
    "_reset",
    "_isscalar",
    "_tuple_shape",
]


def get_tensor_wrapable_method():
    return BUILTIN_TENSOR_WRAP_METHOD + BUILTIN_ARRAY_METHOD


def active_module_tracer():
    return _active_module_tracer


def set_active_module_tracer(tracer):
    global _active_module_tracer
    _active_module_tracer = tracer


class module_tracer:

    # builtin types
    _opaque_types = set()

    _active_scopes = None

    def __init__(self, wrap_fn):
        self._active_scopes = []
        self.checker = TracedModuleChecker(self)
        self.patcher = Patcher(wrap_fn)
        self._activate_constant_cache = []

    @classmethod
    def register_as_builtin(cls, mod):
        assert issubclass(mod, Module)
        cls._opaque_types.add(mod)
        return mod

    @classmethod
    def is_builtin(cls, mod):
        return type(mod) in cls._opaque_types

    def push_scope(self, scope):
        self._active_scopes.append(scope)
        self.checker.push_scope()
        self._activate_constant_cache.append([])

    def pop_scope(self):
        self._active_scopes.pop()
        self.checker.pop_scope()
        cache = self._activate_constant_cache.pop()
        for obj in cache:
            if hasattr(obj, "_NodeMixin__node"):
                delattr(obj, "_NodeMixin__node")

    def current_scope(self):
        if self._active_scopes:
            return self._active_scopes[-1]
        return None

    def current_constant_cache(self):
        if self._activate_constant_cache:
            return self._activate_constant_cache[-1]
        return None

    def top_scope(self):
        if self._active_scopes:
            return self._active_scopes[0]
        return None


class NotExist:
    pass


class PatchedFn:
    frame_dict = None
    name = None
    origin_fn = None

    def __init__(self, frame_dict, name):
        self.frame_dict = frame_dict
        self.name = name
        self.origin_fn = (
            self.frame_dict[name]
            if isinstance(frame_dict, collections.abc.Mapping)
            else getattr(frame_dict, name, NotExist)
        )

    def set_func(self, func):
        if isinstance(self.frame_dict, collections.abc.Mapping):
            self.frame_dict[self.name] = func
        else:
            if func is not NotExist:
                setattr(self.frame_dict, self.name, func)
            else:
                delattr(self.frame_dict, self.name)


class Patcher:

    _builtin_functions = []
    _builtin_modules = [
        F,
        F.distributed,
        F.elemwise,
        F.inplace,
        F.loss,
        F.math,
        F.metric,
        F.nn,
        F.quantized,
        F.tensor,
        F.utils,
        F.vision,
    ]
    _builtin_methods = [
        Tensor,
        ArrayMethodMixin,
    ]

    def __init__(self, wrap_fn):
        self.patched_fn_ids = set()
        self.patched_fn = []
        self.visited_frames_ids = set()
        self.wrap_fn = wrap_fn
        for module in self._builtin_modules:
            self.patch_module(module)
        # some functions in F.nn are import from other module, and not in __all__
        self.auto_patch(F.nn.__dict__, False)
        for meth in BUILTIN_ARRAY_METHOD:
            self.patch_method(ArrayMethodMixin, meth, self.wrap_fn)
        self.patch_method(Tensor, "detach", self.wrap_fn)
        self.patch_method(Tensor, "__new__", self.wrap_fn)
        self.patch_method(QATModule, "_apply_fakequant_with_observer", self.wrap_fn)
        for i, j in self._builtin_functions:
            if id(i) not in self.visited_frames_ids:
                self.patch_function(i, j, self.wrap_fn)

        for m in module_tracer._opaque_types:
            self.auto_patch(getattr(getattr(m, "forward", m), "__globals__", {}))

    def patch_function(self, frame_dict, fn, wrap_fn):
        patched_fn = PatchedFn(frame_dict, fn)
        self.patched_fn_ids.add(id(patched_fn.origin_fn))
        patched_fn.set_func(wrap_fn(patched_fn.origin_fn))
        self.patched_fn.append(patched_fn)

    def patch_method(self, cls, name, wrap_fn):
        self.patch_function(cls, name, wrap_fn)

    def patch_cls(self, cls):
        import inspect

        if id(cls) not in self.visited_frames_ids:
            for k, v in cls.__dict__.items():
                if inspect.isfunction(v) and not k.startswith("_"):
                    self.patch_function(cls, k, self.wrap_fn)
            self.visited_frames_ids.add(id(cls))

    def patch_module(self, module):
        import inspect

        if id(module.__dict__) not in self.visited_frames_ids:
            keys = (
                getattr(module, "__all__")
                if hasattr(module, "__all__")
                else module.__dict__.keys()
            )
            for k in keys:
                v = getattr(module, k)
                if inspect.isfunction(v) and not k.startswith("_"):
                    self.patch_function(module.__dict__, k, self.wrap_fn)
            self.visited_frames_ids.add(id(module.__dict__))

    def auto_patch(self, frame_dict, check_frame_id=True):
        if id(frame_dict) not in self.visited_frames_ids or not check_frame_id:
            for k, v in frame_dict.items():
                if id(v) in self.patched_fn_ids:
                    self.patch_function(frame_dict, k, self.wrap_fn)
        self.visited_frames_ids.add(id(frame_dict))

    def __enter__(self):
        return self

    def __exit__(self, type, vlaue, trace):
        while self.patched_fn:
            pf = self.patched_fn.pop()
            pf.set_func(pf.origin_fn)
        self.visited_frames_ids.clear()
