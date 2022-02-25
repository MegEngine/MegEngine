from . import compat
from ._passes import optimize
from .pytree import register_supported_type
from .tm_config import disable_default_checker, enable_expr_checker
from .traced_module import (
    TracedModule,
    _register_all_builtin_module,
    register_as_builtin,
    trace_module,
    wrap,
)

_register_all_builtin_module()

__all__ = [
    "register_as_builtin",
    "register_supported_type",
    "trace_module",
    "wrap",
    "TracedModule",
    "optimize",
    "enable_expr_checker",
    "disable_default_checker",
]
