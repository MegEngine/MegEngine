import contextlib

from ..core._imperative_rt.core2 import (
    is_tracing_module,
    set_module_tracing,
    unset_module_tracing,
)

_enable_expr_checker = False
_enable_default_checker = True


def _get_expr_checker():
    return _enable_expr_checker


def _get_default_checker():
    return _enable_default_checker


def enable_expr_checker():
    r"""Call this function to check the result of each expr during tracing."""
    global _enable_expr_checker
    _enable_expr_checker = True
    _enable_default_checker = False


def disable_default_checker():
    r"""Call this function to disable checking the final output of the model after tracing."""
    global _enable_default_checker
    _enable_default_checker = False


_enable_graph_surgery_mode = False


def _graph_surgery_mode():
    return _enable_graph_surgery_mode


def _set_graph_surgery_mode(mode: bool):
    global _enable_graph_surgery_mode
    pre_mode = _enable_graph_surgery_mode
    _enable_graph_surgery_mode = mode
    return pre_mode


@contextlib.contextmanager
def _exclude_from_trace():
    is_tracing = is_tracing_module()
    if is_tracing:
        unset_module_tracing()
    yield
    if is_tracing:
        set_module_tracing()
