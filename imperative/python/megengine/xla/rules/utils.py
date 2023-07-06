import warnings

import numpy as np

from ..lib.mlir import ir

lower_rule = {}


def register_lower_rule(*ops):
    def decorator(rule):
        for op in ops:
            assert op not in lower_rule, f"{op} lower rule has been registered"
            lower_rule[op] = rule

        def wrapper(*args, **kwargs):
            return rule(*args, **kwargs)

        return wrapper

    return decorator


def get_rule(op, use_fake_rule_for_debug=False):
    op_key = op if isinstance(op, str) else type(op)
    if use_fake_rule_for_debug:
        if op_key in lower_rule:
            return lower_rule[op_key]
        else:
            warnings.warn(f"op: {op_key} not register, use fake op rule")
            return lower_rule["fake_op_rule_for_debug"]
    else:
        return lower_rule[op_key]


def _log_mge_opr_attrs(mopr):
    print(f"============ {mopr} ============")
    for k in dir(mopr):
        if not k.startswith("__"):
            attr = getattr(mopr, k)
            if not isinstance(attr, type):
                print(f"    {k}: {type(attr)} = {attr}")


def _shape_equal(lhs_shape, rhs_shape):
    lhs_shape = lhs_shape.tolist() if isinstance(lhs_shape, np.ndarray) else lhs_shape
    rhs_shape = rhs_shape.tolist() if isinstance(rhs_shape, np.ndarray) else rhs_shape
    assert isinstance(lhs_shape, (tuple, list)) and isinstance(
        rhs_shape, (tuple, list)
    ), f"lhs_shape: {lhs_shape}{type(lhs_shape)}, rhs_shape: {rhs_shape}{type(rhs_shape)}"
    if len(lhs_shape) == 0 and len(rhs_shape) == 0:
        return True
    if len(lhs_shape) != 0:
        assert isinstance(lhs_shape[0], int)
    if len(rhs_shape) != 0:
        assert isinstance(rhs_shape[0], int)

    if len(lhs_shape) != len(rhs_shape):
        return False

    for l, r in zip(lhs_shape, rhs_shape):
        if l != r:
            return False

    return True


def _check_shape(actual, ref):
    if ref is not None:
        assert _shape_equal(actual, ref), f"shape error, actual: {actual}, ref: {ref}"


def _check_dtype(actual, ref):
    if ref is not None:
        assert actual == ref, f"dtype error, actual: {actual}, ref: {ref}"


def unwrap_opresult_list(irnode):
    if isinstance(irnode, ir.OpResultList):
        if len(irnode) == 1:
            return irnode[0]
    return irnode


def _parse_var_as_value(var):
    assert isinstance(
        var.bound_data, (np.ndarray, int)
    ), "cannot parse a non-const var as value"
    if tuple(var.bound_data.shape) == (1,):
        return int(var.bound_data)
    else:
        return var.bound_data


def _can_broadcast_to(src, dst, broadcast_dims=None):
    if len(src) > len(dst):
        return False
    if broadcast_dims is None:
        for i in range(-1, -len(src) - 1, -1):
            if src[i] != dst[i] and src[i] != 1:
                return False
    else:
        for idx, dim in enumerate(broadcast_dims):
            if not (src[idx] == dst[dim] or src[idx] == 1):
                return False
    return True
