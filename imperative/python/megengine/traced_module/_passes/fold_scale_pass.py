from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Set

from ... import functional as F
from ... import module as M
from ...core.ops.builtin import GetVarShape
from ...logger import get_logger
from ...tensor import Parameter, Tensor
from ..expr import (
    Expr,
    is_apply_def,
    is_call_function,
    is_call_module,
    is_call_tensor_method,
    is_constant,
    is_getattr,
)
from ..traced_module import InternalGraph
from ..utils import assign_attr, get_subattr
from .matcher import PatternMatcher
from .pass_base import BackwardPass, register_pass
from .pattern import is_const, is_op, is_var
from .utils import get_const_value

logger = get_logger(__name__)


@register_pass("BackwardFoldScale")
class BackwardFoldScale(BackwardPass):
    r"""Backward fold const scaling into weights of conv2d.
    
    For example, the following code
    
    .. code-block::
        
        x = conv(x, w, b)
        x = relu(x)
        x1 = x + 3
        x2 = x + 4
        y = (x1 + x2) * 3
    
    will be changed to
    
    .. code-block::

        x = conv(x, w * 3, b * 3)
        x = relu(x)
        x1 = x + 9
        x2 = x + 12
        y = x1 + x2

    """
    name = "BackwardFoldScale"
    required_pass = ["AttrToConstant", "NormElemWise"]
    run_once = True

    def __init__(self):
        super().__init__()
        # todo : supoort more axis
        self.scale_message = OrderedDict()
        self.used_names = defaultdict(int)

    def run_transform(self, expr: Expr) -> Expr:
        if expr not in self.scale_message:
            return expr

        var = is_var().check_users(False)
        mul_const_pattern = var * is_const() | var * "*" | is_op(F.neg)
        add_const_pattern = var + is_const() | var + "*"
        conv_pattern = is_op(F.conv2d) | is_op(M.Conv2d)
        pattern = conv_pattern | add_const_pattern | mul_const_pattern
        macther = PatternMatcher()

        if not macther.match(pattern, expr):
            return expr
        macther_exprs = macther.matched_exprs

        if conv_pattern in macther_exprs:
            return self.fold_conv_mul(expr)

        if mul_const_pattern in macther_exprs:
            return self.fold_mul(expr)

        if add_const_pattern in macther_exprs:
            return self.fold_add_mul(expr)

        return expr

    def fold_add_mul(self, expr: Expr):
        if self.scale_message[expr] is None:
            return expr
        scale = self.scale_message[expr]
        if len(expr.inputs) == 1:
            const = expr.const_val[0][-1]
        else:
            const = get_const_value(expr.inputs[1])

        const = const * scale
        inp_node = expr.inputs[0]
        graph = expr.top_graph
        with graph.insert_exprs():
            add_node = inp_node + const

        graph.replace_node({expr.outputs[0]: add_node})
        graph.compile()
        add_node.name = expr.outputs[0].name
        return add_node.expr

    def fold_mul(self, expr: Expr):
        if self.scale_message[expr] is None:
            return expr
        graph = expr.top_graph
        graph.replace_node({expr.outputs[0]: expr.inputs[0]})
        graph.compile()
        return expr

    def fold_conv_mul(self, expr: Expr):
        graph = expr.top_graph
        scale = self.scale_message[expr]

        if scale is None:
            return expr

        if is_call_function(expr, F.conv2d):
            named_args = expr.named_args
            weight = get_const_value(named_args["weight"], named_args["weight"]) * scale
            bias = get_const_value(named_args["bias"], named_args["bias"]) * scale
            named_args["weight"] = weight
            named_args["bias"] = bias
            with graph.insert_exprs():
                out_node = F.conv2d(**named_args)
            graph.replace_node({expr.outputs[0]: out_node})
            graph.compile()
            out_node.name = expr.outputs[0].name
            return out_node.expr
        else:
            mnode = expr.inputs[0]
            attr_name = expr.inputs[0].expr.name
            graph = expr.top_graph
            if len(mnode.users) > 1:
                self.used_names[mnode.qualname] += 1
                attr_name = "{}_{}".format(attr_name, self.used_names[mnode.qualname])
                logger.warning(
                    "{} is used {} times and its name will be reset to {}.{}".format(
                        mnode.qualname, len(mnode.users), graph.qualname, attr_name
                    )
                )
            conv_module = mnode.owner
            if len(mnode.users) > 1:
                conv_module = deepcopy(conv_module)
                conv_module._name = None
            conv_module.weight = Parameter(conv_module.weight * scale)
            if conv_module.bias is not None:
                conv_module.bias = Parameter(conv_module.bias * scale)

            if len(mnode.users) > 1:
                self_node = mnode.expr.inputs[0]
                assign_attr(conv_module, self_node.owner, attr_name)
                with graph.insert_exprs(mnode.expr):
                    new_conv_node = get_subattr(self_node, attr_name)
                expr.replace_inputs({mnode: new_conv_node})
            return expr

    def reset_expr_message_to_none(
        self, expr: Expr, scale_message: Dict[Expr, Any], skip_exprs: Set[Expr],
    ):
        if expr in skip_exprs:
            return
        scale_message[expr] = None
        if is_call_function(expr, F.conv2d) or is_call_module(expr, M.Conv2d):
            return
        for out_node in expr.outputs:
            for user in out_node.users:
                if user in scale_message:
                    self.reset_expr_message_to_none(user, scale_message, skip_exprs)

    def before_visit_graph(self, graph: InternalGraph):
        var = is_var().check_users(False)
        mul_const_pattern = var * is_const() | var * "*" | is_op(F.neg)
        relu_pattern = (
            is_op(F.relu) | is_op(M.ReLU) | is_op(F.leaky_relu) | is_op(M.LeakyReLU)
        )

        # The param of conv must be const, not support dynamic conv
        conv_pattern = (
            is_op(F.conv2d)(var, is_const(), is_const())
            | is_op(F.conv2d)(var, is_const())
            | is_op(M.Conv2d)
        )

        pattern = mul_const_pattern | relu_pattern | conv_pattern
        for op in [
            "__add__",
            F.reshape,
            "reshape",
            F.transpose,
            "tranpose",
            F.min,
            "min",
            F.max,
            "max",
            F.max_pool2d,
            M.MaxPool2d,
            F.avg_pool2d,
            M.AvgPool2d,
            F.adaptive_avg_pool2d,
            M.AdaptiveAvgPool2d,
            F.adaptive_max_pool2d,
            M.AdaptiveMaxPool2d,
            F.expand_dims,
            F.concat,
            "__getitem__",
        ]:
            pattern |= is_op(op)

        matcher = PatternMatcher()

        scale_message = OrderedDict()
        mem_conv_scale_message = OrderedDict()
        skip_exprs = self.init_skip_exprs(graph)
        for expr in reversed(graph._exprs):
            if expr in skip_exprs:
                continue

            if len(expr.outputs) > 1 or not matcher.match(pattern, expr):
                self.reset_expr_message_to_none(expr, scale_message, skip_exprs)
                if is_call_function(expr, F.conv2d):
                    for user in expr.outputs[0].users:
                        self.reset_expr_message_to_none(user, scale_message, skip_exprs)
                continue

            matched_exprs = matcher.matched_exprs

            const = None
            if mul_const_pattern in matched_exprs:
                if is_call_function(expr, F.neg):
                    const = -1
                elif len(expr.inputs) == 1:
                    const = expr.const_val[0][-1]
                else:
                    const = get_const_value(expr.inputs[1])

            if isinstance(const, Tensor) and const._tuple_shape not in [(1,), tuple()]:
                self.reset_expr_message_to_none(expr, scale_message, skip_exprs)
                continue

            users_const = [
                scale_message[e] for e in expr.outputs[0].users if e not in skip_exprs
            ]

            if len(users_const) == 0:
                scale_message[expr] = const
                continue

            if any(c is None or c != users_const[0] for c in users_const):
                self.reset_expr_message_to_none(expr, scale_message, skip_exprs)
                scale_message[expr] = const
                continue

            const = 1 if const is None else const
            const = const * users_const[0]
            if relu_pattern in matched_exprs and const < 0:
                self.reset_expr_message_to_none(expr, scale_message, skip_exprs)
                continue

            if conv_pattern in matched_exprs:
                self.reset_expr_message_to_none(expr, scale_message, skip_exprs)
                mem_conv_scale_message[expr] = const
                continue

            scale_message[expr] = const

        self.scale_message.update(scale_message)
        self.scale_message.update(mem_conv_scale_message)

    def init_skip_exprs(self, graph: InternalGraph):
        skip_exprs = set()
        for expr in graph._exprs:
            if is_apply_def(expr, GetVarShape):
                skip_exprs.add(expr)
            elif is_call_tensor_method(expr, "__getitem__") and expr in skip_exprs:
                skip_exprs.add(expr)
            elif is_getattr(expr):
                skip_exprs.add(expr)
            elif is_constant(expr):
                skip_exprs.add(expr)
            elif all(n.expr in skip_exprs for n in expr.inputs):
                skip_exprs.add(expr)
        return skip_exprs
