import operator
from collections import defaultdict
from typing import Any, Callable, List

from ... import functional as F
from ... import module as M
from ...logger import get_logger
from ...tensor import Parameter, Tensor
from ...utils.bn_fusion import fold_weight_bias
from ..expr import Expr, is_call_function
from ..utils import assign_attr, get_subattr
from .matcher import PatternMatcher
from .pass_base import BackwardPass, register_pass
from .pattern import ExprPattern, any_node, is_const, is_op, is_var
from .utils import get_const_value, register_obj

logger = get_logger(__name__)


@register_pass("FuseAddMul")
class FuseAddMul(BackwardPass):
    """Fold adjacent const add or mul binary operations.
    
    For example, the following code

    .. code-block::

        x = x + 1
        x = 2 + x
        x = x * 4
        x = x * 0.25

    will be changed to

    .. code-block::

        x = x + 3
    """

    name = "FuseAddMul"
    required_pass = ["NormElemWise"]
    run_once = False

    def __init__(self,):
        super().__init__()

        def _make_pattern(op_0, op_1) -> ExprPattern:
            x = is_var().check_users(False)
            if op_0 not in [operator.add, operator.mul]:
                op_0 = is_op(op_0)
            if op_1 not in [operator.add, operator.mul]:
                op_1 = is_op(op_1)
            pattern = op_0(x, is_const()) | op_0(x, "*")
            pattern = op_1(pattern, is_const()) | op_1(pattern, "*")
            return pattern

        self.pattern_dict = {}

        for op, func in zip([operator.add, F.pow], [self.fold_add, self.fold_pow],):
            self.pattern_dict[_make_pattern(op, op)] = func

        for op_0 in [F.neg, operator.mul]:
            for op_1 in [F.neg, operator.mul]:
                self.pattern_dict[_make_pattern(op_0, op_1)] = self.fold_mul

    def run_transform(self, expr: Expr):
        matcher = PatternMatcher()
        for pattern, func in self.pattern_dict.items():
            res = matcher.match(pattern, expr)
            if res:
                break
        if not res:
            return expr
        return func(expr)

    def _fold_helper(self, expr: Expr, op_c: Callable, op_t: Callable):
        const_0 = self.get_const_value(expr)
        # todo: support more shape
        if isinstance(const_0, Tensor) and const_0._tuple_shape not in [(1,), tuple()]:
            return expr

        const_1 = self.get_const_value(expr.inputs[0].expr)
        if isinstance(const_1, Tensor) and const_1._tuple_shape not in [(1,), tuple()]:
            return expr

        inp_node = expr.inputs[0].expr.inputs[0]
        const = op_c(const_0, const_1)
        graph = expr.top_graph

        if (const == 1 and op_t in [operator.pow, operator.mul]) or (
            const == 0 and op_t in [operator.add]
        ):
            graph.replace_node({expr.outputs[0]: inp_node})
            graph.compile()
            return expr

        with expr.top_graph.insert_exprs():
            out_node = op_t(inp_node, const)
        graph.replace_node({expr.outputs[0]: out_node})
        graph.compile()
        return out_node.expr

    def fold_add(self, expr: Expr):
        return self._fold_helper(expr, operator.add, operator.add)

    def fold_mul(self, expr):
        return self._fold_helper(expr, operator.mul, operator.mul)

    def fold_pow(self, expr):
        return self._fold_helper(expr, operator.mul, F.pow)

    def get_const_value(self, expr: Expr):
        if is_call_function(expr, F.neg):
            return -1
        if len(expr.inputs) == 2:
            value = get_const_value(expr.inputs[1].expr, None)
            assert value is not None, " "
            return value
        value = expr.const_val[0][-1]
        return value


@register_pass("FuseConvBn")
class FuseConvBn(BackwardPass):
    r"""Fuse BN layers into conv2d."""
    name = "FuseConvBn"
    required_pass = ["AttrToConstant"]
    run_once = True

    def __init__(self):
        super().__init__()
        self.used_name = defaultdict(int)

    def run_transform(self, expr: Expr):
        conv_pat_0 = is_op(M.Conv2d)
        conv_pat_1 = is_op(F.conv2d)
        bn_pat_0 = is_op(M.BatchNorm2d)(conv_pat_0 | conv_pat_1)
        bn_pat_1 = is_op(F.batch_norm)
        # inp, running_mean, running_var, weight, bias
        bn_inps = (
            conv_pat_0 | conv_pat_1,
            is_const(),
            is_const(),
            is_const(),
            is_const(),
        )
        bn_pat = (
            (bn_pat_1(*bn_inps[:3]))
            | (bn_pat_1(*bn_inps[:4]))
            | (bn_pat_1(*bn_inps))
            | bn_pat_0
        )

        matcher = PatternMatcher()
        if not matcher.match(bn_pat, expr):
            return expr

        matched_exprs = matcher.matched_exprs
        if conv_pat_0 in matched_exprs:
            return self.fold_convm_bn(matched_exprs[conv_pat_0], matched_exprs[bn_pat])
        else:
            return self.fold_convf_bn(matched_exprs[conv_pat_1], matched_exprs[bn_pat])

    def fold_convm_bn(self, conv: Expr, bn: Expr):
        mnode, inp_node = conv.inputs[:2]
        self_node = mnode.expr.inputs[0]
        attr_name = conv.inputs[0].expr.name
        graph = conv.top_graph
        if len(mnode.users) > 1:
            self.used_name[mnode.qualname] += 1
            attr_name = "{}_{}".format(attr_name, self.used_name[mnode.qualname])
            logger.warning(
                "{} is used {} times and its name will be reset to {}.{}".format(
                    mnode.qualname, len(mnode.users), graph.qualname, attr_name
                )
            )

        conv_module = mnode.owner
        weight, bias = conv_module.weight, conv_module.bias
        mean, var, gamma, beta, eps = self.get_bn_params(bn)
        weight, bias = fold_weight_bias(weight, bias, gamma, beta, mean, var, eps)
        new_conv = M.Conv2d(
            in_channels=conv_module.in_channels,
            out_channels=conv_module.out_channels,
            kernel_size=conv_module.kernel_size,
            stride=conv_module.stride,
            padding=conv_module.padding,
            dilation=conv_module.dilation,
            groups=conv_module.groups,
            bias=conv_module.bias is not None,
            conv_mode=conv_module.conv_mode,
            compute_mode=conv_module.compute_mode,
            name=conv_module.name,
        )
        new_conv.weight = Parameter(weight)
        new_conv.bias = Parameter(bias)
        new_conv.training = conv_module.training
        assign_attr(new_conv, self_node.owner, attr_name)
        with graph.insert_exprs(mnode.expr):
            out_node = get_subattr(self_node, attr_name)(inp_node)
        graph.replace_node({bn.outputs[0]: out_node})
        graph.compile()
        out_node.name = conv.outputs[0].name
        return out_node.expr

    def fold_convf_bn(self, conv: Expr, bn: Expr):
        named_args = conv.named_args
        weight = get_const_value(named_args["weight"], named_args["weight"])
        bias = get_const_value(named_args["bias"], named_args["bias"])
        mean, var, gamma, beta, eps = self.get_bn_params(bn)
        weight, bias = fold_weight_bias(weight, bias, gamma, beta, mean, var, eps)
        named_args["weight"] = weight
        named_args["bias"] = bias
        graph = conv.top_graph
        with graph.insert_exprs():
            out_node = F.conv2d(**named_args)
        graph.replace_node({bn.outputs[0]: out_node})
        graph.compile()
        out_node.name = conv.outputs[0].name
        return out_node.expr

    def get_bn_params(self, bn: Expr):
        if is_call_function(bn):
            named_args = bn.named_args
            mean = get_const_value(
                named_args["running_mean"], named_args["running_mean"]
            )
            var = get_const_value(named_args["running_var"], named_args["running_var"])
            gamma = get_const_value(named_args["weight"], named_args["weight"])
            beta = get_const_value(named_args["bias"], named_args["bias"])
            eps = named_args["eps"]
            return mean, var, gamma, beta, eps
        else:
            bn_module = bn.inputs[0].owner
            mean = bn_module.running_mean
            var = bn_module.running_var
            gamma = bn_module.weight
            beta = bn_module.bias
            eps = bn_module.eps
        return mean, var, gamma, beta, eps
