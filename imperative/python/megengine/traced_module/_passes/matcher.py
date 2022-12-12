from collections import OrderedDict, defaultdict
from functools import partial

from ...logger import get_logger
from ..expr import (
    Expr,
    is_apply_def,
    is_call_function,
    is_call_module,
    is_call_tensor_method,
    is_constant,
)
from .pattern import (
    AnyPattern,
    ApplyDefPattern,
    CallPattern,
    ConstantPattern,
    ExprPattern,
    FunctionPattern,
    ModulePattern,
    OrPattern,
    TensorMethodPattern,
    VarPattern,
)
from .utils import register_obj

logger = get_logger(__name__)


class PatternMatcher:

    method_dict = {}
    register_visiter_func = partial(register_obj, _dict=method_dict)

    def __init__(self) -> None:
        self.matched_patterns = []
        self.matched_exprs = OrderedDict()

    def match(self, pattern: ExprPattern, expr: Expr) -> bool:
        self.matched_exprs.clear()
        self.matched_patterns.clear()
        pattern.check_users(False)
        res = self.visit_pattern(pattern, expr)
        if res and not self._check_users():
            self.clear_map(0)
            res = False
        self._clear_pattern_users()
        return res

    def clear_map(self, mark):
        for _ in range(len(self.matched_patterns) - mark):
            p = self.matched_patterns.pop()
            self.matched_exprs.pop(p)
            p._clear_users()

    def _clear_pattern_users(self):
        for p in self.matched_patterns:
            p._clear_users()

    def _check_users(self) -> bool:
        for pat, expr in self.matched_exprs.items():
            if pat._check_users:
                pattern_users = pat._users
                if len(expr.outputs) != 1:
                    logger.warning(
                        "only support single output, and the matching "
                        "result may be wrong"
                    )
                    continue
                expr_users = expr.outputs[0].users
                if len(pattern_users) != len(expr_users):
                    return False
                for pat, expr in zip(pattern_users, expr_users):
                    if self.matched_exprs[pat] != expr:
                        return False
        return True

    def visit_pattern(self, pattern: ExprPattern, expr: Expr) -> bool:
        if pattern in self.matched_exprs:
            if self.matched_exprs[pattern] is expr:
                if isinstance(pattern, (OrPattern)):
                    assert self._visit_or_pattern(pattern, expr) == True
                return True
            else:
                return False
        else:
            mark = len(self.matched_patterns)
            visiter = self.method_dict.get(type(pattern))
            matched = visiter(self, pattern, expr)
            if matched:
                self.matched_patterns.append(pattern)
                self.matched_exprs[pattern] = expr
            else:
                self.clear_map(mark)
            return matched

    @register_visiter_func(OrPattern)
    def _visit_or_pattern(self, pattern: OrPattern, expr: Expr) -> bool:
        if self.visit_pattern(pattern.left, expr):
            if pattern._users:
                pattern.left._add_users(pattern._users[-1])
            return True
        if self.visit_pattern(pattern.right, expr):
            if pattern._users:
                pattern.right._add_users(pattern._users[-1])
            return True
        return False

    @register_visiter_func(CallPattern)
    def _visit_call_pattern(self, pattern: CallPattern, expr: Expr) -> bool:
        mark = len(self.matched_patterns)
        match_res = self.visit_pattern(pattern.op, expr)
        if not match_res:
            self.clear_map(mark)
            return False
        inputs = expr.inputs
        if isinstance(pattern.op, ModulePattern):
            inputs = inputs[1:]
        if (pattern._match_all_args and len(pattern.args) != len(inputs)) or (
            not pattern._match_all_args and len(pattern.args) > len(inputs)
        ):
            self.clear_map(mark)
            return False
        for i, pat in enumerate(pattern.args):
            pat._add_users(pattern)
            match_res = self.visit_pattern(pat, inputs[i].expr)
            if not match_res:
                pat._clear_users()
                self.clear_map(mark)
                return False
        return True

    @register_visiter_func(ModulePattern)
    def _visit_module_pattern(self, pattern: ModulePattern, expr: Expr) -> bool:
        if not is_call_module(expr, pattern.target):
            return False
        module = expr.inputs[0].owner
        for key, target in pattern.attrs.items():
            value = getattr(module, key, None)
            if target != value:
                return False
        return True

    @register_visiter_func(FunctionPattern)
    def _visit_function_pattern(self, pattern: FunctionPattern, expr: Expr) -> bool:
        if not is_call_function(expr, pattern.target):
            return False
        kwargs = expr.named_args
        for key, target in pattern.params.items():
            value = kwargs.get(key, None)
            if target != value:
                return False
        return True

    @register_visiter_func(TensorMethodPattern)
    def _visit_tensor_method_pattern(
        self, pattern: TensorMethodPattern, expr: Expr
    ) -> bool:
        return is_call_tensor_method(expr, pattern.target)

    @register_visiter_func(ApplyDefPattern)
    def _visit_apply_pattern(self, pattern: ApplyDefPattern, expr: Expr) -> bool:
        return is_apply_def(expr, pattern.target)

    @register_visiter_func(ConstantPattern)
    def _visit_const_pattern(self, pattern: ConstantPattern, expr: Expr) -> bool:
        return is_constant(expr)

    @register_visiter_func(VarPattern)
    def _visit_var_pattern(self, pattern: VarPattern, expr: Expr) -> bool:
        return not is_constant(expr)

    @register_visiter_func(AnyPattern)
    def _visit_any_pattern(self, pattern: AnyPattern, expr: Expr) -> bool:
        return True
