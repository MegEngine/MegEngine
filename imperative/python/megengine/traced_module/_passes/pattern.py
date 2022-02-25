from abc import abstractmethod
from typing import Any, Callable, Dict, List

from ...core._imperative_rt import OpDef
from ...logger import get_logger
from ...module import Module
from ..expr import Expr
from ..node import Node

logger = get_logger(__name__)


class ExprPattern:
    def __init__(self):
        self._check_users = True
        self._users = []

    def __call__(self, *args):
        args = list(args)
        if len(args) == 1 and args[0] is None:
            args = None
        return CallPattern(self, *args)

    def __add__(self, other):
        return is_op("__add__")(self, other)

    def __iadd__(self, other):
        return is_op("__iadd__")(self, other)

    def __radd__(self, other):
        return is_op("__radd__")(self, other)

    def __sub__(self, other):
        return is_op("__sub__")(self, other)

    def __isub__(self, other):
        return is_op("__isub__")(self, other)

    def __rsub__(self, other):
        return is_op("__rsub__")(self, other)

    def __mul__(self, other):
        return is_op("__mul__")(self, other)

    def __imul__(self, other):
        return is_op("__imul__")(self, other)

    def __rmul__(self, other):
        return is_op("__rmul__")(self, other)

    def __truediv__(self, other):
        return is_op("__truediv__")(self, other)

    def __itruediv__(self, other):
        return is_op("__itruediv__")(self, other)

    def __rtruediv__(self, other):
        return is_op("__rtruediv__")(self, other)

    def __or__(self, other):
        assert isinstance(other, ExprPattern)
        return OrPattern(self, other)

    def get_output(self, index):
        raise NotImplementedError

    def check_users(self, check: bool = True):
        self._check_users = check
        return self

    def _add_users(self, pattern: "ExprPattern"):
        self._users.append(pattern)

    def _clear_users(self,):
        self._users.clear()

    def __getitem__(self, index):
        return is_op("__getitem__")(self, index)

    def has_attr(self, **attrs):
        logger.warning("has_param only support ModulePattern")
        return self

    def has_param(self, **params):
        logger.warning("has_param only support FunctionPattern")
        return self

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError


class CallPattern(ExprPattern):
    def __init__(self, op: ExprPattern, *args: List[ExprPattern]):
        super().__init__()
        self.op = op
        self.args = list(filter(lambda x: isinstance(x, ExprPattern), args))
        self._match_all_args = True

    def __repr__(self) -> str:
        return "{}({})".format(self.op, ",".join(str(x) for x in self.args))

    def not_all_args(self):
        self._match_all_args = False

    def check_users(self, check: bool = True):
        self._check_users = check
        self.op.check_users(check)
        return self

    def _add_users(self, pattern: "ExprPattern"):
        self._users.append(pattern)
        self.op._add_users(pattern)

    def _clear_users(self):
        self._users.clear()
        self.op._clear_users()


class OrPattern(ExprPattern):
    def __init__(self, left: ExprPattern, right: ExprPattern):
        super().__init__()
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return "({}|{})".format(self.left, self.right)

    def check_users(self, check: bool = True):
        self._check_users = check
        self.left.check_users(check)
        self.right.check_users(check)
        return self

    def _clear_users(self):
        self._users.clear()
        self.left._clear_users()
        self.right._clear_users()


class GetOutputPaterrn(ExprPattern):
    def __init__(self, op, index):
        super().__init__()
        self.op = op
        self.index = index

    def __repr__(self) -> str:
        return "{}[{}]".format(self.op, self.index)


class ModulePattern(ExprPattern):
    def __init__(self, module_cls: Module) -> None:
        super().__init__()
        self.attrs = {}
        self.target = module_cls

    def has_attr(self, **attrs):
        self.attrs.update(attrs)
        return self

    def __repr__(self) -> str:
        return "{}".format(self.target.__name__)


class FunctionPattern(ExprPattern):
    def __init__(self, func: Callable):
        super().__init__()
        self.params = {}
        self.target = func

    def has_params(self, **params):
        self.params.update(params)
        return self

    def __repr__(self) -> str:
        return "{}".format(self.target.__name__)


class TensorMethodPattern(ExprPattern):
    def __init__(self, method: str):
        super().__init__()
        self.target = method

    def __repr__(self) -> str:
        return self.target


class ApplyDefPattern(ExprPattern):
    def __init__(self, opdef: OpDef):
        super().__init__()
        self.target = opdef

    def __repr__(self) -> str:
        return "{}".format(self.target.__name__)


class VarPattern(ExprPattern):
    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return "var"


class ConstantPattern(ExprPattern):
    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return "const"


class AnyPattern(ExprPattern):
    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return "any"


def is_op(target):
    if isinstance(target, type):
        if issubclass(target, Module):
            return ModulePattern(target)
        if issubclass(target, OpDef):
            return ApplyDefPattern(target)
    elif callable(target):
        return FunctionPattern(target)
    elif isinstance(target, str):
        return TensorMethodPattern(target)
    else:
        raise ValueError("not support")


def is_const():
    return ConstantPattern().check_users(False)


def any_node():
    return AnyPattern()


def is_var():
    return VarPattern()
