import copy
from abc import abstractmethod
from collections import OrderedDict, namedtuple
from functools import partial
from re import T
from typing import Any, Callable, Dict, Iterable, List, Union

from ...logger import get_logger
from ..expr import Expr
from ..traced_module import InternalGraph, TracedModule
from .utils import register_obj

logger = get_logger(__name__)


class PassContext:
    def __init__(
        self, disabled_pass: Iterable[str] = None, pass_config: Dict[str, Any] = None
    ):
        self._disabled_pass = set()
        self._config = pass_config
        self._handle = None
        if disabled_pass:
            self.add_diabled_pass(disabled_pass)

    def add_diabled_pass(self, passes: Iterable[str]):
        if isinstance(passes, str):
            passes = [passes]
        for pas in passes:
            self._disabled_pass.add(pas)

    def pass_enabled(self, pas: Union["BasePass", str]):
        pass_name = pas.name if isinstance(pas, BasePass) else pas
        return pass_name not in self._disabled_pass


_default_context = PassContext()


def get_default_pass_context():
    return _default_context


_pass_dict = OrderedDict()
register_pass = partial(register_obj, _dict=_pass_dict)


def get_registered_pass(pass_name: str):
    pas = _pass_dict.get(pass_name, None)
    assert (
        pas is not None
    ), "{} is not found, please call `register_pass` to register it".format(pass_name)
    return pas


class BasePass:
    run_once = True  # bool
    required_pass = []  # Iterable[str]
    name = ""  # str

    def __init__(self):
        super().__init__()

    def __call__(
        self, mod: TracedModule, pass_ctx: PassContext = get_default_pass_context()
    ) -> TracedModule:
        assert isinstance(pass_ctx, PassContext)
        return self.apply_optimization(mod, pass_ctx)

    def apply_optimization(
        self, mod: TracedModule, pass_ctx: PassContext
    ) -> TracedModule:
        new_mod = mod
        for pass_name in self.required_pass + [self.name]:
            if not pass_ctx.pass_enabled(pass_name):
                logger.warning(
                    "Since {} is disabled, {} will skipped".format(pass_name, self.name)
                )
                return mod

        for pass_name in self.required_pass:
            pass_func = get_registered_pass(pass_name)()
            new_mod = pass_func(new_mod, pass_ctx)

        iter_num = 1
        graph_changed = self.visit_graph(new_mod.graph)
        while not self.run_once and graph_changed:
            graph_changed = self.visit_graph(new_mod.graph)
            iter_num += 1
            if iter_num == 100:
                break
        assert iter_num < 100, "{} was run 100 times, plase check for pass conflict."

        return new_mod

    @abstractmethod
    def visit_graph(self, graph: InternalGraph):
        raise NotImplementedError

    def before_visit_graph(self, graph: InternalGraph):
        pass

    def run_transform(self, expr: Expr) -> Expr:
        return expr

    def __repr__(self) -> str:
        return self.name


class ForwardPass(BasePass):
    def visit_graph(self, graph: InternalGraph):
        class Item:
            def __init__(self, expr: Expr, child_expanded: bool = False):
                self.expr = expr
                self.child_expanded = child_expanded

        self.before_visit_graph(graph)
        graph_changed = False
        queue = [Item(n.expr) for n in graph.outputs]
        visited_expr, visited_graph = set(), set()
        while queue:
            item = queue[-1]
            if item.expr in visited_expr:
                queue.pop()
            elif item.child_expanded:
                if item.expr not in graph._exprs:
                    queue.pop()
                    continue
                new_expr = self.run_transform(item.expr)
                if new_expr is not item.expr:
                    graph_changed = True
                    assert new_expr not in visited_expr
                    queue.append(Item(new_expr))
                    continue
                if (
                    hasattr(item.expr, "graph")
                    and item.expr.graph is not None
                    and item.expr.graph not in visited_graph
                ):
                    graph_changed |= self.visit_graph(item.expr.graph)
                    visited_graph.add(item.expr.graph)
                visited_expr.add(item.expr)
            else:
                item.child_expanded = True
                for i in item.expr.inputs:
                    expr = i.expr
                    if expr not in queue and expr not in visited_expr:
                        queue.append(Item(expr))
        return graph_changed


class BackwardPass(BasePass):
    def visit_graph(self, graph: InternalGraph):
        self.before_visit_graph(graph)
        graph_changed = False
        queue = [n.expr for n in graph.outputs]
        visited_expr, visited_graph = set(), set()
        while queue:
            expr = queue.pop()
            if expr not in graph._exprs:
                continue
            new_expr = self.run_transform(expr)
            if new_expr is not expr:
                graph_changed = True
                queue.append(new_expr)
                continue
            else:
                visited_expr.add(expr)

            if (
                hasattr(expr, "graph")
                and expr.graph is not None
                and expr.graph not in visited_graph
            ):
                graph_changed |= self.visit_graph(expr.graph)
                visited_graph.add(expr.graph)

            for i in expr.inputs:
                expr = i.expr
                if expr not in queue and expr not in visited_expr:
                    queue.append(expr)
        return graph_changed
