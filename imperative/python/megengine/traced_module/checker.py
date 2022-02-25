import traceback
from typing import Sequence

import numpy as np

from ..core._imperative_rt.core2 import apply
from ..core._imperative_rt.ops import ROIAlign, ROIPooling
from ..core.ops.builtin import Copy
from ..tensor import Tensor
from .tm_config import _exclude_from_trace


class TracedModuleChecker:
    def __init__(self, tracer):
        self._active_node2values = []
        self.tracer = tracer

        self.node_without_tensor_info = {}

    def push_scope(self):
        self._active_node2values.append({})

    def pop_scope(self):
        self._active_node2values.pop()

    def current_node2values(self):
        return self._active_node2values[-1]

    def reset_checker(self):
        self._active_node2values = []

    def check_node_not_in_scope(self):
        if self.node_without_tensor_info:
            for node, info in self.node_without_tensor_info.items():
                for expr in info[0]._exprs:
                    if node in expr.inputs or node in expr.outputs:
                        traceback.print_list(info[1])
                        raise ValueError(
                            "node({}) not in the graph:\n{}".format(node, info[0])
                        )
            return True
        else:
            return False

    def check_net_outputs(self, tm_res, gt_res):
        if isinstance(tm_res, Tensor):
            np.testing.assert_allclose(tm_res.numpy(), gt_res.numpy())
        elif isinstance(tm_res, Sequence):
            for i, j in zip(tm_res, gt_res):
                np.testing.assert_allclose(i.numpy(), j.numpy())
        else:
            for k in tm_res.__dict__.keys():
                np.testing.assert_allclose(
                    getattr(tm_res, k).numpy(), getattr(gt_res, k).numpy()
                )

    def record_nodemixin(self, node, value):
        self.current_node2values()[node] = value

    def record_node2value(self, node, value):
        with _exclude_from_trace():
            self.current_node2values()[node] = apply(
                Copy(comp_node=value.device), value
            )[0]

    def check_apply_special_cases(self, opdef, num_outputs):
        indexs = list(range(num_outputs))
        if isinstance(opdef, ROIAlign) and opdef.mode == ROIAlign.Mode.AVERAGE:
            indexs.pop(-1)
        if isinstance(opdef, ROIPooling) and opdef.mode == ROIPooling.Mode.AVERAGE:
            indexs.pop(-1)
        return indexs

    def check_expr_results(self, expr_outputs, gt_outputs, indexs=None):
        expr_outputs = (
            (expr_outputs,) if not isinstance(expr_outputs, Sequence) else expr_outputs
        )
        gt_outputs = (
            (gt_outputs,) if not isinstance(gt_outputs, Sequence) else gt_outputs
        )
        if indexs is not None:
            for i in indexs:
                np.testing.assert_allclose(
                    expr_outputs[i].numpy(), gt_outputs[i].numpy()
                )
        else:
            np.testing.assert_allclose(expr_outputs, gt_outputs)

    def get_node2value(self, inputs, start_idx=0):
        inp_values = []
        has_node_not_in_scope = False
        for i in range(start_idx, len(inputs)):
            try:
                inp_values.append(self.current_node2values()[inputs[i]])
            except:
                has_node_not_in_scope = True
                self.node_without_tensor_info[inputs[i]] = [
                    self.tracer.current_scope(),
                    traceback.extract_stack(),
                ]
        return inp_values, has_node_not_in_scope

    def check_expr_interpret(self, expr, gt_outputs):
        ori_in, has_node_not_in_scope = self.get_node2value(expr.inputs)
        if not has_node_not_in_scope:
            expr_res = expr.interpret(*ori_in)
            try:
                self.check_expr_results(expr_res, gt_outputs)
            except:
                raise ValueError("Error occurred when checking expr: {}".format(expr))

    def check_apply(self, expr, gt_outputs, opdef):
        ori_in, has_node_not_in_scope = self.get_node2value(expr.inputs)
        if not has_node_not_in_scope:
            expr_res = expr.interpret(*ori_in)
            indexs = self.check_apply_special_cases(opdef, len(gt_outputs))
            try:
                self.check_expr_results(expr_res, gt_outputs, indexs=indexs)
            except:
                raise ValueError("Error occurred when checking expr: {}".format(expr))

    def check_builtin_module(self, module, expr, gt_outputs):
        ori_in, has_node_not_in_scope = self.get_node2value(expr.inputs, start_idx=1)
        if not has_node_not_in_scope:
            ori_in.insert(0, module)
            expr_res = expr.interpret(*ori_in)
            try:
                self.check_expr_results(expr_res, gt_outputs)
            except:
                raise ValueError(
                    "{}, Error occurred when checking expr: {}".format(expr)
                )
