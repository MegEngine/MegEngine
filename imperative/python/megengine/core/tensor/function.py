# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ..ops.builtin import OpDef
from .core import TensorBase, TensorWrapperBase, apply


class Function:
    """
    Defines a block of operations with customizable differentiation.

    The computation should be defined in ``forward`` method, with gradient
    computation defined in ``backward`` method.

    Each instance of ``Function`` should be used only once during forwardding.

    Examples:

    .. testcode::

        class Sigmoid(Function):
            def forward(self, x):
                y = 1 / (1 + F.exp(-x))
                self.y = y
                return y

            def backward(self, output_grads):
                y = self.y
                return output_grads * y * (1-y)

    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args):
        ret = apply(self, *args)
        if type(ret) == tuple and len(ret) == 1:
            return ret[0]
        return ret

    def forward(self, *args, **kwargs):
        """
        Applies operations to ``inputs`` and returns results. It must be overriden by all subclasses.

        :param input: input tensors.
        :return: a tuple of Tensor or a single Tensor.

        .. note::

            This method should return a tuple of Tensor or a single Tensor representing the output
            of the function.
        """
        raise NotImplementedError

    def backward(self, *output_grads):
        """
        Compute the gradient of the forward function. It must be overriden by all subclasses.

        :param output_grads: gradients of outputs that are returned by :meth:`~.function.Function.forward`.

        .. note::

            In case when some tensors of outputs are not related to loss function, the corresponding
            values in ``output_grads`` would be ``None``.

        .. note::

            This method should return a tuple which containing the gradients of all inputs, in the same order
            as the ``inputs`` argument of :meth:`~.function.Function.forward` . A ``Tensor`` could be returned
            instead if there is only one input. If users want to stop the propagation of some gradients,
            the corresponding returned values should be set ``None`` .

        """
        raise NotImplementedError

    def get_backward_fn(self):
        if self.backward is None:
            return None

        def _backward(*output_grads):
            if type(output_grads) is tuple:
                _output_grads = [
                    TensorWrapper(i) if i is not None else i for i in output_grads
                ]
            else:
                _output_grads = (
                    TensorWrapper(output_grads)
                    if output_grads is not None
                    else output_grads,
                )
            ret = self.backward(*_output_grads)
            if type(ret) is not tuple:
                ret = (ret,)
            ret = tuple(
                i.__wrapped__ if isinstance(i, TensorWrapper) else i for i in ret
            )
            return ret

        return _backward


Function.apply = Function.__call__


@apply.register()
def _(op: Function, *args: TensorWrapperBase):
    assert args
    Wrapper = type(args[0])

    # compute the value for self define function
    extra_data_dic = {}
    for arg in args:
        extra_data_dic[arg.__wrapped__] = arg.__wrapped__._extra_data
        arg.__wrapped__._extra_data = {}

    rets = op.forward(*args)

    for arg in args:
        arg.__wrapped__._extra_data = extra_data_dic[arg.__wrapped__]

    # update the gradient information for self define function
    inputs = tuple(map(lambda i: i.__wrapped__, args))
    outputs = (
        tuple(map(lambda i: i.__wrapped__, rets))
        if type(rets) is tuple
        else (rets.__wrapped__,)
    )

    for output in outputs:
        if output not in inputs:
            output._extra_data = {}

    with push_context() as ctx:
        ctx.inputs = inputs
        ctx.outputs = outputs
        for k in set().union(*(i._extra_data for i in inputs if isinstance(i, Tensor))):
            ctx.key = k
            data = tuple(
                i._extra_data.get(k) if isinstance(i, Tensor) else i for i in inputs
            )
            # data are instances of Tracer
            # dispatched to apply.add@grad.py
            rets = apply(op, *data)
            if rets is not None:
                assert len(outputs) == len(rets)
                for t, i in zip(outputs, rets):
                    t._extra_data[k] = i

    return tuple(map(Wrapper, outputs))
