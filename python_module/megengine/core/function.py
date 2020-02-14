# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from abc import ABCMeta, abstractmethod
from typing import Iterable, Tuple, Union

import megengine._internal as mgb

from .tensor import Tensor


class _OverrideGradientCraniotome(mgb.craniotome.CraniotomeBase):
    __nr_inputs__ = None
    __nr_outputs__ = None
    __expand_single_outputs__ = False
    __allow_duplicate__ = False

    grad_func = None

    def setup(self, nr_inputs, nr_outputs, grad_func):
        self.__nr_inputs__ = nr_inputs + nr_outputs
        self.__nr_outputs__ = nr_outputs
        self.grad_func = grad_func

    def infer_shape(self, inp_shapes):
        return inp_shapes[-self.__nr_outputs__ :]

    def init_output_dtype(self, input_dtypes):
        return input_dtypes[-self.__nr_outputs__ :]

    def execute(self, inputs, outputs):
        for ivar, ovar in zip(inputs[-self.__nr_outputs__ :], outputs):
            ovar.set_value(ivar)

    def grad(self, wrt_idx, inputs, outputs, out_grad):
        # TODO: Make sure grad_values really have values in eager mode.
        # Porting to the new imperative engine would solve this, but if it
        # don't happen, EagerEvalManager should be changed.
        grads = self.grad_func(
            *(Tensor(x) if x is not None else None for x in out_grad)
        )
        # pylint: disable=literal-comparison
        if isinstance(grads, Tensor) or grads is None or grads is 0:
            grads = (grads,)
        assert (
            len(grads) == self.__nr_inputs__ - self.__nr_outputs__
        ), "Function.backward should return a tuple with len = {}, got {}".format(
            self.__nr_inputs__ - self.__nr_outputs__, len(grads)
        )
        # pylint: disable=literal-comparison
        return (
            list(x._symvar if x is not None and x is not 0 else 0 for x in grads)
            + [0] * self.__nr_outputs__
        )

    def get_serialize_params(self):
        raise NotImplementedError("Serialization of Function is not implemented")


class Function(metaclass=ABCMeta):
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
                self.save_for_backward(y)
                return y

            def backward(self. output_grads):
                (y, ) = self.saved_tensors
                return output_grads * y * (1-y)

    """

    _has_saved_state = False
    saved_tensors = None

    def __init__(self):
        self.saved_tensors = ()

    @abstractmethod
    def forward(self, *inputs: Iterable[Tensor]) -> Union[Tuple[Tensor], Tensor]:
        """
        Applies operations to ``inputs`` and returns results. It must be overriden by all subclasses.
        Users can call :meth:`~.function.Function.save_for_backward` in this method to save tensors.

        :param input: Input tensors.

        .. note::

            This method should return a tuple of Tensor representing the output
            of the function.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(
        self, *output_grads: Iterable[Union[Tensor, None]]
    ) -> Union[Tuple[Tensor], Tensor]:
        """
        Compute the gradient of the function. It must be overriden by all subclasses.

        :param output_grads: gradients of outputs that are returned by :meth:`~.function.Function.forward`

            .. note::

                In case when some tensors of outputs are not related to loss function, the corresponding
                values in ``output_grads`` would be ``None``.

        .. note::

            This method should return a tuple containing gradients of all
            inputs, in the same order as the ``inputs`` argument of :meth:`~.function.Function.forward` . A
            ``Tensor`` could be returned instead if there is only one input.
            If users want to stop the propagation of some gradients, the corresponding returned values should be ``None`` .

        """
        raise NotImplementedError

    def save_for_backward(self, *tensors: Iterable[Tensor]):
        """
        Saves tensors for gradient computation. This method should be called only
        once in :meth:`~.function.Function.forward`, additional calls will replace values saved previously.

        The saved tensors can be accessed through the ``saved_tensors`` attribute.
        """
        self.saved_tensors = tensors

    def __call__(self, *inputs):
        assert (
            not self._has_saved_state
        ), "A Function instance should not be called multiple times"
        outputs = self.forward(*inputs)
        if isinstance(outputs, Tensor):
            outputs = (outputs,)
        self._has_saved_state = True
        sv = (x._symvar for x in inputs + outputs)
        outputs = _OverrideGradientCraniotome.make(
            *sv, nr_inputs=len(inputs), nr_outputs=len(outputs), grad_func=self.backward
        )
        outputs = tuple(map(Tensor, outputs))
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs
