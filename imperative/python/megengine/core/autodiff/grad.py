# -*- coding: utf-8 -*-
import weakref

from .._imperative_rt import core2

_grad_count = 0
_grad_manager_dict = weakref.WeakValueDictionary()


def get_grad_managers():
    return [_grad_manager_dict[key] for key in _grad_manager_dict]


class GradKey(core2.GradKey):
    def __init__(self, name=None):
        if name:
            self.name = name

    def backward(self, ys, dys):
        return core2.backward(self, ys, dys)


class Grad:
    stack = []
    grouping = False
    key2grad = weakref.WeakValueDictionary()

    def __init__(self, name=None):
        global _grad_count
        if name is None:
            name = "grad_%d" % _grad_count
            _grad_count += 1
        self._refkeeper = []
        self._impl = GradKey(name)
        Grad.key2grad[self._impl] = self
        _grad_manager_dict[self._name] = self
        self._group = [weakref.ref(self)]

    @property
    def _name(self):
        return self._impl.name

    def _is_attached_to(self, tensor):
        return self._impl.is_attached_to(tensor)

    def wrt(self, *tensors, callback=None):
        for x in tensors:
            self._impl.attach(x, callback)
        return self

    def __call__(self, ys, dys):
        from collections.abc import Sequence

        if not isinstance(ys, Sequence):
            ys = [ys]

        if not isinstance(dys, Sequence):
            dys = [dys]

        group = [ref() for ref in self._group]

        for grad in group:
            if grad is self:
                continue
            grad.suppress()

        self._impl.backward(ys, dys)

        for grad in group:
            if grad is self:
                continue
            grad.resume()

        self._refkeeper = None
        return None

    def __enter__(self):
        ref = weakref.ref(self)
        self._impl.enter()
        if Grad.grouping:
            group = Grad.stack[-1]
            self._group = group
            group.append(ref)
        else:
            Grad.stack.append(self._group)
        return self

    def __exit__(self, _1, _2, _3):
        self._impl.exit()
        self._refkeeper = None
        del Grad.key2grad[self._impl]
        self._impl = None
        self._group.remove(weakref.ref(self))
        if len(self._group) == 0:
            Grad.stack.remove(self._group)

    @staticmethod
    def begin_group():
        assert not Grad.grouping
        Grad.grouping = True

    @staticmethod
    def end_group():
        group = Grad.stack[-1]
        assert len(group) > 0
        assert Grad.grouping
        Grad.grouping = False

    def suppress(self):
        if self._impl is not None:
            self._impl.suppress()

    def resume(self):
        if self._impl is not None:
            self._impl.resume()


class Function:
    r"""Defines a block of operations with customizable differentiation.

    The computation should be defined in ``forward`` method, with gradient
    computation defined in ``backward`` method.

    Each instance of ``Function`` should be used only once during forwardding.

    Examples:

        .. code-block::

            class Sigmoid(Function):
                def forward(self, x):
                    y = 1 / (1 + F.exp(-x))
                    self.y = y
                    return y

                def backward(self, dy):
                    y = self.y
    """

    def forward(self, *args, **kwargs):
        r"""Applies operations to ``inputs`` and returns results. It must be overriden by all subclasses.

        Args:
            input: input tensors.

        Returns:
            a tuple of Tensor or a single Tensor.

        Note:
            * This method should return a tuple of Tensor or a single Tensor representing the output
              of the function.
            * positional arguments should all be Tensor
        """
        raise NotImplementedError

    def backward(self, *output_grads):
        r"""Compute the gradient of the forward function. It must be overriden by all subclasses.

        Args:
            output_grads: gradients of outputs that are returned by :meth:`forward`.

        Note:
            * In case when some tensors of outputs are not related to loss function, the corresponding
              values in ``output_grads`` would be ``None``.
            * This method should return a tuple which containing the gradients of all inputs, in the same order
              as the ``inputs`` argument of :meth:`forward` . A ``Tensor`` could be returned
              instead if there is only one input. If users want to stop the propagation of some gradients,
              the corresponding returned values should be set ``None`` .
        """
        raise NotImplementedError

    def _default_rule(self, *args):
        ret = self.forward(*args)
        self.__single_output = isinstance(ret, core2.Tensor)
        return ret

    def _grad_rule(self, *args):
        return self._default_rule(*args), self.backward

    def __call__(self, *args):
        from ...tensor import Tensor

        for arg in args:
            if not isinstance(arg, Tensor):
                raise TypeError(
                    "op Function expect type Tensor as inputs, got {}".format(type(arg))
                )

        grad_key = core2.get_grad_key(args)
        if grad_key is None:
            return self._default_rule(*args)

        grad = Grad.key2grad[grad_key]
        group = [ref() for ref in grad._group]

        origin_args = [Tensor(arg) for arg in args]

        for grad in group:
            grad.suppress()
        outputs, backward = self._grad_rule(*args)
        for grad in reversed(group):
            grad.resume()

        def normalized_backward(*output_grads):
            input_grads = backward(*output_grads)
            if isinstance(input_grads, Tensor) or input_grads is None:
                input_grads = (input_grads,)
            return input_grads

        if self.__single_output:
            outputs = (outputs,)
        outputs = core2.set_grad(normalized_backward, origin_args, outputs)
        if self.__single_output:
            (outputs,) = outputs
        return outputs

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
