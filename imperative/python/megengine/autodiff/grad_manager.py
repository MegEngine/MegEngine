from collections import defaultdict
from contextlib import contextmanager
from typing import Callable

from ..core.autodiff.grad import Grad
from ..logger import get_logger
from ..tensor import Tensor
from ..utils.future import Future

logger = get_logger(__name__)

backwarding_grad_manager = None


def get_backwarding_grad_manager():
    return backwarding_grad_manager


class GradManager:
    r"""
    GradManager manages auto differentiation and all resources required to perform it.

    Our auto differentiation framework requires that the user explicitly indicates when
    the forward operations start and when all resources should be released. A typical usage of
    GradManager is as follows:

    .. code-block::

        gm = GradManager()
        gm.attach(model.parameters())
        with gm:
            # forward operations
            ...
            # backward gradients
            gm.backward(loss)

    You can also use ``record()`` and ``release()`` method instead of ``with`` context:

    .. code-block::

        gm = GradManager()
        gm.attach(model.parameters())

        gm.record()

        # forward operations
        ...
        # backward gradients
        gm.backward(loss)

        gm.release()

    Typically, in data parallel, we would like to average the gradients across
    processes. Users will finally get the averaged gradients if an "AllReduce"
    callback is registered as follows:

    .. code-block::

        import megengine.distributed as dist

        gm = GradManager()
        gm.attach(model.parameters(), callback=dist.make_allreduce_cb("MEAN"))

    """

    def __init__(self):
        self._call_back_dict = defaultdict(list)
        self._param_dict = dict()
        self._recording = False
        self._grad = None
        self._after_backward_callback = []
        self._gradients = dict()

    def attach(self, params: list, callbacks=None):
        r"""
        Registers parameters that gradients should be calculated with respect to.
        Callback Functions should have a signature like this:

            .. code-block::

                def cb(param: Tensor, grad: Tensor) -> Tensor:
                    # do something
                    return grad

        :param params: to be registered parameters
        :param callbacks: list of callback functions
        """
        if callbacks is None:
            callbacks = []
        if isinstance(callbacks, Callable):
            callbacks = [callbacks]
        if isinstance(params, Tensor):
            params = [params]
        for p in params:
            self._param_dict[id(p)] = p
            for cb in callbacks:
                self._call_back_dict[id(p)].append(cb)
        if self._grad is not None:
            for p in params:
                self._record_param(id(p))
        return self

    def detach(self, params: list):
        r"""
        Remove specific registered parameters and callback functions.

        :param params: registered parameters
        """
        if isinstance(params, Tensor):
            params = [params]
        for idx, param in enumerate(params):
            if id(param) in self._param_dict:
                self._param_dict.pop(id(param))
                self._call_back_dict.pop(id(param))
            else:
                logger.warning("params with index {} is not attached.".format(idx))

    def _register_after_backward_callback(self, callback):
        self._after_backward_callback.append(callback)
        return self

    def backward(self, ys=None, dys=None):
        r"""
        Performs back-propagation and computes gradients.

        :param ys: outputs of forward operators, e.g., the loss tensor
        :param dys: derivatives of ys
        """
        from ..functional import ones_like

        global backwarding_grad_manager
        cache = backwarding_grad_manager
        backwarding_grad_manager = self
        if not self._recording:
            raise RuntimeError(
                "no computation history. "
                "did you forget record() or "
                "call a method that clears the history?"
            )
        assert self._grad is not None
        if ys is None:
            ys = []
        if not isinstance(ys, (tuple, list)):
            ys = [ys]
        if dys is None:
            dys = [ones_like(y) for y in ys]
        if not isinstance(dys, (tuple, list)):
            dys = [dys]
        try:
            self._grad(ys, dys)
            for callback in self._after_backward_callback:
                callback()
            for p, grad in self._gradients.items():
                if isinstance(grad, Future):
                    grad = grad.get()
                param = self._param_dict[p]
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad += grad
        finally:
            self.release()
            backwarding_grad_manager = cache

    def record(self):
        r"""
        Starts recording forward operations.
        """
        if self._recording:
            raise RuntimeError("already recording")
        grad = Grad()
        self._recording = True
        self._grad = grad
        for param_id in self._param_dict.keys():
            self._record_param(param_id)
        grad.__enter__()

    def _record_param(self, param_id):
        param_wrapper = self._param_dict[param_id]
        callbacks = self._call_back_dict[param_id]

        def callback(param, grad, callbacks=callbacks, p=param_wrapper, gm=self):
            ret = grad
            for cb in callbacks:
                ret = cb(param, ret)
            gm._gradients[id(p)] = ret

        # NOTE: override prev callback wrt when called serval times
        self._grad.wrt(param_wrapper, callback=callback)

    def release(self):
        r"""
        Stops recording and releases resources for gradients calculation.
        """
        if self._grad is not None:
            self._grad.__exit__(None, None, None)
            self._grad = None
        self._recording = False
        self._gradients = dict()

    def __enter__(self):
        self.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
