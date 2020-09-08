from collections import defaultdict
from contextlib import contextmanager

from ..core.autodiff.grad import Grad
from ..tensor import tensor
from ..utils.future import Future

backwarding_grad_manager = None


def get_backwarding_grad_manager():
    return backwarding_grad_manager


class GradManager:
    def __init__(self):
        self._call_back_dict = defaultdict(list)
        self._param_dict = dict()
        self._recording = False
        self._grad = None
        self._after_backward_callback = []
        self._gradients = dict()

    def register(self, params, callbacks=[]):
        for p in params:
            self._param_dict[id(p)] = p
            for cb in callbacks:
                self._call_back_dict[id(p)].append(cb)
        return self

    def register_after_backward_callback(self, callback):
        self._after_backward_callback.append(callback)
        return self

    def backward(self, ys, dys=None):
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
        if not isinstance(ys, (tuple, list)):
            ys = [ys]
        if dys is None:
            dys = [tensor(1.0).broadcast(y.shape) for y in ys]
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
                if getattr(param, "grad", None) is None:
                    param.grad = grad
                else:
                    param.grad += grad
        finally:
            self._grad = None
            self._gradients = dict()
            backwarding_grad_manager = cache

    def record(self):
        @contextmanager
        def recorder():
            grad = Grad()
            if self._recording:
                raise RuntimeError("already recording!")
            try:
                self._recording = True
                self._grad = grad
                for param_id in self._param_dict.keys():
                    param_wrapper = self._param_dict[param_id]
                    callbacks = self._call_back_dict[param_id]

                    def callback(
                        param, grad, callbacks=callbacks, p=param_wrapper, gm=self
                    ):
                        ret = grad
                        for cb in callbacks:
                            ret = cb(param, ret)
                        gm._gradients[id(p)] = ret

                    grad.wrt(param_wrapper, callback=callback)
                with grad:
                    yield
            finally:
                self._recording = False
                self._grad = None
                self._gradients = dict()

        return recorder()
