from contextlib import contextmanager

from ..core.autodiff.grad import Grad
from ..tensor import tensor


class GradManager:
    def __init__(self):
        self._call_back_pair = []
        self._recording = False
        self._grad = None

    def register(self, params, callbacks=None):
        self._call_back_pair.append([list(params), callbacks or []])

    def backward(self, ys, dys=None):
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
            dys = [tensor(1.0) for y in ys]
        if not isinstance(dys, (tuple, list)):
            dys = [dys]
        try:
            self._grad(ys, dys)
        finally:
            self._grad = None

    def record(self):
        @contextmanager
        def recorder():
            grad = Grad()
            if self._recording:
                raise RuntimeError("already recording!")
            try:
                self._recording = True
                self._grad = grad
                for params, callbacks in self._call_back_pair:
                    for p in params:

                        def callback(param, grad, callbacks=callbacks, p=p):
                            ret = grad
                            for cb in callbacks:
                                ret = cb(param, ret)
                            p.grad = ret

                        grad.wrt(p, callback=callback)
                with grad:
                    yield
            finally:
                self._recording = False
                self._grad = None

        return recorder()
