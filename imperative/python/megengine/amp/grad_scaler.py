from typing import Iterable, List, Union

import numpy as np

from ..autodiff import GradManager
from ..functional import full_like
from ..functional.math import _check_non_finite
from ..tensor import Tensor


class GradScaler:
    r"""A helper class that performs grad scaling to prevent from data overflow in
    :class:`~.autocast` mode.

    Args:
        init_scale: Initial scale factor.
        growth_factor: Factor that the scale is multiplied by in actual
            :meth:`update` stage. If growth_factor is 0, scale_factor will not update.
        backoff_factor: Factor that the scale is multiplied by when encountering
            overflow grad.
        growth_interval: The interval between two scale update stages.

    Example:
        .. code-block::

           gm = GradManager()
           opt = ...
           scaler = GradScaler()

           gm.attach(model.parameters())

           @autocast()
           def train_step(image, label):
               with gm:
                   logits = model(image)
                   loss = F.nn.cross_entropy(logits, label)
                   scaler.backward(gm, loss)
               opt.step().clear_grad()
               return loss

        If need more flexible usage, could split ``scaler.backward`` into three lines:

        .. code-block::

           @autocast()
           def train_step(image, label):
               with gm:
                   logits = model(image)
                   loss = F.nn.cross_entropy(logits, label)
                   gm.backward(loss， dy=megengine.tensor(scaler.scale_factor))
               scaler.unscale(gm.attached_tensors())
               scaler.update()
               opt.step().clear_grad()
               return loss

        This is useful when need to accumulate grads for multi batches.
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 4,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self.scale_factor = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = growth_interval

        self._growth_tracker = 0
        self._found_non_finite = False

    def backward(
        self,
        gm: GradManager,
        y: Union[Tensor, List[Tensor]] = None,
        dy: Union[Tensor, List[Tensor]] = None,
        *,
        unscale_grad: bool = True,
        update_scale: bool = "if_unscale_grad"
    ):
        r"""A wrapper of GradManager's :meth:`~.GradManager.backward`, used to scale
        ``y``'s grad and unscale parameters' grads.

        Args:
            gm: The to be wrapped GradManager.
            y: Same as GradManager backward's ``y``.
            dy: Same as GradManager backward's ``dy``. Will be multiplied
                by ``scale_factor``.
            unscale_grad: Whether do :meth:`unscale` at the same time. Could be
                ``False`` if needs to accumulate grads.
            update_scale: Same as :meth:`unscale`'s ``update``. Will be ignored
                if ``unscale_grad`` is ``False``.
        """
        # These checks should be consistent with GradManager's
        if y is None:
            ys = []
        elif isinstance(y, (tuple, list)):
            ys = y
        else:
            ys = [y]
        if dy is None:
            dys = [full_like(y, self.scale_factor) for y in ys]
        elif isinstance(dy, (tuple, list)):
            dys = [dy_ * self.scale_factor for dy_ in dy]
        else:
            dys = [dy * self.scale_factor]

        gm.backward(y=ys, dy=dys)

        if unscale_grad:
            self.unscale(gm.attached_tensors())
            if update_scale:
                self.update()

    def unscale(self, grad_tensors: Iterable[Tensor]):
        r"""Unscale all ``grad_tensors``'s grad.

        Args:
            grad_tensors: Tensors needed to unscale grads. Should be all tensors
                that are affected by ``target`` tensor in GradManager's backward.
        """
        if self.growth_interval == 0:
            # use float64 for better precision
            inv_scale = Tensor(1.0 / self.scale_factor)
            for tensor in grad_tensors:
                if tensor is None or getattr(tensor, "grad", None) is None:
                    continue
                tensor.grad *= inv_scale
            return self

        # to support tracing, _check_gradients should be applied to every grad.
        if self._check_gradients(
            [x.grad for x in grad_tensors], 1.0 / self.scale_factor
        ):
            self._found_non_finite = True
            for tensor in grad_tensors:
                if tensor is None or getattr(tensor, "grad", None) is None:
                    continue
                tensor.grad = None
        return self

    def _check_gradients(self, grads, scale):
        if len(grads) == 0:
            return False
        rst = _check_non_finite(grads, scale)
        rst = rst.numpy()
        return rst

    def update(self, new_scale: float = None):
        r"""Update the scale factor according to whether encountered overflow grad.
        If ``new_scale`` is provided, internal update mechanism will be ignored.
        """
        if self.growth_interval == 0:
            return

        if new_scale is not None:
            self.scale_factor = float(new_scale)
        else:
            if self._found_non_finite:
                self.scale_factor *= self.backoff_factor
                self._growth_tracker = 0
            else:
                self._growth_tracker += 1
                if self._growth_tracker >= self.growth_interval:
                    self.scale_factor *= self.growth_factor
                    self._growth_tracker = 0
        self._found_non_finite = False

    def state_dict(self):
        return {
            "scale_factor": self.scale_factor,
            "growth_factor": self.growth_factor,
            "backoff_factor": self.backoff_factor,
            "growth_interval": self.growth_interval,
            "_growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state):
        self.scale_factor = state["scale_factor"]
        self.growth_factor = state["growth_factor"]
        self.backoff_factor = state["backoff_factor"]
        self.growth_interval = state["growth_interval"]
        self._growth_tracker = state["_growth_tracker"]
