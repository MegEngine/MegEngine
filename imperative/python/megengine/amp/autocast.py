# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools

from ..core.tensor import amp


class autocast:
    r"""A class to control autocast mode for amp as a context manager or a decorator.

    Args:
        enabled: Whether autocast mode is enabled.
        low_prec_dtype: Set amp autocast mode's lower precision dtype. It will change
            the target dtype in tensor casting for better speed and memory. Default: float16.
        high_prec_dtype: Set amp autocast mode's higher precision dtype. It will
            change the target dtype in tensor casting for better precision. Default: float32.

    Examples:
        .. code-block::

           # used as decorator
           @autocast()
           def train_step(image, label):
               with gm:
                   logits = model(image)
                   loss = F.nn.cross_entropy(logits, label)
                   gm.backward(loss)
               opt.step().clear_grad()
               return loss

           # used as context manager
           def train_step(image, label):
               with autocast():
                   with gm:
                       logits = model(image)
                       loss = F.nn.cross_entropy(logits, label)
                       gm.backward(loss)
               opt.step().clear_grad()
               return loss
    """

    def __init__(
        self,
        enabled: bool = True,
        low_prec_dtype: str = "float16",
        high_prec_dtype: str = "float32",
    ):
        self.enabled = enabled
        self.high_prec_dtype = high_prec_dtype
        self.low_prec_dtype = low_prec_dtype
        self._origin_enabled = None
        self._origin_high = None
        self._origin_low = None

    def __enter__(self):
        self._origin_enabled, amp._enabled = amp._enabled, self.enabled
        self._origin_high = amp._high_prec_dtype
        amp._high_prec_dtype = self.high_prec_dtype
        self._origin_low = amp._low_prec_dtype
        amp._low_prec_dtype = self.low_prec_dtype

    def __exit__(self, *args):
        amp._enabled = self._origin_enabled
        amp._high_prec_dtype = self._origin_high
        amp._low_prec_dtype = self._origin_low

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper
