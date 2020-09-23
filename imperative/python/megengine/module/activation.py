# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from ..functional import leaky_relu, prelu, relu, sigmoid, softmax
from ..tensor import Parameter
from .module import Module


class Softmax(Module):
    r"""
    Applies a softmax function. Softmax is defined as:

    .. math::
            \text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}

    It is applied to all elements along axis, and rescales elements so that
    they stay in the range `[0, 1]` and sum to 1.

    :param axis: Along which axis softmax will be applied. By default,
        softmax will apply along the highest ranked axis.

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.module as M

        data = mge.tensor(np.array([-2,-1,0,1,2]).astype(np.float32))
        softmax = M.Softmax()
        output = softmax(data)
        with np.printoptions(precision=6):
            print(output.numpy())

    Outputs:

    .. testoutput::

        [0.011656 0.031685 0.086129 0.234122 0.636409]

    """

    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def forward(self, inputs):
        return softmax(inputs, self.axis)

    def _module_info_string(self) -> str:
        return "axis={axis}".format(axis=self.axis)


class Sigmoid(Module):
    r"""
    Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.module as M

        data = mge.tensor(np.array([-2,-1,0,1,2,]).astype(np.float32))
        sigmoid = M.Sigmoid()
        output = sigmoid(data)
        with np.printoptions(precision=6):
            print(output.numpy())

    Outputs:

    .. testoutput::

        [0.119203 0.268941 0.5      0.731059 0.880797]

    """

    def forward(self, inputs):
        return sigmoid(inputs)


class ReLU(Module):
    r"""
    Applies the element-wise function:

    .. math::
        \text{ReLU}(x) = \max(x, 0)

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.module as M
        data = mge.tensor(np.array([-2,-1,0,1,2,]).astype(np.float32))
        relu = M.ReLU()
        output = relu(data)
        with np.printoptions(precision=6):
            print(output.numpy())

    Outputs:

    .. testoutput::

        [0. 0. 0. 1. 2.]

    """

    def forward(self, x):
        return relu(x)


class PReLU(Module):
    r"""
    Applies the element-wise function:

    .. math::
        \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

    or

    .. math::
        \text{PReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        ax, & \text{ otherwise }
        \end{cases}

    Here :math:`a` is a learnable parameter. When called without arguments, `PReLU()` uses
    a single paramter :math:`a` across all input channel. If called with `PReLU(num_of_channels)`, each input channle will has it's own :math:`a`.

    :param num_parameters: number of :math:`a` to learn, there is only two
        values are legitimate: 1, or the number of channels at input. Default: 1
    :param init: the initial value of :math:`a`. Default: 0.25

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.module as M
        data = mge.tensor(np.array([-1.2, -3.7, 2.7]).astype(np.float32))
        prelu = M.PReLU()
        output = prelu(data)
        print(output.numpy())

    Outputs:

    .. testoutput::

        [-0.3   -0.925  2.7  ]

    """

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.num_parameters = num_parameters
        if num_parameters > 1:
            # Assume format is NCHW
            self.weight = Parameter(
                data=np.full((1, num_parameters, 1, 1), init, dtype=np.float32)
            )
        else:
            self.weight = Parameter(data=[init])

    def forward(self, inputs):
        assert self.weight.shape == (1,) or self.weight.shape == (
            1,
            int(inputs.shape[1]),
            1,
            1,
        ), "invalid weight's shape"
        return prelu(inputs, self.weight)


class LeakyReLU(Module):
    r"""
    Applies the element-wise function:

    .. math::
        \text{LeakyReLU}(x) = \max(0,x) + negative\_slope \times \min(0,x)

    or

    .. math::
        \text{LeakyReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        negative\_slope \times x, & \text{ otherwise }
        \end{cases}

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.module as M
        data = mge.tensor(np.array([-8, -12, 6, 10]).astype(np.float32))

        leakyrelu = M.LeakyReLU(0.01)
        output = leakyrelu(data)
        print(output.numpy())

    Outputs:

    .. testoutput::

        [-0.08 -0.12  6.   10.  ]

    """

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, inputs):
        return leaky_relu(inputs, self.negative_slope)
