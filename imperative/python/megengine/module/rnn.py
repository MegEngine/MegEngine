# -*- coding: utf-8 -*-
import math
import numbers
from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np

from ..core._imperative_rt.core2 import apply
from ..core.ops import builtin
from ..core.ops.builtin import BatchNorm
from ..functional import stack, zeros
from ..tensor import Parameter, Tensor
from . import init
from .module import Module


class RNNCellBase(Module):
    def __init__(
        self, input_size: int, hidden_size: int, bias: bool, num_chunks: int,
    ) -> None:
        # num_chunks indicates the number of gates
        super(RNNCellBase, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # initialize weights
        self.gate_hidden_size = num_chunks * hidden_size
        self.weight_ih = Parameter(
            np.zeros((self.gate_hidden_size, input_size), dtype=np.float32)
        )
        self.weight_hh = Parameter(
            np.zeros((self.gate_hidden_size, hidden_size), dtype=np.float32)
        )
        if bias:
            self.bias_ih = Parameter(
                np.zeros((self.gate_hidden_size), dtype=np.float32)
            )
            self.bias_hh = Parameter(
                np.zeros((self.gate_hidden_size), dtype=np.float32)
            )
        else:
            self.bias_ih = zeros(shape=(self.gate_hidden_size))
            self.bias_hh = zeros(shape=(self.gate_hidden_size))
        self.reset_parameters()
        # if bias is False self.bias will remain zero

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    @abstractmethod
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError("forward not implemented !")


class RNNCell(RNNCellBase):

    r"""An Elman RNN cell with tanh or ReLU non-linearity.

    .. math::

        h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})

    If :attr:`nonlinearity` is `'relu'`, then ReLU is used in place of tanh.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``

    Inputs: input, hidden
        - **input** of shape `(batch, input_size)`: tensor containing input features
        - **hidden** of shape `(batch, hidden_size)`: tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch

    Shape:
        - Input1: :math:`(N, H_{in})` tensor containing input features where
          :math:`H_{in}` = `input_size`
        - Input2: :math:`(N, H_{out})` tensor containing the initial hidden
          state for each element in the batch where :math:`H_{out}` = `hidden_size`
          Defaults to zero if not provided.
        - Output: :math:`(N, H_{out})` tensor containing the next hidden state
          for each element in the batch


    Examples:

        .. code-block::

            import numpy as np
            import megengine as mge
            import megengine.module as M

            m = M.RNNCell(10, 20)
            inp = mge.tensor(np.random.randn(3, 10), dtype=np.float32)
            hx = mge.tensor(np.random.randn(3, 20), dtype=np.float32)
            out = m(inp, hx)
            print(out.numpy().shape)

        Outputs:

        .. code-block::

            (3, 20)

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ) -> None:
        self.nonlinearity = nonlinearity
        super(RNNCell, self).__init__(input_size, hidden_size, bias, num_chunks=1)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = zeros(shape=(input.shape[0], self.gate_hidden_size),)
        op = builtin.RNNCell(nonlineMode=self.nonlinearity)
        return apply(
            op, input, self.weight_ih, self.bias_ih, hx, self.weight_hh, self.bias_hh
        )[0]


class LSTMCell(RNNCellBase):

    r"""A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_size)`: tensor containing input features
        - **h_0** of shape `(batch, hidden_size)`: tensor containing the initial hidden
          state for each element in the batch.
        - **c_0** of shape `(batch, hidden_size)`: tensor containing the initial cell state
          for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

    Outputs: (h_1, c_1)
        - **h_1** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch
        - **c_1** of shape `(batch, hidden_size)`: tensor containing the next cell state
          for each element in the batch

    Examples:

        .. code-block::

            import numpy as np
            import megengine as mge
            import megengine.module as M

            m = M.LSTMCell(10, 20)
            inp = mge.tensor(np.random.randn(3, 10), dtype=np.float32)
            hx = mge.tensor(np.random.randn(3, 20), dtype=np.float32)
            cx = mge.tensor(np.random.randn(3, 20), dtype=np.float32)
            hy, cy = m(inp, (hx, cx))
            print(hy.numpy().shape)
            print(cy.numpy().shape)

        Outputs:

        .. code-block::

            (3, 20)
            (3, 20)

    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,) -> None:
        super(LSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)

    def forward(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        # hx: (h, c)
        if hx is None:
            h = zeros(shape=(input.shape[0], self.hidden_size))
            c = zeros(shape=(input.shape[0], self.hidden_size))
        else:
            h, c = hx
        op = builtin.LSTMCell()
        return apply(
            op, input, self.weight_ih, self.bias_ih, h, self.weight_hh, self.bias_hh, c
        )[:2]


class RNNBase(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0,
        bidirectional: bool = False,
        proj_size: int = 0,
    ) -> None:
        super(RNNBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.proj_size = proj_size

        # check validity of dropout
        if (
            not isinstance(dropout, numbers.Number)
            or not 0 <= dropout <= 1
            or isinstance(dropout, bool)
        ):
            raise ValueError(
                "Dropout should be a float in [0, 1], which indicates the probability "
                "of an element to be zero"
            )

        if proj_size < 0:
            raise ValueError(
                "proj_size should be a positive integer or zero to disable projections"
            )
        elif proj_size >= hidden_size:
            raise ValueError("proj_size has to be smaller than hidden_size")

        self.cells = []
        for layer in range(self.num_layers):
            self.cells.append([])
            for _ in range(self.num_directions):
                self.cells[layer].append(self.create_cell(layer))
        # parameters have been initialized during the creation of the cells
        # if flatten, then delete cells
        self._flatten_parameters(self.cells)

    def _flatten_parameters(self, cells):
        gate_hidden_size = cells[0][0].gate_hidden_size
        size_dim1 = 0
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                size_dim1 += cells[layer][direction].weight_ih.shape[1]
                size_dim1 += cells[layer][direction].weight_hh.shape[1]
        if self.bias:
            size_dim1 += 2 * self.num_directions * self.num_layers

        self._flatten_weights = Parameter(
            np.zeros((gate_hidden_size, size_dim1), dtype=np.float32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    @abstractmethod
    def create_cell(self, layer):
        raise NotImplementedError("Cell not implemented !")

    @abstractmethod
    def init_hidden(self):
        raise NotImplementedError("init_hidden not implemented !")

    @abstractmethod
    def get_output_from_hidden(self, hx):
        raise NotImplementedError("get_output_from_hidden not implemented !")

    @abstractmethod
    def apply_op(self, input, hx):
        raise NotImplementedError("apply_op not implemented !")

    def _apply_fn_to_hx(self, hx, fn):
        return fn(hx)

    def _stack_h_n(self, h_n):
        return stack(h_n, axis=0)

    def forward(self, input: Tensor, hx=None):
        if self.batch_first:
            batch_size = input.shape[0]
            input = input.transpose((1, 0, 2))  # [seq_len, batch_size, dim]
        else:
            batch_size = input.shape[1]
        if hx is None:
            hx = self.init_hidden(batch_size)

        output, h = self.apply_op(input, hx)
        if self.batch_first:
            output = output.transpose((1, 0, 2))
        return output, h


class RNN(RNNBase):

    r"""Applies a multi-layer Elman RNN with :math:`\tanh` or :math:`\text{ReLU}` non-linearity to an
    input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two RNNs together to form a `stacked RNN`,
            with the second RNN taking in outputs of the first RNN and
            computing the final results. Default: 1
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

    Inputs: input, h_0
        * **input**: tensor of shape :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})` containing the initial hidden
          state for each element in the batch. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the RNN, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})` containing the final hidden state
          for each element in the batch.


    Examples:

        .. code-block::

            import numpy as np
            import megengine as mge
            import megengine.module as M

            m = M.RNN(10,20,2,batch_first=False,nonlinearity="relu",bias=True,bidirectional=True)
            inp = mge.tensor(np.random.randn(6, 30, 10), dtype=np.float32)
            hx = mge.tensor(np.random.randn(4, 30, 20), dtype=np.float32)
            out, hn = m(inp, hx)
            print(out.numpy().shape)

        Outputs:

        .. code-block::

            (6, 30, 40)

    """

    def __init__(self, *args, **kwargs) -> None:
        self.nonlinearity = kwargs.pop("nonlinearity", "tanh")
        super(RNN, self).__init__(*args, **kwargs)

    def create_cell(self, layer):
        if layer == 0:
            input_size = self.input_size
        else:
            input_size = self.num_directions * self.hidden_size
        return RNNCell(input_size, self.hidden_size, self.bias, self.nonlinearity)

    def init_hidden(self, batch_size):
        hidden_shape = (
            self.num_directions * self.num_layers,
            batch_size,
            self.hidden_size,
        )
        return zeros(shape=hidden_shape)

    def get_output_from_hidden(self, hx):
        return hx

    def apply_op(self, input, hx):
        fwd_mode = (
            BatchNorm.FwdMode.TRAINING if self.training else BatchNorm.FwdMode.INFERENCE
        )

        op = builtin.RNN(
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            bias=self.bias,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            nonlineMode=self.nonlinearity,
            fwd_mode=fwd_mode,
        )
        output, h = apply(op, input, hx, self._flatten_weights)[:2]
        output = output + h.sum() * 0
        h = h + output.sum() * 0
        return output, h


class LSTM(RNNBase):

    r"""Applies a multi-layer long short-term memory LSTM to an input
    sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{t-1}`
    is the hidden state of the layer at time `t-1` or the initial hidden
    state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,
    :math:`o_t` are the input, forget, cell, and output gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    In a multilayer LSTM, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    If ``proj_size > 0`` is specified, LSTM with projections will be used. This changes
    the LSTM cell in the following way. First, the dimension of :math:`h_t` will be changed from
    ``hidden_size`` to ``proj_size`` (dimensions of :math:`W_{hi}` will be changed accordingly).
    Second, the output hidden state of each layer will be multiplied by a learnable projection
    matrix: :math:`h_t = W_{hr}h_t`. Note that as a consequence of this, the output
    of LSTM network will be of different shape as well. See Inputs/Outputs sections below for exact
    dimensions of all variables. You can find more details in https://arxiv.org/abs/1402.1128.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0

    Inputs: input, (h_0, c_0)
        * **input**: tensor of shape :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})` containing the
          initial hidden state for each element in the batch.
          Defaults to zeros if (h_0, c_0) is not provided.
        * **c_0**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{cell})` containing the
          initial cell state for each element in the batch.
          Defaults to zeros if (h_0, c_0) is not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{cell} ={} & \text{hidden\_size} \\
                H_{out} ={} & \text{proj\_size if } \text{proj\_size}>0 \text{ otherwise hidden\_size} \\
            \end{aligned}

    Outputs: output, (h_n, c_n)
        * **output**: tensor of shape :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the LSTM, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})` containing the
          final hidden state for each element in the batch.
        * **c_n**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{cell})` containing the
          final cell state for each element in the batch.

    Examples:

        .. code-block::

            import numpy as np
            import megengine as mge
            import megengine.module as M

            m = M.LSTM(10, 20, 2, batch_first=False, bidirectional=True, bias=True)
            inp = mge.tensor(np.random.randn(6, 30, 10), dtype=np.float32)
            hx = mge.tensor(np.random.randn(4, 30, 20), dtype=np.float32)
            cx = mge.tensor(np.random.randn(4, 30, 20), dtype=np.float32)
            out, (hn, cn) = m(inp,(hx,cx))
            print(out.numpy().shape)

        Outputs:

        .. code-block::

            (6, 30, 40)

    """

    def __init__(self, *args, **kwargs) -> None:
        super(LSTM, self).__init__(*args, **kwargs)

    def create_cell(self, layer):
        if layer == 0:
            input_size = self.input_size
        else:
            input_size = self.num_directions * self.hidden_size
        return LSTMCell(input_size, self.hidden_size, self.bias)

    def init_hidden(self, batch_size):
        hidden_shape = (
            self.num_directions * self.num_layers,
            batch_size,
            self.hidden_size,
        )
        h = zeros(shape=hidden_shape)
        c = zeros(shape=hidden_shape)
        return (h, c)

    def get_output_from_hidden(self, hx):
        return hx[0]

    def apply_op(self, input, hx):
        fwd_mode = (
            BatchNorm.FwdMode.TRAINING if self.training else BatchNorm.FwdMode.INFERENCE
        )
        op = builtin.LSTM(
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            bias=self.bias,
            hidden_size=self.hidden_size,
            proj_size=self.proj_size,
            dropout=self.dropout,
            fwd_mode=fwd_mode,
        )
        output, h, c = apply(op, input, hx[0], hx[1], self._flatten_weights)[:3]
        placeholders = [output.sum() * 0, h.sum() * 0, c.sum() * 0]
        output = output + placeholders[1] + placeholders[2]
        h = h + placeholders[0] + placeholders[2]
        c = c + placeholders[0] + placeholders[1]
        return output, (h, c)

    def _apply_fn_to_hx(self, hx, fn):
        return (fn(hx[0]), fn(hx[1]))

    def _stack_h_n(self, h_n):
        h = [tup[0] for tup in h_n]
        c = [tup[1] for tup in h_n]
        return (stack(h, axis=0), stack(c, axis=0))
