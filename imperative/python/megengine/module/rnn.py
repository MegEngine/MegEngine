# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import math
import numbers
from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np

from ..core._imperative_rt.core2 import apply
from ..core.ops import builtin
from ..device import is_cuda_available
from ..functional import concat, expand_dims, repeat, stack, zeros
from ..functional.nn import concat
from ..tensor import Parameter, Tensor
from . import init
from .module import Module


class RNNCellBase(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        device=None,
        dtype=None,
    ) -> None:
        # num_chunks indicates the number of gates
        super(RNNCellBase, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # initialize weights
        common_kwargs = {"device": device, "dtype": dtype}
        self.gate_hidden_size = num_chunks * hidden_size
        self.weight_ih = Parameter(
            np.random.uniform(size=(self.gate_hidden_size, input_size)).astype(
                np.float32
            ),
            **common_kwargs,
        )
        self.weight_hh = Parameter(
            np.random.uniform(size=(self.gate_hidden_size, hidden_size)).astype(
                np.float32
            ),
            **common_kwargs,
        )
        if bias:
            self.bias_ih = Parameter(
                np.random.uniform(size=(self.gate_hidden_size)).astype(np.float32),
                **common_kwargs,
            )
            self.bias_hh = Parameter(
                np.random.uniform(size=(self.gate_hidden_size)).astype(np.float32),
                **common_kwargs,
            )
        else:
            self.bias_ih = zeros(shape=(self.gate_hidden_size), **common_kwargs)
            self.bias_hh = zeros(shape=(self.gate_hidden_size), **common_kwargs)
        self.reset_parameters()
        # if bias is False self.bias will remain zero

    def get_op(self):
        return builtin.RNNCell()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = zeros(
                shape=(input.shape[0], self.gate_hidden_size),
                dtype=input.dtype,
                device=input.device,
            )
        op = self.get_op()
        return apply(
            op, input, self.weight_ih, self.bias_ih, hx, self.weight_hh, self.bias_hh
        )[0]
        # return linear(input, self.weight_ih, self.bias_ih) + linear(hx, self.weight_hh, self.bias_hh)


class RNNCell(RNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
    ) -> None:
        self.nonlinearity = nonlinearity
        super(RNNCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=1, device=device, dtype=dtype
        )
        # self.activate = tanh if nonlinearity == "tanh" else relu

    def get_op(self):
        return builtin.RNNCell(nonlineMode=self.nonlinearity)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        return super().forward(input, hx)


class LSTMCell(RNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(LSTMCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=4, device=device, dtype=dtype
        )

    def get_op(self):
        return builtin.LSTMCell()

    def forward(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        # hx: (h, c)
        if hx is None:
            h = zeros(
                shape=(input.shape[0], self.hidden_size),
                dtype=input.dtype,
                device=input.device,
            )
            c = zeros(
                shape=(input.shape[0], self.hidden_size),
                dtype=input.dtype,
                device=input.device,
            )
        else:
            h, c = hx
        op = self.get_op()
        return apply(
            op, input, self.weight_ih, self.bias_ih, h, self.weight_hh, self.bias_hh, c
        )[:2]


def is_gpu(device: str) -> bool:
    if "xpux" in device and is_cuda_available():
        return True
    if "gpu" in device:
        return True
    return False


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
        device=None,
        dtype=None,
    ) -> None:
        super(RNNBase, self).__init__()
        # self.mode = mode
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
                self.cells[layer].append(self.create_cell(layer, device, dtype))
        # parameters have been initialized during the creation of the cells
        # if flatten, then delete cells
        self._flatten_parameters(device, dtype, self.cells)

    def _flatten_parameters(self, device, dtype, cells):
        gate_hidden_size = cells[0][0].gate_hidden_size
        size_dim1 = 0
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                size_dim1 += cells[layer][direction].weight_ih.shape[1]
                size_dim1 += cells[layer][direction].weight_hh.shape[1]
        # if self.bias:
        #     size_dim1 += 2 * self.num_directions * self.num_layers
        size_dim1 += 2 * self.num_directions * self.num_layers
        self._flatten_weights = Parameter(
            np.zeros((gate_hidden_size, size_dim1), dtype=np.float32)
        )
        self.reset_parameters()
        # TODO: if no bias, set the bias to zero

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    @abstractmethod
    def create_cell(self, layer, device, dtype):
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
            hx = self.init_hidden(batch_size, input.device, input.dtype)

        output, h = self.apply_op(input, hx)
        if self.batch_first:
            output = output.transpose((1, 0, 2))
        return output, h

        if is_gpu(str(input.device)) or True:
            # return output, h_n
            output, h = self.apply_op(input, hx)
            if self.batch_first:
                output = output.transpose((1, 0, 2))
            return output, h

        order_settings = [(0, input.shape[0]), (input.shape[0] - 1, -1, -1)]
        h_n = []
        for layer in range(self.num_layers):
            layer_outputs = []
            for direction in range(self.num_directions):
                direction_outputs = [None for _ in range(input.shape[0])]
                cell = self.cells[layer][direction]
                hidden = self._apply_fn_to_hx(
                    hx, lambda x: x[layer * self.num_directions + direction]
                )
                for step in range(*(order_settings[direction])):
                    hidden = cell(input[step], hidden)  # [batch_size, hidden_size]
                    direction_outputs[step] = self.get_output_from_hidden(hidden)
                direction_output = stack(
                    direction_outputs, axis=0
                )  # [seq_len, batch_size, hidden_size]
                layer_outputs.append(direction_output)
                h_n.append(hidden)
            layer_output = concat(
                layer_outputs, axis=-1
            )  # [seq_len, batch_size, D*hidden_size]
            input = layer_output
        if self.batch_first:
            layer_output = layer_output.transpose((1, 0, 2))
        return layer_output, self._stack_h_n(h_n)


class RNN(RNNBase):
    def __init__(self, *args, **kwargs) -> None:
        self.nonlinearity = kwargs.pop("nonlinearity", "tanh")
        super(RNN, self).__init__(*args, **kwargs)

    def create_cell(self, layer, device, dtype):
        if layer == 0:
            input_size = self.input_size
        else:
            input_size = self.num_directions * self.hidden_size
        return RNNCell(
            input_size, self.hidden_size, self.bias, self.nonlinearity, device, dtype
        )

    def init_hidden(self, batch_size, device, dtype):
        hidden_shape = (
            self.num_directions * self.num_layers,
            batch_size,
            self.hidden_size,
        )
        return zeros(shape=hidden_shape, dtype=dtype, device=device)

    def get_output_from_hidden(self, hx):
        return hx

    def apply_op(self, input, hx):
        op = builtin.RNN(
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            bias=self.bias,
            hidden_size=self.hidden_size,
            proj_size=self.proj_size,
            dropout=self.dropout,
            nonlineMode=self.nonlinearity,
        )
        output, h = apply(op, input, hx, self._flatten_weights)[:2]
        output = output + h.sum() * 0
        h = h + output.sum() * 0
        return output, h


class LSTM(RNNBase):
    def __init__(self, *args, **kwargs) -> None:
        super(LSTM, self).__init__(*args, **kwargs)

    def create_cell(self, layer, device, dtype):
        if layer == 0:
            input_size = self.input_size
        else:
            input_size = self.num_directions * self.hidden_size
        return LSTMCell(input_size, self.hidden_size, self.bias, device, dtype)

    def init_hidden(self, batch_size, device, dtype):
        hidden_shape = (
            self.num_directions * self.num_layers,
            batch_size,
            self.hidden_size,
        )
        h = zeros(shape=hidden_shape, dtype=dtype, device=device)
        c = zeros(shape=hidden_shape, dtype=dtype, device=device)
        return (h, c)

    def get_output_from_hidden(self, hx):
        return hx[0]

    def apply_op(self, input, hx):
        op = builtin.LSTM(
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            bias=self.bias,
            hidden_size=self.hidden_size,
            proj_size=self.proj_size,
            dropout=self.dropout,
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
