# -*- coding: utf-8 -*-
import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
from megengine.device import get_device_count
from megengine.module import LSTM, RNN, LSTMCell, RNNCell


def assert_tuple_equal(src, ref):
    assert len(src) == len(ref)
    for i, j in zip(src, ref):
        assert i == j


@pytest.mark.skipif(get_device_count("gpu") > 0, reason="no algorithm on cuda")
@pytest.mark.parametrize(
    "batch_size, input_size, hidden_size, init_hidden",
    [(3, 10, 20, True), (3, 10, 20, False), (1, 10, 20, False)],
)
def test_rnn_cell(batch_size, input_size, hidden_size, init_hidden):
    rnn_cell = RNNCell(input_size, hidden_size)
    x = mge.random.normal(size=(batch_size, input_size))
    if init_hidden:
        h = F.zeros(shape=(batch_size, hidden_size))
    else:
        h = None
    h_new = rnn_cell(x, h)
    assert_tuple_equal(h_new.shape, (batch_size, hidden_size))


@pytest.mark.skipif(get_device_count("gpu") > 0, reason="no algorithm on cuda")
@pytest.mark.parametrize(
    "batch_size, input_size, hidden_size, init_hidden",
    [(3, 10, 20, True), (3, 10, 20, False), (1, 10, 20, False)],
)
def test_lstm_cell(batch_size, input_size, hidden_size, init_hidden):
    rnn_cell = LSTMCell(input_size, hidden_size)
    x = mge.random.normal(size=(batch_size, input_size))
    if init_hidden:
        h = F.zeros(shape=(batch_size, hidden_size))
        hx = (h, h)
    else:
        hx = None
    h_new, c_new = rnn_cell(x, hx)
    assert_tuple_equal(h_new.shape, (batch_size, hidden_size))
    assert_tuple_equal(c_new.shape, (batch_size, hidden_size))


@pytest.mark.skipif(get_device_count("gpu") > 0, reason="no algorithm on cuda")
@pytest.mark.parametrize(
    "batch_size, seq_len, input_size, hidden_size, num_layers, bidirectional, init_hidden, batch_first",
    [
        (3, 6, 10, 20, 2, False, False, True),
        pytest.param(
            3,
            3,
            10,
            10,
            1,
            True,
            True,
            False,
            marks=pytest.mark.skip(reason="bidirectional will cause cuda oom"),
        ),
    ],
)
def test_rnn(
    batch_size,
    seq_len,
    input_size,
    hidden_size,
    num_layers,
    bidirectional,
    init_hidden,
    batch_first,
):
    rnn = RNN(
        input_size,
        hidden_size,
        batch_first=batch_first,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    if batch_first:
        x_shape = (batch_size, seq_len, input_size)
    else:
        x_shape = (seq_len, batch_size, input_size)
    x = mge.random.normal(size=x_shape)
    total_hidden_size = num_layers * (2 if bidirectional else 1) * hidden_size
    if init_hidden:
        h = mge.random.normal(size=(batch_size, total_hidden_size))
    else:
        h = None
    output, h_n = rnn(x, h)
    num_directions = 2 if bidirectional else 1
    if batch_first:
        assert_tuple_equal(
            output.shape, (batch_size, seq_len, num_directions * hidden_size)
        )
    else:
        assert_tuple_equal(
            output.shape, (seq_len, batch_size, num_directions * hidden_size)
        )
    assert_tuple_equal(
        h_n.shape, (num_directions * num_layers, batch_size, hidden_size)
    )


@pytest.mark.skipif(get_device_count("gpu") > 0, reason="no algorithm on cuda")
@pytest.mark.parametrize(
    "batch_size, seq_len, input_size, hidden_size, num_layers, bidirectional, init_hidden, batch_first",
    [
        (3, 10, 20, 20, 1, False, False, True),
        pytest.param(
            3,
            3,
            10,
            10,
            1,
            True,
            True,
            False,
            marks=pytest.mark.skip(reason="bidirectional will cause cuda oom"),
        ),
    ],
)
def test_lstm(
    batch_size,
    seq_len,
    input_size,
    hidden_size,
    num_layers,
    bidirectional,
    init_hidden,
    batch_first,
):
    rnn = LSTM(
        input_size,
        hidden_size,
        batch_first=batch_first,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    if batch_first:
        x_shape = (batch_size, seq_len, input_size)
    else:
        x_shape = (seq_len, batch_size, input_size)
    x = mge.random.normal(size=x_shape)
    total_hidden_size = num_layers * (2 if bidirectional else 1) * hidden_size
    if init_hidden:
        h = mge.random.normal(size=(batch_size, total_hidden_size))
        h = (h, h)
    else:
        h = None
    output, h_n = rnn(x, h)
    num_directions = 2 if bidirectional else 1
    if batch_first:
        assert_tuple_equal(
            output.shape, (batch_size, seq_len, num_directions * hidden_size)
        )
    else:
        assert_tuple_equal(
            output.shape, (seq_len, batch_size, num_directions * hidden_size)
        )
    assert_tuple_equal(
        h_n[0].shape, (num_directions * num_layers, batch_size, hidden_size)
    )
    assert_tuple_equal(
        h_n[1].shape, (num_directions * num_layers, batch_size, hidden_size)
    )
