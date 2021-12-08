# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import pytest

import megengine.functional as F
from megengine import tensor


def test_cross_entropy_with_logits():
    data = tensor([[0, 50], [0, -150]]).astype(np.float32)
    label = tensor([1, 0]).astype(np.int32)
    loss = F.nn.cross_entropy(data, label)
    np.testing.assert_allclose(loss.numpy(), 0.0)
    label = tensor([0, 1]).astype(np.int32)
    loss = F.nn.cross_entropy(data, label)
    np.testing.assert_allclose(loss.numpy(), 100)

    label = np.array([1, 0])
    loss = F.nn.cross_entropy(data, label)
    np.testing.assert_allclose(loss.numpy(), 0.0)


def test_cross_entropy():
    def softmax(x):
        x = np.exp(x)
        x /= x.sum(1, keepdims=True)
        return x

    def ref(x, y):
        return np.mean([-np.log(x[i, y[i]]) for i in range(len(y))])

    x = (np.random.rand(5, 10) - 0.5) * 4
    y = np.random.randint(10, size=(5,))
    for i in range(len(x)):
        x[i, y[i]] += np.random.rand() * 2
    x = softmax(x)
    l_ref = ref(x, y)
    l = F.nn.cross_entropy(tensor(x, "float32"), tensor(y, "int32"), with_logits=False)
    np.testing.assert_allclose(l.numpy(), l_ref, 1e-6, 1e-6)


def test_cross_entropy_reduction():
    logits = np.random.randn(16, 10)
    label = np.random.randint(10, size=[16])
    logits = tensor(logits, dtype="float32")
    label = tensor(label, dtype="int32")

    perm = np.random.permutation(16)
    logits_perm = tensor(logits[perm], dtype="float32")
    label_perm = tensor(label[perm], dtype="int32")

    loss = F.nn.cross_entropy(logits, label, reduction="none")
    loss_perm = F.nn.cross_entropy(logits_perm, label_perm, reduction="none")
    np.testing.assert_allclose(loss.numpy()[perm], loss_perm.numpy())

    loss_sum = F.nn.cross_entropy(logits, label, reduction="sum")
    np.testing.assert_allclose(loss.numpy().sum(), loss_sum.numpy(), rtol=2e-7)

    loss_mean = F.nn.cross_entropy(logits, label, reduction="mean")
    np.testing.assert_allclose(loss_mean.numpy(), loss_sum.numpy() / 16)

    loss_ls = F.nn.cross_entropy(logits, label, reduction="mean", label_smooth=0.1)
    loss_ls_none_reduce = F.nn.cross_entropy(
        logits, label, reduction="none", label_smooth=0.1
    )
    np.testing.assert_allclose(
        loss_ls.numpy(), loss_ls_none_reduce.numpy().mean(), rtol=2e-7
    )

    with pytest.raises(ValueError):
        F.nn.cross_entropy(logits, label, reduction="MEAN")

    with pytest.raises(ValueError):
        F.nn.cross_entropy(logits, label, reduction="max")


def ctc_nll_naive_npy(
    pred,
    pred_lengths,
    label,
    label_lengths,
    blank=0,
    reduction="mean",
    time_major=False,
):
    """naive :func:`ctc_nll` using numpy arrays. Used for testing and helping
    our user to understand how CTC works. Only ``LABEL_COMPACT`` mode is
    supported."""

    pred = np.asarray(pred, dtype=np.float32)
    pred_lengths = np.asarray(pred_lengths, dtype=np.int8)
    label = np.asarray(label, dtype=np.int32)
    label_lengths = np.asarray(label_lengths, dtype=np.int32)

    if time_major:
        pred = np.transpose(pred, (1, 0, 2))
    # pred in (N, T, P) format

    batch_size, time_len, nr_class = pred.shape
    assert pred_lengths.shape == (batch_size,) and pred_lengths.max() <= pred.shape[1]
    assert label_lengths.shape == (batch_size,)
    assert label.shape == (label_lengths.sum(),) and label.max() < nr_class

    ret = np.empty((batch_size,), dtype=np.float32)
    label_start = 0
    for i in range(batch_size):
        label_end = label_start + label_lengths[i]
        ret[i] = _ctc_npy_single_seq(
            pred[i][: pred_lengths[i]], label[label_start:label_end], blank
        )
        label_start = label_end

    if reduction == "mean":
        return (ret / label_lengths).mean()
    elif reduction == "sum":
        return ret.sum()
    elif reduction == "none":
        return ret
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))


def _ctc_npy_single_seq(pred, label, blank):
    def safelog(x):
        eps = np.finfo(x.dtype).tiny
        return np.log(np.maximum(x, eps))

    def log_sum_exp(x, y):
        x, y = np.maximum(x, y), np.minimum(x, y)
        return x + np.log1p(np.exp(y - x))

    assert np.abs(pred.sum(axis=1) - 1).max() <= 1e-3
    len_pred, alphabet_size = pred.shape
    (len_label,) = label.shape

    len_ex_label = len_label * 2 + 1
    ex_label = (np.zeros(len_ex_label)).astype(np.int32) + blank
    ex_label[1::2] = label

    prob = np.zeros(len_ex_label, dtype=np.float32)
    prob[0] = pred[0][ex_label[0]]
    prob[1] = pred[0][ex_label[1]]
    prob = safelog(prob)  # compute on log scale

    ex_label_pmask = ex_label[2:] != ex_label[:-2]
    for t in range(1, len_pred):
        # enter loop: prob[i] = log(p(pred[:t+1], label[:i+1]))
        new_prob = prob.copy()
        new_prob[1:] = log_sum_exp(new_prob[1:], prob[:-1])
        new_prob[2:] = (
            new_prob[2:] * (1 - ex_label_pmask)
            + log_sum_exp(new_prob[2:], prob[:-2]) * ex_label_pmask
        )
        new_prob += safelog(pred[t, ex_label])
        prob = new_prob

    return -log_sum_exp(prob[-1], prob[-2])


def test_ctc_loss():
    def test_func(T, C, N):
        input = np.random.randn(T, N, C)
        input = F.softmax(tensor(input), axis=-1).numpy()
        input_lengths = np.ones(N, dtype=np.int32) * T
        target_lengths = np.random.randint(low=1, high=T + 1, size=(N,), dtype=np.int32)
        target = np.random.randint(
            low=1, high=C, size=(sum(target_lengths)), dtype=np.int32
        )

        input_mge = tensor(input)
        input_lengths_mge = tensor(input_lengths)

        target_mge = tensor(target)
        target_lengths_mge = tensor(target_lengths)

        blank = np.random.randint(C)
        for method in ["mean", "sum", "none"]:
            np_out = ctc_nll_naive_npy(
                input,
                input_lengths,
                target,
                target_lengths,
                blank=blank,
                reduction=method,
                time_major=True,
            )
            mge_out = F.nn.ctc_loss(
                input_mge,
                input_lengths_mge,
                target_mge,
                target_lengths_mge,
                blank=blank,
                reduction=method,
            )
            np.testing.assert_allclose(mge_out.numpy(), np_out, rtol=2e-6)

    cases = [[1, 2, 1], [100, 50, 200], [100, 5, 1]]
    for case in cases:
        test_func(*case)
