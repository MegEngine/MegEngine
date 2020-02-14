# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
from typing import Iterable, Optional, Union

import megengine._internal as mgb

from ..core.graph import get_default_graph
from ..core.tensor import Tensor, wrap_io_tensor
from ..jit import barrier, mark_impure


@wrap_io_tensor
def grad(
    target: Tensor,
    wrt: Union[Tensor, Iterable[Tensor]],
    warn_mid_wrt: bool = True,
    use_virtual_grad: bool = None,
    return_zero_for_nodep: bool = True,
) -> Union[Tensor, Iterable[Optional[Tensor]], None]:
    r"""compute symbolic grad

    :param target: grad target var
    :param wrt: with respect to which to compute the grad
    :param warn_mid_wrt: whether to give warning if ``wrt`` is not endpoint
    :param use_virtual_grad: whether to use virtual grad opr, so fwd graph can
        be optimized before applying grad; if ``None`` is given, then virtual
        grad would be used if ``graph_opt_level >= 2``
    :param return_zero_for_nodep: if ``target`` does not depend on ``wrt``, set to True to return
        a zero-valued :class:`~.Tensor` rather than ``None``; can't be set to False when using
        virtual grad opr.
    :return: :math:`\partial\text{target} / \partial\text{wrt}`
    """
    if not isinstance(wrt, mgb.SymbolVar):
        assert isinstance(wrt, collections.Iterable)
        wrt = [w._symvar for w in wrt]

    return mgb.grad(target, wrt, warn_mid_wrt, use_virtual_grad, return_zero_for_nodep)


_add_update_cache = {}  # type: dict

_dummy = mgb.SharedScalar(0)


def add_update(
    dest: Tensor,
    delta: Tensor,
    *,
    alpha: Union[Tensor, float, int] = 1.0,
    beta: Union[Tensor, float, int] = 1.0,
    bias: Union[Tensor, float, int] = 0.0
):
    r"""Inplace modify ``dest`` as follows:

    .. math::
        dest = alpha * dest +  beta * delta + bias

    :param dest: input data that will be inplace modified.
    :param delta: update value that will be added to ``dest``.
    :param alpha: weight ratio of ``dest``. Default: 1.0
    :param beta: weight ratio of ``delta``. Default: 1.0
    :param bias: bias value appended to the result. Default: 0.0
    """

    if isinstance(beta, Tensor) or isinstance(alpha, Tensor):
        delta *= beta
        beta = 1.0
    if isinstance(alpha, Tensor):
        delta += (alpha - 1.0) * dest
        alpha = 1.0
    if isinstance(bias, Tensor):
        delta += bias
        bias = 0.0

    comp_graph = dest._comp_graph or get_default_graph()
    comp_node = dest._comp_node

    if not isinstance(delta, Tensor):
        _delta = mgb.make_immutable(
            value=delta, comp_node=comp_node, comp_graph=comp_graph
        )
    else:
        _delta = delta._attach(comp_graph)

    _dest = dest._attach(comp_graph)

    # use (dest, delta) as the key, so we could not add the same delta to dest in static graph
    key = (comp_graph._id(), _dest.id, _delta.id)
    if key in _add_update_cache:
        _alpha, _beta, _bias, config = _add_update_cache[key]
        mgb.mgb._mgb.SharedScalar__set(_alpha, alpha)
        mgb.mgb._mgb.SharedScalar__set(_beta, beta)
        mgb.mgb._mgb.SharedScalar__set(_bias, bias)
    else:
        _alpha = mgb.SharedScalar(alpha)
        _beta = mgb.SharedScalar(beta)
        _bias = mgb.SharedScalar(bias)
        config = mgb.helper.gen_config(None, comp_node, None)
        _add_update_cache[key] = (_alpha, _beta, _bias, config)

    u = mgb.mgb._Opr.add_update(
        _dest, barrier(_delta), _alpha, _beta, _bias, _dummy, config
    )
    mark_impure(u)

    return Tensor(u)


@wrap_io_tensor
def add_extra_vardep(oup: Tensor, dep: Tensor):
    r"""Explicitly set the dependency that tensor ``oup`` depends on tensor ``dep``.
    """
    return mgb.config.add_extra_vardep(oup, dep)
