# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Optional, Union

import megengine._internal as mgb
from megengine._internal.opr_param_defs import CollectiveComm as CollParam

from .util import (
    get_backend,
    get_group_id,
    get_master_ip,
    get_master_port,
    get_rank,
    get_world_size,
)


def collective_comm_symvar(
    inp: Union[mgb.SymbolVar, mgb.CompGraph],
    key: Optional[str] = None,
    op: CollParam.Mode = None,
    nr_ranks: Optional[int] = None,
    is_root: Optional[bool] = None,
    rank: Optional[int] = None,
    local_grad: Optional[bool] = False,
    dtype: Optional[type] = None,
    device: Optional[mgb.CompNode] = None,
    comp_graph: Optional[mgb.CompGraph] = None,
) -> mgb.SymbolVar:
    """Helper function for creating collective_comm operators

    :param inp: tensor or comp_graph
    :param key: unique identifier for collective communication
    :param op: mode of collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param is_root: whether this node is root node
    :param rank: rank of this node
    :param local_grad: whether use local grad
    :param dtype: output data type, use dtype of inp as default
    :param device: output comp node, use comp node of inp as default
    :param comp_graph: output comp graph, use comp graph of inp as default
    """
    return mgb.opr.collective_comm(
        inp,
        key=key if key is not None else ("collective_comm_" + str(get_group_id())),
        nr_devices=nr_ranks if nr_ranks is not None else get_world_size(),
        is_root=is_root if is_root is not None else (get_rank() == 0),
        rank=rank if rank is not None else get_rank(),
        local_grad=local_grad,
        server_addr=get_master_ip(),
        port=get_master_port(),
        param=CollParam(mode=op),
        dtype=dtype,
        backend=get_backend(),
        comp_node=device,
        comp_graph=comp_graph,
    )
