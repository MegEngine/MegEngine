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

from .util import get_backend, get_master_ip, get_master_port, get_rank, get_world_size


def collective_comm_symvar(
    inp: Union[mgb.SymbolVar, mgb.CompGraph],
    key: str,
    op: CollParam.Mode,
    nr_ranks: Optional[int] = None,
    rank: Optional[int] = None,
    root: Optional[int] = 0,
    dtype: Optional[type] = None,
    device: Optional[mgb.CompNode] = None,
    comp_graph: Optional[mgb.CompGraph] = None,
) -> mgb.SymbolVar:
    """Helper function for creating collective_comm operators

    :param inp: tensor or comp_graph
    :param key: unique identifier for collective communication
    :param op: mode of collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param rank: rank of the current process, use util.get_rank() as default
    :param root: rank of root node, use 0 as default
    :param dtype: output data type, use dtype of inp as default
    :param device: output comp node, use comp node of inp as default
    :param comp_graph: output comp graph, use comp graph of inp as default
    """
    return mgb.opr.collective_comm(
        inp,
        key=str(key),
        nr_devices=nr_ranks if nr_ranks is not None else get_world_size(),
        rank=rank if rank is not None else get_rank(),
        root=root,
        server_addr=get_master_ip(),
        port=get_master_port(),
        param=CollParam(mode=op),
        dtype=dtype,
        backend=get_backend(),
        comp_node=device,
        comp_graph=comp_graph,
    )
