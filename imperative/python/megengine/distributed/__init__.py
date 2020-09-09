# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from .group import (
    WORLD,
    get_backend,
    get_client,
    get_mm_server_addr,
    get_py_server_addr,
    get_rank,
    get_world_size,
    group_barrier,
    init_process_group,
    is_distributed,
    new_group,
)
from .helper import bcast_list_, make_allreduce_cb, synchronized
from .launcher import launcher
from .server import Client, Server
from .util import get_free_ports
