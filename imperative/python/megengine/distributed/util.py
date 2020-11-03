# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
import socket
from typing import List


def get_free_ports(num: int) -> List[int]:
    """
    Get one or more free ports.
    """
    socks, ports = [], []
    for i in range(num):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        socks.append(sock)
        ports.append(sock.getsockname()[1])
    for sock in socks:
        sock.close()
    return ports
