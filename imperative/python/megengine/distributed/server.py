# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import multiprocessing as mp
import threading
import time
from collections import defaultdict
from functools import partial
from socketserver import ThreadingMixIn
from xmlrpc.client import ServerProxy
from xmlrpc.server import SimpleXMLRPCServer

from ..core._imperative_rt.utils import create_mm_server
from ..utils.future import Future
from .util import get_free_ports


class Methods:
    def __init__(self, mm_server_port):
        self.lock = threading.Lock()
        self.mm_server_port = mm_server_port
        self.dict_is_grad = defaultdict(partial(Future, True))
        self.dict_remote_tracer = defaultdict(partial(Future, True))
        self.dict_pack_list = defaultdict(partial(Future, False))
        self.dict_barrier_counter = defaultdict(int)
        self.dict_barrier_event = defaultdict(threading.Event)

    def connect(self):
        return True

    def get_mm_server_port(self):
        return self.mm_server_port

    def set_is_grad(self, rank_peer, is_grad):
        with self.lock:
            future = self.dict_is_grad[rank_peer]
        future.set(is_grad)
        return True

    def check_is_grad(self, rank_peer):
        with self.lock:
            future = self.dict_is_grad[rank_peer]
        ret = future.get()
        with self.lock:
            del self.dict_is_grad[rank_peer]
        return ret

    def set_remote_tracer(self, rank_peer, tracer_set):
        with self.lock:
            future = self.dict_remote_tracer[rank_peer]
        future.set(tracer_set)
        return True

    def check_remote_tracer(self, rank_peer):
        with self.lock:
            future = self.dict_remote_tracer[rank_peer]
        ret = future.get()
        with self.lock:
            del self.dict_remote_tracer[rank_peer]
        return ret

    def set_pack_list(self, key, pack_list):
        with self.lock:
            future = self.dict_pack_list[key]
        future.set(pack_list)
        return True

    def get_pack_list(self, key):
        with self.lock:
            future = self.dict_pack_list[key]
        return future.get()

    def group_barrier(self, key, size):
        with self.lock:
            self.dict_barrier_counter[key] += 1
            counter = self.dict_barrier_counter[key]
            event = self.dict_barrier_event[key]
        if counter == size:
            del self.dict_barrier_counter[key]
            del self.dict_barrier_event[key]
            event.set()
        else:
            event.wait()
        return True


class ThreadXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


def start_server(py_server_port, mm_server_port):
    server = ThreadXMLRPCServer(("0.0.0.0", py_server_port), logRequests=False)
    server.register_instance(Methods(mm_server_port))
    server.serve_forever()


class Server:
    def __init__(self, port):
        self.py_server_port = get_free_ports(1)[0] if port == 0 else port
        self.mm_server_port = create_mm_server("0.0.0.0", 0)
        self.proc = mp.Process(
            target=start_server,
            args=(self.py_server_port, self.mm_server_port),
            daemon=True,
        )
        self.proc.start()


class Client:
    def __init__(self, master_ip, port):
        self.master_ip = master_ip
        self.port = port
        self.connect()

    def connect(self):
        while True:
            try:
                self.proxy = ServerProxy(
                    "http://{}:{}".format(self.master_ip, self.port)
                )
                if self.proxy.connect():
                    break
            except:
                time.sleep(1)

    def get_mm_server_port(self):
        return self.proxy.get_mm_server_port()

    def set_is_grad(self, rank_peer, is_grad):
        self.proxy.set_is_grad(rank_peer, is_grad)

    def check_is_grad(self, rank_peer):
        return self.proxy.check_is_grad(rank_peer)

    def set_remote_tracer(self, rank_peer, tracer_set):
        self.proxy.set_remote_tracer(rank_peer, tracer_set)

    def check_remote_tracer(self, rank_peer):
        return self.proxy.check_remote_tracer(rank_peer)

    def set_pack_list(self, key, pack_list):
        self.proxy.set_pack_list(key, pack_list)

    def get_pack_list(self, key):
        return self.proxy.get_pack_list(key)

    def group_barrier(self, key, size):
        self.proxy.group_barrier(key, size)
