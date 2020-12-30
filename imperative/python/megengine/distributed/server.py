# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import threading
import time
from collections import defaultdict
from functools import partial
from queue import Queue
from socketserver import ThreadingMixIn
from xmlrpc.client import ServerProxy
from xmlrpc.server import SimpleXMLRPCServer

from ..core._imperative_rt.utils import create_mm_server
from ..utils.future import Future
from .util import get_free_ports


class Methods:
    """
    Distributed Server Method.
    Used for exchange information between distributed nodes.

    :param mm_server_port: multiple machine rpc server port.
    """

    def __init__(self, mm_server_port):
        self.lock = threading.Lock()
        self.mm_server_port = mm_server_port
        self.dict_is_grad = defaultdict(partial(Future, True))
        self.dict_remote_tracer = defaultdict(partial(Future, True))
        self.dict_pack_list = defaultdict(partial(Future, False))
        self.dict_barrier_counter = defaultdict(int)
        self.dict_barrier_event = defaultdict(threading.Event)
        self.user_dict = defaultdict(partial(Future, False))

    def connect(self):
        """Method for checking connection success."""
        return True

    def get_mm_server_port(self):
        """Get multiple machine rpc server port."""
        return self.mm_server_port

    def set_is_grad(self, key, is_grad):
        """
        Mark send/recv need gradiants by key.

        :param key: key to match send/recv op.
        :param is_grad: whether this op need grad.
        """
        with self.lock:
            future = self.dict_is_grad[key]
        future.set(is_grad)
        return True

    def check_is_grad(self, key):
        """
        Check whether send/recv need gradiants.

        :param key: key to match send/recv op.
        """
        with self.lock:
            future = self.dict_is_grad[key]
        ret = future.get()
        with self.lock:
            del self.dict_is_grad[key]
        return ret

    def set_remote_tracer(self, key, tracer_set):
        """
        Set tracer dict for tracing send/recv op.

        :param key: key to match send/recv op.
        :param tracer_set: valid tracer set.
        """
        with self.lock:
            future = self.dict_remote_tracer[key]
        future.set(tracer_set)
        return True

    def check_remote_tracer(self, key):
        """
        Get tracer dict for send/recv op.

        :param key: key to match send/recv op.
        """
        with self.lock:
            future = self.dict_remote_tracer[key]
        ret = future.get()
        with self.lock:
            del self.dict_remote_tracer[key]
        return ret

    def group_barrier(self, key, size):
        """
        A barrier wait for all group member.

        :param key: group key to match each other.
        :param size: group size.
        """
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

    def user_set(self, key, val):
        """Set user defined key-value pairs across processes."""
        with self.lock:
            future = self.user_dict[key]
        future.set(val)
        return True

    def user_get(self, key):
        """Get user defined key-value pairs across processes."""
        with self.lock:
            future = self.user_dict[key]
        return future.get()


class ThreadXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


def _start_server(py_server_port, mm_server_port, queue):
    """
    Start python distributed server and multiple machine server.

    :param py_server_port: python server port.
    :param mm_server_port: multiple machine server port.
    :param queue: server port will put in this queue, puts exception when process fails.
    """
    try:
        server = ThreadXMLRPCServer(("0.0.0.0", py_server_port), logRequests=False)
        server.register_instance(Methods(mm_server_port))
        _, port = server.server_address
        queue.put(port)
        server.serve_forever()
    except Exception as e:
        queue.put(e)


class Server:
    """
    Distributed Server for distributed training.
    Should be running at master node.

    :param port: python server port.
    """

    def __init__(self, port=0):
        self.mm_server_port = create_mm_server("0.0.0.0", 0)
        q = Queue()
        self.proc = threading.Thread(
            target=_start_server, args=(port, self.mm_server_port, q), daemon=True,
        )
        self.proc.start()
        ret = q.get()
        if isinstance(ret, Exception):
            raise ret
        else:
            self.py_server_port = ret


class Client:
    """
    Distributed Client for distributed training.

    :param master_ip: ip address of master node.
    :param port: port of server at master node.
    """

    def __init__(self, master_ip, port):
        self.master_ip = master_ip
        self.port = port
        self.connect()

    def connect(self):
        """Check connection success."""
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
        """Get multiple machine server port."""
        return self.proxy.get_mm_server_port()

    def set_is_grad(self, key, is_grad):
        """
        Mark send/recv need gradiants by key.

        :param key: key to match send/recv op.
        :param is_grad: whether this op need grad.
        """
        self.proxy.set_is_grad(key, is_grad)

    def check_is_grad(self, key):
        """
        Check whether send/recv need gradiants.

        :param key: key to match send/recv op.
        """
        return self.proxy.check_is_grad(key)

    def set_remote_tracer(self, key, tracer_set):
        """
        Set tracer dict for tracing send/recv op.

        :param key: key to match send/recv op.
        :param tracer_set: valid tracer set.
        """
        self.proxy.set_remote_tracer(key, tracer_set)

    def check_remote_tracer(self, key):
        """
        Get tracer dict for send/recv op.

        :param key: key to match send/recv op.
        """
        return self.proxy.check_remote_tracer(key)

    def group_barrier(self, key, size):
        """
        A barrier wait for all group member.

        :param key: group key to match each other.
        :param size: group size.
        """
        self.proxy.group_barrier(key, size)

    def user_set(self, key, val):
        """Set user defined key-value pairs across processes."""
        self.proxy.user_set(key, val)

    def user_get(self, key):
        """Get user defined key-value pairs across processes."""
        return self.proxy.user_get(key)


def main(port=0, verbose=True):
    mm_server_port = create_mm_server("0.0.0.0", 0)
    server = ThreadXMLRPCServer(("0.0.0.0", port), logRequests=verbose)
    server.register_instance(Methods(mm_server_port))
    _, port = server.server_address
    print("serving on port", port)
    server.serve_forever()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", type=int, default=0)
    ap.add_argument("-v", "--verbose", type=bool, default=True)
    args = ap.parse_args()
    main(port=args.port, verbose=args.verbose)
