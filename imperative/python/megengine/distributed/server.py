# -*- coding: utf-8 -*-
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


class Methods:
    r"""Distributed Server Method.
    Used for exchange information between distributed nodes.

    Args:
        mm_server_port: multiple machine rpc server port.
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
        self.bcast_dict = {}

    def connect(self):
        r"""Method for checking connection success."""
        return True

    def get_mm_server_port(self):
        r"""Get multiple machine rpc server port."""
        return self.mm_server_port

    def set_is_grad(self, key, is_grad):
        r"""Mark send/recv need gradiants by key.

        Args:
            key: key to match send/recv op.
            is_grad: whether this op need grad.
        """
        with self.lock:
            future = self.dict_is_grad[key]
        future.set(is_grad)
        return True

    def check_is_grad(self, key):
        r"""Check whether send/recv need gradiants.

        Args:
            key: key to match send/recv op.
        """
        with self.lock:
            future = self.dict_is_grad[key]
        ret = future.get()
        with self.lock:
            del self.dict_is_grad[key]
        return ret

    def set_remote_tracer(self, key, tracer_set):
        r"""Set tracer dict for tracing send/recv op.

        Args:
            key: key to match send/recv op.
            tracer_set: valid tracer set.
        """
        with self.lock:
            future = self.dict_remote_tracer[key]
        future.set(tracer_set)
        return True

    def check_remote_tracer(self, key):
        r"""Get tracer dict for send/recv op.

        Args:
            key: key to match send/recv op.
        """
        with self.lock:
            future = self.dict_remote_tracer[key]
        ret = future.get()
        with self.lock:
            del self.dict_remote_tracer[key]
        return ret

    def group_barrier(self, key, size):
        r"""A barrier wait for all group member.

        Args:
            key: group key to match each other.
            size: group size.
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
        r"""Set user defined key-value pairs across processes."""
        with self.lock:
            future = self.user_dict[key]
        future.set(val)
        return True

    def user_get(self, key):
        r"""Get user defined key-value pairs across processes."""
        with self.lock:
            future = self.user_dict[key]
        return future.get()

    def bcast_val(self, val, key, size):
        with self.lock:
            if key not in self.bcast_dict:
                self.bcast_dict[key] = [Future(False), size]
            arr = self.bcast_dict[key]
        if val is not None:
            arr[0].set(val)
            val = None
        else:
            val = arr[0].get()
        with self.lock:
            cnt = arr[1] - 1
            arr[1] = cnt
            if cnt == 0:
                del self.bcast_dict[key]
        return val

    def _del(self, key):
        with self.lock:
            del self.user_dict[key]

    # thread safe function
    def user_pop(self, key):
        ret = self.user_get(key)
        self._del(key)
        return ret


class ThreadXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


def _start_server(py_server_port, queue):
    r"""Start python distributed server and multiple machine server.

    Args:
        py_server_port: python server port.
        mm_server_port: multiple machine server port.
        queue: server port will put in this queue, puts exception when process fails.
    """
    try:
        mm_server_port = create_mm_server("0.0.0.0", 0)
        server = ThreadXMLRPCServer(
            ("0.0.0.0", py_server_port), logRequests=False, allow_none=True
        )
        server.register_instance(Methods(mm_server_port))
        _, py_server_port = server.server_address
        queue.put((py_server_port, mm_server_port))
        server.serve_forever()
    except Exception as e:
        queue.put(e)


class Server:
    r"""Distributed Server for distributed training.
    Should be running at master node.

    Args:
        port: python server port.
    """

    def __init__(self, port=0):
        q = mp.Queue()
        self.proc = mp.Process(target=_start_server, args=(port, q), daemon=True)
        self.proc.start()
        ret = q.get()
        if isinstance(ret, Exception):
            raise ret
        else:
            self.py_server_port, self.mm_server_port = ret

    def __del__(self):
        self.proc.terminate()


class Client:
    r"""Distributed Client for distributed training.

    Args:
        master_ip: ip address of master node.
        port: port of server at master node.
    """

    def __init__(self, master_ip, port):
        self.master_ip = master_ip
        self.port = port
        self.connect()
        self.bcast_dict = defaultdict(lambda: 0)

    def connect(self):
        r"""Check connection success."""
        while True:
            try:
                self.proxy = ServerProxy(
                    "http://{}:{}".format(self.master_ip, self.port), allow_none=True
                )
                if self.proxy.connect():
                    break
            except:
                time.sleep(1)

    def get_mm_server_port(self):
        r"""Get multiple machine server port."""
        while True:
            try:
                return self.proxy.get_mm_server_port()
            except:
                time.sleep(0.5)

    def set_is_grad(self, key, is_grad):
        r"""Mark send/recv need gradiants by key.

        Args:
            key: key to match send/recv op.
            is_grad: whether this op need grad.
        """
        self.proxy.set_is_grad(key, is_grad)

    def check_is_grad(self, key):
        r"""Check whether send/recv need gradiants.

        Args:
            key: key to match send/recv op.
        """
        return self.proxy.check_is_grad(key)

    def set_remote_tracer(self, key, tracer_set):
        r"""Set tracer dict for tracing send/recv op.

        Args:
            key: key to match send/recv op.
            tracer_set: valid tracer set.
        """
        self.proxy.set_remote_tracer(key, tracer_set)

    def check_remote_tracer(self, key):
        r"""Get tracer dict for send/recv op.

        Args:
            key: key to match send/recv op.
        """
        return self.proxy.check_remote_tracer(key)

    def group_barrier(self, key, size):
        r"""A barrier wait for all group member.

        Args:
            key: group key to match each other.
            size: group size.
        """
        # FIXME: group_barrier is not idempotent
        while True:
            try:
                self.proxy.group_barrier(key, size)
                return
            except:
                time.sleep(0.5)

    def user_set(self, key, val):
        r"""Set user defined key-value pairs across processes."""
        return self.proxy.user_set(key, val)

    def user_get(self, key):
        r"""Get user defined key-value pairs across processes."""
        return self.proxy.user_get(key)

    def user_pop(self, key):
        r"""Get user defined key-value pairs and delete the resources when the get is done"""
        return self.proxy.user_pop(key)

    def bcast_val(self, val, key, size):
        idx = self.bcast_dict[key] + 1
        self.bcast_dict[key] = idx
        key = key + "_bcast_" + str(idx)
        return self.proxy.bcast_val(val, key, size)


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
