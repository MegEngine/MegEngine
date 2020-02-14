# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import binascii
import os
import queue
import subprocess
from multiprocessing import Queue

import pyarrow.plasma as plasma

MGE_PLASMA_MEMORY = int(os.environ.get("MGE_PLASMA_MEMORY", 4000000000))  # 4GB


class _PlasmaStoreManager:
    def __init__(self):
        self.socket_name = "/tmp/mge_plasma_{}".format(
            binascii.hexlify(os.urandom(8)).decode()
        )
        debug_flag = bool(os.environ.get("MGE_DATALOADER_PLASMA_DEBUG", 0))
        self.plasma_store = subprocess.Popen(
            ["plasma_store", "-s", self.socket_name, "-m", str(MGE_PLASMA_MEMORY),],
            stdout=None if debug_flag else subprocess.DEVNULL,
            stderr=None if debug_flag else subprocess.DEVNULL,
        )

    def __del__(self):
        if self.plasma_store and self.plasma_store.returncode is None:
            self.plasma_store.kill()


# Each process only need to start one plasma store, so we set it as a global variable.
# TODO: how to share between different processes?
MGE_PLASMA_STORE_MANAGER = _PlasmaStoreManager()


class PlasmaShmQueue:
    def __init__(self, maxsize: int = 0):
        r"""Use pyarrow in-memory plasma store to implement shared memory queue.

        Compared to native `multiprocess.Queue`, `PlasmaShmQueue` avoid pickle/unpickle
        and communication overhead, leading to better performance in multi-process
        application.

        :type maxsize: int
        :param maxsize: maximum size of the queue, `None` means no limit. (default: ``None``)
        """

        self.socket_name = MGE_PLASMA_STORE_MANAGER.socket_name

        # TODO: how to catch the exception happened in `plasma.connect`?
        self.client = None

        # Used to store the header for the data.(ObjectIDs)
        self.queue = Queue(maxsize)  # type: Queue

    def put(self, data, block=True, timeout=None):
        if self.client is None:
            self.client = plasma.connect(self.socket_name)
        try:
            object_id = self.client.put(data)
        except plasma.PlasmaStoreFull:
            raise RuntimeError("plasma store out of memory!")
        try:
            self.queue.put(object_id, block, timeout)
        except queue.Full:
            self.client.delete([object_id])
            raise queue.Full

    def get(self, block=True, timeout=None):
        if self.client is None:
            self.client = plasma.connect(self.socket_name)
        object_id = self.queue.get(block, timeout)
        if not self.client.contains(object_id):
            raise RuntimeError(
                "ObjectID: {} not found in plasma store".format(object_id)
            )
        data = self.client.get(object_id)
        self.client.delete([object_id])
        return data

    def qsize(self):
        return self.queue.qsize()

    def empty(self):
        return self.queue.empty()

    def join(self):
        self.queue.join()

    def disconnect_client(self):
        if self.client is not None:
            self.client.disconnect()

    def close(self):
        self.queue.close()
        self.disconnect_client()

    def cancel_join_thread(self):
        self.queue.cancel_join_thread()
