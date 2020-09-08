# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import threading


class Future:
    def __init__(self, ack=True):
        self.ready = threading.Event()
        self.ack = threading.Event() if ack else None

    def set(self, value):
        self.value = value
        self.ready.set()
        if self.ack:
            self.ack.wait()

    def get(self):
        self.ready.wait()
        if self.ack:
            self.ack.set()
        return self.value
