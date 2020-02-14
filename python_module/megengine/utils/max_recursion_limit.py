# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import resource
import sys
import threading


class AlternativeRecursionLimit:
    r"""A reentrant context manager for setting global recursion limits.
    """

    def __init__(self, new_py_limit):
        self.new_py_limit = new_py_limit
        self.count = 0
        self.lock = threading.Lock()

        self.orig_py_limit = 0
        self.orig_rlim_stack_soft = 0
        self.orig_rlim_stack_hard = 0

    def __enter__(self):
        with self.lock:
            if self.count == 0:
                self.orig_py_limit = sys.getrecursionlimit()
                (
                    self.orig_rlim_stack_soft,
                    self.orig_rlim_stack_hard,
                ) = resource.getrlimit(resource.RLIMIT_STACK)
                resource.setrlimit(
                    resource.RLIMIT_STACK,
                    (self.orig_rlim_stack_hard, self.orig_rlim_stack_hard),
                )
                # increase recursion limit
                sys.setrecursionlimit(self.new_py_limit)
            self.count += 1

    def __exit__(self, type, value, traceback):
        with self.lock:
            self.count -= 1
            if self.count == 0:
                sys.setrecursionlimit(self.orig_py_limit)
                resource.setrlimit(
                    resource.RLIMIT_STACK,
                    (self.orig_rlim_stack_soft, self.orig_rlim_stack_hard),
                )


_max_recursion_limit_context_manager = AlternativeRecursionLimit(2 ** 31 - 1)


def max_recursion_limit():
    r"""set recursion limit to max possible value
    """
    return _max_recursion_limit_context_manager
