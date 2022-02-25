# -*- coding: utf-8 -*-
import platform
import sys
import threading

# Windows do not imp resource package
if platform.system() != "Windows":
    import resource


class AlternativeRecursionLimit:
    r"""A reentrant context manager for setting global recursion limits."""

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
            if platform.system() != "Windows":
                (
                    self.orig_rlim_stack_soft,
                    self.orig_rlim_stack_hard,
                ) = resource.getrlimit(resource.RLIMIT_STACK)
                # FIXME: https://bugs.python.org/issue34602, python3 release version
                # on Macos always have this issue, not all user install python3 from src
                try:
                    resource.setrlimit(
                        resource.RLIMIT_STACK,
                        (self.orig_rlim_stack_hard, self.orig_rlim_stack_hard),
                    )
                except ValueError as exc:
                    if platform.system() != "Darwin":
                        raise exc

            # increase recursion limit
            sys.setrecursionlimit(self.new_py_limit)
            self.count += 1

    def __exit__(self, type, value, traceback):
        with self.lock:
            self.count -= 1
            if self.count == 0:
                sys.setrecursionlimit(self.orig_py_limit)

            if platform.system() != "Windows":
                try:
                    resource.setrlimit(
                        resource.RLIMIT_STACK,
                        (self.orig_rlim_stack_soft, self.orig_rlim_stack_hard),
                    )
                except ValueError as exc:
                    if platform.system() != "Darwin":
                        raise exc


_max_recursion_limit_context_manager = AlternativeRecursionLimit(2 ** 31 - 1)


def max_recursion_limit():
    r"""Sets recursion limit to the max possible value."""
    return _max_recursion_limit_context_manager
