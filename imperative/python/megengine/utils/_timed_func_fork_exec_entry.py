# -*- coding: utf-8 -*-
import argparse
import os
import sys

from megengine.core._imperative_rt.utils import _timed_func_exec_cb

try:
    from setproctitle import setproctitle
except ImportError:
    setproctitle = None


def main():
    parser = argparse.ArgumentParser(
        description="entry point for fork-exec callback in TimedFuncInvoker;"
        " this file should not be used directly by normal user."
    )
    parser.add_argument("user_data")
    args = parser.parse_args()

    if setproctitle:
        setproctitle("megbrain:timed_func_exec:ppid={}".format(os.getppid()))
    _timed_func_exec_cb(args.user_data)
    raise SystemError("_timed_func_exec_cb returned")


if __name__ == "__main__":
    main()
