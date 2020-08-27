# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
