#!/usr/bin/env python3
# This file is part of MegBrain.
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import argparse
import os
import re
import subprocess
import tempfile
from functools import partial
from multiprocessing import Manager

from tqdm.contrib.concurrent import process_map

# change workspace to MegBrain root dir
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

failed_files = Manager().list()


def process_file(file, clang_format, write):
    original_source = open(file, "r").read()
    source = original_source
    source = re.sub(
        r"MGB_DEFINE(?P<r>([^\\]|\n)*?)// *{", r"class MGB_DEFINE\g<r>{", source
    )
    source, count = re.subn(
        r"(?<!#define )MGB_DEFINE(.*) +\\", r"class MGB_DEFINE\1{\\", source
    )

    result = subprocess.check_output(
        [
            clang_format,
            "-style=file",
            "-verbose",
            "-assume-filename={}".format(file),
            # file,
        ],
        input=bytes(source.encode("utf-8")),
    )

    result = result.decode("utf-8")
    if count:
        result = re.sub(
            r"class MGB_DEFINE(.*){( *)\\", r"MGB_DEFINE\1\2       \\", result
        )
    result = re.sub(r"class MGB_DEFINE((.|\n)*?){", r"MGB_DEFINE\1// {", result)

    if write and original_source != result:
        with tempfile.NamedTemporaryFile(
            dir=os.path.dirname(file), delete=False
        ) as tmp_file:
            tmp_file.write(result.encode("utf-8"))
        os.rename(tmp_file.name, file)
    else:
        ret_code = subprocess.run(
            ["diff", "--color=always", file, "-"], input=bytes(result.encode("utf-8")),
        ).returncode

        # man diff: 0 for same, 1 for different, 2 if trouble.
        if ret_code == 2:
            raise RuntimeError("format process (without overwrite) failed")
        if ret_code != 0:
            print(file)
            global failed_files
            failed_files.append(file)


def main():
    parser = argparse.ArgumentParser(
        description="Format source files using clang-format, eg: `./tools/format.py src -w`. \
        Require clang-format version == 12.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "path", nargs="+", help="file name or path based on MegBrain root dir."
    )
    parser.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="use formatted file to replace original file.",
    )
    parser.add_argument(
        "--clang-format",
        default=os.getenv("CLANG_FORMAT", "clang-format"),
        help="clang-format executable name; it can also be "
        "modified via the CLANG_FORMAT environment var",
    )
    args = parser.parse_args()

    format_type = [".cpp", ".c", ".h", ".cu", ".cuh", ".inl"]

    def getfiles(path):
        rst = []
        for p in os.listdir(path):
            p = os.path.join(path, p)
            if os.path.isdir(p):
                rst += getfiles(p)
            elif (
                os.path.isfile(p)
                and not os.path.islink(p)
                and os.path.splitext(p)[1] in format_type
            ):
                rst.append(p)
        return rst

    files = []
    for path in args.path:
        if os.path.isdir(path):
            files += getfiles(path)
        elif os.path.isfile(path):
            files.append(path)
        else:
            raise ValueError("Invalid path {}".format(path))

    # check version, we only support 12.0.1 now
    version = subprocess.check_output([args.clang_format, "--version",],)
    version = version.decode("utf-8")

    need_version = "12.0.1"
    if version.find(need_version) < 0:
        print(
            "We only support {} now, please install {} version, find version: {}".format(
                need_version, need_version, version
            )
        )
        raise RuntimeError("clang-format version not equal {}".format(need_version))

    process_map(
        partial(process_file, clang_format=args.clang_format, write=args.write,),
        files,
        chunksize=10,
    )

    if failed_files:
        raise RuntimeError("above files are not properly formatted!")


if __name__ == "__main__":
    main()
