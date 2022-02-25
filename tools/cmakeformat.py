#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
from pathlib import Path

CMAKE_FILS_DIRS = [
    "test",
    "dnn",
    "tools",
    "sdk",
    "src",
    "imperative",
    "lite",
    "cmake",
    "toolchains",
]


def main():
    os.chdir(str(Path(__file__).resolve().parent.parent))
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--check", action="store_true", help="check model")
    parser.add_argument(
        "--cmake_files",
        nargs="+",
        default=None,
        dest="cmake_files",
        help="cmake files to format, please split with space",
    )
    args = parser.parse_args()

    handle_files = []
    if args.cmake_files:
        handle_files = args.cmake_files
        for cmake_file in handle_files:
            assert os.path.isfile(
                cmake_file
            ), "error input --cmake_files, can not find file: {}".format(cmake_file)
    else:
        handle_files.append("CMakeLists.txt")
        for cmake_file_dir in CMAKE_FILS_DIRS:
            assert os.path.isdir(
                cmake_file_dir
            ), "{} is not a directory, may config error for CMAKE_FILS_DIRS".format(
                cmake_file_dir
            )
            for cmake_file in [
                os.path.join(root, file)
                for root, dirs, files in os.walk(cmake_file_dir)
                for file in files
                if file.endswith("CMakeLists.txt") or file.endswith(".cmake")
            ]:
                print("find cmake_file: {}".format(cmake_file))
                assert os.path.isfile(cmake_file), "code issue happened!!"
                handle_files.append(cmake_file)

    for cmake_file in handle_files:
        handle_type = ["format", "--in-place"]
        if args.check:
            handle_type = ["check", "--check"]
        cmd = "cmake-format -c tools/cmake_format_config.json {} {}".format(
            handle_type[1], cmake_file
        )
        print("try {}: {} with command: {}".format(handle_type[0], cmake_file, cmd))
        try:
            subprocess.check_call(cmd, shell=True)
        except Exception as exc:
            print("run cmd {} failed".format(cmd))
            if args.check:
                print(
                    'please run: "python3 tools/cmakeformat.py" to format cmake files'
                )
            else:
                print("code issue happened!!, please FIXME!!")
            raise exc


if __name__ == "__main__":
    subprocess.check_call("python3 -m pip install cmakelang==0.6.13 --user", shell=True)
    main()
