#! /usr/bin/env python3
import argparse
import getopt
import os
import runpy
import sys

from megengine.logger import get_logger
from megengine.utils.profiler import Profiler, merge_trace_events


def main():
    parser = argparse.ArgumentParser(
        prog="megengine.tools.profiler", description="Profiling megengine program"
    )
    parser.add_argument(
        "-m", "--module", action="store_true", help="whether launch program as module"
    )
    parser.add_argument("-o", "--output", type=str, help="output file location")
    parser.add_argument(
        "-f",
        "--format",
        action="append",
        type=str,
        help="output file format",
        choices=Profiler.valid_formats,
    )
    parser.add_argument(
        "--merge_trace_events", action="store_true",
    )
    parser.add_argument(
        "--clean", action="store_true",
    )
    for opt in Profiler.valid_options:
        parser.add_argument("--" + opt, type=int, default=None)
    args, extras = parser.parse_known_args(sys.argv[1:])
    prof_args = {}
    for opt in Profiler.valid_options:
        optval = getattr(args, opt, None)
        if optval is not None:
            prof_args[opt] = optval

    if args.output is not None:
        prof_args["path"] = args.output

    if args.format:
        prof_args["formats"] = args.format

    if len(extras) == 0:
        if not args.merge_trace_events:
            parser.print_usage()
            exit(1)
    else:
        filename = extras[0]
        if not args.module:
            if not os.path.exists(filename):
                get_logger().fatal("cannot find file {}".format(filename))
                exit(1)
            filename = os.path.realpath(filename)
            # Replace profiler's dir with script's dir in front of module search path.
            sys.path[0] = os.path.dirname(filename)

        sys.argv[:] = [filename, *extras[1:]]

        profiler = Profiler(**prof_args)

        if args.clean:
            for file in os.listdir(profiler.directory):
                os.remove(os.path.join(profiler.directory, file))

        with profiler:
            if args.module:
                run_module(filename)
            else:
                run_script(filename)
        profiler.dump()

    if args.merge_trace_events:
        merge_trace_events(profiler.directory)


def run_module(modulename):
    runpy.run_module(modulename, None, "__main__", True)


def run_script(filename):
    runpy.run_path(filename, None, "__main__")


if __name__ == "__main__":
    main()
