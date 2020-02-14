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
import pathlib
import subprocess

from megengine.utils.profile_analyze import main as profiler

home = pathlib.Path(__file__).parent.absolute()
script_path = os.path.join(str(home), "resnet50_perf.py")
script_path = "python3 " + script_path

prof_path = "prof.json"

log_path = "log.txt"


def print_log(msg: str, log: str = log_path):
    print(msg)
    with open(log, "a") as f:
        print(msg, file=f)


def run_cmd(cmd: str, log: str = log_path) -> bool:
    stdout = subprocess.getoutput(cmd)
    token = "Wall time"
    gpu_msg = "GPU Usage"
    run_finished = False
    for line in stdout.split("\n"):
        if token in line:
            print(line)
            print_log("Run status: finished")
            run_finished = True
        if gpu_msg in line:
            print(line)
    if not run_finished:
        print_log("Run status: failed")
    with open(log, "a") as f:
        print(stdout, file=f)

    return run_finished


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ResNet50 train performance")
    parser.add_argument(
        "--run-debug-tool", action="store_true", help="run profiler and valgrind"
    )
    parser.add_argument(
        "--run-parallel", action="store_true", help="run data parallel performance"
    )
    parser.add_argument("--run-eager", action="store_false", help="run eager graph")
    args = parser.parse_args()

    f = open(log_path, "w")
    f.close()

    print_log("**************************************")
    print_log("Run ResNet 50 performance test with batch size = 64")

    print_log("**************************************")
    print_log("Run static graph with default opt level")
    cmd = script_path
    run_cmd(cmd)

    print_log("**************************************")
    print_log("Run static graph with conv fastrun")
    cmd = script_path + " --conv-fastrun=yes"
    run_cmd(cmd)

    print_log("**************************************")
    print_log("Run static graph with conv fastrun and JIT")
    cmd = script_path + " --conv-fastrun=yes --opt-level=3"
    run_cmd(cmd)

    print_log("**************************************")
    print_log("Run static graph with JIT, conv fastrun and without running step")
    cmd = script_path + " --conv-fastrun=yes --opt-level=3 --run-step=no"
    run_cmd(cmd)

    if args.run_eager:
        print_log("**************************************")
        print_log("Run static graph with default opt level and batch-size=8")
        cmd = script_path + " --batch-size=8"
        run_cmd(cmd)
        print_log("**************************************")
        print_log("Run eager graph with default opt level and batch-size=8")
        cmd = script_path
        run_cmd("MGE_DISABLE_TRACE=1 " + cmd + " --eager=yes")

    if args.run_debug_tool:

        print_log("**************************************")
        print_log("Run with dump_prof")
        cmd = script_path + " --dump-prof=" + prof_path
        if run_cmd(cmd):
            print("Printing profiling result")
            profiler([prof_path, "--aggregate-by=type", "--aggregate=sum", "-t 10"])

        print_log("**************************************")
        print_log("Run with valgrind massif")
        massif_out = "massif.out"
        # Use 0.01% as valgrind massif threashold
        # A smaller value reports more details but it may take longer time to analyze the log
        # Change it accordingly.
        mem_threshold = 0.01
        cmd = (
            "valgrind --tool=massif --threshold={} --massif-out-file=".format(
                mem_threshold
            )
            + massif_out
            + " "
        )
        cmd = cmd + script_path + " --warm-up=no --run-iter=20"
        run_cmd(cmd)
        ms_print_file = "massif.out.ms_print"
        cmd = (
            "ms_print --threshold={} ".format(mem_threshold)
            + massif_out
            + " > "
            + ms_print_file
        )
        os.system(cmd)
        cmd = "head -n 33 " + ms_print_file
        os.system(cmd)
        print_log("Read {} for detailed massif output".format(ms_print_file))

    if args.run_parallel:
        print_log("**************************************")
        tmp_out = "/dev/null"
        # Change server and port to run at your system
        server = "localhost"
        port = "2222"
        for num_gpu in (2, 4, 8):
            print_log("Run with {} GPUs".format(num_gpu))

            cmd = script_path + " --num-gpu={} --server={} --port={} ".format(
                num_gpu, server, port
            )
            for i in range(num_gpu - 1):
                irank = num_gpu - 1 - i
                os.system(
                    cmd
                    + " --device={}".format(irank)
                    + " 1>{} 2>{} &".format(tmp_out, tmp_out)
                )
            if not run_cmd(cmd):
                break

    print_log("**************************************")
    print_log("**************************************")
    print("Finish run, summary:")
    cmd = 'grep "Run with\|Wall time\|Run status\|Error\|GPU Usage" ' + log_path
    os.system(cmd)
