#!/usr/bin/env python3

"""
purpose: use to test whether a model have good parallelism, if a model have good
parallelism it will get high performance improvement.
"""
import argparse
import logging
import os
import re
import subprocess

# test device
device = {
    "name": "hwmt40p",
    "login_name": "hwmt40p-K9000-maliG78",
    "ip": "box86.br.megvii-inc.com",
    "port": 2200,
    "thread_number": 3,
}


class SshConnector:
    """imp ssh control master connector"""

    ip = None
    port = None
    login_name = None

    def setup(self, login_name, ip, port):
        self.ip = ip
        self.login_name = login_name
        self.port = port

    def copy(self, src_list, dst_dir):
        assert isinstance(src_list, list), "code issue happened!!"
        assert isinstance(dst_dir, str), "code issue happened!!"
        for src in src_list:
            cmd = 'rsync --progress -a -e "ssh -p {}" {} {}@{}:{}'.format(
                self.port, src, self.login_name, self.ip, dst_dir
            )
            logging.debug("ssh run cmd: {}".format(cmd))
            subprocess.check_call(cmd, shell=True)

    def cmd(self, cmd):
        output = ""
        assert isinstance(cmd, list), "code issue happened!!"
        for sub_cmd in cmd:
            p_cmd = 'ssh -p {} {}@{} "{}" '.format(
                self.port, self.login_name, self.ip, sub_cmd
            )
            logging.debug("ssh run cmd: {}".format(p_cmd))
            output = output + subprocess.check_output(p_cmd, shell=True).decode("utf-8")
        return output


def get_finally_bench_resulut_from_log(raw_log) -> float:
    # raw_log --> avg_time=23.331ms -->23.331ms
    h = re.findall(r"avg_time=.*ms ", raw_log)[-1][9:]
    # to 23.331
    h = h[: h.find("ms")]
    # to float
    h = float(h)
    return h


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model_file", help="model file", required=True)
    parser.add_argument(
        "--load_and_run_file", help="path for load_and_run", required=True
    )
    args = parser.parse_args()

    # init device
    ssh = SshConnector()
    ssh.setup(device["login_name"], device["ip"], device["port"])
    # create test dir
    workspace = "model_parallelism_test"
    ssh.cmd(["mkdir -p {}".format(workspace)])
    # copy load_and_run_file
    ssh.copy([args.load_and_run_file], workspace)
    # call test
    model_file = args.model_file
    # copy model file
    ssh.copy([args.model_file], workspace)
    m = model_file.split("\\")[-1]
    # run single thread
    result = []
    thread_number = [1, 2, 4]
    for b in thread_number:
        cmd = []
        cmd1 = "cd {} && ./load_and_run {} -multithread {} --fast-run --fast_run_algo_policy fastrun.cache --iter 1 --warmup-iter 1 --no-sanity-check --weight-preprocess".format(
            workspace, m, b
        )
        cmd2 = "cd {} && ./load_and_run {} -multithread {} --fast_run_algo_policy fastrun.cache --iter 20 --warmup-iter 5 --no-sanity-check --weight-preprocess ".format(
            workspace, m, b
        )
        cmd.append(cmd1)
        cmd.append(cmd2)
        raw_log = ssh.cmd(cmd)
        # logging.debug(raw_log)
        ret = get_finally_bench_resulut_from_log(raw_log)
        logging.debug("model: {} with backend: {} result is: {}".format(m, b, ret))
        result.append(ret)

    thread_2 = result[0] / result[1]
    thread_4 = result[0] / result[2]
    if thread_2 > 1.6 or thread_4 > 3.0:
        print(
            "model: {} can has good parallelism. 2 thread is {}, 4 thread is {}".format(
                m, thread_2, thread_4
            )
        )
    else:
        print(
            "model: {} can has bad parallelism. 2 thread is {}, 4 thread is {}".format(
                m, thread_2, thread_4
            )
        )


if __name__ == "__main__":
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y/%m/%d %H:%M:%S"
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    main()
