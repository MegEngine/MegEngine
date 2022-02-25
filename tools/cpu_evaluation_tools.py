#!/usr/bin/env python3

"""
purpose: Used to simply measure CPU performance by running several basic models.
how to use: python3 cpu_evaluation_tools.py --help for more details, now need to args:
    --load_and_run_file: path of load_and_run binary, please refs to ../scripts/cmake-build/BUILD_README.md to build it.
    --models_dir: path of model directory.
how to config test device info: config device[name/login_name/ip/port/thread_number].
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
# test models
test_cpu_models = [
    "inceptionv2",
    "mobilenetv1",
    "mobilenetv2",
    "resnet18",
    "resnet50",
    "shufflenetv2",
    "vgg16",
]


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
    parser.add_argument("--models_dir", help="models dir", required=True)
    parser.add_argument(
        "--load_and_run_file", help="path for load_and_run", required=True
    )
    args = parser.parse_args()
    assert os.path.isdir(
        args.models_dir
    ), "invalid args for models_dir, need a dir for models"
    assert os.path.isfile(args.load_and_run_file), "invalid args for load_and_run_file"
    for m in test_cpu_models:
        assert os.path.isfile(
            os.path.join(args.models_dir, m)
        ), "invalid args for models_dir, need put model: {} to args.models_dir".format(
            test_cpu_models
        )

    # init device
    ssh = SshConnector()
    ssh.setup(device["login_name"], device["ip"], device["port"])
    # create test dir
    workspace = "cpu_evaluation_workspace"
    ssh.cmd(["mkdir -p {}".format(workspace)])
    # copy load_and_run_file
    ssh.copy([args.load_and_run_file], workspace)
    # call test
    result = []
    for m in test_cpu_models:
        m_path = os.path.join(args.models_dir, m)
        # copy model file
        ssh.copy([m_path], workspace)
        # run single thread
        sub_b = ["-cpu", "-multithread {}".format(device["thread_number"])]
        for b in sub_b:
            cmd = []
            cmd0 = "cd {} && rm -rf fastrun.cache".format(workspace)
            cmd1 = "cd {} && ./load_and_run {} --fast-run --fast_run_algo_policy fastrun.cache --iter 1 --warmup-iter 1 --no-sanity-check --weight-preprocess".format(
                workspace, m, b
            )
            cmd2 = "cd {} && ./load_and_run {} {} --fast_run_algo_policy fastrun.cache --iter 20 --warmup-iter 5 --no-sanity-check --weight-preprocess --record-comp-seq".format(
                workspace, m, b
            )
            cmd.append(cmd0)
            cmd.append(cmd1)
            cmd.append(cmd2)
            raw_log = ssh.cmd(cmd)
            # logging.debug(raw_log)
            ret = get_finally_bench_resulut_from_log(raw_log)
            logging.debug("model: {} with backend: {} result is: {}".format(m, b, ret))
            result.append(ret)

    total_time = 0.0
    for r in result:
        total_time += r
    logging.debug("total time is: {}".format(total_time))
    score = 100000.0 / total_time * 1000
    logging.debug("device: {} score is: {}".format(device["name"], score))


if __name__ == "__main__":
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y/%m/%d %H:%M:%S"
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    main()
