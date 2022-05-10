#!/usr/bin/env python3

"""
purpose: use to test whether a model contain dynamic operator, if no dynamic
operator the model is static, other wise the model is dynamic.
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
        assert isinstance(cmd, list), "code issue happened!!"
        try:
            for sub_cmd in cmd:
                p_cmd = 'ssh -p {} {}@{} "{}" '.format(
                    self.port, self.login_name, self.ip, sub_cmd
                )
                logging.debug("ssh run cmd: {}".format(p_cmd))
                subprocess.check_call(p_cmd, shell=True)
        except:
            raise


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model_file", help="megengine model", required=True)
    parser.add_argument(
        "--load_and_run_file", help="path for load_and_run", required=True
    )
    args = parser.parse_args()
    assert os.path.isfile(
        args.model_file
    ), "invalid args for models_file, need a file for model"
    assert os.path.isfile(args.load_and_run_file), "invalid args for load_and_run_file"

    # init device
    ssh = SshConnector()
    ssh.setup(device["login_name"], device["ip"], device["port"])
    # create test dir
    workspace = "model_static_evaluation_workspace"
    ssh.cmd(["mkdir -p {}".format(workspace)])
    # copy load_and_run_file
    ssh.copy([args.load_and_run_file], workspace)

    model_file = args.model_file
    # copy model file
    ssh.copy([model_file], workspace)
    m = model_file.split("\\")[-1]
    # run single thread
    cmd = "cd {} && ./load_and_run {} --fast-run --record-comp-seq --iter 1 --warmup-iter 1".format(
        workspace, m
    )
    try:
        raw_log = ssh.cmd([cmd])
    except:
        print("model: {} is not static model, it has dynamic operator.".format(m))
        raise

    print("model: {} is static model.".format(m))


if __name__ == "__main__":
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y/%m/%d %H:%M:%S"
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    main()
