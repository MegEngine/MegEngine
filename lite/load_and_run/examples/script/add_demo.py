#!/usr/bin/env python3
import argparse
import math

import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import numpy as np
from megengine import jit, tensor, Parameter

class Simple(M.Module):
    def __init__(self):
        super().__init__()
        self.a = Parameter([0,1,2], dtype=np.float32)

    def forward(self, x):
        x = x + self.a
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="dump mge model for add_demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        help="set the inputs data to get a model with testcase",
        default="",
        type=str,
    )

    parser.add_argument(
        "--dir",
        help="set the dir where the model to dump",
        default=".",
        type=str,
    )

    args = parser.parse_args()
    net = Simple()
    net.eval()

    @jit.trace(symbolic=True, capture_as_const=True)
    def fun(data):
        return net(data)
    data = tensor([3.0,4.0,5.0])
    fun(data)
    if args.inputs == "":
        fun.dump(
            args.dir + "/add_demo_f32_without_data.mge", arg_names=["data"],
            no_assert=True,    
        )
    else:
        fun.dump(
            args.dir + "/add_demo_f32_with_data.mge", arg_names=["data"],
            input_data=[args.inputs], no_assert=True,
        )