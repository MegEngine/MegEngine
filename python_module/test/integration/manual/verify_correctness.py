# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import subprocess
import sys

import numpy as np


def fwd_test(backend):

    model_path = "../examples/cifar10/resnet_example/checkpoint/pretrained_model_82.mge"

    # Change the reference number if the change is from numerical rounding-off
    # FIXME! Need to use different number depending on CPU/GPU
    loss_ref = np.array([7.315978]).astype(np.float32)

    if backend == "megengine-dynamic":
        os.environ["MGE_DISABLE_TRACE"] = "true"

    import megengine
    from megengine.functional.debug_param import set_conv_execution_strategy
    from megengine.test import assertTensorClose
    from megengine.core import Graph

    sys.path.append(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples")
    )
    from cifar10.resnet_example.main import Example as resnet18_config
    from cifar10.resnet_example.main import eval_one_iter_mge

    mge_root = os.path.dirname(megengine.__file__)
    model_path = os.path.join(mge_root, model_path)
    run_case = resnet18_config(backend=backend, mode="eval")
    run_case.init_net()
    run_case.load_model(model_path)

    np.random.seed(0)
    inputs = np.random.rand(run_case.train_batch_size, 3, 32, 32)
    targets = np.random.randint(10, size=(run_case.train_batch_size,))
    max_err = 0.0

    run_case.net_context["net"].eval()
    loss, _ = eval_one_iter_mge(inputs, targets, config=run_case)
    try:
        loss = loss.numpy()
        assertTensorClose(loss, loss_ref, max_err=max_err)
    except:
        print("calculated loss:", loss)
        print("expect:", loss_ref)
        sys.exit(1)


def train_test(backend):

    model_path = "../examples/cifar10/resnet_example/checkpoint/pretrained_model_82.mge"

    # Change the reference number if the change is from numerical rounding-off
    # FIXME! Need to use different number depending on CPU/GPU
    if backend == "megengine-dynamic":
        os.environ["MGE_DISABLE_TRACE"] = "true"
        loss_ref = np.array([3.4709125, 12.46342]).astype(np.float32)
    else:
        loss_ref = np.array([3.4709125, 12.463419]).astype(np.float32)

    import megengine
    from megengine.functional.debug_param import set_conv_execution_strategy
    from megengine.test import assertTensorClose
    from megengine.core import Graph

    sys.path.append(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples")
    )
    from cifar10.resnet_example.main import Example as resnet18_config
    from cifar10.resnet_example.main import train_one_iter_mge

    mge_root = os.path.dirname(megengine.__file__)
    model_path = os.path.join(mge_root, model_path)
    set_conv_execution_strategy("HEURISTIC_REPRODUCIBLE")
    run_case = resnet18_config(backend=backend, mode="train")
    run_case.init_net()
    run_case.load_model(model_path)

    max_err = 0.0

    loss = []
    np.random.seed(0)
    inputs = np.random.rand(run_case.train_batch_size, 3, 32, 32)
    targets = np.random.randint(10, size=(run_case.train_batch_size,))

    run_case.set_optimizer(0.0)
    opt = run_case.net_context["optimizer"]

    for lr in (1.0, 1.0):
        run_case.set_optimizer(lr)
        opt.zero_grad()
        loss_batch, _ = train_one_iter_mge(inputs, targets, config=run_case)
        opt.step()
        loss.append(loss_batch.numpy()[0])
    try:
        assertTensorClose(np.array(loss).astype(np.float32), loss_ref, max_err=1e-5)
    except:
        print("calculated loss:", loss)
        print("expect:", loss_ref)
        sys.exit(1)


def run_func(func):
    cmd_start = ["python3", "-c"]
    cmd_head = "from verify_correctness import fwd_test, train_test\n"
    cmd = cmd_start + [cmd_head + func]
    ret = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if ret.returncode != 0:
        print("Failed!!!")
        print(ret.stdout)
        print(ret.stderr)
        raise
    print("Success")


if __name__ == "__main__":

    print("Running fwd static ...")
    run_func('fwd_test(backend="megengine-static")')

    print("Running fwd dynamic ...")
    run_func('fwd_test(backend="megengine-dynamic")')

    print("Running train static ...")
    run_func('train_test(backend="megengine-static")')

    print("Running train dynamic ...")
    run_func('train_test(backend="megengine-dynamic")')
