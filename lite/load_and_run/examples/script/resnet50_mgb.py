#!/usr/bin/env python3

import megbrain as mgb
from megskull.graph import FpropEnv
import megskull as mgsk
from megskull.opr.compatible.caffepool import CaffePooling2D
from megskull.opr.arith import ReLU
from megskull.opr.all import (
    DataProvider, Conv2D, Pooling2D, FullyConnected,
    Softmax, Dropout, BatchNormalization, CrossEntropyLoss,
    ElementwiseAffine, WarpPerspective, WarpPerspectiveWeightProducer,
    WeightDecay, ParamProvider, ConvBiasActivation, ElemwiseMultiType)
from megskull.network import RawNetworkBuilder
from megskull.utils.debug import CallbackInjector
import megskull.opr.helper.param_init as pinit
from megskull.opr.helper.elemwise_trans import Identity
from megskull.opr.netsrc import DataProvider
from megskull.opr.cnn import Conv2D, Pooling2D, FullyConnected, Softmax, Conv2DImplHelper
from megskull.opr.loss import CrossEntropyLoss
from megskull.opr.regularizer import Dropout, BatchNormalization

from megskull.opr.arith import Add, ReLU
from megskull.opr.netsrc import ConstProvider
from megskull.network import RawNetworkBuilder
import numpy as np
from megskull.network import RawNetworkBuilder, NetworkVisitor
from megskull.graph import iter_dep_opr
from megskull.utils.misc import get_2dshape
import functools
import re
import fnmatch
import argparse
import sys

def create_bn_relu_float(conv_name, f_in, ksize, stride, pad, num_outputs,
                         has_relu, args):
    f = Conv2D(conv_name, f_in, kernel_shape=ksize, stride=stride,
               padding=pad, output_nr_channel=num_outputs,
               nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())
    if has_relu:
        f = ReLU(f)
    return f


def get_num_inputs(feature, format):
    if format == 'NCHW':
        return feature.partial_shape[1]
    else:
        assert format == 'NCHW4'
        return feature.partial_shape[1] * 4


def create_bn_relu(prefix, f_in, ksize, stride, pad, num_outputs,
                   has_relu, conv_name_fun, args):
    if conv_name_fun:
        conv_name = conv_name_fun(prefix)
    else:
        conv_name = prefix
    return create_bn_relu_float(conv_name, f_in, ksize, stride, pad,
                                    num_outputs, has_relu, args)




def create_bottleneck(prefix, f_in, stride, num_outputs1, num_outputs2, args,
                      has_proj=False):
    proj = f_in
    if has_proj:
        proj = create_bn_relu(prefix, f_in, ksize=1, stride=stride, pad=0,
                              num_outputs=num_outputs2, has_relu=False,
                              conv_name_fun=lambda p: "interstellar{}_branch1".format(
                                  p), args=args)

    f = create_bn_relu(prefix, f_in, ksize=1, stride=1, pad=0,
                       num_outputs=num_outputs1, has_relu=True,
                       conv_name_fun=lambda p: "interstellar{}_branch2a".format(
                           p), args=args)

    f = create_bn_relu(prefix, f, ksize=3, stride=stride, pad=1,
                       num_outputs=num_outputs1, has_relu=True,
                       conv_name_fun=lambda p: "interstellar{}_branch2b".format(
                           p), args=args)

    f = create_bn_relu(prefix, f, ksize=1, stride=1, pad=0,
                       num_outputs=num_outputs2, has_relu=False,
                       conv_name_fun=lambda p: "interstellar{}_branch2c".format(
                           p), args=args)

    f = ReLU(f + proj)

    return f


def get(args):
    img_size = 224
    num_inputs = 3
    data = DataProvider('data', shape=(args.batch_size, num_inputs,
                                           img_size, img_size))

    inp = data
    f = create_bn_relu("conv1", inp, ksize=7, stride=2, pad=3, num_outputs=64,
                       has_relu=True, conv_name_fun=None,
                       args=args)
    f = Pooling2D("pool1", f, window=3, stride=2, padding=1, mode="MAX",
                  format=args.format)

    pre = [2, 3, 4, 5]
    stages = [3, 4, 6, 3]
    mid_outputs = [64, 128, 256, 512]
    enable_stride = [False, True, True, True]

    for p, s, o, es in zip(pre, stages, mid_outputs, enable_stride):
        for i in range(s):
            has_proj = False if i > 0 else True
            stride = 1 if not es or i > 0 else 2
            prefix = "{}{}".format(p, chr(ord("a") + i))
            f = create_bottleneck(prefix, f, stride, o, o * 4, args, has_proj)
            print("{}\t{}".format(prefix, f.partial_shape))

    f = Pooling2D("pool5", f, window=7, stride=7, padding=0, mode="AVERAGE",
                  format=args.format)

    f = FullyConnected("fc1000", f, output_dim=1000,
                       nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())

    f = Softmax("cls_softmax", f)
    f.init_weights()

    net = RawNetworkBuilder(inputs=[data], outputs=[f])

    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='dump pkl model for resnet50',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size',
                        help='batch size of the model', default=1)
    parser.add_argument('-f', '--format', choices=['NCHW', 'NCHW4'],
                        help='format of conv',
                        default='NCHW')
    parser.add_argument('-o', '--output',
                        help='output pkl path', required=True)
    args = parser.parse_args()
    if args.format != 'NCHW':
        print('Only suppprt NCHW for float model')
        parser.print_help()
        sys.exit(1)

    from meghair.utils import io
    io.dump(get(args), args.output)

