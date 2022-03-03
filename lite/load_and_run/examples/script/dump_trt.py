#!/usr/bin/env python3
from megskull.network import RawNetworkBuilder
import megskull.opr.all as O
from megskull.opr.external import TensorRTRuntimeOpr
from meghair.utils.io import dump
import argparse

def str2tuple(x):
    x = x.split(',')
    x = [int(a) for a in x]
    x = tuple(x)
    return x

def make_network(model, isize):
    data = [O.DataProvider('input{}'.format(i), shape=isizes[i])
            for i in range(len(isizes))]
    f = open(model, 'rb')
    engine = f.read()

    opr = TensorRTRuntimeOpr(data, engine, 1)

    net = RawNetworkBuilder(inputs=[data], outputs=opr.outputs)

    return net

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest = 'model')
    parser.add_argument(dest = 'output')
    parser.add_argument('--isize', help='input sizes. '
            'e.g. for models with two (1,3,224,224) inputs, '
            'the option --isize="1,3,224,224;1,3,224,224" should be used')
    
    args = parser.parse_args()
    isizes = [str2tuple(x) for x in args.isize.split(';')]
    net = make_network(args.model, isizes)
    dump(net, args.output)