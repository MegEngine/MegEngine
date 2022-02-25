# -*- coding: utf-8 -*-

from megskull.graph import NodeFilter, FpropEnv
from megskull.opr.all import AssertEqual, DataProvider, BatchNormalization
from megskull.utils.logconf import get_logger
from meghair.utils import io
import megbrain as mgb

import argparse
import struct
import re
import os

import numpy as np
import cv2

logger = get_logger(__name__)

def optimize_for_inference(args, outputs):
    args_map = {
        'enable_io16xc32': 'f16_io_f32_comp',
        'enable_ioc16': 'f16_io_comp',
        'enable_hwcd4': 'use_nhwcd4',
        'enable_nchw4': 'use_nchw4',
        'enable_nchw88': 'use_nchw88',
        'enable_nchw44': 'use_nchw44',
        'enable_nchw44_dot': 'use_nchw44_dot',
        'enable_nchw32': 'use_nchw32',
        'enable_chwn4': 'use_chwn4',
        'enable_fuse_conv_bias_nonlinearity': 'fuse_conv_bias_nonlinearity',
        'enable_fuse_conv_bias_with_z': 'fuse_conv_bias_with_z',
    }
    kwargs = {}
    for k, v in args_map.items():
        if getattr(args, k):
            assert args.optimize_for_inference, (
                'optimize_for_inference should be set when {} is given'.format(
                    k))
            kwargs[v] = True

    if args.optimize_for_inference:
        return mgb.optimize_for_inference(outputs, **kwargs)

    return outputs

def main():
    parser = argparse.ArgumentParser(
        description='Dump the Python Megbrain model to C++ model, by the way '
        'optimizing for inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', help='input pkl model file ')
    parser.add_argument('-o', '--output', help='output file', required=True)
    parser.add_argument('--init-bn', action='store_true',
                        help='initialize untrained batch-normalization, to '
                        'avoid NaN or Inf results')
    parser.add_argument('--silent', action='store_true',
                        help='set verbose to False in AssertEqual opr')
    parser.add_argument('--optimize-for-inference', action='store_true',
                        help='enbale optimization for inference')
    parser.add_argument('--discard-var-name', action='store_true',
                        help='discard variable and param names in the '
                        'generated output')
    parser.add_argument('--output-strip-info', action='store_true',
                        help='output code strip information')
    parser.add_argument('--enable-io16xc32', action='store_true',
                        help='transform the mode to float16 io float32 compute')
    parser.add_argument('--enable-ioc16', action='store_true',
                        help='transform the dtype of the model to float16 io '
                        'and compute')
    parser.add_argument('--enable-fuse-conv-bias-nonlinearity',
                        action='store_true',
                        help='fuse convolution bias and nonlinearity opr to a '
                        'conv_bias opr and compute')
    parser.add_argument('--enable-hwcd4', action='store_true',
                        help='transform the model format from NCHW to NHWCD4 '
                        'for inference; you may need to disable CUDA and set '
                        'MGB_USE_MEGDNN_DBG=2')
    parser.add_argument('--enable-nchw4', action='store_true',
                        help='transform the model format from NCHW to NCHW4 '
                        'for inference')
    parser.add_argument('--enable-nchw88', action='store_true',
                        help='transform the model format from NCHW to NCHW88 '
                        'for inference')
    parser.add_argument('--enable-nchw44', action='store_true',
                        help='transform the model format from NCHW to NCHW44 '
                        'for inference')
    parser.add_argument('--enable-nchw44-dot', action='store_true',
                        help='transform the model format from NCHW to NCHW44_DOT '
                        'for optimizing armv8.2 dot in inference')
    parser.add_argument('--enable-chwn4', action='store_true',
                        help='transform the model format to CHWN4 '
                        'for inference, mainly used for nvidia tensorcore')
    parser.add_argument('--enable-nchw32', action='store_true',
                        help='transform the model format from NCHW4 to NCHW32 '
                        'for inference on nvidia TensoCore')
    parser.add_argument('--enable-fuse-conv-bias-with-z', action='store_true',
                        help='fuse conv_bias with z input for inference on '
                        'nvidia GPU (this optimization pass will result in mismatch '
                        'of the precision of output of training and inference)')
    args = parser.parse_args()

    env = FpropEnv(verbose_fprop=False)


    outputs = io.load_network(args.input).outputs

    output_mgbvars = list(map(env.get_mgbvar, outputs))

    output_mgbvars = optimize_for_inference(args, output_mgbvars)

    if args.discard_var_name:
        sereg_kwargs = dict(keep_var_name=0, keep_param_name=False)
    else:
        sereg_kwargs = dict(keep_var_name=2, keep_param_name=True)

    stat = mgb.serialize_comp_graph_to_file(
        args.output, output_mgbvars, append=False,
        output_strip_info=args.output_strip_info,
        **sereg_kwargs)
    logger.info('graph dump sizes: tot_size={:.3f}KiB overhead={:.3f}KiB'.
                format(stat.tot_bytes / 1024,
                       (stat.tot_bytes - stat.tensor_value_bytes) / 1024))

if __name__ == '__main__':
    main()
