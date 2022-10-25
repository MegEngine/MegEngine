from copy import deepcopy

from ..functional import ones, sqrt, zeros
from ..module import (
    BatchNorm2d,
    Conv2d,
    ConvBn2d,
    ConvBnRelu2d,
    ConvRelu2d,
    ConvTranspose2d,
    ConvTransposeBn2d,
    ConvTransposeBnRelu2d,
    ConvTransposeRelu2d,
    ReLU,
)
from ..tensor import Parameter

_MAP_TO_FUSED_MODULE = {
    (Conv2d, BatchNorm2d, ReLU, False): ConvRelu2d,
    (Conv2d, BatchNorm2d, ReLU, True): ConvBnRelu2d,
    (ConvTranspose2d, BatchNorm2d, ReLU, False): ConvTransposeRelu2d,
    (ConvTranspose2d, BatchNorm2d, ReLU, True): ConvTransposeBnRelu2d,
    (Conv2d, BatchNorm2d, False): Conv2d,
    (Conv2d, BatchNorm2d, True): ConvBn2d,
    (Conv2d, ReLU): ConvRelu2d,
    (ConvTranspose2d, BatchNorm2d, False): ConvTranspose2d,
    (ConvTranspose2d, BatchNorm2d, True): ConvTransposeBn2d,
    (ConvTranspose2d, ReLU): ConvTransposeRelu2d,
}


def fold_weight_bias(
    weight, bias, gamma, beta, bn_mean, bn_var, eps=1e-5, transpose=False
):
    shape = (-1, 1, 1, 1)
    if transpose:
        shape = (1, -1, 1, 1)

    kernel_shape = weight.shape
    if len(kernel_shape) == 5:
        if transpose:
            groups, num_features = kernel_shape[0], kernel_shape[2]
        else:
            groups, num_features = kernel_shape[0], kernel_shape[1]
    else:
        if transpose:
            groups, num_features = 1, kernel_shape[1]
        else:
            groups, num_features = 1, kernel_shape[0]

    out_channels = groups * num_features
    if gamma is None:
        gamma = ones((out_channels,), dtype="float32")
    gamma = gamma.reshape(1, -1, 1, 1)
    if beta is None:
        beta = zeros((out_channels,), dtype="float32")
    beta = beta.reshape(1, -1, 1, 1)

    if bn_mean is None:
        bn_mean = zeros((1, out_channels, 1, 1), dtype="float32")
    if bn_var is None:
        bn_var = ones((1, out_channels, 1, 1), dtype="float32")

    if bias is None:
        bias = zeros((1, out_channels, 1, 1), dtype="float32")

    bn_istd = 1.0 / sqrt(bn_var + eps)
    scale_factor = gamma * bn_istd

    if groups == 1:
        w_fold = weight * scale_factor.reshape(*shape)
    else:
        w_fold = weight * scale_factor.reshape(groups, *shape)

    b_fold = beta + gamma * (bias - bn_mean) * bn_istd
    return w_fold, b_fold


def fuse_conv_bn_relu_module(conv: Conv2d, bn: BatchNorm2d, relu: ReLU):
    module_key = tuple([type(m) for m in [conv, bn, relu] if m])
    if bn:
        assert (
            conv.training == bn.training
        ), "Conv and BN both must be in the same mode (train or eval)."
        assert (
            bn.num_features == conv.out_channels
        ), "Output channel of Conv2d must match num_features of BatchNorm2d"
        module_key = module_key + (conv.training,)
    module = _MAP_TO_FUSED_MODULE[module_key](
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        conv_mode=conv.conv_mode,
        compute_mode=conv.compute_mode,
        name=conv.name,
    )
    if isinstance(conv, ConvTranspose2d):
        module.output_padding = conv.output_padding
        new_conv = (
            module if bn is None or not conv.training else module.conv_transpose2d
        )
    else:
        new_conv = module if bn is None or not conv.training else module.conv

    weight, bias = conv.weight, conv.bias
    if not conv.training and bn is not None:
        if isinstance(conv, ConvTranspose2d):
            weight, bias = fold_weight_bias(
                weight,
                bias,
                bn.weight,
                bn.bias,
                bn.running_mean,
                bn.running_var,
                bn.eps,
                transpose=True,
            )
        else:
            weight, bias = fold_weight_bias(
                weight,
                bias,
                bn.weight,
                bn.bias,
                bn.running_mean,
                bn.running_var,
                bn.eps,
            )
    new_conv.weight = Parameter(weight)
    if bias is not None:
        new_conv.bias = Parameter(bias)
    if bn is not None and conv.training:
        module.bn = deepcopy(bn)
    new_conv.training = conv.training
    return module
