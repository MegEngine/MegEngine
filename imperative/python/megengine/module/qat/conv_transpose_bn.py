from ...functional import ones, relu, sqrt, sum, zeros
from .. import conv_transpose_bn as Float
from .module import QATModule


class _ConvTransposeBnActivation2d(Float._ConvTransposeBnActivation2d, QATModule):
    def get_batch_mean_var(self, inp):
        def _sum_channel(inp, axis=0, keepdims=True):
            if isinstance(axis, int):
                out = sum(inp, axis=axis, keepdims=keepdims)
            elif isinstance(axis, tuple):
                for idx, elem in enumerate(axis):
                    out = sum(inp if idx == 0 else out, axis=elem, keepdims=keepdims)
            return out

        sum1 = _sum_channel(inp, (0, 2, 3))
        sum2 = _sum_channel(inp ** 2, (0, 2, 3))
        reduce_size = inp.size / inp.shape[1]
        batch_mean = sum1 / reduce_size
        batch_var = (sum2 - sum1 ** 2 / reduce_size) / reduce_size
        return batch_mean, batch_var

    def fold_weight_bias(self, bn_mean, bn_var):
        # get fold bn conv_transpose2d param
        gamma = self.bn.weight
        if gamma is None:
            gamma = ones((1, self.bn.num_features, 1, 1), dtype="float32")
        beta = self.bn.bias
        if beta is None:
            beta = zeros((1, self.bn.num_features, 1, 1), dtype="float32")

        if bn_mean is None:
            bn_mean = zeros((1, self.bn.num_features, 1, 1), dtype="float32")
        if bn_var is None:
            bn_var = ones((1, self.bn.num_features, 1, 1), dtype="float32")

        conv_transpose2d_bias = self.conv_transpose2d.bias
        if conv_transpose2d_bias is None:
            conv_transpose2d_bias = zeros(
                self.conv_transpose2d._infer_bias_shape(), dtype="float32"
            )

        bn_istd = 1.0 / sqrt(bn_var + self.bn.eps)
        scale_factor = gamma * bn_istd
        if self.conv_transpose2d.groups == 1:
            w_fold = self.conv_transpose2d.weight * scale_factor.reshape(1, -1, 1, 1)
        else:
            w_fold = self.conv_transpose2d.weight * scale_factor.reshape(
                self.conv_transpose2d.groups, 1, -1, 1, 1
            )

        w_fold = self.apply_quant_weight(w_fold)
        b_fold = beta + gamma * (conv_transpose2d_bias - bn_mean) * bn_istd
        return w_fold, b_fold

    def update_running_mean_and_running_var(
        self, bn_mean, bn_var, num_elements_per_channel
    ):
        # update running mean and running var. no grad, use unbiased bn var
        bn_mean = bn_mean.detach()
        bn_var = (
            bn_var.detach() * num_elements_per_channel / (num_elements_per_channel - 1)
        )
        exponential_average_factor = 1 - self.bn.momentum
        self.bn.running_mean *= self.bn.momentum
        self.bn.running_mean += exponential_average_factor * bn_mean
        self.bn.running_var *= self.bn.momentum
        self.bn.running_var += exponential_average_factor * bn_var

    def calc_conv_transpose2d_bn_qat(self, inp, approx=True):
        if self.training and not approx:
            conv_transpose2d = self.conv_transpose2d(inp)
            bn_mean, bn_var = self.get_batch_mean_var(conv_transpose2d)
            num_elements_per_channel = conv_transpose2d.size / conv_transpose2d.shape[1]
            self.update_running_mean_and_running_var(
                bn_mean, bn_var, num_elements_per_channel
            )
        else:
            bn_mean, bn_var = self.bn.running_mean, self.bn.running_var

        # get gamma and beta in BatchNorm
        gamma = self.bn.weight
        if gamma is None:
            gamma = ones((self.bn.num_features), dtype="float32")
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = self.bn.bias
        if beta is None:
            beta = zeros((self.bn.num_features), dtype="float32")
        beta = beta.reshape(1, -1, 1, 1)
        # conv_transpose2d_bias
        conv_transpose2d_bias = self.conv_transpose2d.bias
        if conv_transpose2d_bias is None:
            conv_transpose2d_bias = zeros(
                self.conv_transpose2d._infer_bias_shape(), dtype="float32"
            )

        bn_istd = 1.0 / sqrt(bn_var + self.bn.eps)
        scale_factor = gamma * bn_istd
        if self.conv_transpose2d.groups == 1:
            w_fold = self.conv_transpose2d.weight * scale_factor.reshape(1, -1, 1, 1)
        else:
            w_fold = self.conv_transpose2d.weight * scale_factor.reshape(
                self.conv_transpose2d.groups, 1, -1, 1, 1
            )
        b_fold = None
        if not (self.training and approx):
            b_fold = beta + gamma * (conv_transpose2d_bias - bn_mean) * bn_istd

        w_qat = self.apply_quant_weight(w_fold)
        b_qat = self.apply_quant_bias(b_fold, inp, w_qat)
        conv_transpose2d = self.conv_transpose2d.calc_conv_transpose2d(
            inp, w_qat, b_qat
        )
        if not (self.training and approx):
            return conv_transpose2d

        # rescale conv_transpose2d to get original conv_transpose2d output
        orig_conv_transpose2d = conv_transpose2d / scale_factor.reshape(1, -1, 1, 1)
        if self.conv_transpose2d.bias is not None:
            orig_conv_transpose2d = orig_conv_transpose2d + self.conv_transpose2d.bias
        # calculate batch norm
        conv_transpose2d = self.bn(orig_conv_transpose2d)
        return conv_transpose2d

    @classmethod
    def from_float_module(cls, float_module: Float._ConvTransposeBnActivation2d):
        qat_module = cls(
            float_module.conv_transpose2d.in_channels,
            float_module.conv_transpose2d.out_channels,
            float_module.conv_transpose2d.kernel_size,
            float_module.conv_transpose2d.stride,
            float_module.conv_transpose2d.padding,
            float_module.conv_transpose2d.output_padding,
            float_module.conv_transpose2d.dilation,
            float_module.conv_transpose2d.groups,
            float_module.conv_transpose2d.bias is not None,
            float_module.conv_transpose2d.conv_mode,
            float_module.conv_transpose2d.compute_mode,
            name=float_module.name,
        )
        qat_module.conv_transpose2d.weight = float_module.conv_transpose2d.weight
        qat_module.conv_transpose2d.bias = float_module.conv_transpose2d.bias
        qat_module.bn = float_module.bn
        return qat_module


class ConvTransposeBn2d(_ConvTransposeBnActivation2d):
    r"""A fused :class:`~.QATModule` including :class:`~.module.ConvTranspose2d` and :class:`~.module.BatchNorm2d` with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.quantization.fake_quant.FakeQuantize`.
    """

    def forward(self, inp):
        return self.apply_quant_activation(self.calc_conv_transpose2d_bn_qat(inp))


class ConvTransposeBnRelu2d(_ConvTransposeBnActivation2d):
    r"""A fused :class:`~.QATModule` including :class:`~.module.ConvTranspose2d`, :class:`~.module.BatchNorm2d` and :func:`~.relu` with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.quantization.fake_quant.FakeQuantize`.
    """

    def forward(self, inp):
        return self.apply_quant_activation(relu(self.calc_conv_transpose2d_bn_qat(inp)))
