from ...functional import linear, ones, relu, sqrt, sum, zeros
from .. import linear_bn as Float
from .module import QATModule


class _LinearBnActivation1d(Float._LinearBnActivation1d, QATModule):
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
        weight_shape = [1] * len(self.linear.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.linear.weight.shape)
        bias_shape[1] = -1

        # get fold bn linear param
        gamma = self.bn.weight
        if gamma is None:
            gamma = ones((self.bn.num_features,), dtype="float32")
        gamma = gamma.reshape(-1)
        beta = self.bn.bias
        if beta is None:
            beta = zeros((self.bn.num_features,), dtype="float32")
        beta = beta.reshape(-1)

        if bn_mean is None:
            bn_mean = zeros((self.bn.num_features,), dtype="float32")
        bn_mean = bn_mean.reshape(-1)
        if bn_var is None:
            bn_var = ones((self.bn.num_features,), dtype="float32")
        bn_var = bn_var.reshape(-1)

        linear_bias = self.linear.bias
        if linear_bias is None:
            linear_bias = zeros(beta.shape(), dtype="float32")

        bn_istd = 1.0 / sqrt(bn_var + self.bn.eps)
        scale_factor = gamma * bn_istd
        w_fold = self.linear.weight * scale_factor.reshape(weight_shape)
        w_fold = self.apply_quant_weight(w_fold)
        b_fold = beta + gamma * (linear_bias - bn_mean) * bn_istd
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

    def calc_linear_bn_qat(self, inp, approx=True):
        if self.training and not approx:
            linear = self.linear(inp)
            bn_mean, bn_var = self.get_batch_mean_var(linear)
            num_elements_per_channel = linear.size / linear.shape[1]
            self.update_running_mean_and_running_var(
                bn_mean, bn_var, num_elements_per_channel
            )
        else:
            bn_mean, bn_var = self.bn.running_mean, self.bn.running_var

        bn_mean, bn_var = (
            self.bn.running_mean.reshape(-1),
            self.bn.running_var.reshape(-1),
        )

        weight_shape = [1] * len(self.linear.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.linear.weight.shape)
        bias_shape[1] = -1

        # get gamma and beta in BatchNorm
        gamma = self.bn.weight
        if gamma is None:
            gamma = ones((self.bn.num_features,), dtype="float32")
        gamma = gamma.reshape(-1)
        beta = self.bn.bias
        if beta is None:
            beta = zeros((self.bn.num_features,), dtype="float32")
        beta = beta.reshape(-1)

        # linear_bias
        linear_bias = self.linear.bias
        if linear_bias is None:
            linear_bias = zeros(beta.shape, dtype="float32")

        bn_istd = 1.0 / sqrt(bn_var + self.bn.eps)
        scale_factor = gamma * bn_istd

        w_fold = self.linear.weight * scale_factor.reshape(weight_shape)
        b_fold = None

        if not (self.training and approx):
            b_fold = beta + gamma * (linear_bias - bn_mean) * bn_istd

        w_qat = self.apply_quant_weight(w_fold)
        b_qat = self.apply_quant_bias(b_fold, inp, w_qat)
        linear = self.linear.calc_linear(inp, w_qat, b_qat)
        if not (self.training and approx):
            return linear

        # rescale linear to get original linear output
        orig_linear = linear / scale_factor.reshape(bias_shape)
        if self.linear.bias is not None:
            orig_linear = orig_linear + self.linear.bias.reshape(bias_shape)
        # calculate batch norm
        linear = self.bn(orig_linear)
        return linear

    @classmethod
    def from_float_module(cls, float_module: Float._LinearBnActivation1d):
        qat_module = cls(
            float_module.linear.in_features,
            float_module.linear.out_features,
            float_module.linear.bias is not None,
            float_module.linear.compute_mode,
            float_module.bn.eps,
            float_module.bn.momentum,
            float_module.bn.affine,
            float_module.bn.track_running_stats,
            name=float_module.name,
        )
        qat_module.linear.weight = float_module.linear.weight
        qat_module.linear.bias = float_module.linear.bias
        qat_module.bn = float_module.bn
        return qat_module


class LinearBn1d(_LinearBnActivation1d):
    r"""A fused :class:`~.QATModule` including :class:`~.module.Linear` and :class:`~.module.BatchNorm1d` with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.quantization.fake_quant.FakeQuantize`.
    """

    def forward(self, inp):
        return self.apply_quant_activation(self.calc_linear_bn_qat(inp))


class LinearBnRelu1d(_LinearBnActivation1d):
    r"""A fused :class:`~.QATModule` including :class:`~.module.Linear`, :class:`~.module.BatchNorm1d` and :func:`~.relu` with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.quantization.fake_quant.FakeQuantize`.
    """

    def forward(self, inp):
        return self.apply_quant_activation(relu(self.calc_linear_bn_qat(inp)))
