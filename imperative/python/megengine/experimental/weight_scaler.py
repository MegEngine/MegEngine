import types
from functools import partial

from .. import functional as F
from .. import module as M
from ..utils.module_utils import set_module_mode_safe


def get_norm_mod_value(weight, norm_value):
    weight = weight.reshape(-1)
    norm = F.norm(weight)
    scale = norm_value / norm
    round_log = F.floor(F.log(scale) / F.log(2))
    rounded_scale = 2 ** round_log
    return rounded_scale.detach()


def get_scaled_model(model, scale_submodel, input_shape=None):
    submodule_list = None
    scale_value = None
    accumulated_scale = 1.0

    def scale_calc(mod_calc_func):
        def calcfun(self, inp, weight, bias):
            scaled_weight = weight
            scaled_bias = bias
            if self.training:
                scaled_weight = (
                    weight * self.weight_scale if weight is not None else None
                )
                scaled_bias = bias * self.bias_scale if bias is not None else None
            return mod_calc_func(inp, scaled_weight, scaled_bias)

        return calcfun

    def scale_module_structure(
        scale_list: list = None, scale_value: tuple = None,
    ):
        nonlocal accumulated_scale
        for i in range(len(scale_list)):
            key, mod = scale_list[i]
            w_scale_value = scale_value[1]
            if scale_value[0] is not "CONST":
                w_scale_value = get_norm_mod_value(mod.weight, scale_value[1])

            accumulated_scale *= w_scale_value

            mod.weight_scale = w_scale_value
            mod.bias_scale = accumulated_scale

            if isinstance(mod, M.conv.Conv2d):
                mod.calc_conv = types.MethodType(scale_calc(mod.calc_conv), mod)
            else:
                mod._calc_linear = types.MethodType(scale_calc(mod._calc_linear), mod)

    def forward_hook(submodel, inputs, outpus, modelname=""):
        nonlocal submodule_list
        nonlocal scale_value
        nonlocal accumulated_scale
        if modelname in scale_submodel:
            scale_value = scale_submodel[modelname]
            if isinstance(submodel, (M.conv.Conv2d, M.linear.Linear)):
                scale_module_structure([(modelname, submodel)], scale_value)
            else:
                submodule_list = []

        if isinstance(submodel, (M.conv.Conv2d, M.linear.Linear)) and (
            submodule_list is not None
        ):
            submodule_list.append((modelname, submodel))

        if isinstance(submodel, M.batchnorm.BatchNorm2d) and (
            submodule_list is not None
        ):
            scale_module_structure(submodule_list, scale_value)
            submodule_list = None
            scale_value = None
            accumulated_scale = 1.0

    if input_shape is None:
        raise ValueError("input_shape is required for calculating scale value")

    input = F.zeros(input_shape)

    hooks = []
    for modelname, submodel in model.named_modules():
        hooks.append(
            submodel.register_forward_pre_hook(
                partial(forward_hook, modelname=modelname, outpus=None)
            )
        )

    with set_module_mode_safe(model, training=False) as model:
        model(input)

    for hook in hooks:
        hook.remove()

    return model
