#include "./opr_impl.h"
#if MGB_CUSTOM_OP
#include "megbrain/custom/custom.h"

using namespace custom;

CUSTOM_OP_REG_BEGIN(dilate_forward)

#if MGB_CUDA
void dilate_forward_cuda(
        const std::vector<Tensor>& inputs, const Param& params,
        std::vector<Tensor>& outputs) {
    const Tensor& inp = inputs[0];
    const Tensor& kernel = inputs[1];
    Tensor& output = outputs[0];
    Tensor& workspace = outputs[1];
    const int iterations = params["iterations"].as<int>();
    const std::string border_type = params["border_type"].as<std::string>();
    const double border_value = params["border_value"].as<double>();

    launch_dilate_kernel(
            inp, kernel, output, workspace, iterations, border_type, border_value);
}
#endif

CUSTOM_OP_REG(dilate_forward)
        .add_inputs(2)
        .add_outputs(2)
#if MGB_CUDA
        .set_compute("cuda", dilate_forward_cuda)
#endif
        .add_param("iterations", 1)
        .add_param("border_type", "bilinear")
        .add_param("border_value", 0.0);

CUSTOM_OP_REG_END(dilate_forward)
#endif