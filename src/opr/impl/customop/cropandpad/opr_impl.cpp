#include "./opr_impl.h"
#if MGB_CUSTOM_OP
#include "megbrain/custom/custom.h"

using namespace custom;

CUSTOM_OP_REG_BEGIN(cropandpad_forward)

#if MGB_CUDA
void cropandpad_forward_cuda(
        const std::vector<Tensor>& inputs, const Param& params,
        std::vector<Tensor>& outputs) {
    launch_cropandpad_kernel(
            inputs[0], inputs[1], inputs[2], outputs[0],
            params["align_corners"].as<bool>(), params["mode"].as<std::string>());
}
#endif

CUSTOM_OP_REG(cropandpad_forward)
        .add_inputs(3)
        .add_outputs(1)
#if MGB_CUDA
        .set_compute("cuda", cropandpad_forward_cuda)
#endif
        .add_param("align_corners", false)
        .add_param("mode", "bilinear");

CUSTOM_OP_REG_END(cropandpad_forward)
#endif