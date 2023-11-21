
#include "./opr_impl.h"
#if MGB_CUSTOM_OP
#include "megbrain/custom/custom.h"

using namespace custom;

CUSTOM_OP_REG_BEGIN(fillpoly_forward)

#if MGB_CUDA

void fillpoly_forward_cuda(
        const std::vector<Tensor>& inputs, const Param& params,
        std::vector<Tensor>& outputs) {
    const Tensor& img = inputs[0];
    const Tensor& points = inputs[1];
    const Tensor& lens = inputs[2];
    const Tensor& color = inputs[3];
    Tensor& output = outputs[0];
    launch_fillpoly_kernel(img, points, lens, color, output);
}

#endif

CUSTOM_OP_REG(fillpoly_forward)
        .add_input("img", "img", {"int32", "float32"})
        .add_input("points", "points", {"int32"})
        .add_input("lens", "lens", {"int32"})
        .add_input("color", "color", {"int32", "float32"})
        .add_output("output", "output", {"int32", "float32"})
#if MGB_CUDA
        .set_compute("cuda", fillpoly_forward_cuda)
#endif
        ;

CUSTOM_OP_REG_END(fillpoly_forward)

#endif