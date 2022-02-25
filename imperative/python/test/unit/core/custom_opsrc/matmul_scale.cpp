#include "./matmul_scale.h"
#include "megbrain/custom/custom.h"

CUSTOM_OP_REG_BEGIN(MatMulScale)

void forward_shape_infer(
        const std::vector<Shape>& inputs, const Param& params,
        std::vector<Shape>& outputs) {
    outputs[0] = {inputs[0][0], inputs[1][1]};
}

void forward_compute(
        const std::vector<Tensor>& inputs, const Param& params,
        std::vector<Tensor>& outputs) {
    matmul_forward_helper(
            inputs[0], inputs[1], outputs[0], inputs[0].shape()[0],
            inputs[0].shape()[1], inputs[1].shape()[1], params["scale"].as<float>());
}

CUSTOM_OP_REG(MatMulScaleForward)
        .add_inputs(2)
        .add_outputs(1)
        .add_param("scale", 1.0f)
        .set_shape_infer(forward_shape_infer)
        .set_compute("cuda", forward_compute);

void backward_shape_infer(
        const std::vector<Shape>& ograd_and_inputs, const Param& params,
        std::vector<Shape>& outputs) {
    outputs[0] = ograd_and_inputs[1];
    outputs[1] = ograd_and_inputs[2];
}

void backward_compute(
        const std::vector<Tensor>& ograd_and_inputs, const Param& params,
        std::vector<Tensor>& igrads) {
    matmul_backward_lhs_helper(
            ograd_and_inputs[2], ograd_and_inputs[0], igrads[0],
            ograd_and_inputs[1].shape()[0], ograd_and_inputs[1].shape()[1],
            ograd_and_inputs[2].shape()[1], params["scale"].as<float>());
    matmul_backward_rhs_helper(
            ograd_and_inputs[1], ograd_and_inputs[0], igrads[1],
            ograd_and_inputs[1].shape()[0], ograd_and_inputs[1].shape()[1],
            ograd_and_inputs[2].shape()[1], params["scale"].as<float>());
}

CUSTOM_OP_REG(MatMulScaleBackward)
        .add_inputs(3)
        .add_outputs(2)
        .add_param("scale", 1.0f)
        .set_shape_infer(backward_shape_infer)
        .set_compute("cuda", backward_compute);

CUSTOM_OP_REG_END(MatMulScale)
