#include "megbrain/custom/custom.h"

CUSTOM_OP_REG_BEGIN(ElemAddSmooth)

void forward_device_infer(
        const std::vector<Device>& inputs, const Param& params,
        std::vector<Device>& outputs) {
    outputs[0] = inputs[0];
}

void forward_shape_infer(
        const std::vector<Shape>& inputs, const Param& params,
        std::vector<Shape>& outputs) {
    outputs[0] = inputs[0];
}

void forward_dtype_infer(
        const std::vector<DType>& inputs, const Param& params,
        std::vector<DType>& outputs) {
    outputs[0] = inputs[0];
}

void forward_format_infer(
        const std::vector<Format>& inputs, const Param& params,
        std::vector<Format>& outputs) {
    outputs[0] = inputs[0];
}

template <typename scalar_t>
void forward_kernel(
        const scalar_t* input0, const scalar_t* input1, scalar_t* output, size_t len,
        float smooth) {
    for (size_t i = 0; i < len; ++i) {
        output[i] = input0[i] + input1[i];
        if (output[i] < 0)
            output[i] += smooth;
        else
            output[i] -= smooth;
    }
}

void forward_compute(
        const std::vector<Tensor>& inputs, const Param& params,
        std::vector<Tensor>& outputs) {
    DISPATCH_SIGN_INT_AND_FLOAT_TYPES(
            outputs[0].dtype(), "forward_compute", ([&]() {
                forward_kernel<scalar_t>(
                        inputs[0].data<scalar_t>(), inputs[1].data<scalar_t>(),
                        outputs[0].data<scalar_t>(), outputs[0].size(),
                        params["smooth"].as<float>());
            }));
}

CUSTOM_OP_REG(ElemAddSmoothForward)
        .set_description(
                "Custom ElemAdd Operator With a Smooth Parameter, "
                "which is used to verify the CPU kernel")
        .add_input("lhs")
        .add_input("rhs")
        .add_output("output")
        .add_param("smooth", 0.f)
        .set_device_infer(forward_device_infer)
        .set_shape_infer(forward_shape_infer)
        .set_dtype_infer(forward_dtype_infer)
        .set_format_infer(forward_format_infer)
        .set_compute(forward_compute);

void backward_device_infer(
        const std::vector<Device>& ograds, const Param& params,
        std::vector<Device>& igrads) {
    igrads[0] = ograds[0];
    igrads[1] = ograds[0];
}

void backward_shape_infer(
        const std::vector<Shape>& ograds, const Param& params,
        std::vector<Shape>& igrads) {
    igrads[0] = ograds[0];
    igrads[1] = ograds[0];
}

void backward_dtype_infer(
        const std::vector<DType>& ograds, const Param& params,
        std::vector<DType>& igrads) {
    igrads[0] = ograds[0];
    igrads[1] = ograds[0];
}

void backward_format_infer(
        const std::vector<Format>& ograds, const Param& params,
        std::vector<Format>& igrads) {
    igrads[0] = ograds[0];
    igrads[1] = ograds[0];
}

template <typename scalar_t>
void backward_kernel(
        const scalar_t* ograd, scalar_t* igrad0, scalar_t* igrad1, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        igrad0[i] = ograd[i];
        igrad1[i] = ograd[i];
    }
}

void backward_compute(
        const std::vector<Tensor>& ograds, const Param& params,
        std::vector<Tensor>& igrads) {
    DISPATCH_SIGN_INT_AND_FLOAT_TYPES(
            igrads[0].dtype(), "backward_compute", ([&]() {
                backward_kernel<scalar_t>(
                        ograds[0].data<scalar_t>(), igrads[0].data<scalar_t>(),
                        igrads[1].data<scalar_t>(), igrads[0].size());
            }));
}

CUSTOM_OP_REG(ElemAddSmoothBackward)
        .set_description(
                "Custom ElemAdd Operator With a Smooth Parameter, "
                "which is used to verify the CPU kernel")
        .add_input("ograd")
        .add_output("igrad_lhs")
        .add_output("igrad_rhs")
        .set_device_infer(backward_device_infer)
        .set_shape_infer(backward_shape_infer)
        .set_dtype_infer(backward_dtype_infer)
        .set_format_infer(backward_format_infer)
        .set_compute(backward_compute);

CUSTOM_OP_REG_END(ElemAddSmooth)
