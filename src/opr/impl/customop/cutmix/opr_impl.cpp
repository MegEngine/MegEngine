#include "./opr_impl.h"
#include <algorithm>
#if MGB_CUSTOM_OP
#include "megbrain/custom/custom.h"

using namespace custom;

template <typename T>
T x86_clip(const T& value, const T& lower_bound, const T& upper_bound) {
    return std::min(std::max(value, lower_bound), upper_bound);
}

template <typename T>
void get_bbox_x86(
        T* cx, T* cy, T* cut_h, T* cut_w, T* bbx1, T* bbx2, T* bby1, T* bby2, size_t B,
        size_t H, size_t W) {
    for (size_t i = 0; i < B; i++) {
        T cuh = static_cast<int>(cut_h[i] / 2);
        T cuw = static_cast<int>(cut_w[i] / 2);
        bbx1[i] = x86_clip<T>(cx[i] - cuh, 0, H);
        bbx2[i] = x86_clip<T>(cx[i] + cuh, 0, H);
        bby1[i] = x86_clip<T>(cy[i] - cuw, 0, W);
        bby2[i] = x86_clip<T>(cy[i] + cuw, 0, W);
    }
}

template <typename T>
void cutmix_kernel_x86(
        T* inp1, T* inp2, const T* bbx1, const T* bbx2, const T* bby1, const T* bby2,
        T* out, size_t total, size_t N, size_t C, size_t H, size_t W) {
    for (size_t i = 0; i < total; i++) {
        int inp_n = i / (C * H * W);
        int h_id = i % (C * H * W);
        h_id = h_id % (H * W);
        h_id = h_id / W;
        int w_id = i % W;
        if ((static_cast<int>(bbx1[inp_n]) <= h_id &&
             h_id < static_cast<int>(bbx2[inp_n])) &&
            (static_cast<int>(bby1[inp_n]) <= w_id &&
             w_id < static_cast<int>(bby2[inp_n]))) {
            out[i] = inp2[i];
        } else {
            out[i] = inp1[i];
        }
    }
}

void launch_x86_kernel(
        const Tensor& inp1, const Tensor& inp2, const Tensor& cx, const Tensor& cy,
        const Tensor& cut_h, const Tensor& cut_w, Tensor& output, Tensor& bbx1,
        Tensor& bbx2, Tensor& bby1, Tensor& bby2) {
    auto inp_shape = inp1.shape();
    size_t b = inp_shape[0];
    size_t c = inp_shape[1];
    size_t h = inp_shape[2];
    size_t w = inp_shape[3];
    size_t total_elem = inp1.size();
    DISPATCH_INT_AND_FLOAT_TYPES(
            cx.dtype(), "get_bbox_x86", ([&]() {
                get_bbox_x86(
                        cx.data<scalar_t>(), cy.data<scalar_t>(),
                        cut_h.data<scalar_t>(), cut_w.data<scalar_t>(),
                        bbx1.data<scalar_t>(), bbx2.data<scalar_t>(),
                        bby1.data<scalar_t>(), bby2.data<scalar_t>(), b, h, w);
            }));

    DISPATCH_INT_AND_FLOAT_TYPES(
            inp1.dtype(), "cutmix_forward_x86", ([&]() {
                cutmix_kernel_x86(
                        inp1.data<scalar_t>(), inp2.data<scalar_t>(),
                        bbx1.data<scalar_t>(), bbx2.data<scalar_t>(),
                        bby1.data<scalar_t>(), bby2.data<scalar_t>(),
                        output.data<scalar_t>(), total_elem, b, c, h, w);
            }));
}

CUSTOM_OP_REG_BEGIN(cutmix_forward)

void forward_shape_infer(
        const std::vector<Shape>& inputs, const Param& params,
        std::vector<Shape>& outputs) {
    outputs[0] = inputs[0];
    outputs[1] = {inputs[0][0]};
    outputs[2] = {inputs[0][0]};
    outputs[3] = {inputs[0][0]};
    outputs[4] = {inputs[0][0]};
}

void cutmix_forward_naive(
        const std::vector<Tensor>& inputs, const Param& params,
        std::vector<Tensor>& outputs) {
    launch_x86_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
            outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]);
}

#if MGB_CUDA
void cutmix_forward_cuda(
        const std::vector<Tensor>& inputs, const Param& params,
        std::vector<Tensor>& outputs) {
    launch_cuda_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
            outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]);
}
#endif

CUSTOM_OP_REG(cutmix_forward)
        .add_inputs(6)
        .add_outputs(5)
        .set_shape_infer(forward_shape_infer)
#if MGB_CUDA
        .set_compute("cuda", cutmix_forward_cuda)
#endif
        .set_compute("x86", cutmix_forward_naive);

CUSTOM_OP_REG_END(cutmix_forward)

#endif