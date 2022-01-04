/**
 * \file dnn/src/arm_common/lstm_cell/cell_kernel.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./cell_kernel.h"
#include "src/arm_common/lstm_cell/opr_impl.h"
#include "src/common/lstm_cell.h"
#include "src/common/opr_delegate.h"
#include "src/naive/handle.h"

#include "src/arm_common/elemwise_helper/kimpl/sigmoid.h"
#include "src/arm_common/elemwise_helper/kimpl/tanh.h"

using namespace megdnn;
using namespace arm_common;

namespace {
template <class Op, bool bias>
struct ElemwiseCompute {
    static Op op;

    static inline float32x4x2_t compute_8(
            float* dst, float* tmp, float* ih, float* hh) {
        float32x4_t dst0 = vld1q_f32(dst);
        float32x4_t dst1 = vld1q_f32(dst + 4);
        float32x4_t tmp0 = vld1q_f32(tmp);
        float32x4_t tmp1 = vld1q_f32(tmp + 4);

        auto mid0 = vaddq_f32(dst0, tmp0);
        auto mid1 = vaddq_f32(dst1, tmp1);
        float32x4_t out0, out1;
        if (bias) {
            float32x4_t ih0 = vld1q_f32(ih);
            float32x4_t ih1 = vld1q_f32(ih + 4);
            float32x4_t hh0 = vld1q_f32(hh);
            float32x4_t hh1 = vld1q_f32(hh + 4);
            auto midd0 = vaddq_f32(ih0, hh0);
            auto midd1 = vaddq_f32(ih1, hh1);

            out0 = vaddq_f32(mid0, midd0);
            out1 = vaddq_f32(mid1, midd1);
        } else {
            out0 = mid0;
            out1 = mid1;
        }
        return {{op(out0), op(out1)}};
    }

    static inline float32x4_t compute_4(float* dst, float* tmp, float* ih, float* hh) {
        float32x4_t dst0 = vld1q_f32(dst);
        float32x4_t tmp0 = vld1q_f32(tmp);

        auto mid0 = vaddq_f32(dst0, tmp0);
        float32x4_t out0;
        if (bias) {
            float32x4_t ih0 = vld1q_f32(ih);
            float32x4_t hh0 = vld1q_f32(hh);
            auto midd0 = vaddq_f32(ih0, hh0);

            out0 = vaddq_f32(mid0, midd0);
        } else {
            out0 = mid0;
        }
        return op(out0);
    }

    static inline float compute_1(float* dst, float* tmp, float* ih, float* hh) {
        float out;
        if (bias) {
            out = dst[0] + tmp[0] + ih[0] + hh[0];
        } else {
            out = dst[0] + tmp[0];
        }
        return op(out);
    }
};

template <class Op, bool bias>
Op ElemwiseCompute<Op, bias>::op = Op();

template <bool bias>
void rnn_cell_elemwise_compute(
        _megdnn_tensor_out dst, _megdnn_tensor_in tmp, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in bias_hh, _megdnn_tensor_in cx, _megdnn_tensor_out h_new,
        _megdnn_tensor_out c_new) {
    size_t batch = dst.layout[0];
    size_t batch_length = dst.layout.total_nr_elems() / batch;
    size_t base_length = batch_length / 4;
    float *ih_ptr_ = nullptr, *hh_ptr_ = nullptr;
    float* dst_ptr_ = dst.ptr<float>();
    float* tmp_ptr_ = tmp.ptr<float>();
    if (bias) {
        ih_ptr_ = bias_ih.ptr<float>();
        hh_ptr_ = bias_hh.ptr<float>();
    }
    float* cx_ptr_ = cx.ptr<float>();
    float* h_new_ptr_ = h_new.ptr<float>();
    float* c_new_ptr_ = c_new.ptr<float>();

    ElemwiseCompute<SigmoidOp<dt_float32>, bias> sigmoid_compute;
    ElemwiseCompute<TanhOp<dt_float32>, bias> tanh_compute;
    TanhOp<dt_float32> tanh_op;

    for (size_t b = 0; b < batch; b++) {
        float* dst_ptr = dst_ptr_ + b * batch_length;
        float* tmp_ptr = tmp_ptr_ + b * batch_length;
        float* ih_ptr = ih_ptr_;
        float* hh_ptr = hh_ptr_;
        float* cx_ptr = cx_ptr_ + b * base_length;
        float* h_new_ptr = h_new_ptr_ + b * base_length;
        float* c_new_ptr = c_new_ptr_ + b * base_length;
        size_t index = 0;
        for (; index + 7 < base_length; index += 8) {
            auto out_i = sigmoid_compute.compute_8(dst_ptr, tmp_ptr, ih_ptr, hh_ptr);
            auto out_f = sigmoid_compute.compute_8(
                    dst_ptr + base_length, tmp_ptr + base_length, ih_ptr + base_length,
                    hh_ptr + base_length);
            auto out_g = tanh_compute.compute_8(
                    dst_ptr + 2 * base_length, tmp_ptr + 2 * base_length,
                    ih_ptr + 2 * base_length, hh_ptr + 2 * base_length);
            auto out_o = sigmoid_compute.compute_8(
                    dst_ptr + 3 * base_length, tmp_ptr + 3 * base_length,
                    ih_ptr + 3 * base_length, hh_ptr + 3 * base_length);
            float32x4_t cx_0 = vld1q_f32(cx_ptr);
            float32x4_t cx_1 = vld1q_f32(cx_ptr + 4);

            //! f * cx + i * g
            auto c_new_0 = vaddq_f32(
                    vmulq_f32(out_f.val[0], cx_0),
                    vmulq_f32(out_i.val[0], out_g.val[0]));
            auto c_new_1 = vaddq_f32(
                    vmulq_f32(out_f.val[1], cx_1),
                    vmulq_f32(out_i.val[1], out_g.val[1]));
            vst1q_f32(c_new_ptr, c_new_0);
            vst1q_f32(c_new_ptr + 4, c_new_1);

            auto h_new_0 = vmulq_f32(tanh_op(c_new_0), out_o.val[0]);
            auto h_new_1 = vmulq_f32(tanh_op(c_new_1), out_o.val[1]);

            vst1q_f32(h_new_ptr, h_new_0);
            vst1q_f32(h_new_ptr + 4, h_new_1);

            dst_ptr += 8;
            tmp_ptr += 8;
            ih_ptr += 8;
            hh_ptr += 8;
            cx_ptr += 8;
            c_new_ptr += 8;
            h_new_ptr += 8;
        }
        for (; index + 3 < base_length; index += 4) {
            auto out_i = sigmoid_compute.compute_4(dst_ptr, tmp_ptr, ih_ptr, hh_ptr);
            auto out_f = sigmoid_compute.compute_4(
                    dst_ptr + base_length, tmp_ptr + base_length, ih_ptr + base_length,
                    hh_ptr + base_length);
            auto out_g = tanh_compute.compute_4(
                    dst_ptr + 2 * base_length, tmp_ptr + 2 * base_length,
                    ih_ptr + 2 * base_length, hh_ptr + 2 * base_length);
            auto out_o = sigmoid_compute.compute_4(
                    dst_ptr + 3 * base_length, tmp_ptr + 3 * base_length,
                    ih_ptr + 3 * base_length, hh_ptr + 3 * base_length);
            float32x4_t cx_v = vld1q_f32(cx_ptr);

            //! f * cx + i * g
            auto c_new = vaddq_f32(vmulq_f32(out_f, cx_v), vmulq_f32(out_i, out_g));
            vst1q_f32(c_new_ptr, c_new);

            auto h_new = vmulq_f32(tanh_op(c_new), out_o);

            vst1q_f32(h_new_ptr, h_new);

            dst_ptr += 4;
            tmp_ptr += 4;
            ih_ptr += 4;
            hh_ptr += 4;
            cx_ptr += 4;
            c_new_ptr += 4;
            h_new_ptr += 4;
        }
        for (; index < base_length; index++) {
            auto out_i = sigmoid_compute.compute_1(dst_ptr, tmp_ptr, ih_ptr, hh_ptr);
            auto out_f = sigmoid_compute.compute_1(
                    dst_ptr + base_length, tmp_ptr + base_length, ih_ptr + base_length,
                    hh_ptr + base_length);
            auto out_g = tanh_compute.compute_1(
                    dst_ptr + 2 * base_length, tmp_ptr + 2 * base_length,
                    ih_ptr + 2 * base_length, hh_ptr + 2 * base_length);
            auto out_o = sigmoid_compute.compute_1(
                    dst_ptr + 3 * base_length, tmp_ptr + 3 * base_length,
                    ih_ptr + 3 * base_length, hh_ptr + 3 * base_length);
            c_new_ptr[0] = out_f * cx_ptr[0] + out_i * out_g;
            h_new_ptr[0] = tanh_op(c_new_ptr[0]) * out_o;

            dst_ptr += 1;
            tmp_ptr += 1;
            ih_ptr += 1;
            hh_ptr += 1;
            cx_ptr += 1;
            c_new_ptr += 1;
            h_new_ptr += 1;
        }
    }
}
}  // namespace

void LstmCellCompute::run(
        _megdnn_tensor_in input, _megdnn_tensor_in weight_ih, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in hx, _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_hh,
        _megdnn_tensor_in cx, _megdnn_tensor_out h_new, _megdnn_tensor_out c_new,
        _megdnn_tensor_out gates, _megdnn_workspace workspace, Handle* handle) {
    auto bundle = get_workspace_bundle(
            input.layout, weight_ih.layout, bias_ih.layout, hx.layout, weight_hh.layout,
            bias_hh.layout, cx.layout, h_new.layout, c_new.layout, gates.layout);
    bundle.set(workspace.raw_ptr);
    TensorND tmp{static_cast<void*>(bundle.get(0)), gates.layout};
    auto matmul_workspace =
            megdnn::Workspace{static_cast<dt_byte*>(bundle.get(1)), bundle.get_size(1)};
    auto opr = handle->create_operator<MatrixMul>();
    opr->param().transposeB = true;
    //! the opr will dispatch compute task to device, so record mode
    //! performance will not be effect
    opr->exec(input, weight_ih, tmp, matmul_workspace);
    opr->exec(hx, weight_hh, gates, matmul_workspace);

    //! the optimized post compute, nonlinear(tmp + dst + bias_hx + bias_cx)
    if (bias_ih.layout.ndim != 0 && bias_ih.layout.ndim != 0) {
        MEGDNN_DISPATCH_CPU_KERN(
                static_cast<naive::HandleImpl*>(handle),
                rnn_cell_elemwise_compute<true>(
                        gates, tmp, bias_ih, bias_hh, cx, h_new, c_new));
    } else {
        megdnn_assert(bias_ih.layout.ndim == 0 && bias_ih.layout.ndim == 0);
        MEGDNN_DISPATCH_CPU_KERN(
                static_cast<naive::HandleImpl*>(handle),
                rnn_cell_elemwise_compute<false>(
                        gates, tmp, bias_ih, bias_hh, cx, h_new, c_new));
    }
}

WorkspaceBundle LstmCellCompute::get_workspace_bundle(
        const TensorLayout& input, const TensorLayout& weight_ih, const TensorLayout&,
        const TensorLayout& hx, const TensorLayout& weight_hh, const TensorLayout&,
        const TensorLayout&, const TensorLayout&, const TensorLayout&,
        const TensorLayout& gates) {
    auto opr = inplace_cpu_handle()->create_operator<MatrixMul>();
    opr->param().transposeB = true;
    size_t matmul_workspace = std::max(
            opr->get_workspace_in_bytes(input, weight_ih, gates),
            opr->get_workspace_in_bytes(hx, weight_hh, gates));
    return WorkspaceBundle{nullptr, {gates.span().dist_byte(), matmul_workspace}};
}

bool LstmCellCompute::is_optimized(
        const TensorLayout& input, const TensorLayout&, const TensorLayout& bias_ih,
        const TensorLayout&, const TensorLayout&, const TensorLayout& bias_hh,
        const TensorLayout&, const TensorLayout&, const TensorLayout&,
        const TensorLayout& gates) {
    if (input.dtype.enumv() == DTypeEnum::Float32 && gates[1] == bias_ih[1] &&
        bias_ih[0] == 1 && bias_ih.eq_layout(bias_hh)) {
        return true;
    } else {
        return false;
    }
}

// vim: syntax=cpp.doxygen
