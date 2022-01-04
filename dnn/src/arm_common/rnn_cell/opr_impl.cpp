/**
 * \file dnn/src/arm_common/rnn_cell/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/rnn_cell/opr_impl.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include "src/arm_common/elemwise_helper/kimpl/none.h"
#include "src/arm_common/elemwise_helper/kimpl/relu.h"
#include "src/arm_common/elemwise_helper/kimpl/tanh.h"

#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_rnn_cell)

using namespace megdnn;
using namespace arm_common;

namespace {
ElemwiseForward* get_elemwise_opr() {
    static CpuOprDelegationStorage<1> storage;
    return storage.get<ElemwiseForward>();
}

template <typename Op>
void elemwise_compute(
        float* dst_ptr, float* tmp_ptr, float* ih_ptr, float* hh_ptr, size_t batch,
        size_t length) {
    const constexpr size_t SIMD_8 = 8;
    const constexpr size_t SIMD_4 = 4;
    Op op;
    for (size_t b = 0; b < batch; b++) {
        float* dst = dst_ptr + b * length;
        float* tmp = tmp_ptr + b * length;
        float* ih = ih_ptr;
        float* hh = hh_ptr;
        size_t index = 0;
        for (; index + SIMD_8 - 1 < length; index += SIMD_8) {
            float32x4_t dst0 = vld1q_f32(dst);
            float32x4_t dst1 = vld1q_f32(dst + 4);

            float32x4_t tmp0 = vld1q_f32(tmp);
            float32x4_t tmp1 = vld1q_f32(tmp + 4);

            float32x4_t ih0 = vld1q_f32(ih);
            float32x4_t ih1 = vld1q_f32(ih + 4);

            float32x4_t hh0 = vld1q_f32(hh);
            float32x4_t hh1 = vld1q_f32(hh + 4);

            auto mid0 = vaddq_f32(dst0, tmp0);
            auto mid1 = vaddq_f32(dst1, tmp1);
            auto midd0 = vaddq_f32(ih0, hh0);
            auto midd1 = vaddq_f32(ih1, hh1);
            auto out0 = vaddq_f32(mid0, midd0);
            auto out1 = vaddq_f32(mid1, midd1);

            vst1q_f32(dst, op(out0));
            vst1q_f32(dst + 4, op(out1));

            dst += SIMD_8;
            tmp += SIMD_8;
            ih += SIMD_8;
            hh += SIMD_8;
        }
        for (; index + SIMD_4 - 1 < length; index += SIMD_4) {
            float32x4_t dst0 = vld1q_f32(dst);
            float32x4_t tmp0 = vld1q_f32(tmp);
            float32x4_t ih0 = vld1q_f32(ih);
            float32x4_t hh0 = vld1q_f32(hh);

            auto mid0 = vaddq_f32(dst0, tmp0);
            auto midd0 = vaddq_f32(ih0, hh0);
            auto out0 = vaddq_f32(mid0, midd0);

            vst1q_f32(dst, op(out0));

            dst += SIMD_4;
            tmp += SIMD_4;
            ih += SIMD_4;
            hh += SIMD_4;
        }
        for (; index < length; index++) {
            auto out = dst[0] + tmp[0] + ih[0] + hh[0];
            dst[0] = op(out);
            dst++;
            tmp++;
            ih++;
            hh++;
        }
    }
}

void rnn_cell_post_compute(
        _megdnn_tensor_out dst, _megdnn_tensor_in tmp, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in bias_hh, param::RNNCell::NonlineMode nonline_mode,
        Handle* handle) {
    using NonlineMode = param::RNNCell::NonlineMode;
    megdnn_assert(
            nonline_mode == NonlineMode::RELU || nonline_mode == NonlineMode::TANH ||
                    nonline_mode == NonlineMode::IDENTITY,
            "Now arm only support nonlinear mode Relu, TANH, IDENTITY.");
    if (dst.layout.dtype.enumv() == DTypeEnum::Float32 &&
        dst.layout[1] == bias_ih.layout[1] && bias_ih.layout[0] == 1 &&
        bias_ih.layout.eq_layout(bias_hh.layout)) {
        auto run = [=]() {
            size_t batch = dst.layout[0];
            size_t length = bias_ih.layout.total_nr_elems();
            float* dst_ptr = dst.ptr<float>();
            float* tmp_ptr = tmp.ptr<float>();
            float* ih_ptr = bias_ih.ptr<float>();
            float* hh_ptr = bias_hh.ptr<float>();
            if (nonline_mode == NonlineMode::RELU) {
                elemwise_compute<ReluOp<dt_float32>>(
                        dst_ptr, tmp_ptr, ih_ptr, hh_ptr, batch, length);
            } else if (nonline_mode == NonlineMode::TANH) {
                elemwise_compute<TanhOp<dt_float32>>(
                        dst_ptr, tmp_ptr, ih_ptr, hh_ptr, batch, length);
            } else {
                elemwise_compute<NoneOp<dt_float32>>(
                        dst_ptr, tmp_ptr, ih_ptr, hh_ptr, batch, length);
            }
        };
        MEGDNN_DISPATCH_CPU_KERN(static_cast<naive::HandleImpl*>(handle), run());
    } else {
        //! this opr must be created by inplace handle
        auto elem_opr = get_elemwise_opr();
        auto run = [=]() {
            elem_opr->param().mode = Elemwise::Param::Mode::ADD;
            elem_opr->exec({dst, tmp}, dst);
            elem_opr->exec({dst, bias_ih}, dst);
            elem_opr->exec({dst, bias_hh}, dst);
            // activation
            switch (nonline_mode) {
#define cb(_mode)                                              \
    case NonlineMode::_mode: {                                 \
        elem_opr->param().mode = Elemwise::Param::Mode::_mode; \
        elem_opr->exec({dst}, dst);                            \
        break;                                                 \
    }
                cb(RELU);
                cb(TANH);
#undef cb
                case NonlineMode::IDENTITY:
                    break;
                default:
                    megdnn_throw("unsupport nonlinear mode.");
            }
        };
        MEGDNN_DISPATCH_CPU_KERN(static_cast<naive::HandleImpl*>(handle), run());
    }
}
}  // namespace

WorkspaceBundle RNNCellImpl::get_workspace_bundle(
        const TensorLayout& input, const TensorLayout& weight_ih, const TensorLayout&,
        const TensorLayout& hx, const TensorLayout& weight_hh, const TensorLayout&,
        const TensorLayout& dst) {
    MIDOUT_BEGIN(megdnn_arm_common_rnn_cell, midout_iv(0)) {
        auto opr = handle()->create_operator<MatrixMulForward>();
        opr->param().transposeB = true;
        auto matmul_workspace = std::max(
                opr->get_workspace_in_bytes(input, weight_ih, dst),
                opr->get_workspace_in_bytes(hx, weight_hh, dst));
        auto tmp_workspace = dst.span().dist_byte();
        return WorkspaceBundle{nullptr, {tmp_workspace, matmul_workspace}};
    }
    MIDOUT_END();
}

size_t RNNCellImpl::get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& bias_ih, const TensorLayout& hx,
        const TensorLayout& weight_hh, const TensorLayout& bias_hh,
        const TensorLayout& dst) {
    return get_workspace_bundle(input, weight_ih, bias_ih, hx, weight_hh, bias_hh, dst)
            .total_size_in_bytes();
}

void RNNCellImpl::exec(
        _megdnn_tensor_in input, _megdnn_tensor_in weight_ih, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in hx, _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_hh,
        _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    MIDOUT_BEGIN(megdnn_arm_common_rnn_cell, midout_iv(1)) {
        auto bundle = get_workspace_bundle(
                input.layout, weight_ih.layout, bias_ih.layout, hx.layout,
                weight_hh.layout, bias_hh.layout, dst.layout);
        bundle.set(workspace.raw_ptr);
        auto nonline_mode = param().nonlineMode;

        TensorND tmp{static_cast<void*>(bundle.get(0)), dst.layout};
        auto new_workspace =
                Workspace{static_cast<dt_byte*>(bundle.get(1)), bundle.get_size(1)};
        //! this opr can't be created by inplace handle
        auto opr = handle()->create_operator<MatrixMulForward>();

        opr->param().transposeB = true;
        //! the opr will dispatch compute task to device, so record mode
        //! performance will not be effect
        opr->exec(input, weight_ih, tmp, new_workspace);
        opr->exec(hx, weight_hh, dst, new_workspace);

        //! the optimized post compute, nonlinear(tmp + dst + bias_hx + bias_cx)

        rnn_cell_post_compute(dst, tmp, bias_ih, bias_hh, nonline_mode, handle());
    }
    MIDOUT_END();
}

// vim: syntax=cpp.doxygen
