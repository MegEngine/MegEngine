/**
 * \file dnn/src/naive/lrn/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/lrn/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"
#include <cstring>

namespace {

using namespace megdnn;
using Param = param::LRN;

template <typename T>
void forward(_megdnn_tensor_in src, _megdnn_tensor_out dst,
        const Param &param)
{
    auto N = src.layout.shape[0], C = src.layout.shape[1],
         H = src.layout.shape[2], W = src.layout.shape[3];
    auto sptr = src.ptr<T>(), dptr = dst.ptr<T>();
    auto half_window = param.n / 2;
    rep(n, N) rep(hw, H*W) {
        rep(dc, C) {
            auto didx = n*C*H*W + dc*H*W + hw;
            size_t c_start = (dc >= half_window ? dc - half_window : 0u);
            size_t c_end = std::min(dc + half_window, C - 1);
            float suma2 = 0.0f;
            for (size_t sc = c_start; sc <= c_end; ++sc) {
                auto sidx = n*C*H*W + sc*H*W + hw;
                suma2 += sqr(sptr[sidx]);
            }
            float multiplicand = std::pow(
                    param.k + param.alpha * suma2,
                    -param.beta);
            float multiplier = sptr[didx];
            dptr[didx] = T(multiplier * multiplicand);
        }
    }
}

template <typename T>
void backward(_megdnn_tensor_in src,
        _megdnn_tensor_in /* dst */,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        const Param &param)
{
    auto N = src.layout.shape[0], C = src.layout.shape[1],
         H = src.layout.shape[2], W = src.layout.shape[3];
    auto half_window = param.n / 2;
    auto k = param.k, alpha = param.alpha, beta = param.beta;
    auto sptr = src.ptr<T>(),
         hptr = diff.ptr<T>(),
         gptr = grad.ptr<T>();
    std::fill(gptr, gptr + N*C*H*W, 0);
    rep(n, N) rep(hw, H*W) {
        rep(dc, C) {
            auto didx = n*C*H*W + dc*H*W + hw;
            size_t sc_start = (dc >= half_window ? dc - half_window: 0u);
            size_t sc_end = std::min(dc + half_window, C - 1);
            float tmp = k;
            for (size_t sc = sc_start; sc <= sc_end; ++sc) {
                auto sidx = n*C*H*W + sc*H*W + hw;
                tmp += alpha * sqr(sptr[sidx]);
            }
            for (size_t sc = sc_start; sc <= sc_end; ++sc) {
                auto sidx = n*C*H*W + sc*H*W + hw;
                float res = sptr[didx] *
                    -beta * std::pow(tmp, -beta-1.0f) *
                    2.0f * sptr[sidx] *alpha;
                if (sc == dc) res += std::pow(tmp, -beta);
                gptr[sidx] += T(res * hptr[didx]);
            }
        }
    }
}

} // anonymous namespace

namespace megdnn {
namespace naive {

void LRNForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
#define cb(DType) \
    if (src.layout.dtype == DType()) { \
        MEGDNN_DISPATCH_CPU_KERN_OPR( \
                forward<typename DTypeTrait<DType>::ctype>(src, dst, param())); \
        return; \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_assert_internal(0);
}

void LRNBackwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in dst,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, diff.layout, grad.layout, workspace.size);
#define cb(DType) \
    if (src.layout.dtype == DType()) { \
        MEGDNN_DISPATCH_CPU_KERN_OPR( \
                backward<typename DTypeTrait<DType>::ctype>(\
                    src, dst, diff, grad, param())); \
        return; \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_assert_internal(0);
}

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen
