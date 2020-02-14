/**
 * \file dnn/src/x86/lrn/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/x86/lrn/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "src/x86/utils.h"
#include "src/x86/simd_helper.h"

namespace {

using namespace megdnn;
using namespace x86;

template <SIMDType simd_type>
void lrn_single_instance(const float * __restrict src,
        float * __restrict dst,
        size_t C, size_t H, size_t W,
        size_t n, float k, float alpha, float beta)
{
    using type = typename simd_traits<simd_type>::type;
    static MEGDNN_CONSTEXPR auto width = simd_traits<simd_type>::width;
    auto HW = H*W;
    auto half_n = n / 2;
    auto loadu = &simd_traits<simd_type>::loadu;
    auto storeu = &simd_traits<simd_type>::storeu;
    auto mul = &simd_traits<simd_type>::mul;
    auto fmadd = &simd_traits<simd_type>::fmadd;
    auto set1 = &simd_traits<simd_type>::set1;
    auto exp = &simd_traits<simd_type>::exp;
    auto log = &simd_traits<simd_type>::log;
    type vk = set1(k);
    type valpha = set1(alpha);
    type vnbeta = set1(-beta);
    rep(c, C) {
        auto sptr = src + c*HW;
        auto dptr = dst + c*HW;
        size_t hw = 0u;
        size_t c_start = (c >= half_n ? c - half_n : 0u);
        size_t c_end = std::min(c + half_n, C - 1);
        for (; hw+width <= HW; hw += width, sptr += width, dptr += width) {
            type suma2 = simd_traits<simd_type>::setzero();
            for (size_t sc = c_start; sc <= c_end; ++sc) {
                type sval = loadu(src + (sc*H*W + hw));
                suma2 = fmadd(sval, sval, suma2);
            }
            type a = fmadd(valpha, suma2, vk);
            type b = vnbeta;
            type multiplicand = exp(mul(b, log(a)));
            type multiplier = loadu(sptr);
            type res = mul(multiplier, multiplicand);
            storeu(dptr, res);
        }
        for (; hw < HW; ++hw, ++sptr, ++dptr) {
            float suma2 = 0.0f;
            for (size_t sc = c_start; sc <= c_end; ++sc) {
                float sval = src[sc*HW + hw];
                suma2 += sqr(sval);
            }
            float_t multiplicand = std::pow(
                    k + alpha * suma2,
                    -beta);
            float_t multiplier = *sptr;
            *dptr = multiplicand * multiplier;
        }
    }
}

template MEGDNN_ATTRIBUTE_TARGET("fma")
void lrn_single_instance<SIMDType::FMA>(const float *,
        float *,
        size_t, size_t, size_t,
        size_t, float, float, float);
template MEGDNN_ATTRIBUTE_TARGET("avx")
void lrn_single_instance<SIMDType::AVX>(const float *,
        float *,
        size_t, size_t, size_t,
        size_t, float, float, float);
template MEGDNN_ATTRIBUTE_TARGET("sse")
void lrn_single_instance<SIMDType::SSE>(const float *,
        float *,
        size_t, size_t, size_t,
        size_t, float, float, float);

} // anonymous namespace

namespace megdnn {
namespace x86 {

void LRNImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
    auto N = src.layout.shape[0], C = src.layout.shape[1],
         H = src.layout.shape[2], W = src.layout.shape[3];
    auto sptr_ = src.ptr<dt_float32>(), dptr_ = dst.ptr<dt_float32>();

    std::function<void(const float *, float *,
            size_t, size_t, size_t,
            size_t, float, float, float)> f = nullptr;
    if (is_supported(SIMDType::FMA)) {
        f = &lrn_single_instance<SIMDType::FMA>;
    } else if (is_supported(SIMDType::AVX)) {
        f = &lrn_single_instance<SIMDType::AVX>;
    } else if (is_supported(SIMDType::SSE)) {
        f = &lrn_single_instance<SIMDType::SSE>;
    } else {
        megdnn_throw(megdnn_mangle("no fma/avx/sse detected"));
    }
    auto n = param().n;
    auto k = param().k;
    auto alpha = param().alpha;
    auto beta = param().beta;
    MEGDNN_DISPATCH_CPU_KERN_OPR(
        auto sptr = sptr_;
        auto dptr = dptr_;
        rep(i, N) {
            f(sptr, dptr,
                    C, H, W,
                    n, k, alpha, beta);
            sptr += C*H*W;
            dptr += C*H*W;
        }
    );
}

} // namespace x86
} // namespace megdnn

// vim: syntax=cpp.doxygen
