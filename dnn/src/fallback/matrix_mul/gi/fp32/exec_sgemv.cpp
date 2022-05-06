/**
 * \file dnn/src/fallback/matrix_mul/gi/fp32/exec_sgemv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/fallback/matrix_mul/gi/fp32/exec_sgemv.h"
#include "include/megdnn/oprs.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/general_intrinsic/gi_float.h"

#include "midout.h"
MIDOUT_DECL(megdnn_fp32_gi_sgemv)

using namespace megdnn;
using namespace fallback;

namespace {

void sgemv_gi_naive_n_mk4(
        const float* __restrict A, const float* __restrict B, float* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride) {
    constexpr size_t PACK_SIZE = 4;
    megdnn_assert(
            N == 1 && Bstride == PACK_SIZE && M % PACK_SIZE == 0 && K % PACK_SIZE == 0);
    auto Aptr = A;
    auto Cptr = C;
    size_t m = 0;
    while (m < M) {
        auto Aptr0 = Aptr;
        auto Cptr0 = Cptr;
        GI_FLOAT32_t c[4];
#define INIT(step) c[step] = GiBroadcastFloat32(0.0f);
        UNROLL_CALL_RAW(4, INIT)
#undef INIT
        auto Bptr = B;
        size_t k = 0;
        while (k < K) {
            GI_FLOAT32_t b = GiLoadFloat32(Bptr);
            GI_FLOAT32_V2_t a[2];
#if defined(GI_TEST_NAIVE)
#define LOAD_A(step)                                  \
    a[step].val[0] = GiLoadFloat32(Aptr0 + step * 8); \
    a[step].val[1] = GiLoadFloat32(Aptr0 + step * 8 + 4);
#elif defined(__arm__) || defined(__aarch64__)
#define LOAD_A(step) a[step] = vld1q_f32_x2(Aptr0 + step * 8);
#else
#define LOAD_A(step)                                  \
    a[step].val[0] = GiLoadFloat32(Aptr0 + step * 8); \
    a[step].val[1] = GiLoadFloat32(Aptr0 + step * 8 + 4);
#endif
            UNROLL_CALL_RAW(2, LOAD_A)
#undef LOAD_A

#define COMPT(step) \
    c[step] = GiSimdFmaLane(c[step], a[step / 2].val[step % 2], b, step % 4);
            UNROLL_CALL_RAW(4, COMPT)
#undef COMPT
            Bptr += Bstride;
            Aptr0 += PACK_SIZE * PACK_SIZE;
            k += PACK_SIZE;
        }

#define ADD_C(step, stride) c[step] = GiAddFloat32(c[step], c[step + stride]);
        UNROLL_CALL_RAW(2, ADD_C, 2)
        UNROLL_CALL_RAW(1, ADD_C, 1)
#undef ADD_C
        GiStoreFloat32(Cptr0, c[0]);

        Aptr += Astride;
        Cptr += Cstride;
        m += PACK_SIZE;
    }
}

}  // namespace

namespace megdnn {
namespace fallback {

void gi_gemv_like_mk4(
        const float* __restrict A, const float* __restrict B, float* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert(N == 1 && Bstride == 4);
    MIDOUT_BEGIN(megdnn_fp32_gi_sgemv, midout_iv("F32_GEMV_NCHW_GI_44_N"_hash)) {
        return sgemv_gi_naive_n_mk4(A, B, C, M, N, K, Astride, Bstride, Cstride);
    }
    MIDOUT_END();
}

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
