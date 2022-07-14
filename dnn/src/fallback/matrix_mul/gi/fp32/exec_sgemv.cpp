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
        GI_FLOAT32_V4_t c;
#define INIT(step) GiSetSubVectorFloat32V4(c, step, GiBroadcastFloat32(0.0f));
        UNROLL_CALL_RAW(4, INIT)
#undef INIT
        auto Bptr = B;
        size_t k = 0;
        while (k < K) {
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
//! x86 and rvv GiSimdFmaLane API is slowly, as an alternate, use
//! GiMultiplyAddScalarFloat32
#define MLA(a, b, c, d) GiMultiplyAddScalarFloat32(a, b, *(c + d))
            const float* b = Bptr;
#else
#define MLA(a, b, c, d) GiSimdFmaLane(a, b, c, d)
            GI_FLOAT32_t b = GiLoadFloat32(Bptr);
#endif
            GI_FLOAT32_V4_t a;
#define LOAD_A(step) GiSetSubVectorFloat32V4(a, step, GiLoadFloat32(Aptr0 + step * 4));
            UNROLL_CALL_RAW(4, LOAD_A)
#undef LOAD_A

#define COMPT(step)                                                                \
    t = MLA(GiGetSubVectorFloat32V4(c, step), GiGetSubVectorFloat32V4(a, step), b, \
            step % 4);                                                             \
    GiSetSubVectorFloat32V4(c, step, t);

            GI_FLOAT32_t t;
            UNROLL_CALL_RAW(4, COMPT)
#undef COMPT
            Bptr += Bstride;
            Aptr0 += PACK_SIZE * PACK_SIZE;
            k += PACK_SIZE;
#undef MLA
        }

#define ADD_C(step, stride)                             \
    t = GiAddFloat32(                                   \
            GiGetSubVectorFloat32V4(c, step),           \
            GiGetSubVectorFloat32V4(c, step + stride)); \
    GiSetSubVectorFloat32V4(c, step, t);
        GI_FLOAT32_t t;
        UNROLL_CALL_RAW(2, ADD_C, 2)
        UNROLL_CALL_RAW(1, ADD_C, 1)
#undef ADD_C
        GiStoreFloat32(Cptr0, GiGetSubVectorFloat32V4(c, 0));

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
