#pragma once
#include "src/common/unroll_macro.h"
#include "src/fallback/general_intrinsic/gi_float.h"

#define ADDF    GiAddFloat32
#define ADDFV2  GiAddFloat32V2
#define SUBF    GiSubtractFloat32
#define SUBFV2  GiSubtractFloat32V2
#define MULF    GiMultiplyFloat32
#define MULFV2  GiMultiplyFloat32V2
#define MULSF   GiMultiplyScalerFloat32
#define MULSFV2 GiMultiplyScalerFloat32V2

namespace megdnn {
namespace fallback {
inline void transpose_4x4(const float* src, float* dst, int lda, int ldb) {
    GI_FLOAT32_V2_t a0, a1;
    GiSetSubVectorFloat32V2(a0, 0, GiLoadFloat32(src + 0 * lda));
    GiSetSubVectorFloat32V2(a0, 1, GiLoadFloat32(src + 1 * lda));
    GiSetSubVectorFloat32V2(a1, 0, GiLoadFloat32(src + 2 * lda));
    GiSetSubVectorFloat32V2(a1, 1, GiLoadFloat32(src + 3 * lda));
    GI_FLOAT32_V2_t b0 = GiZipqFloat32(
            GiGetSubVectorFloat32V2(a0, 0), GiGetSubVectorFloat32V2(a1, 0));
    GI_FLOAT32_V2_t b1 = GiZipqFloat32(
            GiGetSubVectorFloat32V2(a0, 1), GiGetSubVectorFloat32V2(a1, 1));
    GI_FLOAT32_V2_t c0 = GiZipqFloat32(
            GiGetSubVectorFloat32V2(b0, 0), GiGetSubVectorFloat32V2(b1, 0));
    GI_FLOAT32_V2_t c1 = GiZipqFloat32(
            GiGetSubVectorFloat32V2(b0, 1), GiGetSubVectorFloat32V2(b1, 1));
    GiStoreFloat32(dst + 0 * ldb, GiGetSubVectorFloat32V2(c0, 0));
    GiStoreFloat32(dst + 1 * ldb, GiGetSubVectorFloat32V2(c0, 1));
    GiStoreFloat32(dst + 2 * ldb, GiGetSubVectorFloat32V2(c1, 0));
    GiStoreFloat32(dst + 3 * ldb, GiGetSubVectorFloat32V2(c1, 1));
}
}  // namespace fallback
}  // namespace megdnn

#define MATRIX_MUL4x4(sum, a, b)                           \
    sum##0 = GiMlaqLowLaneFloat32(sum##0, b##0, a##0, 0);  \
    sum##0 = GiMlaqLowLaneFloat32(sum##0, b##1, a##0, 1);  \
    sum##0 = GiMlaqHighLaneFloat32(sum##0, b##2, a##0, 2); \
    sum##0 = GiMlaqHighLaneFloat32(sum##0, b##3, a##0, 3); \
    sum##1 = GiMlaqLowLaneFloat32(sum##1, b##0, a##1, 0);  \
    sum##1 = GiMlaqLowLaneFloat32(sum##1, b##1, a##1, 1);  \
    sum##1 = GiMlaqHighLaneFloat32(sum##1, b##2, a##1, 2); \
    sum##1 = GiMlaqHighLaneFloat32(sum##1, b##3, a##1, 3); \
    sum##2 = GiMlaqLowLaneFloat32(sum##2, b##0, a##2, 0);  \
    sum##2 = GiMlaqLowLaneFloat32(sum##2, b##1, a##2, 1);  \
    sum##2 = GiMlaqHighLaneFloat32(sum##2, b##2, a##2, 2); \
    sum##2 = GiMlaqHighLaneFloat32(sum##2, b##3, a##2, 3); \
    sum##3 = GiMlaqLowLaneFloat32(sum##3, b##0, a##3, 0);  \
    sum##3 = GiMlaqLowLaneFloat32(sum##3, b##1, a##3, 1);  \
    sum##3 = GiMlaqHighLaneFloat32(sum##3, b##2, a##3, 2); \
    sum##3 = GiMlaqHighLaneFloat32(sum##3, b##3, a##3, 3);

#define CONCAT(a, idx) a##idx

#if MEGDNN_AARCH64
#define TRANSPOSE_8x8(a, ret)                                              \
    do {                                                                   \
        auto b0 = GiZipqFloat32(CONCAT(a, 0).val[0], CONCAT(a, 1).val[0]); \
        auto b1 = GiZipqFloat32(CONCAT(a, 0).val[1], CONCAT(a, 1).val[1]); \
        auto b2 = GiZipqFloat32(CONCAT(a, 2).val[0], CONCAT(a, 3).val[0]); \
        auto b3 = GiZipqFloat32(CONCAT(a, 2).val[1], CONCAT(a, 3).val[1]); \
        auto b4 = GiZipqFloat32(CONCAT(a, 4).val[0], CONCAT(a, 5).val[0]); \
        auto b5 = GiZipqFloat32(CONCAT(a, 4).val[1], CONCAT(a, 5).val[1]); \
        auto b6 = GiZipqFloat32(CONCAT(a, 6).val[0], CONCAT(a, 7).val[0]); \
        auto b7 = GiZipqFloat32(CONCAT(a, 6).val[1], CONCAT(a, 7).val[1]); \
        CONCAT(ret, 0).val[0] = GiReinterpretqS64ToFloat32(GiZip1qS64(     \
                GiReinterpretqFloat32ToS64(b0.val[0]),                     \
                GiReinterpretqFloat32ToS64(b2.val[0])));                   \
        CONCAT(ret, 0).val[1] = GiReinterpretqS64ToFloat32(GiZip1qS64(     \
                GiReinterpretqFloat32ToS64(b4.val[0]),                     \
                GiReinterpretqFloat32ToS64(b6.val[0])));                   \
        CONCAT(ret, 1).val[0] = GiReinterpretqS64ToFloat32(GiZip2qS64(     \
                GiReinterpretqFloat32ToS64(b0.val[0]),                     \
                GiReinterpretqFloat32ToS64(b2.val[0])));                   \
        CONCAT(ret, 1).val[1] = GiReinterpretqS64ToFloat32(GiZip2qS64(     \
                GiReinterpretqFloat32ToS64(b4.val[0]),                     \
                GiReinterpretqFloat32ToS64(b6.val[0])));                   \
        CONCAT(ret, 2).val[0] = GiReinterpretqS64ToFloat32(GiZip1qS64(     \
                GiReinterpretqFloat32ToS64(b0.val[1]),                     \
                GiReinterpretqFloat32ToS64(b2.val[1])));                   \
        CONCAT(ret, 2).val[1] = GiReinterpretqS64ToFloat32(GiZip1qS64(     \
                GiReinterpretqFloat32ToS64(b4.val[1]),                     \
                GiReinterpretqFloat32ToS64(b6.val[1])));                   \
        CONCAT(ret, 3).val[0] = GiReinterpretqS64ToFloat32(GiZip2qS64(     \
                GiReinterpretqFloat32ToS64(b0.val[1]),                     \
                GiReinterpretqFloat32ToS64(b2.val[1])));                   \
        CONCAT(ret, 3).val[1] = GiReinterpretqS64ToFloat32(GiZip2qS64(     \
                GiReinterpretqFloat32ToS64(b4.val[1]),                     \
                GiReinterpretqFloat32ToS64(b6.val[1])));                   \
        CONCAT(ret, 4).val[0] = GiReinterpretqS64ToFloat32(GiZip1qS64(     \
                GiReinterpretqFloat32ToS64(b1.val[0]),                     \
                GiReinterpretqFloat32ToS64(b3.val[0])));                   \
        CONCAT(ret, 4).val[1] = GiReinterpretqS64ToFloat32(GiZip1qS64(     \
                GiReinterpretqFloat32ToS64(b5.val[0]),                     \
                GiReinterpretqFloat32ToS64(b7.val[0])));                   \
        CONCAT(ret, 5).val[0] = GiReinterpretqS64ToFloat32(GiZip2qS64(     \
                GiReinterpretqFloat32ToS64(b1.val[0]),                     \
                GiReinterpretqFloat32ToS64(b3.val[0])));                   \
        CONCAT(ret, 5).val[1] = GiReinterpretqS64ToFloat32(GiZip2qS64(     \
                GiReinterpretqFloat32ToS64(b5.val[0]),                     \
                GiReinterpretqFloat32ToS64(b7.val[0])));                   \
        CONCAT(ret, 6).val[0] = GiReinterpretqS64ToFloat32(GiZip1qS64(     \
                GiReinterpretqFloat32ToS64(b1.val[1]),                     \
                GiReinterpretqFloat32ToS64(b3.val[1])));                   \
        CONCAT(ret, 6).val[1] = GiReinterpretqS64ToFloat32(GiZip1qS64(     \
                GiReinterpretqFloat32ToS64(b5.val[1]),                     \
                GiReinterpretqFloat32ToS64(b7.val[1])));                   \
        CONCAT(ret, 7).val[0] = GiReinterpretqS64ToFloat32(GiZip2qS64(     \
                GiReinterpretqFloat32ToS64(b1.val[1]),                     \
                GiReinterpretqFloat32ToS64(b3.val[1])));                   \
        CONCAT(ret, 7).val[1] = GiReinterpretqS64ToFloat32(GiZip2qS64(     \
                GiReinterpretqFloat32ToS64(b5.val[1]),                     \
                GiReinterpretqFloat32ToS64(b7.val[1])));                   \
    } while (0);

#define TRANSPOSE_8x3(a, ret)                                      \
    auto b0 = GiZipqFloat32(CONCAT(a, 0), CONCAT(a, 1));           \
    auto b1 = GiZipqFloat32(CONCAT(a, 2), CONCAT(a, 3));           \
    auto b2 = GiZipqFloat32(CONCAT(a, 4), CONCAT(a, 5));           \
    auto b3 = GiZipqFloat32(CONCAT(a, 6), CONCAT(a, 7));           \
    CONCAT(ret, 0).val[0] = GiReinterpretqS64ToFloat32(GiZip1qS64( \
            GiReinterpretqFloat32ToS64(b0.val[0]),                 \
            GiReinterpretqFloat32ToS64(b1.val[0])));               \
    CONCAT(ret, 0).val[1] = GiReinterpretqS64ToFloat32(GiZip1qS64( \
            GiReinterpretqFloat32ToS64(b2.val[0]),                 \
            GiReinterpretqFloat32ToS64(b3.val[0])));               \
    CONCAT(ret, 1).val[0] = GiReinterpretqS64ToFloat32(GiZip2qS64( \
            GiReinterpretqFloat32ToS64(b0.val[0]),                 \
            GiReinterpretqFloat32ToS64(b1.val[0])));               \
    CONCAT(ret, 1).val[1] = GiReinterpretqS64ToFloat32(GiZip2qS64( \
            GiReinterpretqFloat32ToS64(b2.val[0]),                 \
            GiReinterpretqFloat32ToS64(b3.val[0])));               \
    CONCAT(ret, 2).val[0] = GiReinterpretqS64ToFloat32(GiZip1qS64( \
            GiReinterpretqFloat32ToS64(b0.val[1]),                 \
            GiReinterpretqFloat32ToS64(b1.val[1])));               \
    CONCAT(ret, 2).val[1] = GiReinterpretqS64ToFloat32(GiZip1qS64( \
            GiReinterpretqFloat32ToS64(b2.val[1]),                 \
            GiReinterpretqFloat32ToS64(b3.val[1])));

#define TRANSPOSE_8x4(a, ret)                                      \
    auto b0 = GiZipqFloat32(CONCAT(a, 0), CONCAT(a, 1));           \
    auto b1 = GiZipqFloat32(CONCAT(a, 2), CONCAT(a, 3));           \
    auto b2 = GiZipqFloat32(CONCAT(a, 4), CONCAT(a, 5));           \
    auto b3 = GiZipqFloat32(CONCAT(a, 6), CONCAT(a, 7));           \
    CONCAT(ret, 0).val[0] = GiReinterpretqS64ToFloat32(GiZip1qS64( \
            GiReinterpretqFloat32ToS64(b0.val[0]),                 \
            GiReinterpretqFloat32ToS64(b1.val[0])));               \
    CONCAT(ret, 0).val[1] = GiReinterpretqS64ToFloat32(GiZip1qS64( \
            GiReinterpretqFloat32ToS64(b2.val[0]),                 \
            GiReinterpretqFloat32ToS64(b3.val[0])));               \
    CONCAT(ret, 1).val[0] = GiReinterpretqS64ToFloat32(GiZip2qS64( \
            GiReinterpretqFloat32ToS64(b0.val[0]),                 \
            GiReinterpretqFloat32ToS64(b1.val[0])));               \
    CONCAT(ret, 1).val[1] = GiReinterpretqS64ToFloat32(GiZip2qS64( \
            GiReinterpretqFloat32ToS64(b2.val[0]),                 \
            GiReinterpretqFloat32ToS64(b3.val[0])));               \
    CONCAT(ret, 2).val[0] = GiReinterpretqS64ToFloat32(GiZip1qS64( \
            GiReinterpretqFloat32ToS64(b0.val[1]),                 \
            GiReinterpretqFloat32ToS64(b1.val[1])));               \
    CONCAT(ret, 2).val[1] = GiReinterpretqS64ToFloat32(GiZip1qS64( \
            GiReinterpretqFloat32ToS64(b2.val[1]),                 \
            GiReinterpretqFloat32ToS64(b3.val[1])));               \
    CONCAT(ret, 3).val[0] = GiReinterpretqS64ToFloat32(GiZip2qS64( \
            GiReinterpretqFloat32ToS64(b0.val[1]),                 \
            GiReinterpretqFloat32ToS64(b1.val[1])));               \
    CONCAT(ret, 3).val[1] = GiReinterpretqS64ToFloat32(GiZip2qS64( \
            GiReinterpretqFloat32ToS64(b2.val[1]),                 \
            GiReinterpretqFloat32ToS64(b3.val[1])));

#else
#define TRANSPOSE_8x4(a, ret)                                           \
    auto b0 = GiZipqFloat32(CONCAT(a, 0), CONCAT(a, 1));                \
    auto b1 = GiZipqFloat32(CONCAT(a, 2), CONCAT(a, 3));                \
    auto b2 = GiZipqFloat32(CONCAT(a, 4), CONCAT(a, 5));                \
    auto b3 = GiZipqFloat32(CONCAT(a, 6), CONCAT(a, 7));                \
    GiSetSubVectorFloat32V2(                                            \
            CONCAT(ret, 0), 0,                                          \
            GiCombineFloat32(                                           \
                    GiGetLowFloat32(GiGetSubVectorFloat32V2(b0, 0)),    \
                    GiGetLowFloat32(GiGetSubVectorFloat32V2(b1, 0))));  \
    GiSetSubVectorFloat32V2(                                            \
            CONCAT(ret, 1), 0,                                          \
            GiCombineFloat32(                                           \
                    GiGetHighFloat32(GiGetSubVectorFloat32V2(b0, 0)),   \
                    GiGetHighFloat32(GiGetSubVectorFloat32V2(b1, 0)))); \
    GiSetSubVectorFloat32V2(                                            \
            CONCAT(ret, 2), 0,                                          \
            GiCombineFloat32(                                           \
                    GiGetLowFloat32(GiGetSubVectorFloat32V2(b0, 1)),    \
                    GiGetLowFloat32(GiGetSubVectorFloat32V2(b1, 1))));  \
    GiSetSubVectorFloat32V2(                                            \
            CONCAT(ret, 3), 0,                                          \
            GiCombineFloat32(                                           \
                    GiGetHighFloat32(GiGetSubVectorFloat32V2(b0, 1)),   \
                    GiGetHighFloat32(GiGetSubVectorFloat32V2(b1, 1)))); \
    GiSetSubVectorFloat32V2(                                            \
            CONCAT(ret, 0), 1,                                          \
            GiCombineFloat32(                                           \
                    GiGetLowFloat32(GiGetSubVectorFloat32V2(b2, 0)),    \
                    GiGetLowFloat32(GiGetSubVectorFloat32V2(b3, 0))));  \
    GiSetSubVectorFloat32V2(                                            \
            CONCAT(ret, 1), 1,                                          \
            GiCombineFloat32(                                           \
                    GiGetHighFloat32(GiGetSubVectorFloat32V2(b2, 0)),   \
                    GiGetHighFloat32(GiGetSubVectorFloat32V2(b3, 0)))); \
    GiSetSubVectorFloat32V2(                                            \
            CONCAT(ret, 2), 1,                                          \
            GiCombineFloat32(                                           \
                    GiGetLowFloat32(GiGetSubVectorFloat32V2(b2, 1)),    \
                    GiGetLowFloat32(GiGetSubVectorFloat32V2(b3, 1))));  \
    GiSetSubVectorFloat32V2(                                            \
            CONCAT(ret, 3), 1,                                          \
            GiCombineFloat32(                                           \
                    GiGetHighFloat32(GiGetSubVectorFloat32V2(b2, 1)),   \
                    GiGetHighFloat32(GiGetSubVectorFloat32V2(b3, 1))));

#endif
// vim: syntax=cpp.doxygen
