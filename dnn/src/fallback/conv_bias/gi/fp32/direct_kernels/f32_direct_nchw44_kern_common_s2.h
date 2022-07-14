#include "megdnn/arch.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"
#include "src/fallback/conv_bias/gi/fp32/f32_direct_nchw44_kern.h"
#include "src/fallback/conv_bias/gi/intrinsic_helper.h"
#include "src/fallback/elemwise_helper/elemwise_op.h"

using namespace megdnn;
using namespace fallback;
namespace {

template <
        int src_idx, int weight_idx, int c_dim, int ow_block, int remain_w, typename T,
        typename T2, typename T3, typename T4>
struct ShiftCalHelper {
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight);
};

template <
        int src_idx, int weight_idx, int c_dim, int ow_block, typename T, typename T2,
        typename T3, typename T4>
struct ShiftCalHelper<src_idx, weight_idx, c_dim, ow_block, 0, T, T2, T3, T4> {
    static MEGDNN_ALWAYS_INLINE void impl(T&, T2&, T3&) {}
};

#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
//! x86 and rvv GiSimdFmaLane API is slowly, as an alternate, use
//! GiMultiplyAddScalarFloat32
#define MLA(a, b, c, d)         \
    GiMultiplyAddScalarFloat32( \
            GiFixLenType2GiFloat32Type(a), GiFixLenType2GiFloat32Type(b), *(c + d))
#else
#define MLA(a, b, c, d)                                                   \
    GiSimdFmaLane(                                                        \
            GiFixLenType2GiFloat32Type(a), GiFixLenType2GiFloat32Type(b), \
            GiFixLenType2GiFloat32Type(c), d)
#endif

#define cb2(step, lane, ow_block)                                                      \
    c[0][step] = GiFloat32Type2FixLenType(                                             \
            MLA(c[0][step], weight[0][lane], src[(step + src_idx) % ow_block], lane)); \
    c[1][step] = GiFloat32Type2FixLenType(                                             \
            MLA(c[1][step], weight[1][lane], src[(step + src_idx) % ow_block], lane));

#define cb(step, lane, ow_block)           \
    c[0][step] = GiFloat32Type2FixLenType( \
            MLA(c[0][step], weight[0][lane], src[(step + src_idx) % ow_block], lane));

#define SHIFT_CAL_HELPER(ow_block, remain_w)                                           \
    template <                                                                         \
            int src_idx, int weight_idx, typename T, typename T2, typename T3,         \
            typename T4>                                                               \
    struct ShiftCalHelper<src_idx, weight_idx, 2, ow_block, remain_w, T, T2, T3, T4> { \
        static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight) {             \
            UNROLL_CALL_RAW(remain_w, cb2, 0, ow_block);                               \
            UNROLL_CALL_RAW(remain_w, cb2, 1, ow_block);                               \
            UNROLL_CALL_RAW(remain_w, cb2, 2, ow_block);                               \
            UNROLL_CALL_RAW(remain_w, cb2, 3, ow_block);                               \
        }                                                                              \
    };                                                                                 \
    template <                                                                         \
            int src_idx, int weight_idx, typename T, typename T2, typename T3,         \
            typename T4>                                                               \
    struct ShiftCalHelper<src_idx, weight_idx, 1, ow_block, remain_w, T, T2, T3, T4> { \
        static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight) {             \
            UNROLL_CALL_RAW(remain_w, cb, 0, ow_block);                                \
            UNROLL_CALL_RAW(remain_w, cb, 1, ow_block);                                \
            UNROLL_CALL_RAW(remain_w, cb, 2, ow_block);                                \
            UNROLL_CALL_RAW(remain_w, cb, 3, ow_block);                                \
        }                                                                              \
    };

SHIFT_CAL_HELPER(8, 1);
SHIFT_CAL_HELPER(8, 2);
SHIFT_CAL_HELPER(8, 3);
SHIFT_CAL_HELPER(8, 4);
SHIFT_CAL_HELPER(8, 5);
SHIFT_CAL_HELPER(8, 6);
SHIFT_CAL_HELPER(8, 7);
SHIFT_CAL_HELPER(8, 8);

SHIFT_CAL_HELPER(4, 1);
SHIFT_CAL_HELPER(4, 2);
SHIFT_CAL_HELPER(4, 3);
SHIFT_CAL_HELPER(4, 4);

#undef SHIFT_CAL_HELPER
#undef cb
#undef cb2
#undef MLA

template <
        int src_idx, int weight_idx, int c_dim, int ow_block, int remain_w, typename T,
        typename T2, typename T3>
MEGDNN_ALWAYS_INLINE void cal_helper(T& c, T2& src, T3& weight) {
    ShiftCalHelper<src_idx, weight_idx, c_dim, ow_block, remain_w, T, T2, T3, int>::
            impl(c, src, weight);
};

template <int oc>
struct OCHelper {
public:
    static const int val = -1;
};

template <>
struct OCHelper<4> {
public:
    static const int val = 1;
};
#if MEGDNN_AARCH64
template <>
struct OCHelper<8> {
public:
    static const int val = 2;
};
#endif
/**
 *  oc8_ow8(m = 8, n = 8) and oc4_ow8(m = 4, n = 8) gemm like kernel
 * */
template <
        BiasMode bias_mode, typename Op, int remain_w, int filter_size, int oc_block,
        int ow_block>
struct KerGiXXs2Nchw44FP32 {
    static void impl(
            const float32_t* src_ptr, const float32_t* weight_ptr,
            const float32_t* bias_ptr, float32_t* dst_ptr, int ic, int ih, int iw,
            int ld_dst_oc, const Op& op, const float32_t* src_ptr_odd);
};

template <BiasMode bias_mode, typename Op, int remain_w, int oc_block, int ow_block>
struct KerGiXXs2Nchw44FP32<bias_mode, Op, remain_w, 2, oc_block, ow_block> {
    static void impl(
            const float32_t* src_ptr_origin, const float32_t* weight_ptr,
            const float32_t* bias_ptr, float32_t* dst_ptr, int ic, int ih, int iw,
            int ld_dst_oc, const Op& op, const float32_t* src_ptr_odd_origin) {
        constexpr int loop_ic_step = 4;
        constexpr int filter_size = 2;
        constexpr int oc_step = 4;
        constexpr int simd_len = 4;

        constexpr int ld_weight = oc_step * oc_step;
        const int ld_bias = bias_mode == BiasMode::BIAS ? ld_dst_oc : oc_step;
        const int ld_weight_oc = oc_step * filter_size * filter_size * ic;
        const int ld_weight_fh = oc_step * oc_step * filter_size;
        const int ld_src_ic = ih * iw;
        const int ld_src_iw = iw * oc_step;
        constexpr int c_dim = OCHelper<oc_block>::val;
        GI_FLOAT32_FIXLEN_t c[c_dim][ow_block];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, ld_bias);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            const float* src_ptr = src_ptr_origin + ic_idx * ld_src_ic;
            const float* src_ptr_odd = src_ptr_odd_origin + ic_idx * ld_src_ic;

            GI_FLOAT32_FIXLEN_t weight[c_dim][4];
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
            const float* src[ow_block];
            load_ptr_helper<ow_block, 0, simd_len, 0>(src, src_ptr, 0);
#else
            GI_FLOAT32_FIXLEN_t src[ow_block];
            load_helper<ow_block, 0, simd_len, 0, Vld1qF32S>(src, src_ptr, 0);
#endif
            /////////row 0/////////////
            load_helper<4, 0, oc_step, c_dim, Vld1qF32S>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<0, 0, c_dim, ow_block, remain_w>(c, src, weight);

#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
            load_ptr_helper<ow_block, 0, simd_len, 0>(src, src_ptr_odd, 0);
#else
            load_helper<ow_block, 0, simd_len, 0, Vld1qF32S>(src, src_ptr_odd, 0);
#endif
            load_helper<4, 1 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<0, 0, c_dim, ow_block, remain_w>(c, src, weight);
            src_ptr += ld_src_iw;
            src_ptr_odd += ld_src_iw;
            weight_ptr += ld_weight_fh;
            /////////row 1/////////////
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
            load_ptr_helper<ow_block, 0, simd_len, 0>(src, src_ptr, 0);
#else
            load_helper<ow_block, 0, simd_len, 0, Vld1qF32S>(src, src_ptr, 0);
#endif
            load_helper<4, 0, oc_step, c_dim, Vld1qF32S>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<0, 0, c_dim, ow_block, remain_w>(c, src, weight);

#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
            load_ptr_helper<ow_block, 0, simd_len, 0>(src, src_ptr_odd, 0);
#else
            load_helper<ow_block, 0, simd_len, 0, Vld1qF32S>(src, src_ptr_odd, 0);
#endif
            load_helper<4, 1 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<0, 0, c_dim, ow_block, remain_w>(c, src, weight);
            src_ptr += ld_src_iw;
            src_ptr_odd += ld_src_iw;
            weight_ptr += ld_weight_fh;
        }
        store_ocx_ow8_remain_static<c_dim, remain_w, Op>(c, op, dst_ptr, ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op, int remain_w, int oc_block, int ow_block>
struct KerGiXXs2Nchw44FP32<bias_mode, Op, remain_w, 3, oc_block, ow_block> {
    static void impl(
            const float32_t* src_ptr_origin, const float32_t* weight_ptr,
            const float32_t* bias_ptr, float32_t* dst_ptr, int ic, int ih, int iw,
            int ld_dst_oc, const Op& op, const float32_t* src_ptr_odd_origin) {
        constexpr int loop_ic_step = 4;
        constexpr int filter_size = 3;
        constexpr int oc_step = 4;
        constexpr int simd_len = 4;

        constexpr int ld_weight = oc_step * oc_step;
        const int ld_bias = bias_mode == BiasMode::BIAS ? ld_dst_oc : oc_step;
        const int ld_weight_oc = oc_step * filter_size * filter_size * ic;
        const int ld_weight_fh = oc_step * oc_step * filter_size;
        const int ld_src_ic = ih * iw;
        const int ld_src_iw = iw * oc_step;
        constexpr int c_dim = OCHelper<oc_block>::val;
        GI_FLOAT32_FIXLEN_t c[c_dim][ow_block];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, ld_bias);
        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            const float* src_ptr = src_ptr_origin + ic_idx * ld_src_ic;
            const float* src_ptr_odd = src_ptr_odd_origin + ic_idx * ld_src_ic;

            GI_FLOAT32_FIXLEN_t weight[c_dim][4];
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
            const float* src[ow_block];
            load_ptr_helper<ow_block, 0, simd_len, 0>(src, src_ptr, 0);
#else
            GI_FLOAT32_FIXLEN_t src[ow_block];
            load_helper<ow_block, 0, simd_len, 0, Vld1qF32S>(src, src_ptr, 0);
#endif
            /////////row 0/////////////
            load_helper<4, 0, oc_step, c_dim, Vld1qF32S>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<0, 0, c_dim, ow_block, remain_w>(c, src, weight);

#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
            src[0] = src_ptr + ow_block * simd_len;
#else
            src[0] = GiFloat32Type2FixLenType(
                    GiLoadFloat32(src_ptr + ow_block * simd_len));
#endif
            load_helper<4, 2 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<1, 0, c_dim, ow_block, remain_w>(c, src, weight);

#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
            load_ptr_helper<ow_block, 0, simd_len, 0>(src, src_ptr_odd, 0);
#else
            load_helper<ow_block, 0, simd_len, 0, Vld1qF32S>(src, src_ptr_odd, 0);
#endif
            load_helper<4, 1 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<0, 0, c_dim, ow_block, remain_w>(c, src, weight);
            src_ptr += ld_src_iw;
            src_ptr_odd += ld_src_iw;
            weight_ptr += ld_weight_fh;
            /////////row 1/////////////
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
            load_ptr_helper<ow_block, 0, simd_len, 0>(src, src_ptr, 0);
#else
            load_helper<ow_block, 0, simd_len, 0, Vld1qF32S>(src, src_ptr, 0);
#endif
            load_helper<4, 0, oc_step, c_dim, Vld1qF32S>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<0, 0, c_dim, ow_block, remain_w>(c, src, weight);
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
            src[0] = src_ptr + ow_block * simd_len;
#else
            src[0] = GiFloat32Type2FixLenType(
                    GiLoadFloat32(src_ptr + ow_block * simd_len));
#endif
            load_helper<4, 2 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<1, 0, c_dim, ow_block, remain_w>(c, src, weight);

#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
            load_ptr_helper<ow_block, 0, simd_len, 0>(src, src_ptr_odd, 0);
#else
            load_helper<ow_block, 0, simd_len, 0, Vld1qF32S>(src, src_ptr_odd, 0);
#endif
            load_helper<4, 1 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<0, 0, c_dim, ow_block, remain_w>(c, src, weight);
            src_ptr += ld_src_iw;
            src_ptr_odd += ld_src_iw;
            weight_ptr += ld_weight_fh;
            //////////row 2/////////////
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
            load_ptr_helper<ow_block, 0, simd_len, 0>(src, src_ptr, 0);
#else
            load_helper<ow_block, 0, simd_len, 0, Vld1qF32S>(src, src_ptr, 0);
#endif
            load_helper<4, 0, oc_step, c_dim, Vld1qF32S>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<0, 0, c_dim, ow_block, remain_w>(c, src, weight);
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
            src[0] = src_ptr + ow_block * simd_len;
#else
            src[0] = GiFloat32Type2FixLenType(
                    GiLoadFloat32(src_ptr + ow_block * simd_len));
#endif

            load_helper<4, 2 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<1, 0, c_dim, ow_block, remain_w>(c, src, weight);

#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
            load_ptr_helper<ow_block, 0, simd_len, 0>(src, src_ptr_odd, 0);
#else
            load_helper<ow_block, 0, simd_len, 0, Vld1qF32S>(src, src_ptr_odd, 0);
#endif
            load_helper<4, 1 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<0, 0, c_dim, ow_block, remain_w>(c, src, weight);
            src_ptr += ld_src_iw;
            src_ptr_odd += ld_src_iw;
            weight_ptr += ld_weight_fh;
        }
        store_ocx_ow8_remain_static<c_dim, remain_w, Op>(c, op, dst_ptr, ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op, int remain_w, int oc_block, int ow_block>
struct KerGiXXs2Nchw44FP32<bias_mode, Op, remain_w, 5, oc_block, ow_block> {
    static void impl(
            const float32_t* src_ptr_origin, const float32_t* weight_ptr,
            const float32_t* bias_ptr, float32_t* dst_ptr, int ic, int ih, int iw,
            int ld_dst_oc, const Op& op, const float32_t* src_ptr_odd_origin) {
        constexpr int loop_ic_step = 4;
        constexpr int filter_size = 5;
        constexpr int oc_step = 4;
        constexpr int simd_len = 4;

        constexpr int ld_weight = oc_step * oc_step;
        const int ld_bias = bias_mode == BiasMode::BIAS ? ld_dst_oc : oc_step;
        const int ld_weight_oc = oc_step * filter_size * filter_size * ic;
        const int ld_weight_fh = oc_step * oc_step * filter_size;
        const int ld_src_ic = ih * iw;
        const int ld_src_iw = iw * oc_step;
        constexpr int c_dim = OCHelper<oc_block>::val;
        GI_FLOAT32_FIXLEN_t c[c_dim][ow_block];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, ld_bias);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            const float* src_ptr = src_ptr_origin + ic_idx * ld_src_ic;
            const float* src_ptr_odd = src_ptr_odd_origin + ic_idx * ld_src_ic;

            for (int fh_idx = 0; fh_idx < filter_size; ++fh_idx) {
                GI_FLOAT32_FIXLEN_t weight[c_dim][4];
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
                const float* src[ow_block];
                load_ptr_helper<ow_block, 0, simd_len, 0>(src, src_ptr, 0);
#else
                GI_FLOAT32_FIXLEN_t src[ow_block];
                load_helper<ow_block, 0, simd_len, 0, Vld1qF32S>(src, src_ptr, 0);
#endif
                // even element
                load_helper<4, 0, oc_step, c_dim, Vld1qF32S>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<0, 0, c_dim, ow_block, remain_w>(c, src, weight);
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
                src[0] = src_ptr + ow_block * simd_len;
#else
                src[0] = GiFloat32Type2FixLenType(
                        GiLoadFloat32(src_ptr + ow_block * simd_len));
#endif
                load_helper<4, 2 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<1, 0, c_dim, ow_block, remain_w>(c, src, weight);
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
                src[1] = src_ptr + (ow_block + 1) * simd_len;
#else
                src[1] = GiFloat32Type2FixLenType(
                        GiLoadFloat32(src_ptr + (ow_block + 1) * simd_len));
#endif
                load_helper<4, 4 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<2, 0, c_dim, ow_block, remain_w>(c, src, weight);
                // odd element
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
                load_ptr_helper<ow_block, 0, simd_len, 0>(src, src_ptr_odd, 0);
#else
                load_helper<ow_block, 0, simd_len, 0, Vld1qF32S>(src, src_ptr_odd, 0);
#endif
                load_helper<4, 1 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<0, 0, c_dim, ow_block, remain_w>(c, src, weight);
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
                src[0] = src_ptr_odd + ow_block * simd_len;
#else
                src[0] = GiFloat32Type2FixLenType(
                        GiLoadFloat32(src_ptr_odd + ow_block * simd_len));
#endif
                load_helper<4, 3 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<1, 0, c_dim, ow_block, remain_w>(c, src, weight);

                src_ptr += ld_src_iw;
                src_ptr_odd += ld_src_iw;
                weight_ptr += ld_weight_fh;
            }
        }
        store_ocx_ow8_remain_static<c_dim, remain_w, Op>(c, op, dst_ptr, ld_dst_oc);
    }
};

/**
 * for kernel[7], calculate sequence is kernel[0], kernel[2], kernel[4],
 * kernel[6], kernel[1], kernel[3], kernel[5]
 * src is packed like 0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9
 **/
template <BiasMode bias_mode, typename Op, int remain_w, int oc_block, int ow_block>
struct KerGiXXs2Nchw44FP32<bias_mode, Op, remain_w, 7, oc_block, ow_block> {
    static void impl(
            const float32_t* src_ptr_origin, const float32_t* weight_ptr,
            const float32_t* bias_ptr, float32_t* dst_ptr, int ic, int ih, int iw,
            int ld_dst_oc, const Op& op, const float32_t* src_ptr_odd_origin) {
        constexpr int loop_ic_step = 4;
        constexpr int filter_size = 7;
        constexpr int oc_step = 4;
        constexpr int simd_len = 4;

        constexpr int ld_weight = oc_step * oc_step;
        const int ld_bias = bias_mode == BiasMode::BIAS ? ld_dst_oc : oc_step;
        const int ld_weight_oc = oc_step * filter_size * filter_size * ic;
        const int ld_weight_fh = oc_step * oc_step * filter_size;
        const int ld_src_ic = ih * iw;
        const int ld_src_iw = iw * oc_step;
        constexpr int c_dim = OCHelper<oc_block>::val;
        GI_FLOAT32_FIXLEN_t c[c_dim][ow_block];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, ld_bias);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            const float* src_ptr = src_ptr_origin + ic_idx * ld_src_ic;
            const float* src_ptr_odd = src_ptr_odd_origin + ic_idx * ld_src_ic;

            for (int fh_idx = 0; fh_idx < filter_size; ++fh_idx) {
                GI_FLOAT32_FIXLEN_t weight[c_dim][4];
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
                const float* src[ow_block];
                load_ptr_helper<ow_block, 0, simd_len, 0>(src, src_ptr, 0);
#else
                GI_FLOAT32_FIXLEN_t src[ow_block];
                load_helper<ow_block, 0, simd_len, 0, Vld1qF32S>(src, src_ptr, 0);
#endif
                // even element
                load_helper<4, 0, oc_step, c_dim, Vld1qF32S>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<0, 0, c_dim, ow_block, remain_w>(c, src, weight);
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
                src[0] = src_ptr + ow_block * simd_len;
#else
                src[0] = GiFloat32Type2FixLenType(
                        GiLoadFloat32(src_ptr + ow_block * simd_len));
#endif
                load_helper<4, 2 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<1, 0, c_dim, ow_block, remain_w>(c, src, weight);
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
                src[1] = src_ptr + (ow_block + 1) * simd_len;
#else
                src[1] = GiFloat32Type2FixLenType(
                        GiLoadFloat32(src_ptr + (ow_block + 1) * simd_len));
#endif
                load_helper<4, 4 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<2, 0, c_dim, ow_block, remain_w>(c, src, weight);
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
                src[2] = src_ptr + (ow_block + 2) * simd_len;
#else
                src[2] = GiFloat32Type2FixLenType(
                        GiLoadFloat32(src_ptr + (ow_block + 2) * simd_len));
#endif
                load_helper<4, 6 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<3, 0, c_dim, ow_block, remain_w>(c, src, weight);
                // odd element
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
                load_ptr_helper<ow_block, 0, simd_len, 0>(src, src_ptr_odd, 0);
#else
                load_helper<ow_block, 0, simd_len, 0, Vld1qF32S>(src, src_ptr_odd, 0);
#endif
                load_helper<4, 1 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<0, 0, c_dim, ow_block, remain_w>(c, src, weight);
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
                src[0] = src_ptr_odd + ow_block * simd_len;
#else
                src[0] = GiFloat32Type2FixLenType(
                        GiLoadFloat32(src_ptr_odd + ow_block * simd_len));
#endif
                load_helper<4, 3 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<1, 0, c_dim, ow_block, remain_w>(c, src, weight);
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
                src[1] = src_ptr_odd + (ow_block + 1) * simd_len;
#else
                src[1] = GiFloat32Type2FixLenType(
                        GiLoadFloat32(src_ptr_odd + (ow_block + 1) * simd_len));
#endif
                load_helper<4, 5 * ld_weight, oc_step, c_dim, Vld1qF32S>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<2, 0, c_dim, ow_block, remain_w>(c, src, weight);

                src_ptr += ld_src_iw;
                src_ptr_odd += ld_src_iw;
                weight_ptr += ld_weight_fh;
            }
        }
        store_ocx_ow8_remain_static<c_dim, remain_w, Op>(c, op, dst_ptr, ld_dst_oc);
    }
};

}  // namespace

template <BiasMode bias_mode, typename Op, int filter_size, int stride>
void conv_bias::conv_direct_fp32_nchw44(
        const float* src, const float* filter, const float* bias, float*, float* dst,
        const int oc, const int ic, const int ih, const int iw, const int oh,
        const int oh_block, const int ow, const Op& op, const int, const int) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int ic_step = 4;
#if MEGDNN_AARCH64
    constexpr int big_oc_step = 8;
#else
    constexpr int big_oc_step = 4;
#endif
    constexpr int oc_step = 4;
    constexpr int ih_step = 1;
    constexpr int oh_step = 1;
    constexpr int ow_step = 8;
    constexpr int stride_h = 2;
    constexpr int stride_w = 2;

    const int img_stride = oh * ow;
    const int ow_end = ow / ow_step * ow_step;
    const int ow_remain = ow - ow_end;
    const int oc_end = oc / big_oc_step * big_oc_step;
    const int oc_remain = oc - oc_end;
    const int ld_dst_oc = oc_step * img_stride;
    const int odd_start = div_ceil(iw, 2);

    using remain_fun = std::function<void(
            const float32_t* src_ptr, const float32_t* weight_ptr,
            const float32_t* bias_ptr, float32_t* dst_ptr, int ic, int ih, int iw,
            int ld_dst_oc, const Op& op, const float32_t* src_ptr_odd_origin)>;
    remain_fun kern_big_oc_remain = nullptr;
    remain_fun kern_small_oc_remain = nullptr;

    switch (ow_remain) {
#define cb(step)                                                               \
    case step:                                                                 \
        kern_big_oc_remain = KerGiXXs2Nchw44FP32<                              \
                bias_mode, Op, step, filter_size, big_oc_step, ow_step>::impl; \
        kern_small_oc_remain = KerGiXXs2Nchw44FP32<                            \
                bias_mode, Op, step, filter_size, oc_step, ow_step>::impl;     \
        break;

        UNROLL_CALL_RAW(8, cb);
        default:
            megdnn_assert(0, "no remain %d for kern", ow_remain);
    }
    for (int oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
        const int weight_offset = oc_idx * ic * fh * fw;
        for (int oh_idx = 0; oh_idx < oh_block; oh_idx += oh_step) {
            for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const int src_offset =
                        (oh_idx * stride_h * iw + ow_idx / 2 * stride_w * ih_step) *
                        ic_step;
                const int src_offset_odd =
                        (oh_idx * stride_h * iw + ow_idx / 2 * stride_w * ih_step +
                         odd_start) *
                        ic_step;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                const int bias_offset =
                        bias_mode == BiasMode::BIAS ? dst_offset : oc_idx;
                KerGiXXs2Nchw44FP32<
                        bias_mode, Op, ow_step, filter_size, big_oc_step, ow_step>::
                        impl(src + src_offset, filter + weight_offset,
                             bias + bias_offset, dst + dst_offset, ic, ih, iw,
                             ld_dst_oc, op, src + src_offset_odd);
            }
            if (ow_remain > 0) {
                const int src_offset =
                        (oh_idx * stride_h * iw + ow_end / 2 * stride_w * ih_step) *
                        ic_step;
                const int src_offset_odd =
                        (oh_idx * stride_h * iw + ow_end / 2 * stride_w * ih_step +
                         odd_start) *
                        ic_step;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                const int bias_offset =
                        bias_mode == BiasMode::BIAS ? dst_offset : oc_idx;
                kern_big_oc_remain(
                        src + src_offset, filter + weight_offset, bias + bias_offset,
                        dst + dst_offset, ic, ih, iw, ld_dst_oc, op,
                        src + src_offset_odd);
            }
        }
    }
    if (oc_remain > 0) {
        int oc_idx = oc_end;
        const int weight_offset = oc_idx * ic * fh * fw;
        for (int oh_idx = 0; oh_idx < oh_block; oh_idx += oh_step) {
            for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const int src_offset =
                        (oh_idx * stride_h * iw + ow_idx / 2 * stride_w * ih_step) *
                        ic_step;
                const int src_offset_odd =
                        (oh_idx * stride_h * iw + ow_idx / 2 * stride_w * ih_step +
                         odd_start) *
                        ic_step;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                const int bias_offset =
                        bias_mode == BiasMode::BIAS ? dst_offset : oc_idx;
                KerGiXXs2Nchw44FP32<
                        bias_mode, Op, ow_step, filter_size, oc_step, ow_step>::
                        impl(src + src_offset, filter + weight_offset,
                             bias + bias_offset, dst + dst_offset, ic, ih, iw,
                             ld_dst_oc, op, src + src_offset_odd);
            }
            if (ow_remain > 0) {
                const int src_offset =
                        (oh_idx * stride_h * iw + ow_end / 2 * stride_w * ih_step) *
                        ic_step;
                const int src_offset_odd =
                        (oh_idx * stride_h * iw + ow_end / 2 * stride_w * ih_step +
                         odd_start) *
                        ic_step;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                const int bias_offset =
                        bias_mode == BiasMode::BIAS ? dst_offset : oc_idx;
                kern_small_oc_remain(
                        src + src_offset, filter + weight_offset, bias + bias_offset,
                        dst + dst_offset, ic, ih, iw, ld_dst_oc, op,
                        src + src_offset_odd);
            }
        }
    }
}

#define INSTANTIATION(filter_size, bias_mode, Op)                                    \
    template void conv_bias::conv_direct_fp32_nchw44<bias_mode, Op, filter_size, 2>( \
            const float* src, const float* filter, const float* bias, float*,        \
            float* dst, const int oc, const int ic, const int ih, const int iw,      \
            const int oh, const int oh_block, const int ow, const Op& op, const int, \
            const int);

#define FOR_OP(filter_size, bias)                          \
    INSTANTIATION(filter_size, bias, NoneOp<dt_float32>)   \
    INSTANTIATION(filter_size, bias, ReluOp<dt_float32>)   \
    INSTANTIATION(filter_size, bias, HSwishOp<dt_float32>) \
    INSTANTIATION(filter_size, bias, SigmoidOp<dt_float32>)

#define INSTANTIATION_CONV_S2_NO_BIAS(filter_size) \
    FOR_OP(filter_size, BiasMode::NO_BIAS)

#define INSTANTIATION_CONV_S2_BROADCAST_CHANNEL_BIAS(filter_size) \
    FOR_OP(filter_size, BiasMode::BROADCAST_CHANNEL_BIAS)

#define INSTANTIATION_CONV_S2_BIAS(filter_size) FOR_OP(filter_size, BiasMode::BIAS)

// vim: syntax=cpp.doxygen
