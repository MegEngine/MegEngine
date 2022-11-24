#pragma once
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "src/arm_common/conv_bias/intrinsic_helper.h"
#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/arm_common/intrinsic_helper.h"
#include "src/arm_common/neon_struct.h"
#include "src/common/unroll_macro.h"
#include "src/fallback/conv_bias/common.h"
namespace megdnn {
namespace arm_common {
namespace {

/**
 * @brief Say src -> (N, IC, IH, IW), weight -> (OC, IC, FH, FW), bias -> (1, OC, 1, 1)
 *      Calculate (n, ic, ih, iw) * (oc : oc + nr_oc_block * oc_block, ic, fh, fw) + (0,
 * oc : oc + nr_oc_block * oc_block, 0, 0)
 *
 * @tparam src_idx  related to ih and iw above
 * @tparam weight_idx   related to fh and fw above
 * @tparam nr_oc_block  number of oc block
 * @tparam stride
 * @tparam nr_ow    This function calculates the value of nr_ow positions at a time.
 * @tparam T1
 * @tparam T2
 * @tparam T3
 */
template <
        int src_idx, int weight_idx, int nr_oc_block, int stride, int nr_ow,
        typename T1, typename T2, typename T3>
struct CalHelper {
    static MEGDNN_ALWAYS_INLINE void impl(T1& bias, const T2& src, const T3& weight);
};

template <
        int src_idx, int weight_idx, int nr_oc_block, int stride, typename T1,
        typename T2, typename T3>
struct CalHelper<src_idx, weight_idx, nr_oc_block, stride, 0, T1, T2, T3> {
    static MEGDNN_ALWAYS_INLINE void impl(T1& bias, const T2& src, const T3& weight){};
};

#if defined(__ARM_FEATURE_FMA)

#if MEGDNN_AARCH64
#define fma_lane_f16(a, b, v, lane) vfmaq_laneq_f16((a), (b), (v), (lane))
#else
#define fma_lane_f16(a, b, v, lane) \
    vfmaq_f16((a), (b), vdupq_n_f16(vgetq_lane_f16(v, lane)))
#endif

#else

#if MEGDNN_AARCH64
#define fma_lane_f16(a, b, v, lane) vaddq_f16((a), vmulq_laneq_f16((b), (v), (lane)))
#else
#define fma_lane_f16(a, b, v, lane) \
    vaddq_f16((a), vmulq_n_f16(b, vgetq_lane_f16(v, lane)))
#endif

#endif

#define cb1(step)                                                                     \
    bias[0][step] = fma_lane_f16(                                                     \
            bias[0][step], weight[0][weight_idx], src[(step * stride + src_idx) / 8], \
            (step * stride + src_idx) % 8);

#define cb2(step)                                                                     \
    bias[0][step] = fma_lane_f16(                                                     \
            bias[0][step], weight[0][weight_idx], src[(step * stride + src_idx) / 8], \
            (step * stride + src_idx) % 8);                                           \
    bias[1][step] = fma_lane_f16(                                                     \
            bias[1][step], weight[1][weight_idx], src[(step * stride + src_idx) / 8], \
            (step * stride + src_idx) % 8);

#define CAL_HELPER(nr_ow)                                                      \
    template <                                                                 \
            int src_idx, int weight_idx, int stride, typename T1, typename T2, \
            typename T3>                                                       \
    struct CalHelper<src_idx, weight_idx, 1, stride, nr_ow, T1, T2, T3> {      \
        static MEGDNN_ALWAYS_INLINE void impl(                                 \
                T1& bias, const T2& src, const T3& weight) {                   \
            UNROLL_CALL_NOWRAPPER(nr_ow, cb1);                                 \
        }                                                                      \
    };                                                                         \
    template <                                                                 \
            int src_idx, int weight_idx, int stride, typename T1, typename T2, \
            typename T3>                                                       \
    struct CalHelper<src_idx, weight_idx, 2, stride, nr_ow, T1, T2, T3> {      \
        static MEGDNN_ALWAYS_INLINE void impl(                                 \
                T1& bias, const T2& src, const T3& weight) {                   \
            UNROLL_CALL_NOWRAPPER(nr_ow, cb2);                                 \
        }                                                                      \
    };

CAL_HELPER(1)
CAL_HELPER(2)
CAL_HELPER(3)
CAL_HELPER(4)
CAL_HELPER(5)
CAL_HELPER(6)
CAL_HELPER(7)
CAL_HELPER(8)

#undef CAL_HELPER
#undef cb2
#undef cb1
#undef fma_lane_f16

template <
        int src_idx, int weight_idx, int nr_oc_block, int stride, int nr_ow,
        typename T1, typename T2, typename T3>
MEGDNN_ALWAYS_INLINE void cal_helper(T1& bias, const T2& src, const T3& weight) {
    CalHelper<src_idx, weight_idx, nr_oc_block, stride, nr_ow, T1, T2, T3>::impl(
            bias, src, weight);
}

template <int oc>
struct OCHelper {
    static constexpr int val = -1;
};
template <>
struct OCHelper<8> {
    static constexpr int val = 1;
};
template <>
struct OCHelper<16> {
    static constexpr int val = 2;
};

template <
        BiasMode bias_mode, typename Op, int nr_ow, int filter, int oc_block,
        int stride>  //! CHECK
struct KernFilterXStrideXNchwNchw88FP16 {
    static void impl(
            const __fp16* src_ptr, const __fp16* weight_ptr, const __fp16* bias_ptr,
            __fp16* dst_ptr, int ic, int ih, int iw, int ld_dst_oc, const Op& op);
};

#define KERNEL_CB(i) cal_helper<i, i, nr_oc_block, stride, nr_ow>(bias, src, weight);

#define KERNEL(step, FILTER_SIZE)                                                      \
    load_helper<src_reg_size, 0, simd_len, 0, Vld1q_f16>(src, src_ptr + step * iw, 0); \
    load_helper<filter_size, 0, simd_len, nr_oc_block, Vld1q_f16>(                     \
            weight, weight_ptr + step * ld_weight_fh, ld_weight_oc);                   \
    UNROLL_CALL_RAW(FILTER_SIZE, KERNEL_CB);

#define INSTANCE_KERN(FILTER_SIZE)                                                  \
    template <BiasMode bias_mode, typename Op, int nr_ow, int oc_block, int stride> \
    struct KernFilterXStrideXNchwNchw88FP16<                                        \
            bias_mode, Op, nr_ow, FILTER_SIZE, oc_block, stride> {                  \
        static void impl(                                                           \
                const __fp16* src_ptr, const __fp16* weight_ptr,                    \
                const __fp16* bias_ptr, __fp16* dst_ptr, int ic, int ih, int iw,    \
                int ld_dst_oc, const Op& op) {                                      \
            constexpr int filter_size = FILTER_SIZE;                                \
            constexpr int oc_step = 8;                                              \
            constexpr int simd_len = 8;                                             \
            constexpr int src_reg_size =                                            \
                    ((nr_ow - 1) * stride + filter_size + simd_len - 1) / simd_len; \
                                                                                    \
            constexpr int ld_weight_fh = filter_size * oc_step;                     \
            constexpr int ld_weight_ic = ld_weight_fh * filter_size;                \
            const int ld_weight_oc = ld_weight_ic * ic;                             \
            const int ld_src_ic = ih * iw;                                          \
                                                                                    \
            constexpr int nr_oc_block = OCHelper<oc_block>::val;                    \
            float16x8_t bias[nr_oc_block][nr_ow];                                   \
            init_ocx_ow8<nr_oc_block, bias_mode, nr_ow>(bias, bias_ptr, oc_step);   \
                                                                                    \
            rep(ic_idx, ic) {                                                       \
                float16x8_t src[src_reg_size], weight[nr_oc_block][filter_size];    \
                UNROLL_CALL_ONE_ARG_RAW(FILTER_SIZE, KERNEL, FILTER_SIZE);          \
                src_ptr += ld_src_ic;                                               \
                weight_ptr += ld_weight_ic;                                         \
            }                                                                       \
            store_ocx_ow8_remain_static<nr_oc_block, nr_ow, Op, 16>(                \
                    bias, op, dst_ptr, ld_dst_oc);                                  \
        }                                                                           \
    };

INSTANCE_KERN(2)
INSTANCE_KERN(3)
INSTANCE_KERN(5)
INSTANCE_KERN(7)

#undef INSTANCE_KERN
#undef KERNEL
#undef KERNEL_CB

template <BiasMode bias_mode, typename Op, int filter_size, int stride>
struct ConvDirectNchwNchw88Fp16 {
    static MEGDNN_ALWAYS_INLINE void impl(
            const __fp16* src, const __fp16* filter, const __fp16* bias, __fp16* dst,
            const int oc, const int ic, const int ih, const int iw, const int oh,
            const int oh_block, const int ow, const Op& op) {
        constexpr int fh = filter_size;
        constexpr int fw = filter_size;
#ifdef MEGDNN_ARMV7
        constexpr int big_oc_step = 8;
#else
        constexpr int big_oc_step = 16;
#endif
        constexpr int oc_step = 8;
        constexpr int ow_step = 8;
        constexpr int sh = stride;
        constexpr int sw = stride;

        const int ld_dst_oc = oh * ow * oc_step;
        const int ow_end = ow / ow_step * ow_step;
        const int ow_remain = ow - ow_end;
        const int oc_end = oc / big_oc_step * big_oc_step;
        const int oc_remain = oc - oc_end;

        using remain_func = std::function<void(
                const __fp16* src_ptr, const __fp16* weight_ptr, const __fp16* bias_ptr,
                __fp16* dst_ptr, int ic, int ih, int iw, int ld_dst_oc, const Op& op)>;

        remain_func big_oc_remain = nullptr, small_oc_remain = nullptr;
        if (ow_remain) {
            switch (ow_remain) {
#define cb(i)                                                                  \
    case i + 1:                                                                \
        big_oc_remain = KernFilterXStrideXNchwNchw88FP16<                      \
                bias_mode, Op, i + 1, filter_size, big_oc_step, stride>::impl; \
        small_oc_remain = KernFilterXStrideXNchwNchw88FP16<                    \
                bias_mode, Op, i + 1, filter_size, oc_step, stride>::impl;     \
        break;
                UNROLL_CALL_NOWRAPPER(7, cb);
#undef cb
                default:
                    megdnn_assert(0, "Don't support remain %d for kern", ow_remain);
            }
        }

        for (int oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
            const __fp16* weight_ptr = filter + oc_idx * ic * fh * fw;
            rep(oh_idx, oh_block) {
                for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                    const __fp16* src_ptr = src + oh_idx * sh * iw + ow_idx * sw;
                    const __fp16* bias_ptr = bias + oc_idx;
                    __fp16* dst_ptr =
                            dst + oc_idx * oh * ow + (oh_idx * ow + ow_idx) * oc_step;
                    KernFilterXStrideXNchwNchw88FP16<
                            bias_mode, Op, ow_step, filter_size, big_oc_step, stride>::
                            impl(src_ptr, weight_ptr, bias_ptr, dst_ptr, ic, ih, iw,
                                 ld_dst_oc, op);
                }
                if (ow_remain > 0) {
                    const __fp16* src_ptr = src + oh_idx * sh * iw + ow_end * sw;
                    const __fp16* bias_ptr = bias + oc_idx;
                    __fp16* dst_ptr =
                            dst + oc_idx * oh * ow + (oh_idx * ow + ow_end) * oc_step;
                    big_oc_remain(
                            src_ptr, weight_ptr, bias_ptr, dst_ptr, ic, ih, iw,
                            ld_dst_oc, op);
                }
            }
        }
        if (oc_remain > 0) {
            const __fp16* weight_ptr = filter + oc_end * ic * fh * fw;
            rep(oh_idx, oh_block) {
                for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                    const __fp16* src_ptr = src + oh_idx * sh * iw + ow_idx * sw;
                    const __fp16* bias_ptr = bias + oc_end;
                    __fp16* dst_ptr =
                            dst + oc_end * oh * ow + (oh_idx * ow + ow_idx) * oc_step;
                    KernFilterXStrideXNchwNchw88FP16<
                            bias_mode, Op, ow_step, filter_size, oc_step, stride>::
                            impl(src_ptr, weight_ptr, bias_ptr, dst_ptr, ic, ih, iw,
                                 ld_dst_oc, op);
                }
                if (ow_remain > 0) {
                    const __fp16* src_ptr = src + oh_idx * sh * iw + ow_end * sw;
                    const __fp16* bias_ptr = bias + oc_end;
                    __fp16* dst_ptr =
                            dst + oc_end * oh * ow + (oh_idx * ow + ow_end) * oc_step;
                    small_oc_remain(
                            src_ptr, weight_ptr, bias_ptr, dst_ptr, ic, ih, iw,
                            ld_dst_oc, op);
                }
            }
        }
    }
};
}  // namespace
}  // namespace arm_common
}  // namespace megdnn
#endif