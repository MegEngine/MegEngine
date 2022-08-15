#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/common/winograd/winograd_helper.h"
#include "src/fallback/conv_bias/gi/fp32/filter_transform.h"
#include "src/fallback/conv_bias/gi/fp32/helper.h"
#include "src/fallback/conv_bias/gi/fp32/strategy.h"
#include "src/fallback/conv_bias/winograd/winograd.h"
#include "src/fallback/elemwise_helper/op_unary.h"

#include "midout.h"
MIDOUT_DECL(megdnn_fallback_winograd_fp32_F43_4x4)

using namespace megdnn;
using namespace fallback;

namespace {
#define MLAF GiMultiplyAddScalarFloat32
#define MLSF GiMultiplySubScalarFloat32
struct InputTransform4X3 {
    /**
     * @brief Convert layout from NCHW to NCHW44(i.e. NC4HW4)
     *
     * @tparam inner Whether all data in [[ih_start, ih_start+6), [iw_start,
     * iw_start+6)] is in @input
     * @param input Pointer which points to all input data(CHW, exclude dim N)
     * @param patch Buffer which size is sizeof(float) * 4 * 6 * 6. Continuous storage
     * of data for the current block, order by C, H, W.
     * @param patchT RETURN
     * @param ih_start The start index of dim H of current block
     * @param iw_start The start index of dim W of current block
     * @param IH Dim H of input
     * @param IW Dim W of input
     * @param ic The index of dim C of input
     * @param IC Dim C of input
     */
    template <bool inner>
    static void transpose(
            const float* input, float* patch, float* patchT, int ih_start, int iw_start,
            size_t IH, size_t IW, size_t ic, size_t IC) {
        constexpr size_t alpha = 4 + 3 - 1;
        if (!inner || ic + 4 > IC) {
            memset(patch, 0, sizeof(float) * 4 * alpha * alpha);
        }

        if (inner) {
            const float* input_ptr = input + ic * IH * IW + ih_start * IW + iw_start;
            for (size_t ico = 0; ico < 4; ++ico) {
                if (ic + ico < IC) {
#define cb(i)                                                         \
    auto v##i##0 = GiLoadFloat32(input_ptr + i * IW);                 \
    GiStoreFloat32(patch + ico * alpha * alpha + i * alpha, v##i##0); \
    auto v##i##1 = GiLoadFloat32LowHalf(input_ptr + i * IW + 4);      \
    GiStoreFloat32(patch + ico * alpha * alpha + i * alpha + 4, v##i##1);
                    UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
                    input_ptr += IH * IW;
                }
            }
        } else {
            size_t ih0 = std::max(0, ih_start), ih1 = std::min(ih_start + alpha, IH),
                   iw0 = std::max(0, iw_start), iw1 = std::min(iw_start + alpha, IW);
            for (size_t ico = 0; ico < 4 && ic + ico < IC; ++ico) {
                for (size_t ih = ih0; ih < ih1; ++ih) {
                    for (size_t iw = iw0; iw < iw1; ++iw) {
                        patch[ico * alpha * alpha + (ih - ih_start) * alpha +
                              (iw - iw_start)] =
                                input[(ic + ico) * IH * IW + ih * IW + iw];
                    }
                }
            }
        }

#define cb(i) transpose_4x4(patch + i * 4, patchT + i * 16, 36, 4);
        UNROLL_CALL_NOWRAPPER(9, cb);
#undef cb
    }

    static void transform(
            const float* patchT, float* input_transform_buf, size_t unit_idx,
            size_t nr_units_in_tile, size_t ic, size_t IC) {
        constexpr size_t alpha = 4 + 3 - 1;
#define cb(m, n) \
    GI_FLOAT32_t d##m##n = GiLoadFloat32(patchT + m * alpha * 4 + n * 4), wd##m##n;
        UNROLL_CALL_NOWRAPPER_D2(6, 6, cb);
#undef cb
        //! BT
        //!    4    0   -5    0    1    0
        //!    0   -4   -4    1    1    0
        //!    0    4   -4   -1    1    0
        //!    0   -2   -1    2    1    0
        //!    0    2   -1   -2    1    0
        //!    0    4    0   -5    0    1

        //! wd0n = 4 * (d0n - d2n) + (d4n - d2n)
        //! wd1n = (d3n + d4n) - 4 * (d1n + d2n)
        //! wd2n = 4 * (d1n - d2n) + (d4n - d3n)
        //! wd3n = (d4n - d2n) - 2 * (d1n - d3n)
        //! wd4n = 2 * (d1n - d3n) + (d4n - d2n)
        //! wd5n = 4 * (d1n - d3n) + (d5n - d3n)
#define cb(n)                                                        \
    {                                                                \
        auto&& d4subd2 = SUBF(d4##n, d2##n);                         \
        auto&& d1subd3 = SUBF(d1##n, d3##n);                         \
        wd0##n = MLAF(d4subd2, SUBF(d0##n, d2##n), 4.0f);            \
        wd1##n = MLSF(ADDF(d3##n, d4##n), ADDF(d1##n, d2##n), 4.0f); \
        wd2##n = MLAF(SUBF(d4##n, d3##n), SUBF(d1##n, d2##n), 4.0f); \
        auto&& double_d1subd3 = MULSF(d1subd3, 2.0f);                \
        wd3##n = SUBF(d4subd2, double_d1subd3);                      \
        wd4##n = ADDF(double_d1subd3, d4subd2);                      \
        wd5##n = MLAF(SUBF(d5##n, d3##n), d1subd3, 4.0f);            \
    }
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        //! B
        //!    4    0    0    0    0    0
        //!    0   -4    4   -2    2    4
        //!   -5   -4   -4   -1   -1    0
        //!    0    1   -1    2   -2   -5
        //!    1    1    1    1    1    0
        //!    0    0    0    0    0    1

        //! dm0 = 4 * (wdm0 - wdm2) + (wdm4 - wdm2)
        //! dm1 = (wdm3 + wdm4) - 4 * (wdm1 + wdm2)
        //! dm2 = 4 * (wdm1 - wdm2) + (wdm4 - wdm3)
        //! dm3 = (wdm4 - wdm2) - 2 * (wdm1 - wdm3)
        //! dm4 = 2 * (wdm1 - wdm3) + (wdm4 - wdm2)
        //! dm5 = 4 * (wdm1 - wdm3) + (wdm5 - wdm3)
#define cb(m)                                                                     \
    {                                                                             \
        auto&& wd4subwd2 = SUBF(wd##m##4, wd##m##2);                              \
        auto&& wd1subwd3 = SUBF(wd##m##1, wd##m##3);                              \
        d##m##0 = MLAF(wd4subwd2, SUBF(wd##m##0, wd##m##2), 4.0f);                \
        d##m##1 = MLSF(ADDF(wd##m##3, wd##m##4), ADDF(wd##m##1, wd##m##2), 4.0f); \
        d##m##2 = MLAF(SUBF(wd##m##4, wd##m##3), SUBF(wd##m##1, wd##m##2), 4.0f); \
        auto&& double_wd1subwd3 = MULSF(wd1subwd3, 2.0f);                         \
        d##m##3 = SUBF(wd4subwd2, double_wd1subwd3);                              \
        d##m##4 = ADDF(double_wd1subwd3, wd4subwd2);                              \
        d##m##5 = MLAF(SUBF(wd##m##5, wd##m##3), wd1subwd3, 4.0f);                \
    }
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        size_t ICB = IC / 4;
        size_t icb = ic / 4;
#define cb(m, n)                                                                 \
    GiStoreFloat32(                                                              \
            input_transform_buf + (m * alpha + n) * ICB * 4 * nr_units_in_tile + \
                    icb * nr_units_in_tile * 4 + unit_idx * 4,                   \
            d##m##n);
        UNROLL_CALL_NOWRAPPER_D2(6, 6, cb);
#undef cb
    }
};  // InputTransform4X3

template <BiasMode bmode, typename Op>
struct OutputTransform4X3 {
    static void transform(
            const float* output_transform_buf, const float* bias, float* output,
            float* transform_mid_buf, size_t oh_start, size_t ow_start, size_t OH,
            size_t OW, size_t oc_start, size_t oc_end, size_t oc_index, size_t unit_idx,
            size_t nr_units_in_tile, const DType& src_dtype, const DType& dst_dtype) {
        Op op(src_dtype, dst_dtype);
        constexpr size_t alpha = 4 + 3 - 1;
        size_t oc = oc_start + oc_index;
        size_t OCB = (oc_end - oc_start) / 4;
        size_t ocb = oc_index / 4;

#define cb(m, n)                                                                  \
    auto v##m##n = GiLoadFloat32(                                                 \
            output_transform_buf + (m * alpha + n) * OCB * nr_units_in_tile * 4 + \
            ocb * nr_units_in_tile * 4 + unit_idx * 4);
        UNROLL_CALL_NOWRAPPER_D2(6, 6, cb);
#undef cb

        //! AT
        //!    1    1    1    1    1    0
        //!    0    1   -1    2   -2    0
        //!    0    1    1    4    4    0
        //!    0    1   -1    8   -8    1

        //! t0n = v0n + (v1n + v2n) +     (v3n + v4n)
        //! t1n =       (v1n - v2n) + 2 * (v3n - v4n)
        //! t2n =       (v1n + v2n) + 4 * (v3n + v4n)
        //! t3n =       (v1n - v2n) + 8 * (v3n - v4n) + v5n
#define cb(m, n) GI_FLOAT32_t t##m##n;
        UNROLL_CALL_NOWRAPPER_D2(4, 6, cb);
#undef cb

#define cb(n)                                              \
    {                                                      \
        auto&& v1addv2 = ADDF(v1##n, v2##n);               \
        auto&& v1subv2 = SUBF(v1##n, v2##n);               \
        auto&& v3addv4 = ADDF(v3##n, v4##n);               \
        auto&& v3subv4 = SUBF(v3##n, v4##n);               \
                                                           \
        t0##n = ADDF(ADDF(v0##n, v1addv2), v3addv4);       \
        t1##n = MLAF(v1subv2, v3subv4, 2.0f);              \
        t2##n = MLAF(v1addv2, v3addv4, 4.0f);              \
        t3##n = ADDF(MLAF(v1subv2, v3subv4, 8.0f), v5##n); \
    }
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

//! A
//!    1    0    0    0
//!    1    1    1    1
//!    1   -1    1   -1
//!    1    2    4    8
//!    1   -2    4   -8
//!    0    0    0    1

// vm0 = tm0 + (tm1 + tm2) +     (tm3 + tm4)
// vm1 =       (tm1 - tm2) + 2 * (tm3 - tm4)
// vm2 =       (tm1 + tm2) + 4 * (tm3 + tm4)
// vm3 =       (tm1 - tm2) + 8 * (tm3 - tm4) + tm5
#define cb(m)                                                  \
    {                                                          \
        auto&& t1addt2 = ADDF(t##m##1, t##m##2);               \
        auto&& t1subt2 = SUBF(t##m##1, t##m##2);               \
        auto&& t3addt4 = ADDF(t##m##3, t##m##4);               \
        auto&& t3subt4 = SUBF(t##m##3, t##m##4);               \
        v##m##0 = ADDF(ADDF(t##m##0, t1addt2), t3addt4);       \
        v##m##1 = MLAF(t1subt2, t3subt4, 2.0f);                \
        v##m##2 = MLAF(t1addt2, t3addt4, 4.0f);                \
        v##m##3 = ADDF(MLAF(t1subt2, t3subt4, 8.0f), t##m##5); \
    }
        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

        GI_FLOAT32_t vbias;
        if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias = GiLoadFloat32(bias + oc);
#define cb(m, n) v##m##n = GiAddFloat32(v##m##n, vbias);
            UNROLL_CALL_NOWRAPPER_D2(4, 4, cb);
#undef cb
        }

        if (bmode != BiasMode::BIAS) {
#define cb(m, n) v##m##n = op(v##m##n);
            UNROLL_CALL_NOWRAPPER_D2(4, 4, cb);
#undef cb
        }

#define cb(m, n) GiStoreFloat32(transform_mid_buf + (4 * m + n) * 4, v##m##n);
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb);
#undef cb

        for (size_t oho = 0; oho < 4 && oh_start + oho < OH; ++oho) {
            for (size_t owo = 0; owo < 4 && ow_start + owo < OW; ++owo) {
                for (size_t oco = 0; oco < 4 && oc + oco < oc_end; ++oco) {
                    float res = transform_mid_buf[oho * 4 * 4 + owo * 4 + oco];
                    size_t oh = oh_start + oho;
                    size_t ow = ow_start + owo;
                    if (bmode == BiasMode::BIAS) {
                        res += bias[(oc + oco) * OH * OW + oh * OW + ow];
                        res = op(res);
                    }
                    output[(oc + oco) * OH * OW + oh * OW + ow] = res;
                }
            }
        }
    }
};  // OutputTransform4X3

#undef MLSF
#undef MLAF
}  // namespace

namespace megdnn {
namespace fallback {
namespace winograd {
MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_4x3_4x4_f)

void winograd_4x3_4x4_f::filter(
        const float* filter, float* filter_transform_buf, float* transform_mid_buf,
        size_t OC, size_t IC, size_t oc_start, size_t oc_end) {
    FilterTransform4X3<megdnn::param::MatrixMul::Format::MK4>::transform(
            filter, filter_transform_buf, transform_mid_buf, OC, IC, oc_start, oc_end);
}

void winograd_4x3_4x4_f::input(
        const float* input, float* input_transform_buf, float* transform_mid_buf,
        size_t IH, size_t IW, size_t IC, size_t PH, size_t PW, size_t unit_start_idx,
        size_t nr_units_in_tile) {
    megdnn_assert(IC % 4 == 0);
    auto unit_w = div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
    float* patch = transform_mid_buf;
    float* patchT = transform_mid_buf + 4 * ALPHA * ALPHA;
    for (size_t ic = 0; ic < IC; ic += 4) {
        for (size_t unit_idx = 0; unit_idx < nr_units_in_tile; ++unit_idx) {
            size_t index = unit_start_idx + unit_idx;
            size_t oht = index / unit_w;
            size_t owt = index % unit_w;
            int ih_start = static_cast<int>(oht * OUTPUT_BLOCK_SIZE - PH);
            int iw_start = static_cast<int>(owt * OUTPUT_BLOCK_SIZE - PW);
            if (ih_start >= 0 && ih_start + 6 <= static_cast<int>(IH) &&
                iw_start >= 0 && iw_start + 6 <= static_cast<int>(IW)) {
                InputTransform4X3::transpose<true>(
                        input, patch, patchT, ih_start, iw_start, IH, IW, ic, IC);
            } else {
                InputTransform4X3::transpose<false>(
                        input, patch, patchT, ih_start, iw_start, IH, IW, ic, IC);
            }
            InputTransform4X3::transform(
                    patchT, input_transform_buf, unit_idx, nr_units_in_tile, ic, IC);
        }
    }
}

void winograd_4x3_4x4_f::output(
        const float* output_transform_buf, const float* bias, float* output,
        float* transform_mid_buf, BiasMode bmode, NonlineMode nonline_mode, size_t OH,
        size_t OW, size_t oc_start, size_t oc_end, size_t unit_start_idx,
        size_t nr_units_in_tile) {
#define cb(_bmode, _nonline_mode, ...) \
    OutputTransform4X3<_bmode, _nonline_mode>::transform(__VA_ARGS__);
    auto unit_w = div_ceil<size_t>(OW, OUTPUT_BLOCK_SIZE);

    for (size_t oc = oc_start; oc < oc_end; oc += 4) {
        size_t oc_index = oc - oc_start;
        for (size_t unit_idx = 0; unit_idx < nr_units_in_tile; ++unit_idx) {
            size_t index = unit_idx + unit_start_idx;
            size_t oht = index / unit_w;
            size_t owt = index % unit_w;
            size_t oh_start = oht * OUTPUT_BLOCK_SIZE;
            size_t ow_start = owt * OUTPUT_BLOCK_SIZE;
            GI_DISPATCH_CONV_WINOGRAD_BIAS(
                    megdnn_fallback_winograd_fp32_F43_4x4, cb, float, float, bmode,
                    nonline_mode, output_transform_buf, bias, output, transform_mid_buf,
                    oh_start, ow_start, OH, OW, oc_start, oc_end, oc_index, unit_idx,
                    nr_units_in_tile, src_dtype, dst_dtype);
        }
    }
#undef cb
}

}  // namespace winograd
}  // namespace fallback
}  // namespace megdnn