#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/common/winograd/winograd_helper.h"
#include "src/fallback/conv_bias/gi/fp32/filter_transform.h"
#include "src/fallback/conv_bias/gi/fp32/helper.h"
#include "src/fallback/conv_bias/gi/fp32/strategy.h"
#include "src/fallback/conv_bias/winograd/winograd.h"
#include "src/fallback/elemwise_helper/op_unary.h"

#include "midout.h"
MIDOUT_DECL(megdnn_fallback_winograd_fp32_F43_mk4)

using namespace megdnn;
using namespace fallback;

namespace {

constexpr size_t alpha = 4 + 3 - 1;
constexpr size_t pack_size = 4;
constexpr float input_parameters[4] = {4.0f, 5.0f, 2.0f, 0.0f};
constexpr float output_parameters[4] = {1.0f, 2.0f, 4.0f, 8.0f};

struct InputTransformF43_NCHW44 {
    template <bool inner>
    static void transform(
            const float* input, float* input_transform_buf, size_t unit_idx,
            size_t nr_units_in_tile, size_t ic, size_t IC, int ih_start, int iw_start,
            size_t IH, size_t IW, const bool* ih_valid, const bool* iw_valid) {
        // BT * d * B
        size_t ICB = IC / pack_size;
        size_t icb = ic / pack_size;

#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
//! x86 and rvv GiSimdFmaLane API is slowly, as an alternate, use
//! GiMultiplyAddScalarFloat32
#define MADD(a, b, c, d) GiMultiplyAddScalarFloat32(a, b, *(c + d))
#define MSUB(a, b, c, d) GiMultiplySubScalarFloat32(a, b, *(c + d))
        const float* v0 = input_parameters;
#else
#define MADD(a, b, c, d) GiSimdFmaLane(a, b, c, d)
#define MSUB(a, b, c, d) GiFmsqLaneQFloat32(a, b, c, d)
        GI_FLOAT32_t v0 = GiLoadFloat32(input_parameters);
#endif
        //    B
        //    4    0    0    0    0    0
        //    0   -4    4   -2    2    4
        //   -5   -4   -4   -1   -1    0
        //    0    1   -1    2   -2   -5
        //    1    1    1    1    1    0
        //    0    0    0    0    0    1

        const float* input_ptr =
                input + ic * IH * IW + ih_start * IW * 4 + iw_start * 4;
        GI_FLOAT32_t zero = GiZeroFloat32();
#define cb(i, j) GI_FLOAT32_t d##i##j;
        UNROLL_CALL_NOWRAPPER_D2(4, 6, cb);
#undef cb
#define cb(i) GI_FLOAT32_t t##i;
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        // load line 0 -> d00 ... d05
        const float* line_ptr = input_ptr;
        if (inner) {
#define cb(i) d0##i = GiLoadFloat32(line_ptr + i * pack_size);
            UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
        } else {
            if (ih_valid[0]) {
#define cb(i) d0##i = iw_valid[i] ? GiLoadFloat32(line_ptr + i * pack_size) : zero;
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            } else {
#define cb(i) d0##i = zero;
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            }
        }

        // load line 4 -> d30 ... t35
        line_ptr = input_ptr + 4 * IW * 4;
        if (inner) {
#define cb(i)                                        \
    d3##i = GiLoadFloat32(line_ptr + i * pack_size); \
    t##i = MADD(d3##i, d0##i, v0, 0);
            UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
        } else {
            if (ih_valid[4]) {
#define cb(i)                                                             \
    d3##i = iw_valid[i] ? GiLoadFloat32(line_ptr + i * pack_size) : zero; \
    t##i = MADD(d3##i, d0##i, v0, 0);
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            } else {
#define cb(i)     \
    d3##i = zero; \
    t##i = MADD(d3##i, d0##i, v0, 0);
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            }
        }

        // load line 2 -> d20 ... d25
        line_ptr = input_ptr + 2 * IW * 4;
        if (inner) {
#define cb(i)                                        \
    d2##i = GiLoadFloat32(line_ptr + i * pack_size); \
    t##i = MSUB(t##i, d2##i, v0, 1);
            UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
        } else {
            if (ih_valid[2]) {
#define cb(i)                                                             \
    d2##i = iw_valid[i] ? GiLoadFloat32(line_ptr + i * pack_size) : zero; \
    t##i = MSUB(t##i, d2##i, v0, 1);
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            } else {
#define cb(i)     \
    d2##i = zero; \
    t##i = MSUB(t##i, d2##i, v0, 1);
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            }
        }

        // load line 3 -> d10 ... d15
        line_ptr = input_ptr + 3 * IW * 4;
        if (inner) {
#define cb(i) d1##i = GiLoadFloat32(line_ptr + i * pack_size);
            UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
        } else {
            if (ih_valid[3]) {
#define cb(i) d1##i = iw_valid[i] ? GiLoadFloat32(line_ptr + i * pack_size) : zero;
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            } else {
#define cb(i) d1##i = zero;
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            }
        }

        float* buf_ptr = input_transform_buf + icb * nr_units_in_tile * pack_size +
                         unit_idx * pack_size;

        d00 = MADD(t4, t0, v0, 0);
        d00 = MSUB(d00, t2, v0, 1);
        GiStoreFloat32(buf_ptr, d00);
        d00 = MSUB(t3, t1, v0, 0);
        d01 = MSUB(t4, t2, v0, 0);
        d02 = ADDF(d00, d01);
        GiStoreFloat32(buf_ptr + 1 * ICB * nr_units_in_tile * pack_size, d02);
        d02 = SUBF(d01, d00);
        GiStoreFloat32(buf_ptr + 2 * ICB * nr_units_in_tile * pack_size, d02);
        d00 = SUBF(t3, t1);
        d01 = SUBF(t4, t2);
        d02 = MADD(d01, d00, v0, 2);
        GiStoreFloat32(buf_ptr + 3 * ICB * nr_units_in_tile * pack_size, d02);
        d02 = MSUB(d01, d00, v0, 2);
        GiStoreFloat32(buf_ptr + 4 * ICB * nr_units_in_tile * pack_size, d02);
        d01 = SUBF(t5, t3);
        d02 = MSUB(d01, d00, v0, 0);
        GiStoreFloat32(buf_ptr + 5 * ICB * nr_units_in_tile * pack_size, d02);

// ln4 - ln2 -> t
#define cb(i) t##i = SUBF(d3##i, d2##i);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        // load line 1 -> d00 ... d05
        line_ptr = input_ptr + IW * 4;
        if (inner) {
#define cb(i) d0##i = GiLoadFloat32(line_ptr + i * pack_size);
            UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
        } else {
            if (ih_valid[1]) {
#define cb(i) d0##i = iw_valid[i] ? GiLoadFloat32(line_ptr + i * pack_size) : zero;
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            } else {
#define cb(i) d0##i = zero;
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            }
        }

// ln4 - 4 * ln2 -> ln4
#define cb(i) d3##i = MSUB(d3##i, d2##i, v0, 0);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

// ln3 - 4 * ln1 -> ln2
#define cb(i) d2##i = MSUB(d1##i, d0##i, v0, 0);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

// ln3 - ln1 -> ln3
#define cb(i) d1##i = SUBF(d1##i, d0##i);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

// (ln4 - 4 * ln2)[ln4] + (ln3 - 4 * ln1)[ln2] -> ln1
#define cb(i) d0##i = ADDF(d3##i, d2##i);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

// (ln4 - 4 * ln2)[ln4] - (ln3 - 4 * ln1)[ln2] -> ln2
#define cb(i) d2##i = SUBF(d3##i, d2##i);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        // ln4(d30 ... d35) is free until now
        buf_ptr = input_transform_buf + 1 * alpha * ICB * nr_units_in_tile * pack_size +
                  icb * nr_units_in_tile * pack_size + unit_idx * pack_size;
        d30 = MADD(d04, d00, v0, 0);
        d30 = MSUB(d30, d02, v0, 1);
        GiStoreFloat32(buf_ptr, d30);
        d30 = MSUB(d03, d01, v0, 0);
        d32 = MSUB(d04, d02, v0, 0);
        d31 = ADDF(d30, d32);
        GiStoreFloat32(buf_ptr + ICB * nr_units_in_tile * pack_size, d31);
        d31 = SUBF(d32, d30);
        GiStoreFloat32(buf_ptr + 2 * ICB * nr_units_in_tile * pack_size, d31);
        d30 = SUBF(d03, d01);
        d31 = SUBF(d04, d02);
        d32 = MADD(d31, d30, v0, 2);
        GiStoreFloat32(buf_ptr + 3 * ICB * nr_units_in_tile * pack_size, d32);
        d32 = MSUB(d31, d30, v0, 2);
        GiStoreFloat32(buf_ptr + 4 * ICB * nr_units_in_tile * pack_size, d32);
        d31 = SUBF(d05, d03);
        d32 = MSUB(d31, d30, v0, 0);
        GiStoreFloat32(buf_ptr + 5 * ICB * nr_units_in_tile * pack_size, d32);

        buf_ptr = input_transform_buf + 2 * alpha * ICB * nr_units_in_tile * pack_size +
                  icb * nr_units_in_tile * pack_size + unit_idx * pack_size;
        d33 = MADD(d24, d20, v0, 0);
        d33 = MSUB(d33, d22, v0, 1);
        GiStoreFloat32(buf_ptr, d33);
        d33 = MSUB(d23, d21, v0, 0);
        d35 = MSUB(d24, d22, v0, 0);
        d34 = ADDF(d33, d35);
        GiStoreFloat32(buf_ptr + ICB * nr_units_in_tile * pack_size, d34);
        d34 = SUBF(d35, d33);
        GiStoreFloat32(buf_ptr + 2 * ICB * nr_units_in_tile * pack_size, d34);
        d33 = SUBF(d23, d21);
        d34 = SUBF(d24, d22);
        d35 = MADD(d34, d33, v0, 2);
        GiStoreFloat32(buf_ptr + 3 * ICB * nr_units_in_tile * pack_size, d35);
        d35 = MSUB(d34, d33, v0, 2);
        GiStoreFloat32(buf_ptr + 4 * ICB * nr_units_in_tile * pack_size, d35);
        d34 = SUBF(d25, d23);
        d35 = MSUB(d34, d33, v0, 0);
        GiStoreFloat32(buf_ptr + 5 * ICB * nr_units_in_tile * pack_size, d35);

// (ln4 - ln2)[t] + (ln3 - ln1)[ln3] * 2 -> ln4
#define cb(i) d3##i = MADD(t##i, d1##i, v0, 2);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

// (ln4 - ln2)[t] - (ln3 - ln1)[ln3] * 2 -> ln3
#define cb(i) d1##i = MSUB(t##i, d1##i, v0, 2);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
        // t is free
        buf_ptr = input_transform_buf + 3 * alpha * ICB * nr_units_in_tile * pack_size +
                  icb * nr_units_in_tile * pack_size + unit_idx * pack_size;
        t0 = MADD(d34, d30, v0, 0);
        t0 = MSUB(t0, d32, v0, 1);
        GiStoreFloat32(buf_ptr, t0);
        t0 = MSUB(d33, d31, v0, 0);
        t2 = MSUB(d34, d32, v0, 0);
        t1 = ADDF(t0, t2);
        GiStoreFloat32(buf_ptr + ICB * nr_units_in_tile * pack_size, t1);
        t1 = SUBF(t2, t0);
        GiStoreFloat32(buf_ptr + 2 * ICB * nr_units_in_tile * pack_size, t1);
        t0 = SUBF(d33, d31);
        t1 = SUBF(d34, d32);
        t2 = MADD(t1, t0, v0, 2);
        GiStoreFloat32(buf_ptr + 3 * ICB * nr_units_in_tile * pack_size, t2);
        t2 = MSUB(t1, t0, v0, 2);
        GiStoreFloat32(buf_ptr + 4 * ICB * nr_units_in_tile * pack_size, t2);
        t1 = SUBF(d35, d33);
        t2 = MSUB(t1, t0, v0, 0);
        GiStoreFloat32(buf_ptr + 5 * ICB * nr_units_in_tile * pack_size, t2);

        buf_ptr = input_transform_buf + 4 * alpha * ICB * nr_units_in_tile * pack_size +
                  icb * nr_units_in_tile * pack_size + unit_idx * pack_size;
        t3 = MADD(d14, d10, v0, 0);
        t3 = MSUB(t3, d12, v0, 1);
        GiStoreFloat32(buf_ptr, t3);
        t3 = MSUB(d13, d11, v0, 0);
        t5 = MSUB(d14, d12, v0, 0);
        t4 = ADDF(t3, t5);
        GiStoreFloat32(buf_ptr + ICB * nr_units_in_tile * pack_size, t4);
        t4 = SUBF(t5, t3);
        GiStoreFloat32(buf_ptr + 2 * ICB * nr_units_in_tile * pack_size, t4);
        t3 = SUBF(d13, d11);
        t4 = SUBF(d14, d12);
        t5 = MADD(t4, t3, v0, 2);
        GiStoreFloat32(buf_ptr + 3 * ICB * nr_units_in_tile * pack_size, t5);
        t5 = MSUB(t4, t3, v0, 2);
        GiStoreFloat32(buf_ptr + 4 * ICB * nr_units_in_tile * pack_size, t5);
        t4 = SUBF(d15, d13);
        t5 = MSUB(t4, t3, v0, 0);
        GiStoreFloat32(buf_ptr + 5 * ICB * nr_units_in_tile * pack_size, t5);

        // load line 5 -> d30 ... d35
        line_ptr = input_ptr + 5 * IW * 4;
        if (inner) {
#define cb(i) d3##i = GiLoadFloat32(line_ptr + i * pack_size);
            UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
        } else {
            if (ih_valid[5]) {
#define cb(i) d3##i = iw_valid[i] ? GiLoadFloat32(line_ptr + i * pack_size) : zero;
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            } else {
#define cb(i) d3##i = zero;
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            }
        }

        // load line 1 -> d0 ... d5
        line_ptr = input_ptr + IW * 4;
        if (inner) {
#define cb(i)                                        \
    d0##i = GiLoadFloat32(line_ptr + i * pack_size); \
    d3##i = MADD(d3##i, d0##i, v0, 0);
            UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
        } else {
            if (ih_valid[1]) {
#define cb(i)                                                             \
    d0##i = iw_valid[i] ? GiLoadFloat32(line_ptr + i * pack_size) : zero; \
    d3##i = MADD(d3##i, d0##i, v0, 0);
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            } else {
#define cb(i)     \
    d0##i = zero; \
    d3##i = MADD(d3##i, d0##i, v0, 0);
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            }
        }

        // load line 3 -> d10 ... d15
        line_ptr = input_ptr + 3 * IW * 4;
        if (inner) {
#define cb(i)                                        \
    d1##i = GiLoadFloat32(line_ptr + i * pack_size); \
    d3##i = MSUB(d3##i, d1##i, v0, 1);
            UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
        } else {
            if (ih_valid[3]) {
#define cb(i)                                                             \
    d1##i = iw_valid[i] ? GiLoadFloat32(line_ptr + i * pack_size) : zero; \
    d3##i = MSUB(d3##i, d1##i, v0, 1);
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            } else {
#define cb(i)     \
    d1##i = zero; \
    d3##i = MSUB(d3##i, d1##i, v0, 1);
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            }
        }

        buf_ptr = input_transform_buf + 5 * alpha * ICB * nr_units_in_tile * pack_size +
                  icb * nr_units_in_tile * pack_size + unit_idx * pack_size;
        t0 = MADD(d34, d30, v0, 0);
        t0 = MSUB(t0, d32, v0, 1);
        GiStoreFloat32(buf_ptr, t0);
        t0 = MSUB(d33, d31, v0, 0);
        t2 = MSUB(d34, d32, v0, 0);
        t1 = ADDF(t0, t2);
        GiStoreFloat32(buf_ptr + ICB * nr_units_in_tile * pack_size, t1);
        t1 = SUBF(t2, t0);
        GiStoreFloat32(buf_ptr + 2 * ICB * nr_units_in_tile * pack_size, t1);
        t0 = SUBF(d33, d31);
        t1 = SUBF(d34, d32);
        t2 = MADD(t1, t0, v0, 2);
        GiStoreFloat32(buf_ptr + 3 * ICB * nr_units_in_tile * pack_size, t2);
        t2 = MSUB(t1, t0, v0, 2);
        GiStoreFloat32(buf_ptr + 4 * ICB * nr_units_in_tile * pack_size, t2);
        t1 = SUBF(d35, d33);
        t2 = MSUB(t1, t0, v0, 0);
        GiStoreFloat32(buf_ptr + 5 * ICB * nr_units_in_tile * pack_size, t2);

#undef MSUB
#undef MADD
    }
};  // InputTransformF43_NCHW44

template <BiasMode bmode, typename Op>
struct OutputTransformF43_NCHW44 {
    static inline void transform(
            const float* output_transform_buf, const float* bias, float* output,
            const size_t oh_start, const size_t ow_start, const size_t OH,
            const size_t OW, const size_t oc_start, const size_t oc_end,
            const size_t oc_index, const size_t unit_idx, const size_t nr_units_in_tile,
            const DType& src_dtype, const DType& dst_dtype) {
        Op op(src_dtype, dst_dtype);
        //! AT * m * A

        size_t oc = oc_start + oc_index;
        size_t OCB = (oc_end - oc_start) / pack_size;
        size_t ocb = oc_index / pack_size;
        size_t col_step = OCB * nr_units_in_tile * 4;
        size_t row_step = alpha * col_step;

#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
//! x86 and rvv GiSimdFmaLane API is slowly, as an alternate, use
//! GiMultiplyAddScalarFloat32
#define MADD(a, b, c, d) GiMultiplyAddScalarFloat32(a, b, *(c + d))
#define MSUB(a, b, c, d) GiMultiplySubScalarFloat32(a, b, *(c + d))
        const float* v0 = output_parameters;
#else
#define MADD(a, b, c, d) GiSimdFmaLane(a, b, c, d)
#define MSUB(a, b, c, d) GiFmsqLaneQFloat32(a, b, c, d)
        GI_FLOAT32_t v0 = GiLoadFloat32(output_parameters);
#endif

        GI_FLOAT32_t vbias = GiZeroFloat32();
#define cb(i, j) GI_FLOAT32_t v##i##j;
        UNROLL_CALL_NOWRAPPER_D2(5, 6, cb);
#undef cb

        const float* buf_base =
                output_transform_buf + ocb * nr_units_in_tile * 4 + unit_idx * 4;
        const float* buf_ptr = nullptr;

        // load line 1 -> v10 ... v15
        buf_ptr = buf_base + row_step;
#define cb(i) v1##i = GiLoadFloat32(buf_ptr + i * col_step);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        // load line 2 -> v20 ... v25
        buf_ptr = buf_base + 2 * row_step;
#define cb(i)                                      \
    v2##i = GiLoadFloat32(buf_ptr + i * col_step); \
    v0##i = ADDF(v1##i, v2##i);                    \
    v1##i = SUBF(v1##i, v2##i);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        // load line 3 -> v30 ... v35
        buf_ptr = buf_base + 3 * row_step;
#define cb(i) v3##i = GiLoadFloat32(buf_ptr + i * col_step);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        // load line 4 -> v40 ... v45
        buf_ptr = buf_base + 4 * row_step;
#define cb(i)                                      \
    v4##i = GiLoadFloat32(buf_ptr + i * col_step); \
    v2##i = ADDF(v3##i, v4##i);                    \
    v3##i = SUBF(v3##i, v4##i);                    \
    v4##i = MADD(v0##i, v2##i, v0, 2);             \
    v2##i = ADDF(v2##i, v0##i);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        float* output_base = output + oc * OH * OW + oh_start * OW * pack_size +
                             ow_start * pack_size;
        float* output_ptr = output_base + 2 * OW * pack_size;
        const float* bias_base = nullptr;
        const float* bias_ptr = nullptr;
        if (bmode == BiasMode::BIAS) {
            bias_base = bias + oc * OH * OW + oh_start * OW * pack_size +
                        ow_start * pack_size;
            bias_ptr = bias_base + 2 * OW * pack_size;
        }
        if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias = GiLoadFloat32(bias + oc);
        }
        v00 = ADDF(v41, v42);
        v01 = ADDF(v43, v44);
        v02 = ADDF(v40, v00);
        v02 = ADDF(v02, v01);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr);
        }
        v02 = ADDF(v02, vbias);
        v02 = op(v02);
        GiStoreFloat32(output_ptr, v02);

        v03 = SUBF(v41, v42);
        v04 = SUBF(v43, v44);
        v05 = MADD(v03, v04, v0, 1);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr + pack_size);
        }
        v05 = ADDF(v05, vbias);
        v05 = op(v05);
        GiStoreFloat32(output_ptr + pack_size, v05);

        v02 = MADD(v00, v01, v0, 2);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr + 2 * pack_size);
        }
        v02 = ADDF(v02, vbias);
        v02 = op(v02);
        GiStoreFloat32(output_ptr + 2 * pack_size, v02);

        v05 = MADD(v03, v04, v0, 3);
        v05 = ADDF(v05, v45);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr + 3 * pack_size);
        }
        v05 = ADDF(v05, vbias);
        v05 = op(v05);
        GiStoreFloat32(output_ptr + 3 * pack_size, v05);

        buf_ptr = buf_base;
#define cb(i)                                      \
    v4##i = GiLoadFloat32(buf_ptr + i * col_step); \
    v4##i = ADDF(v4##i, v2##i);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        output_ptr = output_base;
        if (bmode == BiasMode::BIAS) {
            bias_ptr = bias_base;
        }
        v00 = ADDF(v41, v42);
        v01 = ADDF(v43, v44);
        v02 = ADDF(v40, v00);
        v02 = ADDF(v02, v01);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr);
        }
        v02 = ADDF(v02, vbias);
        v02 = op(v02);
        GiStoreFloat32(output_ptr, v02);

        v03 = SUBF(v41, v42);
        v04 = SUBF(v43, v44);
        v05 = MADD(v03, v04, v0, 1);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr + pack_size);
        }
        v05 = ADDF(v05, vbias);
        v05 = op(v05);
        GiStoreFloat32(output_ptr + pack_size, v05);

        v02 = MADD(v00, v01, v0, 2);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr + 2 * pack_size);
        }
        v02 = ADDF(v02, vbias);
        v02 = op(v02);
        GiStoreFloat32(output_ptr + 2 * pack_size, v02);

        v05 = MADD(v03, v04, v0, 3);
        v05 = ADDF(v05, v45);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr + 3 * pack_size);
        }
        v05 = ADDF(v05, vbias);
        v05 = op(v05);
        GiStoreFloat32(output_ptr + 3 * pack_size, v05);

#define cb(i) v4##i = MADD(v1##i, v3##i, v0, 1);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        output_ptr = output_base + OW * pack_size;
        if (bmode == BiasMode::BIAS) {
            bias_ptr = bias_base + OW * pack_size;
        }
        v00 = ADDF(v41, v42);
        v01 = ADDF(v43, v44);
        v02 = ADDF(v40, v00);
        v02 = ADDF(v02, v01);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr);
        }
        v02 = ADDF(v02, vbias);
        v02 = op(v02);
        GiStoreFloat32(output_ptr, v02);

        v03 = SUBF(v41, v42);
        v04 = SUBF(v43, v44);
        v05 = MADD(v03, v04, v0, 1);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr + pack_size);
        }
        v05 = ADDF(v05, vbias);
        v05 = op(v05);
        GiStoreFloat32(output_ptr + pack_size, v05);

        v02 = MADD(v00, v01, v0, 2);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr + 2 * pack_size);
        }
        v02 = ADDF(v02, vbias);
        v02 = op(v02);
        GiStoreFloat32(output_ptr + 2 * pack_size, v02);

        v05 = MADD(v03, v04, v0, 3);
        v05 = ADDF(v05, v45);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr + 3 * pack_size);
        }
        v05 = ADDF(v05, vbias);
        v05 = op(v05);
        GiStoreFloat32(output_ptr + 3 * pack_size, v05);

        buf_ptr = buf_base + 5 * row_step;
#define cb(i)                                      \
    v2##i = GiLoadFloat32(buf_ptr + i * col_step); \
    v1##i = MADD(v1##i, v3##i, v0, 3);             \
    v2##i = ADDF(v1##i, v2##i);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        output_ptr = output_base + 3 * OW * pack_size;
        if (bmode == BiasMode::BIAS) {
            bias_ptr = bias_base + 3 * OW * pack_size;
        }
        v00 = ADDF(v21, v22);
        v01 = ADDF(v23, v24);
        v02 = ADDF(v20, v00);
        v02 = ADDF(v02, v01);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr);
        }
        v02 = ADDF(v02, vbias);
        v02 = op(v02);
        GiStoreFloat32(output_ptr, v02);

        v03 = SUBF(v21, v22);
        v04 = SUBF(v23, v24);
        v05 = MADD(v03, v04, v0, 1);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr + pack_size);
        }
        v05 = ADDF(v05, vbias);
        v05 = op(v05);
        GiStoreFloat32(output_ptr + pack_size, v05);

        v02 = MADD(v00, v01, v0, 2);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr + 2 * pack_size);
        }
        v02 = ADDF(v02, vbias);
        v02 = op(v02);
        GiStoreFloat32(output_ptr + 2 * pack_size, v02);

        v05 = MADD(v03, v04, v0, 3);
        v05 = ADDF(v05, v25);
        if (bmode == BiasMode::BIAS) {
            vbias = GiLoadFloat32(bias_ptr + 3 * pack_size);
        }
        v05 = ADDF(v05, vbias);
        v05 = op(v05);
        GiStoreFloat32(output_ptr + 3 * pack_size, v05);

#undef MSUB
#undef MADD
    }

    static inline void transform(
            const float* output_transform_buf, const float* bias, float* output,
            const size_t oh_start, const size_t ow_start, const size_t OH,
            const size_t OW, const size_t oc_start, const size_t oc_end,
            const size_t oc_index, const size_t unit_idx, const size_t nr_units_in_tile,
            const DType& src_dtype, const DType& dst_dtype, const size_t num_oh_valid,
            const size_t num_ow_valid) {
        Op op(src_dtype, dst_dtype);
        //! AT * m * A

        size_t oc = oc_start + oc_index;
        size_t OCB = (oc_end - oc_start) / pack_size;
        size_t ocb = oc_index / pack_size;
        size_t col_step = OCB * nr_units_in_tile * 4;
        size_t row_step = alpha * col_step;

#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
//! x86 and rvv GiSimdFmaLane API is slowly, as an alternate, use
//! GiMultiplyAddScalarFloat32
#define MADD(a, b, c, d) GiMultiplyAddScalarFloat32(a, b, *(c + d))
#define MSUB(a, b, c, d) GiMultiplySubScalarFloat32(a, b, *(c + d))
        const float* v0 = output_parameters;
#else
#define MADD(a, b, c, d) GiSimdFmaLane(a, b, c, d)
#define MSUB(a, b, c, d) GiFmsqLaneQFloat32(a, b, c, d)
        GI_FLOAT32_t v0 = GiLoadFloat32(output_parameters);
#endif

        GI_FLOAT32_t vbias = GiZeroFloat32();
#define cb(i, j) GI_FLOAT32_t v##i##j;
        UNROLL_CALL_NOWRAPPER_D2(5, 6, cb);
#undef cb

        const float* buf_base =
                output_transform_buf + ocb * nr_units_in_tile * 4 + unit_idx * 4;
        const float* buf_ptr = nullptr;

        // load line 1 -> v10 ... v15
        buf_ptr = buf_base + row_step;
#define cb(i) v1##i = GiLoadFloat32(buf_ptr + i * col_step);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        // load line 2 -> v20 ... v25
        buf_ptr = buf_base + 2 * row_step;
#define cb(i)                                      \
    v2##i = GiLoadFloat32(buf_ptr + i * col_step); \
    v0##i = ADDF(v1##i, v2##i);                    \
    v1##i = SUBF(v1##i, v2##i);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        // load line 3 -> v30 ... v35
        buf_ptr = buf_base + 3 * row_step;
#define cb(i) v3##i = GiLoadFloat32(buf_ptr + i * col_step);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        // load line 4 -> v40 ... v45
        buf_ptr = buf_base + 4 * row_step;
#define cb(i)                                      \
    v4##i = GiLoadFloat32(buf_ptr + i * col_step); \
    v2##i = ADDF(v3##i, v4##i);                    \
    v3##i = SUBF(v3##i, v4##i);                    \
    v4##i = MADD(v0##i, v2##i, v0, 2);             \
    v2##i = ADDF(v2##i, v0##i);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        // result line 2, v40 ... v45 -> v02 ... v05
        // v40 ... v45 is free.
        v00 = ADDF(v41, v42);
        v01 = ADDF(v43, v44);
        v02 = ADDF(v40, v00);
        v02 = ADDF(v02, v01);

        v04 = MADD(v00, v01, v0, 2);

        v00 = SUBF(v41, v42);
        v01 = SUBF(v43, v44);
        v03 = MADD(v00, v01, v0, 1);

        v05 = MADD(v00, v01, v0, 3);
        v05 = ADDF(v05, v45);

        buf_ptr = buf_base;
#define cb(i)                                      \
    v4##i = GiLoadFloat32(buf_ptr + i * col_step); \
    v4##i = ADDF(v4##i, v2##i);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        // result line 0
        // v40 ... v45 -> v22 ... v25
        v20 = ADDF(v41, v42);
        v21 = ADDF(v43, v44);
        v22 = ADDF(v40, v20);
        v22 = ADDF(v22, v21);

        v24 = MADD(v20, v21, v0, 2);

        v20 = SUBF(v41, v42);
        v21 = SUBF(v43, v44);
        v23 = MADD(v20, v21, v0, 1);

        v25 = MADD(v20, v21, v0, 3);
        v25 = ADDF(v25, v45);

#define cb(i)                          \
    v4##i = MADD(v1##i, v3##i, v0, 1); \
    v3##i = MADD(v1##i, v3##i, v0, 3);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        // result line 1
        // v40 ... v45 -> v12 ... v15
        v10 = ADDF(v41, v42);
        v11 = ADDF(v43, v44);
        v12 = ADDF(v40, v10);
        v12 = ADDF(v12, v11);

        v14 = MADD(v10, v11, v0, 2);

        v10 = SUBF(v41, v42);
        v11 = SUBF(v43, v44);
        v13 = MADD(v10, v11, v0, 1);

        v15 = MADD(v10, v11, v0, 3);
        v15 = ADDF(v15, v45);

        buf_ptr = buf_base + 5 * row_step;
#define cb(i)                                      \
    v4##i = GiLoadFloat32(buf_ptr + i * col_step); \
    v4##i = ADDF(v3##i, v4##i);
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        // result line 3
        // v40 ... v45 -> v32 ... v35
        v30 = ADDF(v41, v42);
        v31 = ADDF(v43, v44);
        v32 = ADDF(v40, v30);
        v32 = ADDF(v32, v31);

        v34 = MADD(v30, v31, v0, 2);

        v30 = SUBF(v41, v42);
        v31 = SUBF(v43, v44);
        v33 = MADD(v30, v31, v0, 1);

        v35 = MADD(v30, v31, v0, 3);
        v35 = ADDF(v35, v45);

        float* output_base = output + oc * OH * OW + oh_start * OW * pack_size +
                             ow_start * pack_size;
        float* output_ptr = nullptr;
        const float* bias_base = nullptr;
        const float* bias_ptr = nullptr;
        if (bmode == BiasMode::BIAS) {
            bias_base = bias + oc * OH * OW + oh_start * OW * pack_size +
                        ow_start * pack_size;
        }
        if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias = GiLoadFloat32(bias + oc);
        }

        switch (num_oh_valid) {
            case 4: {
                output_ptr = output_base + 3 * OW * pack_size;
                if (bmode == BiasMode::BIAS) {
                    bias_ptr = bias_base + 3 * OW * pack_size;
                }
                switch (num_ow_valid) {
                    case 4:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr + 3 * pack_size);
                        }
                        v35 = ADDF(v35, vbias);
                        v35 = op(v35);
                        GiStoreFloat32(output_ptr + 3 * pack_size, v35);
                        MEGDNN_FALLTHRU;
                    case 3:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr + 2 * pack_size);
                        }
                        v34 = ADDF(v34, vbias);
                        v34 = op(v34);
                        GiStoreFloat32(output_ptr + 2 * pack_size, v34);
                        MEGDNN_FALLTHRU;
                    case 2:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr + pack_size);
                        }
                        v33 = ADDF(v33, vbias);
                        v33 = op(v33);
                        GiStoreFloat32(output_ptr + pack_size, v33);
                        MEGDNN_FALLTHRU;
                    case 1:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr);
                        }
                        v32 = ADDF(v32, vbias);
                        v32 = op(v32);
                        GiStoreFloat32(output_ptr, v32);
                }
                MEGDNN_FALLTHRU;
            }
            case 3: {
                output_ptr = output_base + 2 * OW * pack_size;
                if (bmode == BiasMode::BIAS) {
                    bias_ptr = bias_base + 2 * OW * pack_size;
                }
                switch (num_ow_valid) {
                    case 4:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr + 3 * pack_size);
                        }
                        v05 = ADDF(v05, vbias);
                        v05 = op(v05);
                        GiStoreFloat32(output_ptr + 3 * pack_size, v05);
                        MEGDNN_FALLTHRU;
                    case 3:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr + 2 * pack_size);
                        }
                        v04 = ADDF(v04, vbias);
                        v04 = op(v04);
                        GiStoreFloat32(output_ptr + 2 * pack_size, v04);
                        MEGDNN_FALLTHRU;
                    case 2:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr + pack_size);
                        }
                        v03 = ADDF(v03, vbias);
                        v03 = op(v03);
                        GiStoreFloat32(output_ptr + pack_size, v03);
                        MEGDNN_FALLTHRU;
                    case 1:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr);
                        }
                        v02 = ADDF(v02, vbias);
                        v02 = op(v02);
                        GiStoreFloat32(output_ptr, v02);
                }
                MEGDNN_FALLTHRU;
            }
            case 2: {
                output_ptr = output_base + OW * pack_size;
                if (bmode == BiasMode::BIAS) {
                    bias_ptr = bias_base + OW * pack_size;
                }
                switch (num_ow_valid) {
                    case 4:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr + 3 * pack_size);
                        }
                        v15 = ADDF(v15, vbias);
                        v15 = op(v15);
                        GiStoreFloat32(output_ptr + 3 * pack_size, v15);
                        MEGDNN_FALLTHRU;
                    case 3:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr + 2 * pack_size);
                        }
                        v14 = ADDF(v14, vbias);
                        v14 = op(v14);
                        GiStoreFloat32(output_ptr + 2 * pack_size, v14);
                        MEGDNN_FALLTHRU;
                    case 2:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr + pack_size);
                        }
                        v13 = ADDF(v13, vbias);
                        v13 = op(v13);
                        GiStoreFloat32(output_ptr + pack_size, v13);
                        MEGDNN_FALLTHRU;
                    case 1:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr);
                        }
                        v12 = ADDF(v12, vbias);
                        v12 = op(v12);
                        GiStoreFloat32(output_ptr, v12);
                }
                MEGDNN_FALLTHRU;
            }
            case 1: {
                output_ptr = output_base;
                if (bmode == BiasMode::BIAS) {
                    bias_ptr = bias_base;
                }
                switch (num_ow_valid) {
                    case 4:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr + 3 * pack_size);
                        }
                        v25 = ADDF(v25, vbias);
                        v25 = op(v25);
                        GiStoreFloat32(output_ptr + 3 * pack_size, v25);
                        MEGDNN_FALLTHRU;
                    case 3:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr + 2 * pack_size);
                        }
                        v24 = ADDF(v24, vbias);
                        v24 = op(v24);
                        GiStoreFloat32(output_ptr + 2 * pack_size, v24);
                        MEGDNN_FALLTHRU;
                    case 2:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr + pack_size);
                        }
                        v23 = ADDF(v23, vbias);
                        v23 = op(v23);
                        GiStoreFloat32(output_ptr + pack_size, v23);
                        MEGDNN_FALLTHRU;
                    case 1:
                        if (bmode == BiasMode::BIAS) {
                            vbias = GiLoadFloat32(bias_ptr);
                        }
                        v22 = ADDF(v22, vbias);
                        v22 = op(v22);
                        GiStoreFloat32(output_ptr, v22);
                }
            }
        }

#undef MSUB
#undef MADD
    }

    static void wrapper(
            const float* output_transform_buf, const float* bias, float* output,
            const size_t OH, const size_t OW, const size_t oc_start,
            const size_t oc_end, const size_t unit_start_idx,
            const size_t nr_units_in_tile, const DType& src_dtype,
            const DType& dst_dtype) {
        auto units_w = div_ceil<size_t>(OW, 4);
        for (size_t oc = oc_start; oc < oc_end; oc += pack_size) {
            size_t oc_index = oc - oc_start;
            rep(unit_idx, nr_units_in_tile) {
                size_t index = unit_start_idx + unit_idx;
                auto nh = index / units_w;
                auto nw = index % units_w;
                size_t oh_start = nh * 4;
                size_t ow_start = nw * 4;
                megdnn_assert(oh_start < OH);
                megdnn_assert(ow_start < OW);
                size_t num_valid_oh = std::min(static_cast<size_t>(4), OH - oh_start),
                       num_valid_ow = std::min(static_cast<size_t>(4), OW - ow_start);
                if (num_valid_oh == num_valid_ow && num_valid_oh == 4) {
                    transform(
                            output_transform_buf, bias, output, oh_start, ow_start, OH,
                            OW, oc_start, oc_end, oc_index, unit_idx, nr_units_in_tile,
                            src_dtype, dst_dtype);
                } else {
                    transform(
                            output_transform_buf, bias, output, oh_start, ow_start, OH,
                            OW, oc_start, oc_end, oc_index, unit_idx, nr_units_in_tile,
                            src_dtype, dst_dtype, num_valid_oh, num_valid_ow);
                }
            }
        }
    }
};  // OutputTransformF43_NCHW44
}  // namespace

namespace megdnn {
namespace fallback {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_F43_mk4_f_nchw44)

void winograd_F43_mk4_f_nchw44::filter(
        const float* filter, float* filter_transform_buf, float* transform_mid_buf,
        size_t OC, size_t IC, size_t oc_start, size_t oc_end) {
    constexpr size_t pack_size = 4;
    MEGDNN_MARK_USED_VAR(transform_mid_buf);
    megdnn_assert(
            (oc_end - oc_start) % pack_size == 0 && oc_start % pack_size == 0 &&
                    oc_end % pack_size == 0 && IC % pack_size == 0 &&
                    OC % pack_size == 0,
            "NCHW44 Winograd filter transform requires both OC and IC "
            "are times of 4");

    size_t ICB = IC / pack_size;
    for (size_t ocb = oc_start / pack_size; ocb < oc_end / pack_size; ocb++) {
        for (size_t icb = 0; icb < ICB; icb++) {
            for (size_t ic_inner = 0; ic_inner < pack_size; ic_inner++) {
                const float* fptr = filter +
                                    (ocb * ICB + icb) * KERNEL_SIZE * KERNEL_SIZE *
                                            pack_size * pack_size +
                                    ic_inner * pack_size;

#define cb(m, n)           \
    GI_FLOAT32_t g##m##n = \
            GiLoadFloat32(fptr + (m * KERNEL_SIZE + n) * pack_size * pack_size);
                UNROLL_CALL_NOWRAPPER_D2(3, 3, cb)
#undef cb

                //! G
                //    1/4     0     0
                //   -1/6  -1/6  -1/6
                //   -1/6   1/6  -1/6
                //   1/24  1/12   1/6
                //   1/24 -1/12   1/6
                //      0     0     1

#define FILTER_TRANSFORM(n, wd, g)                                       \
    auto wd##n##0 = MULSF(g##0##n, 0.25f);                               \
    tmp0 = MULSF(ADDF(g##0##n, g##2##n), -0.1666667f);                   \
    tmp1 = MULSF(g##1##n, -0.1666667f);                                  \
    auto wd##n##1 = ADDF(tmp0, tmp1);                                    \
    auto wd##n##2 = SUBF(tmp0, tmp1);                                    \
    tmp0 = ADDF(MULSF(g##0##n, 0.0416667f), MULSF(g##2##n, 0.1666667f)); \
    tmp1 = MULSF(g##1##n, 0.0833333f);                                   \
    auto wd##n##3 = ADDF(tmp0, tmp1);                                    \
    auto wd##n##4 = SUBF(tmp0, tmp1);                                    \
    auto wd##n##5 = g##2##n;
                GI_FLOAT32_t tmp0, tmp1;
                UNROLL_CALL_RAW(3, FILTER_TRANSFORM, wd, g);
                UNROLL_CALL_RAW(6, FILTER_TRANSFORM, ret, wd);
#undef FILTER_TRANSFORM
#define cb_save(m, n)                                                                 \
    GiStoreFloat32(                                                                   \
            filter_transform_buf + (m * alpha + n) * OC * IC + ocb * IC * pack_size + \
                    icb * pack_size * pack_size + ic_inner * pack_size,               \
            ret##m##n);
                UNROLL_CALL_NOWRAPPER_D2(6, 6, cb_save)
#undef cb_save
            }
        }
    }
}

void winograd_F43_mk4_f_nchw44::input(
        const float* input, float* input_transform_buf, float* transform_mid_buf,
        size_t IH, size_t IW, size_t IC, size_t PH, size_t PW, size_t unit_start_idx,
        size_t nr_units_in_tile) {
    MEGDNN_MARK_USED_VAR(transform_mid_buf);
    constexpr size_t pack_size = 4;
    megdnn_assert(IC % pack_size == 0);
    constexpr int alpha = 3 + 4 - 1;

    auto units_w = div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);

    bool ih_valid[6], iw_valid[6];

    for (size_t ic = 0; ic < IC; ic += pack_size) {
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            size_t nh = index / units_w;
            size_t nw = index % units_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW;

            if (ih_start >= 0 && ih_start + alpha <= static_cast<int>(IH) &&
                iw_start >= 0 && iw_start + alpha <= static_cast<int>(IW)) {
                InputTransformF43_NCHW44::transform<true>(
                        input, input_transform_buf, unit_idx, nr_units_in_tile, ic, IC,
                        ih_start, iw_start, IH, IW, ih_valid, iw_valid);
            } else {
                for (int iho = 0; iho < alpha; ++iho) {
                    ih_valid[iho] =
                            (iho + ih_start >= 0 &&
                             iho + ih_start < static_cast<int>(IH));
                }
                for (int iwo = 0; iwo < alpha; ++iwo) {
                    iw_valid[iwo] =
                            (iwo + iw_start >= 0 &&
                             iwo + iw_start < static_cast<int>(IW));
                }
                InputTransformF43_NCHW44::transform<false>(
                        input, input_transform_buf, unit_idx, nr_units_in_tile, ic, IC,
                        ih_start, iw_start, IH, IW, ih_valid, iw_valid);
            }
        }
    }
}

void winograd_F43_mk4_f_nchw44::output(
        const float* output_transform_buf, const float* bias, float* output,
        float* transform_mid_buf, BiasMode bmode, NonlineMode nonline_mode, size_t OH,
        size_t OW, size_t oc_start, size_t oc_end, size_t unit_start_idx,
        size_t nr_units_in_tile) {
    MEGDNN_MARK_USED_VAR(transform_mid_buf);
#define cb(_bmode, _nonline_op, ...)                                      \
    OutputTransformF43_NCHW44<_bmode MEGDNN_COMMA _nonline_op>::wrapper(  \
            output_transform_buf, bias, output, OH, OW, oc_start, oc_end, \
            unit_start_idx, nr_units_in_tile, src_dtype, dst_dtype);

    constexpr size_t pack_size = 4;
    size_t OC = oc_end - oc_start;
    megdnn_assert(
            OC % pack_size == 0 && oc_start % pack_size == 0 && oc_end % pack_size == 0,
            "NCHW44 Winograd filter transform requires OC is times of 4");

    GI_DISPATCH_CONV_WINOGRAD_BIAS(
            megdnn_fallback_winograd_fp32_F43_mk4, cb, float, float, bmode,
            nonline_mode);
#undef cb
}

}  // namespace winograd
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
