#include "megdnn/arch.h"
#if MGB_ENABLE_DOT
#include "src/arm_common/simd_macro/marm_neon.h"

#include "src/arm_common/conv_bias/int8/direct_kernels/dot_direct_nchw_large.h"
#include "src/arm_common/conv_bias/int8/direct_kernels/dot_direct_nchw_large_common.h"
#include "src/common/unroll_macro.h"

MEGDNN_ATTRIBUTE_TARGET("dotprod")
void megdnn_dot_nchw_large_chanwise_direct_conv_11x11s1_oh4_ow16(
        const int8_t* src, const int8_t* weight, int32_t bias, int8_t* dst, size_t oh,
        size_t ow, size_t OH, size_t OW, size_t pad_iw, const float scale,
        int8_t relu_val) {
    //! 4x16
    const size_t SH = 1;
    const size_t SW = 1;

    static const uint8_t tbl_array_0[16] = {0, 1, 2, 3, 1, 2, 3, 4,
                                            2, 3, 4, 5, 3, 4, 5, 6};
    static const uint8_t tbl_array_1[16] = {4, 5, 6, 7, 5, 6, 7, 8,
                                            6, 7, 8, 9, 7, 8, 9, 10};
    static const uint8_t tbl_array_2[16] = {8,  9,  10, 11, 9,  10, 11, 12,
                                            10, 11, 12, 13, 11, 12, 13, 14};

    uint8x16_t tbl_reg_0 = vld1q_u8(&tbl_array_0[0]);
    uint8x16_t tbl_reg_1 = vld1q_u8(&tbl_array_1[0]);
    uint8x16_t tbl_reg_2 = vld1q_u8(&tbl_array_2[0]);

    const int8_t* src_n = src + oh * SH * pad_iw + ow * SW;

    //! init
    int32x4_t c[4][4];
#define cb(step)                    \
    c[step][0] = vdupq_n_s32(bias); \
    c[step][1] = vdupq_n_s32(bias); \
    c[step][2] = vdupq_n_s32(bias); \
    c[step][3] = vdupq_n_s32(bias);

    UNROLL_CALL_RAW(4, cb);
#undef cb
#define flt_reg 4
    int8x16_t flt[flt_reg];
    flt[0] = vld1q_s8(weight + 0 * 16);
    flt[1] = vld1q_s8(weight + 1 * 16);
    flt[2] = vld1q_s8(weight + 2 * 16);
    flt[3] = vld1q_s8(weight + 3 * 16);
    //! row 0

    int8x16_t read_w[2];
    read_w[0] = vld1q_s8(src_n + 0 * pad_iw);
    read_w[1] = vld1q_s8(src_n + 0 * pad_iw + 16);
    int8x16_t n0123_0 = vqtbl1q_s8(read_w[0], tbl_reg_0);
    int8x16_t n4567_0 = vqtbl1q_s8(read_w[0], tbl_reg_1);
    int8x16_t n89ab_0 = vqtbl1q_s8(read_w[0], tbl_reg_2);
    int8x16_t ncdef_0 = vqtbl1q_s8(vextq_s8(read_w[0], read_w[1], 12), tbl_reg_0);
    int8x16_t n0123_1 = n4567_0;
    int8x16_t n4567_1 = n89ab_0;
    int8x16_t n89ab_1 = ncdef_0;
    int8x16_t ncdef_1 = vqtbl1q_s8(read_w[1], tbl_reg_0);
    int8x16_t n0123_2 = n89ab_0;
    int8x16_t n4567_2 = ncdef_0;
    int8x16_t n89ab_2 = ncdef_1;
    int8x16_t ncdef_2 = vqtbl1q_s8(read_w[1], tbl_reg_1);
#define CAL_C(oh, flt_start)                                       \
    c[oh][0] = vdotq_laneq_s32(                                    \
            c[oh][0], n0123_0, flt[(flt_start + 0) / 4 % flt_reg], \
            (flt_start + 0) % 4);                                  \
    c[oh][1] = vdotq_laneq_s32(                                    \
            c[oh][1], n4567_0, flt[(flt_start + 0) / 4 % flt_reg], \
            (flt_start + 0) % 4);                                  \
    c[oh][2] = vdotq_laneq_s32(                                    \
            c[oh][2], n89ab_0, flt[(flt_start + 0) / 4 % flt_reg], \
            (flt_start + 0) % 4);                                  \
    c[oh][3] = vdotq_laneq_s32(                                    \
            c[oh][3], ncdef_0, flt[(flt_start + 0) / 4 % flt_reg], \
            (flt_start + 0) % 4);                                  \
    c[oh][0] = vdotq_laneq_s32(                                    \
            c[oh][0], n0123_1, flt[(flt_start + 1) / 4 % flt_reg], \
            (flt_start + 1) % 4);                                  \
    c[oh][1] = vdotq_laneq_s32(                                    \
            c[oh][1], n4567_1, flt[(flt_start + 1) / 4 % flt_reg], \
            (flt_start + 1) % 4);                                  \
    c[oh][2] = vdotq_laneq_s32(                                    \
            c[oh][2], n89ab_1, flt[(flt_start + 1) / 4 % flt_reg], \
            (flt_start + 1) % 4);                                  \
    c[oh][3] = vdotq_laneq_s32(                                    \
            c[oh][3], ncdef_1, flt[(flt_start + 1) / 4 % flt_reg], \
            (flt_start + 1) % 4);                                  \
    c[oh][0] = vdotq_laneq_s32(                                    \
            c[oh][0], n0123_2, flt[(flt_start + 2) / 4 % flt_reg], \
            (flt_start + 2) % 4);                                  \
    c[oh][1] = vdotq_laneq_s32(                                    \
            c[oh][1], n4567_2, flt[(flt_start + 2) / 4 % flt_reg], \
            (flt_start + 2) % 4);                                  \
    c[oh][2] = vdotq_laneq_s32(                                    \
            c[oh][2], n89ab_2, flt[(flt_start + 2) / 4 % flt_reg], \
            (flt_start + 2) % 4);                                  \
    c[oh][3] = vdotq_laneq_s32(                                    \
            c[oh][3], ncdef_2, flt[(flt_start + 2) / 4 % flt_reg], \
            (flt_start + 2) % 4);

    CAL_C(0, 0);

    //! row 1
#define LOAD_SRC(row_id)                                                 \
    read_w[0] = vld1q_s8(src_n + row_id * pad_iw);                       \
    read_w[1] = vld1q_s8(src_n + row_id * pad_iw + 16);                  \
    n0123_0 = vqtbl1q_s8(read_w[0], tbl_reg_0);                          \
    n4567_0 = vqtbl1q_s8(read_w[0], tbl_reg_1);                          \
    n89ab_0 = vqtbl1q_s8(read_w[0], tbl_reg_2);                          \
    ncdef_0 = vqtbl1q_s8(vextq_s8(read_w[0], read_w[1], 12), tbl_reg_0); \
    n0123_1 = n4567_0;                                                   \
    n4567_1 = n89ab_0;                                                   \
    n89ab_1 = ncdef_0;                                                   \
    ncdef_1 = vqtbl1q_s8(read_w[1], tbl_reg_0);                          \
    n0123_2 = n89ab_0;                                                   \
    n4567_2 = ncdef_0;                                                   \
    n89ab_2 = ncdef_1;                                                   \
    ncdef_2 = vqtbl1q_s8(read_w[1], tbl_reg_1);

    LOAD_SRC(1);
    CAL_C(0, 3);
    CAL_C(1, 0);

    //! row 2
    LOAD_SRC(2);
    CAL_C(0, 3 * 2);
    CAL_C(1, 3 * 1);
    CAL_C(2, 3 * 0);

    //! row 3
    LOAD_SRC(3);
    CAL_C(0, 3 * 3);
    CAL_C(1, 3 * 2);
    CAL_C(2, 3 * 1);
    CAL_C(3, 3 * 0);

    //! row 4
    LOAD_SRC(4);
    CAL_C(0, 3 * 4);
    CAL_C(1, 3 * 3);
    CAL_C(2, 3 * 2);
    CAL_C(3, 3 * 1);
    //! update flt 4 -> 0
    flt[0] = vld1q_s8(weight + 4 * 16);

    //! row 5
    LOAD_SRC(5);
    CAL_C(0, 3 * 5);
    CAL_C(1, 3 * 4);
    CAL_C(2, 3 * 3);
    CAL_C(3, 3 * 2);
    //! update flt 5 -> 1
    flt[1] = vld1q_s8(weight + 5 * 16);

    //! row 6
    LOAD_SRC(6);
    CAL_C(0, 3 * 6);
    CAL_C(1, 3 * 5);
    CAL_C(2, 3 * 4);
    CAL_C(3, 3 * 3);
    //! update flt 6 -> 2
    flt[2] = vld1q_s8(weight + 6 * 16);

    //! row 7
    LOAD_SRC(7);
    CAL_C(0, 3 * 7);
    CAL_C(1, 3 * 6);
    CAL_C(2, 3 * 5);
    CAL_C(3, 3 * 4);

    //! row 8
    LOAD_SRC(8);
    CAL_C(3, 3 * 5);
    //! update flt 7 -> 3
    flt[3] = vld1q_s8(weight + 7 * 16);
    CAL_C(2, 3 * 6);
    CAL_C(1, 3 * 7);
    CAL_C(0, 3 * 8);

    //! row 9
    LOAD_SRC(9);
    CAL_C(0, 3 * 9);
    CAL_C(1, 3 * 8);
    CAL_C(2, 3 * 7);
    CAL_C(3, 3 * 6);

    //! row 10
    LOAD_SRC(10);
    //! update flt 8 -> 0
    flt[0] = vld1q_s8(weight + 8 * 16);
    CAL_C(3, 3 * 7);
    CAL_C(2, 3 * 8);
    CAL_C(1, 3 * 9);
    CAL_C(0, 3 * 10);

    //! row 11
    LOAD_SRC(11);
    CAL_C(1, 3 * 10);
    CAL_C(2, 3 * 9);
    CAL_C(3, 3 * 8);

    //! row 12
    LOAD_SRC(12);
    CAL_C(2, 3 * 10);
    CAL_C(3, 3 * 9);

    //! row 13
    LOAD_SRC(13);
    CAL_C(3, 3 * 10);

    float32x4_t dst_reg[4][4];
#define cb(step)                                  \
    dst_reg[step][0] = vcvtq_f32_s32(c[step][0]); \
    dst_reg[step][1] = vcvtq_f32_s32(c[step][1]); \
    dst_reg[step][2] = vcvtq_f32_s32(c[step][2]); \
    dst_reg[step][3] = vcvtq_f32_s32(c[step][3]);

    UNROLL_CALL_RAW(4, cb);
#undef cb

#define cb(step)                                             \
    dst_reg[step][0] = vmulq_n_f32(dst_reg[step][0], scale); \
    dst_reg[step][1] = vmulq_n_f32(dst_reg[step][1], scale); \
    dst_reg[step][2] = vmulq_n_f32(dst_reg[step][2], scale); \
    dst_reg[step][3] = vmulq_n_f32(dst_reg[step][3], scale);

    UNROLL_CALL_RAW(4, cb);
#undef cb
    int8_t* dst_store = dst + oh * OW + ow;
    int8x16_t relu_reg = vdupq_n_s8(relu_val);
#define cb(step)                                                                    \
    quant_store_s8(                                                                 \
            dst_reg[step][0], dst_reg[step][1], dst_reg[step][2], dst_reg[step][3], \
            dst_store + step * OW, relu_reg);

    UNROLL_CALL_RAW(4, cb);
#undef cb
}
#endif