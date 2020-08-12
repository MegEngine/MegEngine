/**
 * \file dnn/src/arm_common/conv_bias/int8x8x16/conv_direct.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/int8x8x16/conv_direct.h"
#include "src/common/utils.h"

#include <cstring>
#include "src/arm_common/simd_macro/marm_neon.h"

using namespace megdnn;
using namespace arm_common;
using namespace conv_bias;

template <bool add_to_dst>
void conv_bias::conv_direct_2x2_sc_int8_int8_int16(const int8_t* src, const int8_t* filter,
                                        int16_t* dst, size_t IH, size_t IW,
                                        size_t OH, size_t OW, size_t PH,
                                        size_t PW) {
    size_t OH_start = PH, OH_stop = OH - PH;
    size_t OW_start = PW, OW_stop = OW - PW;
    auto run_single = [&](size_t oh, size_t ow) {
        if (!add_to_dst) {
            dst[oh * OW + ow] = 0;
        }
        for (size_t fh = 0; fh < 2; ++fh)
            for (size_t fw = 0; fw < 2; ++fw) {
                size_t ih = oh + fh - PH;
                size_t iw = ow + fw - PW;
                if (ih < IH && iw < IW) {
                    dst[oh * OW + ow] +=
                            (int16_t)src[ih * IW + iw] * filter[fh * 2 + fw];
                }
            }
    };
    for (size_t oh = 0; oh < OH_start; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    for (size_t oh = OH_start; oh < OH_stop; ++oh) {
        for (size_t ow = 0; ow < OW_start; ++ow)
            run_single(oh, ow);
        for (size_t ow = OW_stop; ow < OW; ++ow)
            run_single(oh, ow);
    }
    for (size_t oh = OH_stop; oh < OH; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    // 4x8 block
    size_t oh = OH_start;
    for (; oh + 4 <= OH_stop; oh += 4) {
        size_t ih = oh - PH;
        size_t ow = OW_start;
        for (; ow < OW_stop; ow += 8) {
            size_t iw = ow - PW;
            int16_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int8_t* __restrict fptr = filter;
            int16x8_t d0, d1, d2, d3;
            int8x8_t k0, k1, s;
            if (add_to_dst) {
                d0 = vld1q_s16(dptr + 0 * OW);
                d1 = vld1q_s16(dptr + 1 * OW);
                d2 = vld1q_s16(dptr + 2 * OW);
                d3 = vld1q_s16(dptr + 3 * OW);
            } else {
                d0 = vdupq_n_s16(0);
                d1 = vdupq_n_s16(0);
                d2 = vdupq_n_s16(0);
                d3 = vdupq_n_s16(0);
            }

            for (size_t fw = 0; fw < 2; ++fw) {
                k0 = vdup_n_s8(fptr[0 * 2 + fw]);
                k1 = vdup_n_s8(fptr[1 * 2 + fw]);

                s = vld1_s8(sptr + 0 * IW);
                d0 = vmlal_s8(d0, k0, s);

                s = vld1_s8(sptr + 1 * IW);
                d0 = vmlal_s8(d0, k1, s);
                d1 = vmlal_s8(d1, k0, s);

                s = vld1_s8(sptr + 2 * IW);
                d1 = vmlal_s8(d1, k1, s);
                d2 = vmlal_s8(d2, k0, s);

                s = vld1_s8(sptr + 3 * IW);
                d2 = vmlal_s8(d2, k1, s);
                d3 = vmlal_s8(d3, k0, s);

                s = vld1_s8(sptr + 4 * IW);
                d3 = vmlal_s8(d3, k1, s);

                ++sptr;
            }
            vst1q_s16(dptr + 0 * OW, d0);
            vst1q_s16(dptr + 1 * OW, d1);
            vst1q_s16(dptr + 2 * OW, d2);
            vst1q_s16(dptr + 3 * OW, d3);
        }
    }
    if (oh + 3 == OH_stop) {
        size_t ih = oh - PH;
        size_t ow = OW_start;
        for (; ow < OW_stop; ow += 8) {
            size_t iw = ow - PW;
            int16_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int8_t* __restrict fptr = filter;
            int16x8_t d0, d1, d2;
            int8x8_t k0, k1, s;
            if (add_to_dst) {
                d0 = vld1q_s16(dptr + 0 * OW);
                d1 = vld1q_s16(dptr + 1 * OW);
                d2 = vld1q_s16(dptr + 2 * OW);
            } else {
                d0 = vdupq_n_s16(0);
                d1 = vdupq_n_s16(0);
                d2 = vdupq_n_s16(0);
            }
            for (size_t fw = 0; fw < 2; ++fw) {
                k0 = vdup_n_s8(fptr[0 * 2 + fw]);
                k1 = vdup_n_s8(fptr[1 * 2 + fw]);

                s = vld1_s8(sptr + 0 * IW);
                d0 = vmlal_s8(d0, k0, s);

                s = vld1_s8(sptr + 1 * IW);
                d0 = vmlal_s8(d0, k1, s);
                d1 = vmlal_s8(d1, k0, s);

                s = vld1_s8(sptr + 2 * IW);
                d1 = vmlal_s8(d1, k1, s);
                d2 = vmlal_s8(d2, k0, s);

                s = vld1_s8(sptr + 3 * IW);
                d2 = vmlal_s8(d2, k1, s);

                ++sptr;
            }
            vst1q_s16(dptr + 0 * OW, d0);
            vst1q_s16(dptr + 1 * OW, d1);
            vst1q_s16(dptr + 2 * OW, d2);
        }
    } else if (oh + 2 == OH_stop) {
        size_t ih = oh - PH;
        size_t ow = OW_start;
        for (; ow < OW_stop; ow += 8) {
            size_t iw = ow - PW;
            int16_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int8_t* __restrict fptr = filter;
            int16x8_t d0, d1;
            int8x8_t k0, k1, s;
            if (add_to_dst) {
                d0 = vld1q_s16(dptr + 0 * OW);
                d1 = vld1q_s16(dptr + 1 * OW);
            } else {
                d0 = vdupq_n_s16(0);
                d1 = vdupq_n_s16(0);
            }
            for (size_t fw = 0; fw < 2; ++fw) {
                k0 = vdup_n_s8(fptr[0 * 2 + fw]);
                k1 = vdup_n_s8(fptr[1 * 2 + fw]);

                s = vld1_s8(sptr + 0 * IW);
                d0 = vmlal_s8(d0, k0, s);

                s = vld1_s8(sptr + 1 * IW);
                d0 = vmlal_s8(d0, k1, s);
                d1 = vmlal_s8(d1, k0, s);

                s = vld1_s8(sptr + 2 * IW);
                d1 = vmlal_s8(d1, k1, s);

                ++sptr;
            }
            vst1q_s16(dptr + 0 * OW, d0);
            vst1q_s16(dptr + 1 * OW, d1);
        }
    } else if (oh + 1 == OH_stop) {
        size_t ih = oh - PH;
        size_t ow = OW_start;
        for (; ow < OW_stop; ow += 8) {
            size_t iw = ow - PW;
            int16_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int8_t* __restrict fptr = filter;
            int16x8_t d0;
            int8x8_t k0, k1, s;
            if (add_to_dst) {
                d0 = vld1q_s16(dptr + 0 * OW);
            } else {
                d0 = vdupq_n_s16(0);
            }
            for (size_t fw = 0; fw < 2; ++fw) {
                k0 = vdup_n_s8(fptr[0 * 2 + fw]);
                k1 = vdup_n_s8(fptr[1 * 2 + fw]);

                s = vld1_s8(sptr + 0 * IW);
                d0 = vmlal_s8(d0, k0, s);

                s = vld1_s8(sptr + 1 * IW);
                d0 = vmlal_s8(d0, k1, s);

                ++sptr;
            }
            vst1q_s16(dptr + 0 * OW, d0);
        }
    }
}

template <bool add_to_dst>
void conv_bias::conv_direct_3x3_sc_int8_int8_int16(const int8_t* src, const int8_t* filter,
                                        int16_t* dst, size_t IH, size_t IW,
                                        size_t OH, size_t OW, size_t PH,
                                        size_t PW) {
    size_t OH_start = PH, OH_stop = OH - PH;
    size_t OW_start = PW, OW_stop = OW - PW;

    auto run_single = [&](size_t oh, size_t ow) {
        if (!add_to_dst) {
            dst[oh * OW + ow] = 0;
        }
        for (size_t fh = 0; fh < 3; ++fh)
            for (size_t fw = 0; fw < 3; ++fw) {
                size_t ih = oh + fh - PH;
                size_t iw = ow + fw - PW;
                if (ih < IH && iw < IW) {
                    dst[oh * OW + ow] +=
                            (int16_t)src[ih * IW + iw] * filter[fh * 3 + fw];
                }
            }
    };

    for (size_t oh = 0; oh < OH_start; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    for (size_t oh = OH_start; oh < OH_stop; ++oh) {
        for (size_t ow = 0; ow < OW_start; ++ow)
            run_single(oh, ow);
        for (size_t ow = OW_stop; ow < OW; ++ow)
            run_single(oh, ow);
    }
    for (size_t oh = OH_stop; oh < OH; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }

    // 4x8 block
    size_t oh = OH_start;
    for (; oh + 4 <= OH_stop; oh += 4) {
        size_t ih = oh - PH;
        size_t ow = OW_start;
        for (; ow < OW_stop; ow += 8) {
            size_t iw = ow - PW;
            int16_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int8_t* __restrict fptr = filter;
            int16x8_t d0, d1, d2, d3;
            int8x8_t k0, k1, k2, s;
            if (add_to_dst) {
                d0 = vld1q_s16(dptr + 0 * OW);
                d1 = vld1q_s16(dptr + 1 * OW);
                d2 = vld1q_s16(dptr + 2 * OW);
                d3 = vld1q_s16(dptr + 3 * OW);
            } else {
                d0 = vdupq_n_s16(0);
                d1 = vdupq_n_s16(0);
                d2 = vdupq_n_s16(0);
                d3 = vdupq_n_s16(0);
            }
            for (size_t fw = 0; fw < 3; ++fw) {
                k0 = vdup_n_s8(fptr[0 * 3 + fw]);
                k1 = vdup_n_s8(fptr[1 * 3 + fw]);
                k2 = vdup_n_s8(fptr[2 * 3 + fw]);

                s = vld1_s8(sptr + 0 * IW);
                d0 = vmlal_s8(d0, k0, s);

                s = vld1_s8(sptr + 1 * IW);
                d0 = vmlal_s8(d0, k1, s);
                d1 = vmlal_s8(d1, k0, s);

                s = vld1_s8(sptr + 2 * IW);
                d0 = vmlal_s8(d0, k2, s);
                d1 = vmlal_s8(d1, k1, s);
                d2 = vmlal_s8(d2, k0, s);

                s = vld1_s8(sptr + 3 * IW);
                d1 = vmlal_s8(d1, k2, s);
                d2 = vmlal_s8(d2, k1, s);
                d3 = vmlal_s8(d3, k0, s);

                s = vld1_s8(sptr + 4 * IW);
                d2 = vmlal_s8(d2, k2, s);
                d3 = vmlal_s8(d3, k1, s);

                s = vld1_s8(sptr + 5 * IW);
                d3 = vmlal_s8(d3, k2, s);

                ++sptr;
            }
            vst1q_s16(dptr + 0 * OW, d0);
            vst1q_s16(dptr + 1 * OW, d1);
            vst1q_s16(dptr + 2 * OW, d2);
            vst1q_s16(dptr + 3 * OW, d3);
        }
    }

    if (oh + 3 == OH_stop) {
        size_t ih = oh - PH;
        size_t ow = OW_start;
        for (; ow < OW_stop; ow += 8) {
            size_t iw = ow - PW;
            int16_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int8_t* __restrict fptr = filter;
            int16x8_t d0, d1, d2;
            int8x8_t k0, k1, k2, s;
            if (add_to_dst) {
                d0 = vld1q_s16(dptr + 0 * OW);
                d1 = vld1q_s16(dptr + 1 * OW);
                d2 = vld1q_s16(dptr + 2 * OW);
            } else {
                d0 = vdupq_n_s16(0);
                d1 = vdupq_n_s16(0);
                d2 = vdupq_n_s16(0);
            }
            for (size_t fw = 0; fw < 3; ++fw) {
                k0 = vdup_n_s8(fptr[0 * 3 + fw]);
                k1 = vdup_n_s8(fptr[1 * 3 + fw]);
                k2 = vdup_n_s8(fptr[2 * 3 + fw]);

                s = vld1_s8(sptr + 0 * IW);
                d0 = vmlal_s8(d0, k0, s);

                s = vld1_s8(sptr + 1 * IW);
                d0 = vmlal_s8(d0, k1, s);
                d1 = vmlal_s8(d1, k0, s);

                s = vld1_s8(sptr + 2 * IW);
                d0 = vmlal_s8(d0, k2, s);
                d1 = vmlal_s8(d1, k1, s);
                d2 = vmlal_s8(d2, k0, s);

                s = vld1_s8(sptr + 3 * IW);
                d1 = vmlal_s8(d1, k2, s);
                d2 = vmlal_s8(d2, k1, s);

                s = vld1_s8(sptr + 4 * IW);
                d2 = vmlal_s8(d2, k2, s);
                ++sptr;
            }
            vst1q_s16(dptr + 0 * OW, d0);
            vst1q_s16(dptr + 1 * OW, d1);
            vst1q_s16(dptr + 2 * OW, d2);
        }
    } else if (oh + 2 == OH_stop) {
        size_t ih = oh - PH;
        size_t ow = OW_start;
        for (; ow < OW_stop; ow += 8) {
            size_t iw = ow - PW;
            int16_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int8_t* __restrict fptr = filter;
            int16x8_t d0, d1;
            int8x8_t k0, k1, k2, s;
            if (add_to_dst) {
                d0 = vld1q_s16(dptr + 0 * OW);
                d1 = vld1q_s16(dptr + 1 * OW);
            } else {
                d0 = vdupq_n_s16(0);
                d1 = vdupq_n_s16(0);
            }
            for (size_t fw = 0; fw < 3; ++fw) {
                k0 = vdup_n_s8(fptr[0 * 3 + fw]);
                k1 = vdup_n_s8(fptr[1 * 3 + fw]);
                k2 = vdup_n_s8(fptr[2 * 3 + fw]);

                s = vld1_s8(sptr + 0 * IW);
                d0 = vmlal_s8(d0, k0, s);

                s = vld1_s8(sptr + 1 * IW);
                d0 = vmlal_s8(d0, k1, s);
                d1 = vmlal_s8(d1, k0, s);

                s = vld1_s8(sptr + 2 * IW);
                d0 = vmlal_s8(d0, k2, s);
                d1 = vmlal_s8(d1, k1, s);

                s = vld1_s8(sptr + 3 * IW);
                d1 = vmlal_s8(d1, k2, s);

                ++sptr;
            }
            vst1q_s16(dptr + 0 * OW, d0);
            vst1q_s16(dptr + 1 * OW, d1);
        }
    } else if (oh + 1 == OH_stop) {
        size_t ih = oh - PH;
        size_t ow = OW_start;

        for (; ow < OW_stop; ow += 8) {
            size_t iw = ow - PW;
            int16_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int8_t* __restrict fptr = filter;
            int16x8_t d0;
            int8x8_t k0, k1, k2, s;

            if (add_to_dst) {
                d0 = vld1q_s16(dptr + 0 * OW);
            } else {
                d0 = vdupq_n_s16(0);
            }
            for (size_t fw = 0; fw < 3; ++fw) {
                k0 = vdup_n_s8(fptr[0 * 3 + fw]);
                k1 = vdup_n_s8(fptr[1 * 3 + fw]);
                k2 = vdup_n_s8(fptr[2 * 3 + fw]);

                s = vld1_s8(sptr + 0 * IW);
                d0 = vmlal_s8(d0, k0, s);

                s = vld1_s8(sptr + 1 * IW);
                d0 = vmlal_s8(d0, k1, s);

                s = vld1_s8(sptr + 2 * IW);
                d0 = vmlal_s8(d0, k2, s);

                ++sptr;
            }
            vst1q_s16(dptr + 0 * OW, d0);
        }
    }
}

template <bool add_to_dst>
void conv_bias::conv_direct_5x5_sc_int8_int8_int16(const int8_t* src, const int8_t* filter,
                                        int16_t* dst, size_t IH, size_t IW,
                                        size_t OH, size_t OW, size_t PH,
                                        size_t PW) {
    size_t OH_start = PH, OH_stop = OH - PH;
    size_t OW_start = PW, OW_stop = OW - PW;
    auto run_single = [&](size_t oh, size_t ow) {
        if (!add_to_dst) {
            dst[oh * OW + ow] = 0;
        }
        for (size_t fh = 0; fh < 5; ++fh)
            for (size_t fw = 0; fw < 5; ++fw) {
                size_t ih = oh + fh - PH;
                size_t iw = ow + fw - PW;
                if (ih < IH && iw < IW) {
                    dst[oh * OW + ow] +=
                            (int16_t)src[ih * IW + iw] * filter[fh * 5 + fw];
                }
            }
    };
    for (size_t oh = 0; oh < OH_start; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    for (size_t oh = OH_start; oh < OH_stop; ++oh) {
        for (size_t ow = 0; ow < OW_start; ++ow)
            run_single(oh, ow);
        for (size_t ow = OW_stop; ow < OW; ++ow)
            run_single(oh, ow);
    }
    for (size_t oh = OH_stop; oh < OH; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    // 4x8 block
    size_t oh = OH_start;
    for (; oh + 4 <= OH_stop; oh += 4) {
        size_t ih = oh - PH;
        size_t ow = OW_start;
        for (; ow + 8 <= OW_stop; ow += 8) {
            size_t iw = ow - PW;
            int16_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int8_t* __restrict fptr = filter;
            int16x8_t d0, d1, d2, d3;
            int8x8_t k0, k1, k2, k3, k4, s;
            if (add_to_dst) {
                d0 = vld1q_s16(dptr + 0 * OW);
                d1 = vld1q_s16(dptr + 1 * OW);
                d2 = vld1q_s16(dptr + 2 * OW);
                d3 = vld1q_s16(dptr + 3 * OW);
            } else {
                d0 = vdupq_n_s16(0);
                d1 = vdupq_n_s16(0);
                d2 = vdupq_n_s16(0);
                d3 = vdupq_n_s16(0);
            }
            for (size_t fw = 0; fw < 5; ++fw) {
                k0 = vdup_n_s8(fptr[0 * 5 + fw]);
                k1 = vdup_n_s8(fptr[1 * 5 + fw]);
                k2 = vdup_n_s8(fptr[2 * 5 + fw]);
                k3 = vdup_n_s8(fptr[3 * 5 + fw]);
                k4 = vdup_n_s8(fptr[4 * 5 + fw]);

                s = vld1_s8(sptr + 0 * IW);
                d0 = vmlal_s8(d0, k0, s);

                s = vld1_s8(sptr + 1 * IW);
                d0 = vmlal_s8(d0, k1, s);
                d1 = vmlal_s8(d1, k0, s);

                s = vld1_s8(sptr + 2 * IW);
                d0 = vmlal_s8(d0, k2, s);
                d1 = vmlal_s8(d1, k1, s);
                d2 = vmlal_s8(d2, k0, s);

                s = vld1_s8(sptr + 3 * IW);
                d0 = vmlal_s8(d0, k3, s);
                d1 = vmlal_s8(d1, k2, s);
                d2 = vmlal_s8(d2, k1, s);
                d3 = vmlal_s8(d3, k0, s);

                s = vld1_s8(sptr + 4 * IW);
                d0 = vmlal_s8(d0, k4, s);
                d1 = vmlal_s8(d1, k3, s);
                d2 = vmlal_s8(d2, k2, s);
                d3 = vmlal_s8(d3, k1, s);

                s = vld1_s8(sptr + 5 * IW);
                d1 = vmlal_s8(d1, k4, s);
                d2 = vmlal_s8(d2, k3, s);
                d3 = vmlal_s8(d3, k2, s);

                s = vld1_s8(sptr + 6 * IW);
                d2 = vmlal_s8(d2, k4, s);
                d3 = vmlal_s8(d3, k3, s);

                s = vld1_s8(sptr + 7 * IW);
                d3 = vmlal_s8(d3, k4, s);

                ++sptr;
            }
            vst1q_s16(dptr + 0 * OW, d0);
            vst1q_s16(dptr + 1 * OW, d1);
            vst1q_s16(dptr + 2 * OW, d2);
            vst1q_s16(dptr + 3 * OW, d3);
        }
        for (; ow < OW_stop; ++ow) {
            run_single(oh + 0, ow);
            run_single(oh + 1, ow);
            run_single(oh + 2, ow);
            run_single(oh + 3, ow);
        }
    }
    for (; oh < OH_stop; ++oh) {
        for (size_t ow = OW_start; ow < OW_stop; ++ow) {
            run_single(oh, ow);
        }
    }
}

template void conv_bias::conv_direct_2x2_sc_int8_int8_int16<true>(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW);
template void conv_bias::conv_direct_2x2_sc_int8_int8_int16<false>(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW);
template void conv_bias::conv_direct_3x3_sc_int8_int8_int16<true>(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW);
template void conv_bias::conv_direct_3x3_sc_int8_int8_int16<false>(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW);
template void conv_bias::conv_direct_5x5_sc_int8_int8_int16<true>(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW);
template void conv_bias::conv_direct_5x5_sc_int8_int8_int16<false>(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW);

// vim: syntax=cpp.doxygen
