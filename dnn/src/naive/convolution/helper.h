/**
 * \file dnn/src/naive/convolution/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "megdnn/oprs/nn.h"
#include "src/common/utils.h"

#include <cstring>

namespace megdnn {
namespace naive {
namespace convolution {

struct GroupCounter {
    const size_t grp_size;
    size_t cur_grp = 0, cur_off = 0;

    explicit GroupCounter(size_t grp_size) : grp_size{grp_size} {}

    void next() {
        if ((++cur_off) == grp_size) {
            cur_off = 0;
            ++cur_grp;
        }
    }
};

struct StrategyFwd {
    template <typename st, typename ft, typename ct>
    static void on(st& s, ft& f, ct& d, DType, DType, DType) {
        d += static_cast<ct>(s) * static_cast<ct>(f);
    }

    template <typename ct, typename dt>
    static void write(ct& d, dt& dst) {
        dst = static_cast<dt>(d);
    }

    template <typename dt>
    static void init_dval(dt& d) {
        d = static_cast<dt>(0);
    }
};

// Explicit specialization of member function template is not allowed to happen
// in class scope, this is a defect of C++ specification which will be fixed in
// C++17. We workaround this by marking the implmentation as inline and move
// out of class definition.
template <>
inline void StrategyFwd::on(dt_quint8& s, dt_quint8& f, dt_qint32& d,
                            DType src_dt, DType filt_dt, DType) {
    auto cast = [](const dt_quint8& val, DType dt) {
        return dt_qint32(static_cast<int32_t>(val.as_uint8()) -
                         dt.param<dtype::Quantized8Asymm>().zero_point);
    };
    d += cast(s, src_dt) * cast(f, filt_dt);
}

template <>
inline void StrategyFwd::on(dt_qint8& s, dt_qint8& f, dt_float32& d,
                            DType src_dt, DType filt_dt, DType) {
    auto cast = [](const dt_qint8& val, DType dt) {
        return dt.param<dtype::QuantizedS8>().dequantize(val);
    };
    d += cast(s, src_dt) * cast(f, filt_dt);
}

template <>
inline void StrategyFwd::on(dt_qint8& s, dt_qint8& f, dt_qint32& d, DType,
                            DType, DType) {
    auto cast = [](const dt_qint8& val) {
        return dt_qint32(static_cast<int32_t>(val.as_int8()));
    };
    d += cast(s) * cast(f);
}

struct StrategyBwdData {
    template <typename st, typename ft, typename dt>
    static void on(st& s, ft& f, dt& d, DType, DType, DType) {
        s += static_cast<st>(f) * static_cast<st>(d);
    }

    template <typename ct, typename dt>
    static void write(ct&, dt&) {}

    template <typename dt>
    static void init_dval(dt&) {}
};

template <>
inline void StrategyBwdData::on(int& s, signed char& f, signed char& d, DType,
                                DType, DType) {
    auto cast = [](signed char& val) {
        return static_cast<int32_t>(((megdnn::dt_qint8)val).as_int8());
    };
    s += cast(f) * cast(d);
}

template <>
inline void StrategyBwdData::on(dt_qint32& s, dt_quint8& f, dt_quint8& d, DType,
                                DType filt_dt, DType dst_dt) {
    auto cast = [](const dt_quint8& val, DType dt) {
        return dt_qint32(static_cast<int32_t>(val.as_uint8()) -
                         dt.param<dtype::Quantized8Asymm>().zero_point);
    };
    s += cast(f, filt_dt) * cast(d, dst_dt);
}

template <>
inline void StrategyBwdData::on(dt_qint32& s, dt_qint8& f, dt_qint8& d, DType,
                                DType, DType) {
    auto cast = [](const dt_qint8& val) {
        return dt_qint32(static_cast<int32_t>(val.as_int8()));
    };
    s += cast(f) * cast(d);
}

struct StrategyBwdFlt {
    template <typename st, typename ft, typename dt>
    static void on(st& s, ft& f, dt& d, DType, DType, DType) {
        f += static_cast<ft>(s) * static_cast<ft>(d);
    }

    template <typename ct, typename dt>
    static void write(ct&, dt&) {}

    template <typename dt>
    static void init_dval(dt&) {}
};

struct ConvFilterVisitor {
    template <typename ftype>
    static ftype* get_current_ptr(ftype* fptr, size_t /* batch */,
                                  size_t /* oc */, size_t /* oh */,
                                  size_t /* ow */, size_t /* filter_sizes*/) {
        return fptr;
    }
};

template <typename stype, typename ftype, typename dtype, typename comp_type,
          class Strategy, typename FilterMeta,
          typename FilterVisitor = ConvFilterVisitor>
void compute2d(_megdnn_tensor_in src, ftype* __restrict fptr,
               _megdnn_tensor_out dst, const FilterMeta& filter_meta) {
    size_t spatial_start, channel_pos, batch_pos;
    using Format = param::Convolution::Format;
    if (filter_meta.format == Format::NCHW ||
        filter_meta.format == Format::NCHW88 ||
        filter_meta.format == Format::NCHW44 ||
        filter_meta.format == Format::NCHW44_DOT ||
        filter_meta.format == Format::NCHW4 ||
        filter_meta.format == Format::NCHW4_NCHW ||
        filter_meta.format == Format::NCHW4_NCHW32 ||
        filter_meta.format == Format::NCHW8 ||
        filter_meta.format == Format::NCHW32 ||
        filter_meta.format == Format::NCHW32_NCHW4) {
        spatial_start = 2;
        channel_pos = 1;
        batch_pos = 0;
    } else if (filter_meta.format == Format::CHWN4) {
        spatial_start = 1;
        channel_pos = 0;
        batch_pos = 3;
    } else {
        megdnn_assert(filter_meta.format == Format::NHWC,
                      "invalid conv format");
        spatial_start = 1;
        channel_pos = 3;
        batch_pos = 0;
    }

    auto N = src.layout.shape[batch_pos], IH = src.layout.shape[spatial_start],
         IW = src.layout.shape[spatial_start + 1];
    auto FH = filter_meta.spatial[0], FW = filter_meta.spatial[1];
    auto OC = dst.layout.shape[channel_pos],
         OH = dst.layout.shape[spatial_start],
         OW = dst.layout.shape[spatial_start + 1];

    if (filter_meta.format == Format::NCHW4 ||
        filter_meta.format == Format::CHWN4 ||
        filter_meta.format == Format::NCHW44_DOT ||
        filter_meta.format == Format::NCHW44 || 
        filter_meta.format == Format::NCHW32_NCHW4) {
        OC *= 4;
    } else if (filter_meta.format == Format::NCHW8 ||
               filter_meta.format == Format::NCHW88) {
        OC *= 8;
    } else if (filter_meta.format == Format::NCHW32 ||
               filter_meta.format == Format::NCHW4_NCHW32) {
        OC *= 32;
    }

    size_t FS_G, FS_OC, FS_IC, FS_SPATIAL;
    if (filter_meta.format == Format::NCHW ||
        filter_meta.format == Format::NCHW4 ||
        filter_meta.format == Format::NCHW4_NCHW ||
        filter_meta.format == Format::NCHW4_NCHW32 ||
        filter_meta.format == Format::NCHW8 ||
        filter_meta.format == Format::NCHW32 ||
        filter_meta.format == Format::NCHW32_NCHW4) {
        // g, oc, ic, fh, fw
        FS_SPATIAL = 1;
        FS_IC = FH * FW;
        FS_OC = FS_IC * filter_meta.icpg;
        FS_G = FS_OC * filter_meta.ocpg;
    } else if (filter_meta.format == Format::CHWN4) {
        // g, ic, fh, fw, oc, pack_size
        FS_SPATIAL = filter_meta.ocpg * 4;
        FS_IC = FH * FW * FS_SPATIAL;
        FS_OC = 4;
        FS_G = FS_IC * filter_meta.icpg;
    } else if (filter_meta.format == Format::NCHW88) {
        if (filter_meta.group > 1 && filter_meta.icpg == 1 &&
            src.layout.ndim == 5 && filter_meta.ocpg == 1) {
            FS_SPATIAL = 8;
            FS_IC = FH * FW * FS_SPATIAL;
            FS_OC = FS_IC * filter_meta.icpg;
            FS_G = FS_OC * filter_meta.ocpg;
        } else {
            if (src.layout.ndim == 4 && dst.layout.ndim == 5) {
                FS_IC = 8;
                FS_SPATIAL = filter_meta.icpg * FS_IC;
                FS_OC = FH * FW * FS_SPATIAL;
                FS_G = FS_OC * filter_meta.ocpg / 8;
            } else {
                FS_SPATIAL = 8 * 8;
                FS_IC = FH * FW * FS_SPATIAL;
                FS_OC = FS_IC * filter_meta.icpg / 8;
                FS_G = FS_OC * filter_meta.ocpg / 8;
            }
        }
    } else if (filter_meta.format == Format::NCHW44 ||
               filter_meta.format == Format::NCHW44_DOT) {
        if (filter_meta.group > 1 && filter_meta.icpg == 1 &&
            src.layout.ndim == 5 && filter_meta.ocpg == 1) {
            FS_SPATIAL = 4;
            FS_IC = FH * FW * FS_SPATIAL;
            FS_OC = FS_IC * filter_meta.icpg;
            FS_G = FS_OC * filter_meta.ocpg;
        } else {
            if (src.layout.ndim == 4 && dst.layout.ndim == 5) {
                FS_IC = 4;
                FS_SPATIAL = filter_meta.icpg * FS_IC;
                FS_OC = FH * FW * FS_SPATIAL;
                FS_G = FS_OC * filter_meta.ocpg / 4;
            } else {
                FS_SPATIAL = 4 * 4;
                FS_IC = FH * FW * FS_SPATIAL;
                FS_OC = FS_IC * filter_meta.icpg / 4;
                FS_G = FS_OC * filter_meta.ocpg / 4;
            }
        }
    } else {
        // g, oc, fh, fw, ic
        megdnn_assert(filter_meta.format == Format::NHWC);
        FS_IC = 1;
        FS_SPATIAL = filter_meta.icpg;
        FS_OC = FS_SPATIAL * FH * FW;
        FS_G = FS_OC * filter_meta.ocpg;
    }
    int ph = filter_meta.padding[0], pw = filter_meta.padding[1];
    size_t sh = filter_meta.stride[0], sw = filter_meta.stride[1];
    int dh = filter_meta.dilation[0], dw = filter_meta.dilation[1];
    stype* __restrict sptr = src.compatible_ptr<stype>();
    dtype* __restrict dptr = dst.compatible_ptr<dtype>();

    int h_offset = -ph, w_offset = -pw;
    if (filter_meta.should_flip) {
        h_offset += filter_meta.dilated_spatial[0] - 1;
        w_offset += filter_meta.dilated_spatial[1] - 1;
        dh = -dh;
        dw = -dw;
    }

    auto get_linear_addr = [&filter_meta, &src](ptrdiff_t n, ptrdiff_t c,
                                                ptrdiff_t h, ptrdiff_t w,
                                                const TensorLayout& layout,
                                                bool is_output) -> ptrdiff_t {
        if (filter_meta.format == Format::NCHW) {
            return n * layout.stride[0] + c * layout.stride[1] +
                   h * layout.stride[2] + w * layout.stride[3];
        } else if (filter_meta.format == Format::NHWC) {
            return n * layout.stride[0] + h * layout.stride[1] +
                   w * layout.stride[2] + c * layout.stride[3];
        } else if (filter_meta.format == Format::NCHW8 ||
                   filter_meta.format == Format::NCHW88) {
            if (filter_meta.format == Format::NCHW88 && !is_output &&
                src.layout.ndim == 4) {
                return n * layout.stride[0] + c * layout.stride[1] +
                       h * layout.stride[2] + w * layout.stride[3];
            } else {
                return n * layout.stride[0] + (c / 8) * layout.stride[1] +
                       h * layout.stride[2] + w * layout.stride[3] +
                       (c & 0b111) * layout.stride[4];
            }
        } else if (filter_meta.format == Format::NCHW44 ||
                   filter_meta.format == Format::NCHW44_DOT) {
            if (!is_output && src.layout.ndim == 4) {
                return n * layout.stride[0] + c * layout.stride[1] +
                       h * layout.stride[2] + w * layout.stride[3];
            } else {
                return n * layout.stride[0] + (c / 4) * layout.stride[1] +
                       h * layout.stride[2] + w * layout.stride[3] +
                       (c % 4) * layout.stride[4];
            }
        } else if (filter_meta.format == Format::NCHW32) {
            return n * layout.stride[0] + (c >> 5) * layout.stride[1] +
                   h * layout.stride[2] + w * layout.stride[3] +
                   (c & 0x1F) * layout.stride[4];
        } else if (filter_meta.format == Format::NCHW32_NCHW4) {
            if (is_output) {
                return n * layout.stride[0] + (c / 4) * layout.stride[1] +
                       h * layout.stride[2] + w * layout.stride[3] +
                       (c & 0b11) * layout.stride[4];
            } else {
                return n * layout.stride[0] + (c >> 5) * layout.stride[1] +
                       h * layout.stride[2] + w * layout.stride[3] +
                       (c & 0x1F) * layout.stride[4];
            }
        } else if (filter_meta.format == Format::CHWN4) {
            return (c / 4) * layout.stride[0] + h * layout.stride[1] +
                   w * layout.stride[2] + n * layout.stride[3] +
                   (c % 4) * layout.stride[4];
        } else if (filter_meta.format == Format::NCHW4_NCHW) {
            if (is_output) {
                return n * layout.stride[0] + c * layout.stride[1] +
                       h * layout.stride[2] + w * layout.stride[3];
            } else {
                return n * layout.stride[0] + (c / 4) * layout.stride[1] +
                       h * layout.stride[2] + w * layout.stride[3] +
                       (c & 0b11) * layout.stride[4];
            }
        } else if (filter_meta.format == Format::NCHW4_NCHW32) {
            if (is_output) {
                return n * layout.stride[0] + (c >> 5) * layout.stride[1] +
                       h * layout.stride[2] + w * layout.stride[3] +
                       (c & 0x1F) * layout.stride[4];
            } else {
                return n * layout.stride[0] + (c / 4) * layout.stride[1] +
                       h * layout.stride[2] + w * layout.stride[3] +
                       (c & 0b11) * layout.stride[4];
            }
        } else {
            megdnn_assert(filter_meta.format == Format::NCHW4,
                          "invalid conv format");
            return n * layout.stride[0] + (c / 4) * layout.stride[1] +
                   h * layout.stride[2] + w * layout.stride[3] +
                   (c & 0b11) * layout.stride[4];
        }
    };

    auto get_filter_addr = [&](GroupCounter& gc_out, size_t ic, size_t ic0,
                               size_t fh, size_t fw) {
        if (filter_meta.format == Format::NCHW4 ||
            filter_meta.format == Format::NCHW4_NCHW ||
            filter_meta.format == Format::NCHW4_NCHW32) {
            return gc_out.cur_grp * FS_G + gc_out.cur_off * FS_OC +
                   (ic - ic0) / 4 * FS_IC * 4 +
                   (fh * FW + fw) * FS_SPATIAL * 4 + ((ic - ic0) & 0b11);
        } else if (filter_meta.format == Format::NCHW8) {
            return gc_out.cur_grp * FS_G + gc_out.cur_off * FS_OC +
                   (ic - ic0) / 8 * FS_IC * 8 +
                   (fh * FW + fw) * FS_SPATIAL * 8 + ((ic - ic0) & 0b111);
        } else if (filter_meta.format == Format::NCHW32 ||
                   filter_meta.format == Format::NCHW32_NCHW4) {
            return gc_out.cur_grp * FS_G + gc_out.cur_off * FS_OC +
                   (ic - ic0) / 32 * FS_IC * 32 +
                   (fh * FW + fw) * FS_SPATIAL * 32 + ((ic - ic0) & 0x1F);
        } else if (filter_meta.format == Format::CHWN4) {
            return gc_out.cur_grp * FS_G + gc_out.cur_off * FS_OC +
                   (ic - ic0) / 4 * FS_IC + (fh * FW + fw) * FS_SPATIAL +
                   ((ic - ic0) % 4);
        } else if (filter_meta.format == Format::NCHW88 ||
                   filter_meta.format == Format::NCHW44) {
            size_t pack_c_size = 4_z;
            if(filter_meta.format == Format::NCHW88){
                pack_c_size = 8_z;
            }
            if (src.layout.ndim == 4) {
                // ic < 8, input is nchw
                return gc_out.cur_grp * FS_G +
                       gc_out.cur_off / pack_c_size * FS_OC +
                       (fh * FW + fw) * FS_SPATIAL + (ic - ic0) * FS_IC +
                       gc_out.cur_off % pack_c_size;
            } else if (filter_meta.group > 1 && filter_meta.icpg == 1 &&
                       filter_meta.ocpg == 1 && src.layout.ndim == 5) {
                // dw case
                return gc_out.cur_grp / pack_c_size * FS_G +
                       gc_out.cur_off * FS_OC + (ic - ic0) * FS_IC +
                       (fh * FW + fw) * FS_SPATIAL +
                       gc_out.cur_grp % pack_c_size;
            } else if (src.layout.ndim == 5) {
                // normal case
                return gc_out.cur_grp * FS_G +
                       gc_out.cur_off / pack_c_size * FS_OC +
                       (ic - ic0) / pack_c_size * FS_IC +
                       (fh * FW + fw) * FS_SPATIAL +
                       ((ic - ic0) % pack_c_size) * pack_c_size +
                       gc_out.cur_off % pack_c_size;
            } else {
                megdnn_throw(
                        "nchw88/nchw44 naive not support this input and "
                        "output\n");
            }
        } else if (filter_meta.format == Format::NCHW44_DOT) {
            if (src.layout.ndim == 4) {
                // ic < 4, input is nchw
                return gc_out.cur_grp * FS_G + gc_out.cur_off / 4 * FS_OC +
                       (fh * FW + fw) * FS_SPATIAL + (ic - ic0) * FS_IC +
                       gc_out.cur_off % 4;
            } else if (filter_meta.group > 1 && filter_meta.icpg == 1 &&
                       filter_meta.ocpg == 1 && src.layout.ndim == 5) {
                // dw case
                return gc_out.cur_grp / 4 * FS_G + gc_out.cur_off * FS_OC +
                       (ic - ic0) * FS_IC + (fh * FW + fw) * FS_SPATIAL +
                       gc_out.cur_grp % 4;
            } else if (src.layout.ndim == 5) {
                // normal case
                return gc_out.cur_grp * FS_G + gc_out.cur_off / 4 * FS_OC +
                       (ic - ic0) / 4 * FS_IC + (fh * FW + fw) * FS_SPATIAL +
                       (gc_out.cur_off % 4) * 4 + ((ic - ic0) % 4);
            } else {
                megdnn_throw(
                        "nchw44_dot naive not support this input and output\n");
            }
        } else {
            return gc_out.cur_grp * FS_G + gc_out.cur_off * FS_OC +
                   (ic - ic0) * FS_IC + (fh * FW + fw) * FS_SPATIAL;
        }
    };
    size_t filter_sizes = filter_meta.ocpg * filter_meta.icpg * FH * FW;
    for (size_t n = 0; n < N; ++n) {
        GroupCounter gc_out{filter_meta.ocpg};
        for (size_t oc = 0; oc < OC; ++oc, gc_out.next())
            for (size_t oh = 0; oh < OH; ++oh)
                for (size_t ow = 0; ow < OW; ++ow) {
                    comp_type dval = dptr[get_linear_addr(n, oc, oh, ow,
                                                          dst.layout, true)];
                    ftype* fptr_cur = FilterVisitor::template get_current_ptr(
                            fptr, n, oc, oh, ow, filter_sizes);
                    Strategy::init_dval(dval);

                    for (size_t fh = 0; fh < FH; ++fh)
                        for (size_t fw = 0; fw < FW; ++fw) {
                            size_t ih = sh * oh + fh * dh + h_offset,
                                   iw = sw * ow + fw * dw + w_offset;
                            // here ih and iw are represented in unsigned int
                            // they will become very large if underflow occurs
                            if (ih < IH && iw < IW) {
                                size_t ic0 = gc_out.cur_grp * filter_meta.icpg,
                                       ic1 = ic0 + filter_meta.icpg;
                                for (size_t ic = ic0; ic < ic1; ++ic) {
                                    stype& sval = sptr[get_linear_addr(
                                            n, ic, ih, iw, src.layout, false)];
                                    ftype& fval = fptr_cur[get_filter_addr(
                                            gc_out, ic, ic0, fh, fw)];
                                    Strategy::on(sval, fval, dval,
                                                 src.layout.dtype,
                                                 filter_meta.dtype,
                                                 dst.layout.dtype);
                                }
                            }
                        }
                    Strategy::write(dval,
                                    dptr[get_linear_addr(n, oc, oh, ow,
                                                         dst.layout, true)]);
                }
    }
}

template <typename stype, typename ftype, typename dtype, typename comp_type,
          class Strategy, typename FilterMeta,
          typename FilterVisitor = ConvFilterVisitor>
void compute2d_hwcd4(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                     _megdnn_tensor_out dst, const FilterMeta& filter_meta) {
    // The filter's layout is (G, OC/4, FH, FW, IC, 4) when using mad
    // and (G, OC/4, FH, FW, IC/4, 4, 4) when using dot.
    bool use_dot = false;
    if (src.layout.dtype.enumv() == DTypeEnum::QuantizedS8 ||
        src.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm ||
        (src.layout.dtype.enumv() == DTypeEnum::QuantizedS32 &&
         (filter.layout.dtype.enumv() == DTypeEnum::QuantizedS8 ||
          filter.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm)))
        use_dot = true;

    using Format = param::Convolution::Format;
    megdnn_assert(filter_meta.format == Format::NHWCD4);
    auto N = src.layout.shape[0], IH = src.layout.shape[1],
         IW = src.layout.shape[3];
    auto FH = filter_meta.spatial[0], FW = filter_meta.spatial[1];
    auto OC = dst.layout.shape[2] * 4, OH = dst.layout.shape[1],
         OW = dst.layout.shape[3];
    int ph = filter_meta.padding[0], pw = filter_meta.padding[1];
    size_t sh = filter_meta.stride[0], sw = filter_meta.stride[1];
    int dh = filter_meta.dilation[0], dw = filter_meta.dilation[1];
    stype* __restrict sptr = src.compatible_ptr<stype>();
    ftype* __restrict fptr = filter.compatible_ptr<ftype>();
    dtype* __restrict dptr = dst.compatible_ptr<dtype>();

    megdnn_assert(!filter_meta.should_flip);
    int h_offset = -ph, w_offset = -pw;

    auto get_linear_addr = [](size_t n, size_t c, size_t h, size_t w,
                              const TensorLayout& layout) -> size_t {
        return n * layout.stride[0] + h * layout.stride[1] +
               (c / 4) * layout.stride[2] + w * layout.stride[3] +
               c % 4 * layout.stride[4];
    };

    size_t FS_G, FS_OCB, FS_SPATIAL;
    if (!use_dot && filter.layout.ndim == 5) {
        if (filter_meta.ocpg == 1 && filter_meta.icpg == 1) {
            // chanwise conv, (G/4, 1, FH, FW, 4)
            FS_G = filter.layout.stride[0];
            FS_OCB = 0;
            FS_SPATIAL = 4;
        } else {
            // dense conv, (OC/4, FH, FW, IC, 4)
            FS_G = 0;
            FS_OCB = filter.layout.stride[0];
            FS_SPATIAL = filter.layout.stride[2];
        }
    } else if (!use_dot && filter.layout.ndim == 6) {
        // group conv, (G, OC/4, FH, FW, IC, 4)
        FS_G = filter.layout.stride[0];
        FS_OCB = filter.layout.stride[1];
        FS_SPATIAL = filter.layout.stride[3];
    } else if (use_dot && filter.layout.ndim == 6) {
        // dense conv used dot, (OC/4, FH, FW, IC/4, 4, 4)
        FS_G = 0;
        FS_OCB = filter.layout.stride[0];
        FS_SPATIAL = filter.layout.stride[2];
    } else if (use_dot && filter.layout.ndim == 7) {
        // group conv used dot, (G, OC/4, FH, FW, IC/4, 4, 4)
        FS_G = filter.layout.stride[0];
        FS_OCB = filter.layout.stride[1];
        FS_SPATIAL = filter.layout.stride[3];
    } else if (use_dot && filter.layout.ndim == 5 && filter_meta.ocpg == 1 &&
               filter_meta.icpg == 1) {
        // chanwise conv, (G/4, 1, FH, FW, 4)
        FS_G = filter.layout.stride[0];
        FS_OCB = 0;
        FS_SPATIAL = 4;
    } else {
        megdnn_assert(0, "invalid filter layout");
    }

    auto get_filter_addr = [&use_dot, &FS_G, &FS_OCB, &FS_SPATIAL, &FW,
                            &filter_meta](size_t group, size_t offset,
                                          size_t fh, size_t fw,
                                          size_t c) -> size_t {
        if (filter_meta.ocpg == 1 && filter_meta.icpg == 1) {
            return (group / 4) * FS_G + (fh * FW + fw) * FS_SPATIAL +
                   (group % 4);
        } else if (!use_dot) {
            return group * FS_G + (offset / 4) * FS_OCB +
                   (fh * FW + fw) * FS_SPATIAL + c * 4 + (offset % 4);
        } else {
            megdnn_assert(use_dot);
            return group * FS_G + (offset / 4) * FS_OCB +
                   (fh * FW + fw) * FS_SPATIAL + (c / 4) * 16 +
                   (offset % 4) * 4 + (c % 4);
        }
    };

    size_t filter_sizes = filter_meta.ocpg * filter_meta.icpg * FH * FW;
    for (size_t n = 0; n < N; ++n) {
        GroupCounter gc_out{filter_meta.ocpg};
        for (size_t oc = 0; oc < OC; ++oc, gc_out.next())
            for (size_t oh = 0; oh < OH; ++oh)
                for (size_t ow = 0; ow < OW; ++ow) {
                    comp_type dval =
                            dptr[get_linear_addr(n, oc, oh, ow, dst.layout)];
                    Strategy::init_dval(dval);
                    ftype* fptr_cur = FilterVisitor::template get_current_ptr(
                            fptr, n, oc, oh, ow, filter_sizes);

                    for (size_t fh = 0; fh < FH; ++fh)
                        for (size_t fw = 0; fw < FW; ++fw) {
                            size_t ih = sh * oh + fh * dh + h_offset,
                                   iw = sw * ow + fw * dw + w_offset;
                            // here ih and iw are represented in unsigned int
                            // they will become very large if underflow occurs
                            if (ih < IH && iw < IW) {
                                size_t ic0 = gc_out.cur_grp * filter_meta.icpg,
                                       ic1 = ic0 + filter_meta.icpg;
                                for (size_t ic = ic0; ic < ic1; ++ic) {
                                    stype& sval = sptr[get_linear_addr(
                                            n, ic, ih, iw, src.layout)];
                                    ftype& fval = fptr_cur[get_filter_addr(
                                            gc_out.cur_grp, gc_out.cur_off, fh,
                                            fw, ic - ic0)];
                                    Strategy::on(sval, fval, dval,
                                                 src.layout.dtype,
                                                 filter_meta.dtype,
                                                 dst.layout.dtype);
                                }
                            }
                        }
                    Strategy::write(
                            dval,
                            dptr[get_linear_addr(n, oc, oh, ow, dst.layout)]);
                }
    }
}

//! forward with only filter ptr
template <typename stype, typename ftype, typename dtype, typename comp_type>
void forward(_megdnn_tensor_in src, const ftype* fptr, _megdnn_tensor_out dst,
             const Convolution::CanonizedFilterMeta& filter_meta) {
    megdnn_assert(filter_meta.spatial_ndim == 2);
    megdnn_assert(
            filter_meta.format == param::Convolution::Format::NCHW ||
            filter_meta.format == param::Convolution::Format::NHWC ||
            filter_meta.format == param::Convolution::Format::NCHW88 ||
            filter_meta.format == param::Convolution::Format::NCHW44 ||
            filter_meta.format == param::Convolution::Format::NCHW44_DOT ||
            filter_meta.format == param::Convolution::Format::NCHW4 ||
            filter_meta.format == param::Convolution::Format::NCHW4_NCHW ||
            filter_meta.format == param::Convolution::Format::NCHW4_NCHW32 ||
            filter_meta.format == param::Convolution::Format::NCHW32_NCHW4);
    compute2d<stype, ftype, dtype, comp_type, StrategyFwd>(
            src, const_cast<ftype*>(fptr), dst, filter_meta);
}

//! forward with full filter (for API compatibility)
template <typename stype, typename ftype, typename dtype, typename comp_type>
void forward(_megdnn_tensor_in src, _megdnn_tensor_in filter,
             _megdnn_tensor_out dst,
             const Convolution::CanonizedFilterMeta& filter_meta) {
    if (filter_meta.format == param::Convolution::Format::NHWCD4) {
        return compute2d_hwcd4<stype, ftype, dtype, comp_type, StrategyFwd>(
                src, filter, dst, filter_meta);
    }
    return forward<stype, ftype, dtype, comp_type>(
            src, filter.compatible_ptr<ftype>(), dst, filter_meta);
}

template <typename ftype, typename dtype, typename gtype>
void backward_data(_megdnn_tensor_in filter, _megdnn_tensor_in diff,
                   _megdnn_tensor_out grad,
                   const Convolution::CanonizedFilterMeta& filter_meta) {
    megdnn_assert(grad.layout.is_contiguous());
    memset(grad.raw_ptr, 0, grad.layout.span().dist_byte());
    megdnn_assert(filter_meta.spatial_ndim == 2);
    if (filter_meta.format == param::Convolution::Format::NHWCD4) {
        return compute2d_hwcd4<gtype, ftype, dtype, dtype, StrategyBwdData>(
                grad, filter, diff, filter_meta);
    }
    compute2d<gtype, ftype, dtype, dtype, StrategyBwdData>(
            grad, filter.compatible_ptr<ftype>(), diff, filter_meta);
}

template <typename stype, typename dtype, typename gtype>
void backward_filter(_megdnn_tensor_in src, _megdnn_tensor_in diff,
                     _megdnn_tensor_out grad,
                     const Convolution::CanonizedFilterMeta& filter_meta) {
    megdnn_assert(grad.layout.is_contiguous());
    memset(grad.raw_ptr, 0, grad.layout.span().dist_byte());
    megdnn_assert(filter_meta.spatial_ndim == 2);
    compute2d<stype, gtype, dtype, dtype, StrategyBwdFlt>(
            src, grad.compatible_ptr<gtype>(), diff, filter_meta);
}

template <typename stype, typename ftype, typename dtype, typename comp_type,
          typename FilterMeta, typename FilterVisitor = ConvFilterVisitor>
void forward_bias(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                  _megdnn_tensor_in bias, _megdnn_tensor_out dst,
                  dt_byte* /* workspace_ptr */, const FilterMeta& filter_meta) {
    megdnn_assert(filter_meta.spatial_ndim == 2);
    switch (filter_meta.format) {
        case param::Convolution::Format::NCHW:
        case param::Convolution::Format::NCHW88:
        case param::Convolution::Format::NCHW44:
        case param::Convolution::Format::NCHW44_DOT:
        case param::Convolution::Format::NHWC:
        case param::Convolution::Format::NCHW4:
        case param::Convolution::Format::NCHW4_NCHW:
        case param::Convolution::Format::NCHW4_NCHW32:
        case param::Convolution::Format::NCHW8:
        case param::Convolution::Format::NCHW32:
        case param::Convolution::Format::NCHW32_NCHW4:
        case param::Convolution::Format::CHWN4:
            compute2d<stype, ftype, dtype, comp_type, StrategyFwd, FilterMeta,
                      FilterVisitor>(src, filter.compatible_ptr<ftype>(), dst,
                                     filter_meta);
            break;
        case param::Convolution::Format::NHWCD4:
            compute2d_hwcd4<stype, ftype, dtype, comp_type, StrategyFwd,
                            FilterMeta, FilterVisitor>(src, filter, dst,
                                                       filter_meta);
            break;
        default:
            megdnn_assert_internal(0);
    }

    //! we can not decide with bias.raw_ptr, as non bias the raw_ptr is not
    //! nullptr
    if (bias.layout.ndim != 0) {
        if (dst.layout.eq_shape(bias.layout) &&
            dst.layout.dtype.enumv() == bias.layout.dtype.enumv()) {
            dtype* dst_ptr = dst.compatible_ptr<dtype>();
            dtype* bias_ptr = bias.compatible_ptr<dtype>();
            for (size_t i = 0; i < dst.layout.span().dist_elem(); i++) {
                comp_type val = static_cast<comp_type>(dst_ptr[0]) +
                                static_cast<comp_type>(bias_ptr[0]);
                dst_ptr[0] = val;
                dst_ptr++;
                bias_ptr++;
            }
            return;
        }

        using Format = param::ConvBias::Format;
        switch (filter_meta.format) {
            case Format::NCHW:
            case Format::NCHW4_NCHW: {
                int dst_batch = dst.layout.shape[0];
                int dst_channel = dst.layout.shape[1];
                int chann_stride = dst.layout.shape[2] * dst.layout.shape[3];
                dtype* dst_ptr = dst.compatible_ptr<dtype>();

                for (int batch = 0; batch < dst_batch; ++batch) {
                    for (int chan = 0; chan < dst_channel; ++chan) {
                        dtype bias_val = bias.compatible_ptr<dtype>()[chan];
                        for (int i = 0; i < chann_stride; ++i, ++dst_ptr) {
                            comp_type val = static_cast<comp_type>(dst_ptr[0]) +
                                            static_cast<comp_type>(bias_val);
                            dst_ptr[0] = val;
                        }
                    }
                }
                break;
            };
#define BIAS_ADD_NCHWx(_pack_size)                                        \
    do {                                                                  \
        megdnn_assert(dst.layout.is_contiguous());                        \
        int dst_batch = dst.layout.shape[0];                              \
        int dst_channel = dst.layout.shape[1] * (_pack_size);             \
        int chann_stride = dst.layout.shape[2] * dst.layout.shape[3];     \
        dtype* dst_ptr = dst.compatible_ptr<dtype>();                     \
        for (int batch = 0; batch < dst_batch; ++batch) {                 \
            for (int chan = 0; chan < dst_channel; ++chan) {              \
                dtype bias_val = bias.compatible_ptr<dtype>()[chan];      \
                for (int i = 0; i < chann_stride; ++i) {                  \
                    int idx = batch * dst_channel * chann_stride +        \
                              (chan / (_pack_size)) *                     \
                                      (chann_stride * (_pack_size)) +     \
                              i * (_pack_size) + chan % (_pack_size);     \
                    dst_ptr[idx] = static_cast<comp_type>(dst_ptr[idx]) + \
                                   static_cast<comp_type>(bias_val);      \
                }                                                         \
            }                                                             \
        }                                                                 \
    } while (0)
            case Format::NCHW44:
            case Format::NCHW44_DOT:
            case Format::NCHW32_NCHW4:
            case Format::NCHW4: {
                BIAS_ADD_NCHWx(4);
                break;
            };
            case Format::NCHW8: {
                BIAS_ADD_NCHWx(8);
                break;
            };
            case Format::NCHW4_NCHW32: 
            case Format::NCHW32: {
                BIAS_ADD_NCHWx(32);
                break;
            };
            case Format::NCHW88: {
                BIAS_ADD_NCHWx(8);
                break;
            };
#define BIAS_ADD_CHWNx(_pack_size)                                            \
    do {                                                                      \
        megdnn_assert(dst.layout.is_contiguous());                            \
        int dst_batch = dst.layout.shape[3];                                  \
        int dst_channel = dst.layout.shape[0] * (_pack_size);                 \
        int chann_stride =                                                    \
                dst.layout.shape[1] * dst.layout.shape[2] * dst_batch;        \
        dtype* dst_ptr = dst.compatible_ptr<dtype>();                         \
        for (int chan = 0; chan < dst_channel; ++chan) {                      \
            dtype bias_val = bias.compatible_ptr<dtype>()[chan];              \
            for (int i = 0; i < chann_stride; ++i) {                          \
                int idx =                                                     \
                        (chan / (_pack_size)) * chann_stride * (_pack_size) + \
                        i * (_pack_size) + chan % (_pack_size);               \
                dst_ptr[idx] = static_cast<comp_type>(dst_ptr[idx]) +         \
                               static_cast<comp_type>(bias_val);              \
            }                                                                 \
        }                                                                     \
    } while (0)
            case Format::CHWN4: {
                BIAS_ADD_CHWNx(4);
                break;
            }
            case Format::NHWC: {
                int dst_nhw = dst.layout.shape[0] * dst.layout.shape[1] *
                              dst.layout.shape[2];
                int dst_channel = dst.layout.shape[3];
                dtype* dst_ptr = dst.compatible_ptr<dtype>();

                for (int nhw = 0; nhw < dst_nhw; ++nhw) {
                    for (int chan = 0; chan < dst_channel; ++chan, ++dst_ptr) {
                        dtype bias_val = bias.compatible_ptr<dtype>()[chan];
                        comp_type val = static_cast<comp_type>(dst_ptr[0]) +
                                        static_cast<comp_type>(bias_val);
                        dst_ptr[0] = val;
                    }
                }
                break;
            };
            case Format::NHWCD4: {
                dtype* bias_ptr = bias.compatible_ptr<dtype>();
                dtype* dst_ptr = dst.compatible_ptr<dtype>();
                for (size_t n = 0; n < dst.layout[0]; n++) {
                    for (size_t h = 0; h < dst.layout[1]; h++) {
                        for (size_t cb = 0; cb < dst.layout[2]; cb++) {
                            for (size_t w = 0; w < dst.layout[3]; w++) {
                                for (size_t i = 0; i < 4; i++) {
                                    auto ptr = dst_ptr +
                                               n * dst.layout.stride[0] +
                                               h * dst.layout.stride[1] +
                                               cb * dst.layout.stride[2] +
                                               w * dst.layout.stride[3] +
                                               i * dst.layout.stride[4];
                                    comp_type val =
                                            static_cast<comp_type>(ptr[0]) +
                                            static_cast<comp_type>(
                                                    bias_ptr[cb * 4 + i]);
                                    ptr[0] = val;
                                }
                            }
                        }
                    }
                }
                break;
            };
            default:
                megdnn_assert_internal(0);
        }
    }
}

}  // namespace convolution
}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
