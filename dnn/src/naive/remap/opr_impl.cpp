/**
 * \file dnn/src/naive/remap/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/naive/remap/opr_impl.h"
#include "src/common/cv/helper.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;

namespace {
template <param::Remap::Format format>
inline int get_offset(int, int, int, int, int, int);

template <>
inline int get_offset<param::Remap::Format::NCHW>(int height, int width,
                                                  int channel, int h, int w,
                                                  int) {
    return channel * h * w + height * w + width;
}

template <>
inline int get_offset<param::Remap::Format::NHWC>(int height, int width,
                                                  int channel, int, int w,
                                                  int c) {
    return height * w * c + width * c + channel;
}

template <typename DataType, param::Remap::Format format,
          param::Remap::BorderMode bordertype>
struct GetSrcData {
    static inline DataType get(const DataType* src, int height, int width,
                               int channel, int h, int w, int c, float,
                               std::function<DataType(float)>) {
        height = megcv::border_interpolate<bordertype>(height, h);
        width = megcv::border_interpolate<bordertype>(width, w);
        return src[get_offset<format>(height, width, channel, h, w, c)];
    }
};

template <typename DataType, param::Remap::Format format>
struct GetSrcData<DataType, format, param::Remap::BorderMode::CONSTANT> {
    static inline DataType get(const DataType* src, int height, int width,
                               int channel, int h, int w, int c, float scalar,
                               std::function<DataType(float)> round) {
        return (height >= 0 && height < h && width >= 0 && width < w)
                       ? src[get_offset<format>(height, width, channel, h, w,
                                                c)]
                       : static_cast<DataType>(round(scalar));
    }
};

template <typename DataType, param::Remap::Format format,
          param::Remap::BorderMode bordertype>
void remap_LINEAR(const DataType* src, const float* map_xy, DataType* dst,
                  int N, int C, int IH, int IW, int OH, int OW, float scalar,
                  std::function<DataType(float)> round) {
    for (int n = 0; n < N;
         ++n, src += C * IH * IW, dst += C * OH * OW, map_xy += OH * OW * 2) {
        for (int h = 0; h < OH; ++h) {
            for (int w = 0; w < OW; ++w) {
                float index_col = map_xy[h * OW * 2 + w * 2 + 0];
                float index_row = map_xy[h * OW * 2 + w * 2 + 1];
                int col = static_cast<int>(floor(index_col));
                int row = static_cast<int>(floor(index_row));
                float v = index_col - col;
                float u = index_row - row;
                float one = 1.f;
                for (int c = 0; c < C; ++c) {
                    DataType a00 =
                            GetSrcData<DataType, format, bordertype>::get(
                                    src, row + 0, col + 0, c, IH, IW, C, scalar,
                                    round);
                    DataType a01 =
                            GetSrcData<DataType, format, bordertype>::get(
                                    src, row + 0, col + 1, c, IH, IW, C, scalar,
                                    round);
                    DataType a10 =
                            GetSrcData<DataType, format, bordertype>::get(
                                    src, row + 1, col + 0, c, IH, IW, C, scalar,
                                    round);
                    DataType a11 =
                            GetSrcData<DataType, format, bordertype>::get(
                                    src, row + 1, col + 1, c, IH, IW, C, scalar,
                                    round);

                    dst[get_offset<format>(h, w, c, OH, OW, C)] =
                            static_cast<DataType>(
                                    round(a00 * (one - u) * (one - v) +
                                          a01 * (one - u) * v +
                                          a10 * (one - v) * u + a11 * u * v));
                }
            }
        }
    }
}

template <typename DataType, DTypeCategory cat>
struct Round {
    static inline DataType round(float x) { return std::round(x); }
};

template <typename DataType>
struct Round<DataType, DTypeCategory::FLOAT> {
    static inline DataType round(float x) { return static_cast<DataType>(x); }
};

}  // namespace

void RemapImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in map_xy,
                     _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, map_xy.layout, dst.layout, workspace.size);
    int N, C, IH, IW, OH, OW;
    if (param().format == param::Remap::Format::NCHW) {
        N = src.layout.shape[0];
        C = src.layout.shape[1];
        IH = src.layout.shape[2];
        IW = src.layout.shape[3];
    } else if (param().format == param::Remap::Format::NHWC) {
        N = src.layout.shape[0];
        C = src.layout.shape[3];
        IH = src.layout.shape[1];
        IW = src.layout.shape[2];
    } else {
        megdnn_throw("unsupported format");
    }
    OH = map_xy.layout.shape[1];
    OW = map_xy.layout.shape[2];
    switch (src.layout.dtype.enumv()) {
#define cb(dt, fmt, border, interpolation)                                 \
    if (param().format == param::Remap::Format::fmt &&                     \
        param().border_type == param::Remap::BorderMode::border &&         \
        param().imode == param::Remap::InterpolationMode::interpolation) { \
        using ctype = DTypeTrait<dt>::ctype;                               \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                      \
                (remap_##interpolation<ctype, param::Remap::Format::fmt,   \
                                       param::Remap::BorderMode::border>(  \
                        src.compatible_ptr<ctype>(),                       \
                        map_xy.compatible_ptr<dt_float32>(),               \
                        dst.compatible_ptr<ctype>(), N, C, IH, IW, OH, OW, \
                        param().scalar,                                    \
                        Round<ctype, DTypeTrait<dt>::category>::round)));  \
        break;                                                             \
    }

#define support_dtype(dt)                                                   \
    case DTypeTrait<dt>::enumv: {                                           \
        cb(dt, NCHW, CONSTANT, LINEAR);                                     \
        cb(dt, NCHW, REPLICATE, LINEAR);                                    \
        cb(dt, NCHW, REFLECT, LINEAR);                                      \
        cb(dt, NCHW, REFLECT_101, LINEAR);                                  \
        cb(dt, NCHW, WRAP, LINEAR);                                         \
        cb(dt, NHWC, CONSTANT, LINEAR);                                     \
        cb(dt, NHWC, REPLICATE, LINEAR);                                    \
        cb(dt, NHWC, REFLECT, LINEAR);                                      \
        cb(dt, NHWC, REFLECT_101, LINEAR);                                  \
        cb(dt, NHWC, WRAP, LINEAR);                                         \
        megdnn_throw(                                                       \
                "format, border type or imode is incorrect in remap navie " \
                "with dtype = " #dt);                                       \
    }

        support_dtype(dtype::Float32);
        MEGDNN_INC_FLOAT16(support_dtype(dtype::Float16));
        support_dtype(dtype::Int8);
        support_dtype(dtype::Uint8);
#undef cb
#undef support_dtype

        default:
            megdnn_throw("unsupported dtype in remap naive\n");
    }
}
