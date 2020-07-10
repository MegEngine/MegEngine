/**
 * \file dnn/src/naive/winograd_filter_preprocess/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/naive/winograd_filter_preprocess/opr_impl.h"
#include "src/common/utils.h"
#include "src/common/winograd/winograd_helper.h"
#include "src/naive/handle.h"

#include "midout.h"
MIDOUT_DECL(megdnn_naive_winograd_filter_preprocess)

using namespace megdnn;
using namespace naive;

void WinogradFilterPreprocessImpl::exec(_megdnn_tensor_in src,
                                        _megdnn_tensor_out dst,
                                        _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    
    //! nchw88 group conv
    size_t flt_start = 0;
    size_t pack_c_size = 1;
    size_t group = 1;
    //! group conv
    if (src.layout.ndim == 5) {
        flt_start = 1;
        group = src.layout[0];
        //! nchw88 dense conv
    } else if (src.layout.ndim == 6) {
        pack_c_size = src.layout[5];
        //! nchw88 group conv
    } else if (src.layout.ndim == 7) {
        flt_start = 1;
        group = src.layout[0];
        pack_c_size = src.layout[6];
    }
    size_t OC = src.layout[flt_start] * pack_c_size,
           IC = src.layout[flt_start + 1] * pack_c_size,
           FW = src.layout[flt_start + 3];

    size_t m = param().output_block_size;

    bool execed = false;

#define cb(_ctype, _dst_type, _input_filter_compute_type,                    \
           _output_compute_type, _format, rescale)                           \
    if (param().format == _format) {                                         \
        return winograd::StrategyHelper<                                     \
                _ctype, _dst_type, _input_filter_compute_type,               \
                _output_compute_type, param::ConvBias::Format::NCHW,         \
                _format>::filter(src_ptr, dst_ptr, workspace_ptr, OC, IC, 0, \
                                 OC, m, FW, interp_points, src.layout.dtype, \
                                 rescale);                                   \
    }

#define DISPATCH_FORMAT_MK4(_ctype, _dst_type, _input_filter_compute_type,  \
                            _output_compute_type, _rescale)                 \
    cb(_ctype, _dst_type, _input_filter_compute_type, _output_compute_type, \
       param::Winograd::Format::DEFAULT, _rescale);                         \
    cb(_ctype, _dst_type, _input_filter_compute_type, _output_compute_type, \
       param::Winograd::Format::MK4, _rescale);

#define DISPATCH_FORMAT_MK8(_ctype, _dst_type, _input_filter_compute_type,  \
                            _output_compute_type, _rescale)                 \
    cb(_ctype, _dst_type, _input_filter_compute_type, _output_compute_type, \
       param::Winograd::Format::DEFAULT, _rescale);                         \
    cb(_ctype, _dst_type, _input_filter_compute_type, _output_compute_type, \
       param::Winograd::Format::MK8, _rescale);

#define DISPATCH_KERNEL(_ctype, _dst_type, _input_filter_compute_type,     \
                        _output_compute_type, _kern, _rescale, ...)        \
    const _ctype* src_ptr = src.compatible_ptr<_ctype>();                  \
    _input_filter_compute_type* dst_ptr =                                  \
            dst.compatible_ptr<_input_filter_compute_type>();              \
    _input_filter_compute_type* workspace_ptr =                            \
            workspace.ptr<_input_filter_compute_type>();                   \
    MIDOUT_BEGIN(megdnn_naive_winograd_filter_preprocess, ##__VA_ARGS__) { \
        for (size_t g = 0; g < group; g++) {                               \
            auto run = [=]() {                                             \
                _kern(_ctype, _dst_type, _input_filter_compute_type,       \
                      _output_compute_type, _rescale);                     \
            };                                                             \
            MEGDNN_DISPATCH_CPU_KERN_OPR(run());                           \
            src_ptr += src.layout.stride[0];                               \
            dst_ptr += dst.layout.stride[0];                               \
        }                                                                  \
        execed = true;                                                     \
    }                                                                      \
    MIDOUT_END();

#define DISPATCH_DTYPE(_midout_tag)                                          \
    if (src.layout.dtype.enumv() == DTypeEnum::Float32) {                    \
        DISPATCH_KERNEL(dt_float32, dt_float32, dt_float32, dt_float32,      \
                        DISPATCH_FORMAT_MK4, 1.0f, _midout_tag, 0);          \
    }                                                                        \
    if (src.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {                \
        DISPATCH_KERNEL(dt_int8, dt_int8, dt_int16, dt_int32,                \
                        DISPATCH_FORMAT_MK8, 2.0f, _midout_tag, 1);          \
    }                                                                        \
    MEGDNN_INC_FLOAT16(if (src.layout.dtype.enumv() == DTypeEnum::Float16) { \
        DISPATCH_KERNEL(dt_float16, dt_float16, dt_float16, dt_float16,      \
                        DISPATCH_FORMAT_MK8, 1.0f, _midout_tag, 2);          \
    })

    if (src.layout.ndim <= 5) {
        //! dispatch_dtype with consider layout and format.
        if (FW == 3) {
            if (m == 2) {
                std::vector<float> interp_points = {0, 1, -1};
                DISPATCH_DTYPE(0);
            } else if (m == 6) {
                std::vector<float> interp_points = {0, 1, -1, 2, -2, 0.5, -0.5};
                DISPATCH_DTYPE(1);
            }
        } else if (FW == 4) {
            if (m == 5) {
                std::vector<float> interp_points = {0, 0.5, -0.5, 1, -1, 2, -2};
                DISPATCH_DTYPE(2);
            }
        } else if (FW == 5) {
            if (m == 4) {
                std::vector<float> interp_points = {0, 1, -1, 0.5, -0.5, 2, -2};
                DISPATCH_DTYPE(3);
            }
        }
#undef cb
#undef DISPATCH_FORMAT_MK4
#undef DISPATCH_FORMAT_MK8
#undef DISPATCH_DTYPE
    } else {
        megdnn_assert(src.layout.ndim == 6 || src.layout.ndim == 7);
#define cb(_ctype, _dst_type, _input_filter_compute_type,                    \
           _output_compute_type, _format, rescale)                           \
    if (param().format == _format) {                                         \
        return winograd::StrategyHelper<                                     \
                _ctype, _dst_type, _input_filter_compute_type,               \
                _output_compute_type, param::ConvBias::Format::NCHW88,       \
                _format>::filter(src_ptr, dst_ptr, workspace_ptr, OC, IC, 0, \
                                 OC, m, FW, interp_points, src.layout.dtype, \
                                 rescale);                                   \
    }

#define DISPATCH_FORMAT_MK8(_ctype, _dst_type, _input_filter_compute_type,  \
                            _output_compute_type, _rescale)                 \
    cb(_ctype, _dst_type, _input_filter_compute_type, _output_compute_type, \
       param::Winograd::Format::MK8, _rescale);

#define DISPATCH_DTYPE(_midout_tag)                                     \
    if (src.layout.dtype.enumv() == DTypeEnum::Float32) {               \
        DISPATCH_KERNEL(dt_float32, dt_float32, dt_float32, dt_float32, \
                        DISPATCH_FORMAT_MK8, 1.0f, _midout_tag, 0);     \
    }
        if (pack_c_size == 8) {  //! NCHW88
            if (FW == 3) {
                if (m == 2) {
                    std::vector<float> interp_points = {0, 1, -1};
                    DISPATCH_DTYPE(4);
                } else if (m == 6) {
                    std::vector<float> interp_points = {0,  1,   -1,  2,
                                                        -2, 0.5, -0.5};
                    DISPATCH_DTYPE(5);
                }
            }
#undef cb
#undef DISPATCH_DTYPE
        }
        else if (pack_c_size == 4) {  //! NCHW44
#define cb(_ctype, _dst_type, _input_filter_compute_type,                    \
           _output_compute_type, _format, rescale)                           \
    if (param().format == _format) {                                         \
        return winograd::StrategyHelper<                                     \
                _ctype, _dst_type, _input_filter_compute_type,               \
                _output_compute_type, param::ConvBias::Format::NCHW44,       \
                _format>::filter(src_ptr, dst_ptr, workspace_ptr, OC, IC, 0, \
                                 OC, m, FW, interp_points, src.layout.dtype, \
                                 rescale);                                   \
    }

#define DISPATCH_FORMAT_MK4(_ctype, _dst_type, _input_filter_compute_type,  \
                            _output_compute_type, _rescale)                 \
    cb(_ctype, _dst_type, _input_filter_compute_type, _output_compute_type, \
       param::Winograd::Format::MK4, _rescale);

#define DISPATCH_DTYPE(_midout_tag)                                     \
    if (src.layout.dtype.enumv() == DTypeEnum::Float32) {               \
        DISPATCH_KERNEL(dt_float32, dt_float32, dt_float32, dt_float32, \
                        DISPATCH_FORMAT_MK4, 1.0f, _midout_tag, 0);     \
    }                                                                   \
    if (src.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {           \
        if (param().format == param::Winograd::Format::MK4) {           \
            DISPATCH_KERNEL(dt_int8, dt_int8, dt_float32, dt_float32,   \
                            DISPATCH_FORMAT_MK4, 1.0f, _midout_tag, 0); \
        } else if (param().format == param::Winograd::Format::MK8) {    \
            DISPATCH_KERNEL(dt_int8, dt_int8, dt_int16, dt_int32,       \
                            DISPATCH_FORMAT_MK8, 2.0f, _midout_tag, 0); \
        }                                                               \
    }
            if (FW == 3) {
                if (m == 2) {
                    std::vector<float> interp_points = {0, 1, -1};
                    DISPATCH_DTYPE(6);
                } else if (m == 6) {
                    std::vector<float> interp_points = {0,  1,   -1,  2,
                                                        -2, 0.5, -0.5};
                    DISPATCH_DTYPE(7);
                } else if (m == 7) {
                    std::vector<float> interp_points = {0,  1,   -1,   2,
                                                        -2, 0.5, -0.5, 1.5};
                    DISPATCH_DTYPE(8);
                }
            }
#undef cb
#undef DISPATCH_FORMAT_MK8
#undef DISPATCH_FORMAT_MK4
#undef DISPATCH_KERNEL
#undef DISPATCH_DTYPE
        }
    }

    megdnn_assert(execed,
                  "Unsupport winograd filter preprocess. m: %zu src: %s", m,
                  src.layout.to_string().c_str());
}

// vim: syntax=cpp.doxygen
