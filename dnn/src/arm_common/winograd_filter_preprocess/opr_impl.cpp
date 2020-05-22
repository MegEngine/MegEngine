/**
 * \file dnn/src/arm_common/winograd_filter_preprocess/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/winograd_filter_preprocess/opr_impl.h"
#include "src/arm_common/handle.h"
#include "src/common/utils.h"
#include "src/arm_common/conv_bias/fp32/strategy.h"
#include "src/arm_common/conv_bias/int8/strategy.h"
#include "src/arm_common/conv_bias/f16/strategy.h"

#include "midout.h"
MIDOUT_DECL(megdnn_arm_common_winograd_filter_preprocess)

using namespace megdnn;
using namespace arm_common;

void WinogradFilterPreprocessImpl::exec(_megdnn_tensor_in src,
                                        _megdnn_tensor_out dst,
                                        _megdnn_workspace workspace) {
    using namespace winograd;
    check_exec(src.layout, dst.layout, workspace.size);

    size_t flt_start = 0;
    size_t group = 1;
    if (src.layout.ndim == 5) {
        flt_start = 1;
        group = src.layout[0];
    }
    size_t OC = src.layout[flt_start], IC = src.layout[flt_start + 1],
           FW = src.layout[flt_start + 3];
    size_t m = param().output_block_size;

    bool execed = false;

#define DISPATCH(_strategy, _format, ...)                                    \
    MIDOUT_BEGIN(megdnn_arm_common_winograd_filter_preprocess,               \
                 ##__VA_ARGS__) {                                            \
        if (param().format == _format) {                                     \
            for (size_t g = 0; g < group; g++) {                             \
                auto run = [=]() {                                           \
                    _strategy strategy(src.layout.dtype, src.layout.dtype,   \
                                       src.layout.dtype);                    \
                    megdnn::winograd::ConvBias<_strategy, _format>(          \
                            strategy, 1, 1, 1, 1, 1)                         \
                            .filter_process(src_ptr, dst_ptr, workspace_ptr, \
                                            OC, IC);                         \
                };                                                           \
                MEGDNN_DISPATCH_CPU_KERN_OPR(run());                         \
                src_ptr += src.layout.stride[0];                             \
                dst_ptr += dst.layout.stride[0];                             \
            }                                                                \
            execed = true;                                                   \
        }                                                                    \
    }                                                                        \
    MIDOUT_END();

    if (src.layout.dtype.enumv() == DTypeEnum::Float32) {
        const float* src_ptr = src.ptr<float>();
        float* dst_ptr = dst.ptr<float>();
        float* workspace_ptr = workspace.ptr<float>();
        if (FW == 3) {
            if (m == 2) {
                DISPATCH(winograd_2x3_4x4_f, param::Winograd::Format::MK4, 0,
                         0);
            } else if (m == 6) {
                DISPATCH(winograd_6x3_1x1_f, param::Winograd::Format::DEFAULT,
                         0, 1);
                DISPATCH(winograd_6x3_4x4_f, param::Winograd::Format::MK4, 0,
                         2);
            }
        } else if (FW == 4) {
            if (m == 5) {
                DISPATCH(winograd_5x4_1x1_f, param::Winograd::Format::DEFAULT,
                         0, 3);
            }
        } else if (FW == 5) {
            if (m == 4) {
                DISPATCH(winograd_4x5_1x1_f, param::Winograd::Format::DEFAULT,
                         0, 4);
            }
        }
    }
    if (src.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {
        const dt_int8* src_ptr = src.compatible_ptr<dt_int8>();
        dt_int16* dst_ptr = dst.compatible_ptr<dt_int16>();
        dt_int16* workspace_ptr = workspace.ptr<dt_int16>();
        if (FW == 3) {
            if (m == 2) {
                DISPATCH(winograd_2x3_8x8_s8, param::Winograd::Format::MK8, 1,
                         0);
            }
        }
    }
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (src.layout.dtype.enumv() == DTypeEnum::Float16) {
        const dt_float16* src_ptr = src.ptr<dt_float16>();
        dt_float16* dst_ptr = dst.ptr<dt_float16>();
        dt_float16* workspace_ptr = workspace.ptr<dt_float16>();
        if (FW == 3) {
            if (m == 2) {
                DISPATCH(winograd_2x3_4x4_f16, param::Winograd::Format::DEFAULT,
                         2, 0);
                DISPATCH(winograd_2x3_8x8_f16, param::Winograd::Format::MK8, 2,
                         1);
            } else if (m == 6) {
                DISPATCH(winograd_6x3_1x1_f16, param::Winograd::Format::DEFAULT,
                         2, 2);
            }
        } else if (FW == 5) {
            if (m == 4) {
                DISPATCH(winograd_4x5_1x1_f16, param::Winograd::Format::DEFAULT,
                         2, 3);
            }
        }
    }
#endif
#undef DISPATCH

    megdnn_assert(execed,
                  "Unsupport winograd filter preprocess. m: %zu src: %s", m,
                  src.layout.to_string().c_str());
}

// vim: syntax=cpp.doxygen
