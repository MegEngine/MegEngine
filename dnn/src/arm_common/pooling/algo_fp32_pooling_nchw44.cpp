/**
 * \file dnn/src/arm_common/pooling/algo_fp32_pooling_nchw44.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megdnn/opr_param_defs.h"
#include "src/arm_common/pooling/algo.h"
#include "src/arm_common/pooling/kern_fp32_pooling_nchw44.h"

#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_fp32_pooling_nchw44)

namespace megdnn {
namespace arm_common {
bool PoolingImpl::AlgoFp32ModexStridexNCHW44::usable(
        const PoolingKernSizeParam& param) const {
    uint32_t sh = param.stride[0];
    uint32_t sw = param.stride[1];
    uint32_t fh = param.filter[0];
    uint32_t fw = param.filter[1];

    bool avaible = param.src_type.enumv() == DTypeEnum::Float32 &&
                   param.format == Param::Format::NCHW44 &&
                   (param.mode == Mode::MAX || param.mode == Mode::AVERAGE) &&
                   fh == fw && sh == sw &&
                   (fh == 2 || fh == 3 || fh == 4 || fh == 5) &&
                   (sh == 1 || sh == 2);
    return avaible;
}

void PoolingImpl::AlgoFp32ModexStridexNCHW44::exec(
        const PoolingKernParam& param) const {
    int ih = param.isz[0];
    int iw = param.isz[1];
    int oh = param.osz[0];
    int ow = param.osz[1];
    int n = param.n;
    int ic = param.ic;
    int ph = param.padding[0];
    int pw = param.padding[1];
    int sh = param.stride[0];
    int fh = param.filter[0];

    void* src_ptr = param.src_ptr;
    void* dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(filter, stride, mode)                                   \
    MIDOUT_BEGIN(megdnn_arm_common_fp32_pooling_nchw44, midout_iv(0),         \
                 midout_iv(#filter #stride #mode##_hash)) {                   \
        auto run = [ih, iw, oh, ow, ph, pw, src_ptr, dst_ptr](size_t index,   \
                                                              size_t) {       \
            const int c_idx = index;                                          \
            pooling_fp32_nchw44<filter, stride, mode>(                        \
                    static_cast<const float*>(src_ptr) + c_idx * ih * iw * 4, \
                    static_cast<float*>(dst_ptr) + c_idx * oh * ow * 4, ih,   \
                    iw, oh, ow, ph, pw);                                      \
        };                                                                    \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle),      \
                n* ic, run);                                                  \
    }                                                                         \
    MIDOUT_END();

#define DISPATCH_MODE(filter, stride)                                  \
    switch (param.mode) {                                              \
        case PoolingBase::Mode::MAX:                                   \
            DISPATCH_FUNC(filter, stride, PoolingBase::Mode::MAX);     \
            break;                                                     \
        case PoolingBase::Mode::AVERAGE:                               \
            DISPATCH_FUNC(filter, stride, PoolingBase::Mode::AVERAGE); \
            break;                                                     \
        default:                                                       \
            megdnn_assert(0, "invalid mode %u",                        \
                          static_cast<uint32_t>(param.mode));          \
    }

#define DISPATCH_STRIDE(filter)                        \
    switch (sh) {                                      \
        case 1:                                        \
            DISPATCH_MODE(filter, 1);                  \
            break;                                     \
        case 2:                                        \
            DISPATCH_MODE(filter, 2);                  \
            break;                                     \
        default:                                       \
            megdnn_assert(0, "invalid stride %d", sh); \
    }

#define DISPATCH_FILTER()                              \
    switch (fh) {                                      \
        case 2:                                        \
            DISPATCH_STRIDE(2);                        \
            break;                                     \
        case 3:                                        \
            DISPATCH_STRIDE(3);                        \
            break;                                     \
        case 4:                                        \
            DISPATCH_STRIDE(4);                        \
            break;                                     \
        case 5:                                        \
            DISPATCH_STRIDE(5);                        \
            break;                                     \
        default:                                       \
            megdnn_assert(0, "invalid filter %d", fh); \
    }

    DISPATCH_FILTER()

#undef DISPATCH_FILTER
#undef DISPATCH_STRIDE
#undef DISPATCH_MODE
#undef DISPATCH_FUNC
}

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen