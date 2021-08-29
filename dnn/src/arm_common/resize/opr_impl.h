/**
 * \file dnn/src/arm_common/resize/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "megdnn/oprs.h"
#include "src/fallback/resize/opr_impl.h"

namespace megdnn {
namespace arm_common {
class ResizeImpl : public fallback::ResizeImpl {
public:
    using fallback::ResizeImpl::ResizeImpl;

    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&,
                                  const TensorLayout&) override {
        return 0;
    }

private:
    template <typename ctype>
    void kern_nchw44_fp32(const KernParam<ctype>& kern_param);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    template <typename ctype>
    void kern_nchw88_fp16(const KernParam<ctype>& kern_param);
#endif
    void kern_nearest_upsample2_pack_simd_width(_megdnn_tensor_in src,
                                                _megdnn_tensor_out dst);
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
