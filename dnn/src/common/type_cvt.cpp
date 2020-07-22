/**
 * \file dnn/src/common/type_cvt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void TypeCvt::check_exec(const TensorLayout &src, const TensorLayout &dst) {
    megdnn_assert_contiguous(dst);
    megdnn_assert_eq_shape(src, dst);
    auto cat = src.dtype.category();
    megdnn_assert(cat == DTypeCategory::FLOAT || cat == DTypeCategory::INT ||
                  cat == DTypeCategory::QUANTIZED ||
                  cat == DTypeCategory::BOOL);
    cat = dst.dtype.category();
    megdnn_assert(cat == DTypeCategory::FLOAT || cat == DTypeCategory::INT ||
                  cat == DTypeCategory::QUANTIZED ||
                  cat == DTypeCategory::BOOL);
}

} // namespace megdnn

// vim: syntax=cpp.doxygen
