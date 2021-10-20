/**
 * \file dnn/src/naive/elemwise/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/common/elemwise/opr_impl_helper.h"

namespace megdnn {
namespace naive {

class ElemwiseForwardImpl : public ElemwiseForwardImplHelper {
#include "src/common/elemwise/opr_impl_class_def.inl"
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
