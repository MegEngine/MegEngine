/**
 * \file dnn/src/naive/type_cvt/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class TypeCvtImpl : public TypeCvt {
public:
    using TypeCvt::TypeCvt;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) override;
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
