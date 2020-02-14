/**
 * \file dnn/src/naive/winograd_filter_preprocess/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
namespace naive {

class WinogradFilterPreprocessImpl : public WinogradFilterPreprocess {
public:
    using WinogradFilterPreprocess::WinogradFilterPreprocess;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) override;
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
