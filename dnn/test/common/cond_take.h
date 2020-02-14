/**
 * \file dnn/test/common/cond_take.h
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
#include "./checker.h"

namespace megdnn {
namespace test {
class CondTakeTestcase {
    std::unique_ptr<uint8_t> m_mem;
    CondTake::Param m_param;
    TensorND m_data, m_mask;

    CondTakeTestcase(CondTake::Param param, const TensorLayout& data,
                     const TensorLayout& mask)
            : m_param{param}, m_data{nullptr, data}, m_mask{nullptr, mask} {}

public:
    //! pair of (data, idx)
    using Result =
            std::pair<std::shared_ptr<TensorND>, std::shared_ptr<TensorND>>;
    Result run(CondTake* opr);
    static std::vector<CondTakeTestcase> make();
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
