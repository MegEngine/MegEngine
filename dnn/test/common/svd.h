/**
 * \file dnn/test/common/svd.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "./checker.h"
#include "megdnn/oprs.h"

namespace megdnn {
namespace test {
class SVDTestcase {
    std::unique_ptr<dt_float32> m_mem;

    SVDTestcase(const SVDForward::Param& param, const TensorLayout& mat)
            : m_param{param}, m_mat{nullptr, mat} {}

public:
    SVDForward::Param m_param;
    TensorND m_mat;
    struct Result {
        std::shared_ptr<TensorND> u;
        std::shared_ptr<TensorND> s;
        std::shared_ptr<TensorND> vt;
        std::shared_ptr<TensorND> recovered_mat;
    };
    Result run(SVDForward* opr);
    static std::vector<SVDTestcase> make();
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
